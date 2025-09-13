from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from robot_rl.modules import ActorCriticDMO
from robot_rl.storage import RolloutStorage
from robot_rl.utils import string_to_callable

class DMO:
    """Decoupled forward-backward Model-based policy Optimization algorithm (https://machines-in-motion.github.io/DMO/)."""

    policy: ActorCriticDMO
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCriticDMO,
        mode: Literal["bptt", "shac"] = "shac",
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-4,
        actor_lr=None,
        critic_lr=None,
        dynamics_lr=None,
        entropy_lr=None,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # DMO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizers
        actor_params = list(self.policy.actor.parameters()) + [self.policy.std]
        params = [
            {"params": actor_params, "lr": actor_lr, "name": "actor"},
            {"params": self.policy.dynamics.parameters(), "lr": dynamics_lr, "name": "dynamics"},
        ]
        if self.mode != "bptt":
            params.append(
                {"params": self.policy.critic.parameters(), "lr": critic_lr, "name": "critic"},
            )
            params.append(
                {"params": self.policy.alpha, "lr": entropy_lr, "name": "entropy"},
            )
            opt_cls = optim.AdamW
        else:
            opt_cls = optim.Adam
        self.optimizer = opt_cls(params, betas=(0.7, 0.95))

        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # DMO parameters
        self.mode = mode
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma if self.mode != "bptt" else 1.0
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.actor_lr = actor_lr if actor_lr is not None else learning_rate
        self.dynamics_lr = dynamics_lr if dynamics_lr is not None else learning_rate
        if self.mode != "bptt":
            self.critic_lr = critic_lr if critic_lr is not None else learning_rate
            self.entropy_lr = entropy_lr if entropy_lr is not None else learning_rate

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ) -> None:
        # create rollout storage
        self.num_transitions_per_env = num_transitions_per_env
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            None,
            critic_obs_shape,
            self.device,
        )

    def test_mode(self):
        self.policy.test() # type: ignore

    def train_mode(self):
        self.policy.train()
    
    def predict(self, obs, actions):
        return self.policy.estimate(torch.cat((obs, actions), dim=-1))

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states() # type: ignore
        # compute the actions and values
        self.transition.actions = self.policy.act(obs)
        if self.mode == "bptt":
            self.transition.values = torch.zeros_like(self.num_transitions_per_env, device=self.device)
        else:
            self.transition.values = self.policy.evaluate(critic_obs)
        self.transition.actions_log_prob = (0 * self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs, critic_obs, and estimate_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            ) # type: ignore

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        if self.mode == "bptt":
            last_values = torch.zeros(self.storage.num_envs, 1, device=self.device)
        else:
            # compute value for the last step
            last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=False
        )

    def update(self):  # noqa: C901
        mean_policy_loss = 0
        mean_value_loss = 0
        mean_entropy = 0
        mean_estimation_loss = 0
        mean_estimation_errors = torch.zeros(self.policy.n_dynamics, requires_grad=False, device=self.device)

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            _,
            returns_batch,
            _,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            _,
            estimate_obs_batch,
        ) in generator:

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])

            # -- entropy
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    ) # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.actor_lr = max(1e-5, self.actor_lr / 1.5)
                            if self.mode != "bptt":
                                self.critic_lr = max(1e-5, self.critic_lr / 1.5)
                                self.entropy_lr = max(1e-5, self.entropy_lr / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.actor_lr = min(1e-2, self.actor_lr * 1.5)
                            if self.mode != "bptt":
                                self.critic_lr = min(1e-2, self.critic_lr * 1.5)
                                self.entropy_lr = min(1e-2, self.entropy_lr * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        actor_lr_tensor = torch.tensor(self.actor_lr, device=self.device)
                        torch.distributed.broadcast(actor_lr_tensor, src=0)
                        self.actor_lr = actor_lr_tensor.item()

                        if self.mode != "bptt":
                            critic_lr_tensor = torch.tensor(self.critic_lr, device=self.device)
                            torch.distributed.broadcast(critic_lr_tensor, src=0)
                            self.critic_lr = critic_lr_tensor.item()

                            entropy_lr_tensor = torch.tensor(self.entropy_lr, device=self.device)
                            torch.distributed.broadcast(entropy_lr_tensor, src=0)
                            self.entropy_lr = entropy_lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        if param_group.get("name") == "actor":
                            param_group["lr"] = self.actor_lr
                        if self.mode != "bptt":
                            if param_group.get("name") == "critic":
                                param_group["lr"] = self.critic_lr
                            if param_group.get("name") == "entropy":
                                param_group["lr"] = self.entropy_lr

            # Policy loss
            policy_loss = -returns_batch.mean()
            loss = policy_loss

            # Value function loss
            if self.mode != "bptt":
                value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                loss += value_loss

            # Dynamics Loss
            estimates_batch = self.predict(critic_obs_batch, actions_batch)
            estimate_diff = critic_obs_batch - estimates_batch # type: ignore
            estimate_loss = estimate_diff.pow(2) + estimate_diff.abs()  # L2 + L1 loss
            estimate_loss = estimate_loss.mean()
            estimate_errors = estimate_diff.abs().mean(dim=0)
            loss += estimate_loss

            # Entropy Loss
            if self.policy.mode != "bptt":
                entropy_loss = (entropy_batch - self.policy.entropy_ref).pow(2).mean()
                loss += entropy_loss

            # Compute the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Store the losses
            mean_policy_loss += policy_loss.item()
            if self.policy.mode != "bptt":
                mean_value_loss += value_loss.item()
                mean_entropy += entropy_batch.mean().item()
            mean_estimation_loss += estimate_loss.item()
            mean_estimation_errors += estimate_errors.clone()

        # -- Logging
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_policy_loss /= num_updates
        if self.policy.mode != "bptt":
            mean_value_loss /= num_updates
            mean_entropy /= num_updates
        mean_estimation_loss /= num_updates
        mean_estimation_errors /= num_updates
        # -- Clear the storage
        self.storage.clear()
        # self.counter += 1

        # construct the loss dictionary
        loss_dict = {
            "policy": mean_policy_loss,
            "value_function": mean_value_loss,
            "entropy": mean_entropy,
            # "gradient": mean_gradient_loss,
        }
        loss_dict["estimation"] = mean_estimation_loss
        # loss_dict["estimation_errors"] = mean_estimation_errors
        error_dict = {f"estimation_{str(i)}": estimate_errors[i] for i in range(num_est_obs)} # type: ignore

        return loss_dict, error_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
