from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from robot_rl.utils import resolve_nn_activation


class ActorCriticDMO(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_dynamics_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        dynamics_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        mode="shac", # Options["bptt", "shac"]
        init_temperature=1.0,
        # state_dependant_noise: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticDMO.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.mode = mode

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        mlp_input_dim_d = num_dynamics_obs + num_actions
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(resolve_nn_activation(activation))
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(resolve_nn_activation(activation))
        self.actor = nn.Sequential(*actor_layers)
        print(f"Actor MLP: {self.actor}")

        # Value function
        if self.mode != "bptt":
            critic_layers = []
            critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
            critic_layers.append(resolve_nn_activation(activation))
            for layer_index in range(len(critic_hidden_dims)):
                if layer_index == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                    critic_layers.append(resolve_nn_activation(activation))
            self.critic = nn.Sequential(*critic_layers)
            print(f"Critic MLP: {self.critic}")

        # Dynamics
        dynamics_layers = []
        dynamics_layers.append(nn.Linear(mlp_input_dim_d, dynamics_hidden_dims[0]))
        dynamics_layers.append(resolve_nn_activation(activation))
        for layer_index in range(len(dynamics_hidden_dims)):
            if layer_index == len(dynamics_hidden_dims) - 1:
                dynamics_layers.append(nn.Linear(dynamics_hidden_dims[layer_index], 2 * num_dynamics_obs))
            else:
                dynamics_layers.append(nn.Linear(dynamics_hidden_dims[layer_index], dynamics_hidden_dims[layer_index + 1]))
                dynamics_layers.append(resolve_nn_activation(activation))
        self.dynamics = nn.Sequential(*dynamics_layers)
        self.n_dynamics = num_dynamics_obs
        print(f"Dynamics MLP: {self.dynamics}")

        # Action noise
        self.noise_std_type = noise_std_type
        # self.state_dependant_noise = state_dependant_noise
        if self.noise_std_type == "scalar": # and not self.state_dependant_noise:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log": # and not self.state_dependant_noise:
            self.std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        # Entropy
        if self.mode != "bptt":
            init_vec = init_temperature * torch.ones(num_actions)
            softplus_inv = torch.log(torch.exp(init_vec) - 1)
            self.alpha = nn.Parameter(softplus_inv)
            self.entropy_ref = -num_actions / 2

        # Action distribution (populated in update_distribution)
        self.distribution = Normal(0.0, 1.0)
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    @property
    def dynamics_entropy(self):
        return self.dynamics_distribution.entropy().sum(dim=-1)
    
    def get_entropy_bonus(self):
        a = torch.nn.functional.softplus(self.alpha)
        return a * self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def update_dynamics(self, observations):
        # compute mean and std
        out = self.dynamics(observations)
        mean, scale = out[:, :self.n_dynamics], out[:, self.n_dynamics:]
        std = torch.nn.functional.softplus(scale) # psd
        # create distribution
        self.dynamics_distribution = Normal(mean, std)

    def estimate(self, observations):
        self.update_dynamics(observations)
        return self.dynamics_distribution.sample()
    
    def estimate_inference(self, observations):
        dynamics_mean = self.dynamics(observations)[:, :self.n_dynamics]
        return dynamics_mean

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
