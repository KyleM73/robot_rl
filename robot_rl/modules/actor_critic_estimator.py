from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from robot_rl.networks import MLP, EmpiricalNormalization
from robot_rl.utils import resolve_nn_activation


class ActorCriticEstimator(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        num_estimates,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        estimator_index=-1,
        oracle=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticEstimator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        self.num_estimates = num_estimates
        if estimator_index == -1:
            estimator_index = len(actor_hidden_dims) - 1
        # actor - estimator
        shared_layers = []
        actor_layers = []
        shared_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        shared_layers.append(resolve_nn_activation(activation))
        for layer_index in range(len(actor_hidden_dims) - 1):
            if layer_index < estimator_index:
                shared_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                shared_layers.append(resolve_nn_activation(activation))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(resolve_nn_activation(activation))
        self.backbone = nn.Sequential(*shared_layers)
        self.actor = nn.Sequential(*shared_layers, *actor_layers, nn.Linear(actor_hidden_dims[-1], num_actions))

        if oracle:
            oracle_layers = []
            oracle_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
            oracle_layers.append(resolve_nn_activation(activation))
            for layer_index in range(len(actor_hidden_dims) - 1):
                    oracle_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                    oracle_layers.append(resolve_nn_activation(activation))
            self.estimator = nn.Sequential(*oracle_layers, nn.Linear(actor_hidden_dims[-1], num_estimates))
        else:
            self.estimator = nn.Sequential(*shared_layers, nn.Linear(actor_hidden_dims[-1], num_estimates))
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")
        print(f"Estimator MLP: {self.estimator}")

        # critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

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

    def update_distribution(self, obs):
        # compute mean
        mean = self.actor(obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

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
        return True  # training resumes

    def estimate(self, obs, **kwargs) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        estimates = self.estimator(obs)
        return estimates
    
    def get_latents(self, obs, **kwargs) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        latents = self.backbone(obs)
        return latents
    
    def get_last_layer(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # Get the weight matrix of the last layer of the actor
        actor_last_layer = self.actor[-1]
        actor_weights = actor_last_layer.weight if isinstance(actor_last_layer, nn.Linear) else None

        # Get the weight matrix of the last layer of the estimator
        estimator_last_layer = self.estimator[-1]
        estimator_weights = estimator_last_layer.weight if isinstance(estimator_last_layer, nn.Linear) else None

        return actor_weights, estimator_weights
