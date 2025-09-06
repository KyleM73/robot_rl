from __future__ import annotations

from typing import Literal
import math

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

from robot_rl.utils import resolve_nn_activation
from .actor_critic import ActorCritic

class DGN(nn.Module):
    def __init__(
        self,
        num_obs,
        num_actions,
        hidden_dims=[128,128],
        activation="relu",
        dropout=0.2,
        num_heads=1
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.n = num_actions * (num_actions + 1) // 2
        self.n_gmm = num_heads

        if self.is_gmm:
            self.weights = nn.Sequential(
                self.make_mlp(num_obs, self.n_gmm, hidden_dims, activation, dropout),
                nn.Softmax(dim=-1)
            )
            self.cholesky = self.make_mlp(num_obs, self.n_gmm * self.n, hidden_dims, activation, dropout)
        else:
            self.cholesky = self.make_mlp(num_obs, self.n, hidden_dims, activation, dropout)

    @property
    def is_gmm(self) -> bool:
        return self.n_gmm > 1

    def make_mlp(self, n_in, n_out, hidden_dims, activation, dropout) -> nn.Sequential:
        layers, in_dim = [], n_in
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(resolve_nn_activation(activation))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_out))
        return nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Returns the *mixture covariance*:
            Σ_mix(s) = Σ_h π_h(s) Σ_h(s)
        Shape: [B, A, A]
        """
        B = s.shape[0]

        if not self.is_gmm:
            chol_flat = self.cholesky(s)                  # [B, n]
            chol_all = chol_flat.unsqueeze(1)             # [B, 1, n]
            probs = torch.ones(B, 1, device=s.device)     # [B, 1]
        else:
            probs = self.weights(s)                       # [B, H]
            chol_flat = self.cholesky(s).view(B, self.n_gmm, self.n)  # [B, H, n]
            chol_all = chol_flat                          # [B, H, n]

        # Expand flat to lower-triangular matrices [B, H, A, A]
        L = s.new_zeros(B, self.n_gmm, self.num_actions, self.num_actions)
        tri_idx = torch.tril_indices(self.num_actions, self.num_actions, 0)
        L[:, :, tri_idx[0], tri_idx[1]] = chol_all

        # Ensure positive diagonals
        diag_idx = torch.arange(self.num_actions, device=s.device)
        L[:, :, diag_idx, diag_idx] = F.softplus(L[:, :, diag_idx, diag_idx]) + 1e-4

        covs = L @ L.transpose(-1, -2)   # [B, H, A, A]

        # Weighted mixture covariance
        mix_cov = torch.einsum("bh, bhij -> bij", probs, covs)  # [B, A, A]
        return mix_cov
    
    def nll(self, s: torch.Tensor, a: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Computes exact NLL under a Gaussian mixture:
            -log( Σ_h π_h(s) N(a | mu(s), Σ_h(s)) )

        Args:
            s:  [B, obs_dim]
            a:  [B, A]
            mu: [B, A]  (policy mean from actor)

        Returns:
            nll: [B]
        """
        B = s.shape[0]
        probs = self.weights(s)                        # [B, H]
        chol_flat = self.cholesky(s).view(B, self.n_gmm, self.n)

        # Expand to Cholesky [B, H, A, A]
        L = s.new_zeros(B, self.n_gmm, self.num_actions, self.num_actions)
        tri_idx = torch.tril_indices(self.num_actions, self.num_actions, 0)
        L[:, :, tri_idx[0], tri_idx[1]] = chol_flat
        diag_idx = torch.arange(self.num_actions, device=s.device)
        L[:, :, diag_idx, diag_idx] = F.softplus(L[:, :, diag_idx, diag_idx]) + 1e-4

        covs = L @ L.transpose(-1, -2)   # [B, H, A, A]

        # Compute log N(a | mu, Σ_h) for each component
        diff = a.unsqueeze(1) - mu.unsqueeze(1)         # [B, H, A]
        y = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False).squeeze(-1)  # [B,H,A]
        mahal = 0.5 * (y ** 2).sum(-1)                  # [B,H]
        logdet = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # [B,H]
        const = 0.5 * self.num_actions * math.log(2 * math.pi)
        log_probs = -(mahal + logdet + const)           # [B,H]

        # Mixture log likelihood via logsumexp
        weighted_log = log_probs + torch.log(probs + 1e-8)  # [B,H]
        log_likelihood = torch.logsumexp(weighted_log, dim=1)  # [B]
        return -log_likelihood


class ActorCriticDGN(ActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_dgn_obs,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: Literal["scalar"] = "scalar",
        dgn_hidden_dims=[128, 128],
        dgn_activation="relu",
        dgn_gmm_heads=1,
        dgn_dropout=0.2,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticDGN.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__(num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims, critic_hidden_dims, activation, init_noise_std, noise_std_type, **kwargs)

        self.dgn = DGN(num_dgn_obs, num_actions, dgn_hidden_dims, dgn_activation, dgn_dropout, dgn_gmm_heads)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    
    def update_distribution(self, observations, dgn_observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        std = torch.diag_embed(self.std.expand_as(mean))
        dgn_cov = self.dgn(dgn_observations)
        # create distribution
        cov = std @ std.transpose(-1, -2) + dgn_cov
        self.distribution = MultivariateNormal(mean, covariance_matrix=cov)


