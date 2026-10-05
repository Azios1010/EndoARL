"""Separate tiny actor/critic; privileged critic features never enter Actor."""
from __future__ import annotations

import math

import torch
from torch import nn


def _mlp(state_dim, hidden, output_dim, output_gain):
    if state_dim <= 0 or not hidden or any(width <= 0 for width in hidden):
        raise ValueError("Network dimensions must be positive.")
    modules = []
    dimension = state_dim
    for width in hidden:
        layer = nn.Linear(dimension, width)
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
        nn.init.zeros_(layer.bias)
        modules.extend([layer, nn.Tanh()])
        dimension = width
    output = nn.Linear(dimension, output_dim)
    nn.init.orthogonal_(output.weight, gain=output_gain)
    nn.init.zeros_(output.bias)
    return nn.Sequential(*modules, output)


class Actor(nn.Module):
    def __init__(self, state_dim=30, hidden=(64, 64)):
        super().__init__()
        if state_dim != 30:
            raise ValueError("V1 actor must have exactly 30 inference-available features.")
        self.state_dim = state_dim
        self.network = _mlp(state_dim, hidden, 2, 0.01)

    def forward(self, observation):
        if observation.shape[-1] != self.state_dim:
            raise ValueError("Actor observation must end in 30 scalar features.")
        return self.network(observation)


class Critic(nn.Module):
    def __init__(self, state_dim=30, hidden=(64, 64)):
        super().__init__()
        if state_dim not in (30, 34):
            raise ValueError("V1 critic uses 30 standard or 34 explicitly privileged features.")
        self.state_dim = state_dim
        self.network = _mlp(state_dim, hidden, 1, 1.0)

    def forward(self, observation):
        if observation.shape[-1] != self.state_dim:
            raise ValueError("Critic input dimension does not match its configured privilege mode.")
        return self.network(observation).squeeze(-1)


def privileged_critic_observation(actor_observation, gt_features=None):
    """Explicit training-only concatenation: error mean/p90, valid ratio, available bit."""
    if gt_features is None or gt_features.shape != (*actor_observation.shape[:-1], 4):
        raise ValueError("Privileged critic requires exactly four separate GT features.")
    return torch.cat((actor_observation.detach(), gt_features.detach()), dim=-1)
