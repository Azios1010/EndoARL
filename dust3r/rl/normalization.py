"""Versioned rollout-frozen normalization of selected continuous actor features."""
from __future__ import annotations

import torch
from torch import nn

from dust3r.memory.observations import CONTINUOUS_INDICES


class ObservationNormalizer(nn.Module):
    def __init__(self, state_dim=30, epsilon=1e-6, clip=5.0):
        super().__init__()
        if state_dim != 30:
            raise ValueError("Observation normalizer is tied to the v1 30-feature schema.")
        self.epsilon, self.clip = float(epsilon), float(clip)
        self.register_buffer("mean", torch.zeros(state_dim, dtype=torch.float64))
        self.register_buffer("variance", torch.ones(state_dim, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("version", torch.tensor(0, dtype=torch.long))
        self.frozen = False

    def forward(self, observation):
        if observation.shape[-1] != 30 or not torch.isfinite(observation).all():
            raise ValueError("Normalizer requires finite 30-feature observations.")
        result = observation.float().clone()
        indices = list(CONTINUOUS_INDICES)
        if self.count.item() > 0:
            mean = self.mean.to(device=result.device, dtype=result.dtype)
            variance = self.variance.to(device=result.device, dtype=result.dtype)
            result[..., indices] = ((result[..., indices] - mean[indices]) /
                                    (variance[indices] + self.epsilon).sqrt()).clamp(-self.clip, self.clip)
        return result

    def freeze(self):
        self.frozen = True
        return int(self.version.item())

    def unfreeze(self):
        self.frozen = False

    @torch.no_grad()
    def update(self, observations):
        if self.frozen:
            raise RuntimeError("Normalizer snapshot is frozen for rollout collection/PPO epochs.")
        observations = observations.detach().reshape(-1, 30).to(self.mean.device, torch.float64)
        if not observations.numel() or not torch.isfinite(observations).all():
            raise ValueError("Normalizer update requires nonempty finite training observations.")
        n = observations.shape[0]
        batch_mean = observations.mean(0)
        batch_variance = observations.var(0, unbiased=False)
        old_count = self.count.clone()
        total = old_count + n
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (n / total)
        new_variance = ((self.variance * old_count + batch_variance * n +
                         delta.square() * old_count * n / total) / total)
        self.mean.copy_(new_mean)
        self.variance.copy_(new_variance)
        self.count.copy_(total)
        self.version.add_(1)
