"""Binary long-write controllers sharing one causal observation interface."""
from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor
from torch.distributions import Categorical

from .observations import ObservationBuilder
from .types import PolicyDecision


class LongWriteAction(IntEnum):
    TEMP_ONLY = 0
    WRITE_LONG = 1


TEMP_ONLY = LongWriteAction.TEMP_ONLY
WRITE_LONG = LongWriteAction.WRITE_LONG


def valid_action_mask(packet, action_mask=None):
    mask = torch.as_tensor(packet.action_mask, device=packet.observation.device, dtype=torch.bool)
    if action_mask is not None:
        supplied = torch.as_tensor(action_mask, device=packet.observation.device, dtype=torch.bool)
        if supplied.shape != mask.shape:
            raise ValueError("Supplied action mask must match the observation packet mask.")
        mask = mask & supplied
    if mask.shape != (packet.observation.shape[0], 2) or not mask.any(-1).all():
        raise ValueError("Binary action mask must be [B,2] with a valid action in each row.")
    return mask


def _decision(preferred: Tensor, packet, mask: Tensor) -> PolicyDecision:
    preferred = preferred.long()
    valid = mask.gather(1, preferred[:, None]).squeeze(1)
    action = torch.where(valid, preferred, mask.long().argmax(-1))
    forced = mask.sum(-1) == 1
    return PolicyDecision(action=action, log_prob=torch.zeros_like(action, dtype=torch.float32),
                          forced=forced, action_mask=mask)


class LongWritePolicy:
    def __init__(self, observation_builder=None):
        self.observation_builder = observation_builder or ObservationBuilder()

    def observe(self, candidate, state, diagnostics=None):
        confidence = getattr(candidate, "confidence", None)
        if confidence is None:
            raise ValueError("Memory candidate must expose current reconstruction confidence.")
        return self.observation_builder.build({"pts3d": candidate.pointmap, "conf": confidence},
                                              candidate.key, state, diagnostics)

    def act(self, packet, action_mask=None, deterministic=False, critic_observation=None):
        raise NotImplementedError


class HeuristicWritePolicy(LongWritePolicy):
    def __init__(self, threshold=0.92, observation_builder=None):
        super().__init__(observation_builder)
        if not -1 <= threshold <= 1:
            raise ValueError("Descriptor similarity threshold must be in [-1,1].")
        self.threshold = float(threshold)

    def act(self, packet, action_mask=None, deterministic=False):
        mask = valid_action_mask(packet, action_mask)
        similarity = packet.heuristic_similarity
        if similarity is None:
            raise ValueError("Heuristic policy requires cheap recent-long descriptor similarity.")
        preferred = (similarity <= self.threshold).long()
        return _decision(preferred, packet, mask)


class AlwaysWritePolicy(LongWritePolicy):
    def act(self, packet, action_mask=None, deterministic=False):
        mask = valid_action_mask(packet, action_mask)
        return _decision(torch.ones(mask.shape[0], device=mask.device, dtype=torch.long), packet, mask)


class NeverWritePolicy(LongWritePolicy):
    """Seed long memory when forced, then use TEMP_ONLY."""
    def act(self, packet, action_mask=None, deterministic=False):
        mask = valid_action_mask(packet, action_mask)
        return _decision(torch.zeros(mask.shape[0], device=mask.device, dtype=torch.long), packet, mask)


class PPOWritePolicy(LongWritePolicy):
    def __init__(self, actor, critic=None, normalizer=None, observation_builder=None):
        super().__init__(observation_builder)
        self.actor, self.critic, self.normalizer = actor, critic, normalizer

    @torch.no_grad()
    def act(self, packet, action_mask=None, deterministic=False, critic_observation=None):
        mask = valid_action_mask(packet, action_mask)
        observation = packet.observation
        if self.normalizer is not None:
            observation = self.normalizer(observation)
        logits = self.actor(observation).float().masked_fill(~mask, float("-inf"))
        distribution = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else distribution.sample()
        forced = mask.sum(-1) == 1
        log_prob = torch.where(forced, torch.zeros_like(distribution.log_prob(action)), distribution.log_prob(action))
        if self.critic is None:
            value = None
        elif getattr(self.critic, "state_dim", observation.shape[-1]) != observation.shape[-1]:
            if critic_observation is None:
                raise ValueError("Privileged critic needs a separate training-only observation; actor inputs remain 30-D.")
            value = self.critic(critic_observation.detach()).reshape(-1)
        else:
            value = self.critic(observation).reshape(-1)
        return PolicyDecision(action=action, log_prob=log_prob, value=value,
                              forced=forced, action_mask=mask)


def make_policy(kind, *, threshold=0.92, actor=None, critic=None, normalizer=None):
    factories = {"heuristic": lambda: HeuristicWritePolicy(threshold),
                 "always": AlwaysWritePolicy, "never": NeverWritePolicy,
                 "ppo": lambda: PPOWritePolicy(actor, critic, normalizer)}
    if kind not in factories:
        raise ValueError(f"Unknown long-write policy {kind!r}.")
    if kind == "ppo" and actor is None:
        raise ValueError("PPO write policy requires a constructed/loaded actor.")
    return factories[kind]()
