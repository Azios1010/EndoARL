"""Small detached PPO transitions and terminal-aware generalized advantages."""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
from torch import Tensor


@dataclass
class Transition:
    raw_observation: Tensor
    observation: Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    action_mask: Tensor
    forced: bool = False
    truncated: bool = False
    next_value: float | None = None
    critic_observation: Tensor | None = None
    requested_write: bool = False
    successful_write: bool = False
    raft_flag: bool = False
    kda_flag: bool = False
    quality_metrics: dict[str, float] = field(default_factory=dict)
    cost_components: dict[str, float] = field(default_factory=dict)
    episode_id: str = ""
    candidate_index: int = 0
    arrival_index: int = 1
    normalization_version: int = 0
    environment_checkpoint_id: str = ""


def compute_gae(rewards: Tensor, values: Tensor, dones: Tensor, last_value=0.0,
                gamma=0.99, gae_lambda=0.95, next_values: Tensor | None = None,
                truncated: Tensor | None = None):
    """True terminal zeroes bootstrap; rollout truncation bootstraps but breaks recursion."""
    rewards, values = rewards.float(), values.float()
    dones = dones.bool()
    if rewards.shape != values.shape or rewards.shape != dones.shape or rewards.ndim != 1:
        raise ValueError("GAE rewards/values/dones must be matching vectors.")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("GAE discount and lambda must be in [0,1].")
    if next_values is None:
        next_values = torch.cat([values[1:], torch.as_tensor(last_value, device=values.device).reshape(1)])
    if next_values.shape != values.shape:
        raise ValueError("GAE next values must match rewards.")
    truncated = torch.zeros_like(dones) if truncated is None else truncated.bool()
    advantages = torch.zeros_like(values)
    following = values.new_zeros(())
    for index in reversed(range(values.numel())):
        bootstrap = (~dones[index]).float()
        delta = rewards[index] + gamma * bootstrap * next_values[index] - values[index]
        continuation = bootstrap * (~truncated[index]).float()
        following = delta + gamma * gae_lambda * continuation * following
        advantages[index] = following
    return advantages, advantages + values


class RolloutBuffer:
    def __init__(self, capacity=128):
        if capacity <= 0:
            raise ValueError("Rollout capacity must be positive.")
        self.capacity = int(capacity)
        self.transitions: list[Transition] = []

    def __len__(self):
        return len(self.transitions)

    def clear(self):
        self.transitions.clear()

    def add(self, transition: Transition | None = None, **kwargs):
        if len(self) >= self.capacity:
            raise RuntimeError("Rollout is full; update/clear before adding more transitions.")
        transition = transition or Transition(**kwargs)
        for name in ("raw_observation", "observation", "critic_observation", "action_mask"):
            value = getattr(transition, name)
            if value is not None:
                setattr(transition, name, value.detach().cpu().reshape(-1).clone())
        if transition.observation.shape != (30,) or transition.raw_observation.shape != (30,):
            raise ValueError("Rollout observations must have exactly 30 features.")
        if transition.action_mask.shape != (2,) or transition.action not in (0, 1):
            raise ValueError("Rollout requires a binary action and two-element mask.")
        transition.action_mask = transition.action_mask.bool()
        if not transition.action_mask[transition.action]:
            raise ValueError("Stored action is invalid under its original mask.")
        if not torch.isfinite(transition.observation).all() or not torch.isfinite(transition.raw_observation).all():
            raise ValueError("Rollout observations must be finite.")
        if not torch.isfinite(torch.tensor([transition.log_prob, transition.value, transition.reward])).all():
            raise ValueError("Rollout log probabilities, values and rewards must be finite.")
        if transition.next_value is not None and not torch.isfinite(torch.tensor(transition.next_value)):
            raise ValueError("Bootstrap value must be finite.")
        transition.forced = bool(transition.forced or transition.action_mask.sum() == 1)
        if transition.forced and abs(transition.log_prob) > 1e-6:
            raise ValueError("A forced deterministic action has old log_prob=0.")
        if self.transitions:
            first = self.transitions[0]
            if transition.normalization_version != first.normalization_version:
                raise ValueError("One on-policy rollout must use one frozen normalizer snapshot.")
            if transition.environment_checkpoint_id != first.environment_checkpoint_id:
                raise ValueError("Environment checkpoint changes invalidate on-policy samples.")
            previous = self.transitions[-1]
            if (previous.episode_id != transition.episode_id and
                    not previous.done and not previous.truncated):
                raise ValueError("Episode changes must have an explicit terminal/truncation boundary.")
        self.transitions.append(transition)

    def compute_returns(self, last_value=0.0, gamma=0.99, gae_lambda=0.95):
        if not self.transitions:
            raise ValueError("Cannot calculate advantages for an empty rollout.")
        rewards = torch.tensor([t.reward for t in self.transitions], dtype=torch.float32)
        values = torch.tensor([t.value for t in self.transitions], dtype=torch.float32)
        dones = torch.tensor([t.done for t in self.transitions], dtype=torch.bool)
        next_values = []
        for index, transition in enumerate(self.transitions):
            if transition.next_value is not None:
                next_values.append(transition.next_value)
            elif transition.done:
                next_values.append(0.0)
            elif index + 1 < len(self.transitions):
                if transition.truncated:
                    raise ValueError("An interior truncated trajectory needs explicit bootstrap next_value.")
                next_values.append(self.transitions[index + 1].value)
            else:
                next_values.append(float(last_value))
        return compute_gae(rewards, values, dones, last_value, gamma, gae_lambda,
                           torch.tensor(next_values), torch.tensor([t.truncated for t in self.transitions]))

    def tensors(self, device="cpu", *, last_value=0.0, gamma=0.99, gae_lambda=0.95):
        advantages, returns = self.compute_returns(last_value, gamma, gae_lambda)
        return {"observations": torch.stack([t.observation for t in self.transitions]).to(device),
                "raw_observations": torch.stack([t.raw_observation for t in self.transitions]).to(device),
                "critic_observations": torch.stack([t.critic_observation if t.critic_observation is not None
                                                      else t.observation for t in self.transitions]).to(device),
                "actions": torch.tensor([t.action for t in self.transitions], device=device),
                "old_log_probs": torch.tensor([t.log_prob for t in self.transitions], device=device),
                "old_values": torch.tensor([t.value for t in self.transitions], device=device),
                "action_masks": torch.stack([t.action_mask for t in self.transitions]).to(device),
                "policy_mask": torch.tensor([not t.forced for t in self.transitions], device=device),
                "advantages": advantages.to(device), "returns": returns.to(device)}
