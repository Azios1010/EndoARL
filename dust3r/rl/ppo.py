"""Clipped PPO implementation. Construction never starts training."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical
from torch.nn import functional as F


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    value_clip: float | None = 0.2
    max_grad_norm: float = 0.5
    minibatch_size: int = 64
    epochs: int = 4
    target_kl: float = 0.03
    normalize_advantages: bool = True

    def __post_init__(self):
        if min(self.learning_rate, self.max_grad_norm, self.minibatch_size, self.epochs) <= 0:
            raise ValueError("PPO learning rate, norm limit and batch sizes must be positive.")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_ratio < 1:
            raise ValueError("Invalid PPO discount/lambda/clipping defaults.")
        if min(self.entropy_coefficient, self.value_coefficient, self.target_kl) < 0:
            raise ValueError("PPO loss coefficients and KL threshold must be nonnegative.")
        if self.value_clip is not None and self.value_clip <= 0:
            raise ValueError("Value clipping threshold must be positive or None.")


class PPO:
    def __init__(self, actor, critic, config=None):
        self.actor, self.critic = actor, critic
        self.config = config or PPOConfig()
        actor_ids = {id(parameter) for parameter in actor.parameters()}
        if actor_ids & {id(parameter) for parameter in critic.parameters()}:
            raise ValueError("Actor and critic parameters must be separate.")
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.config.learning_rate, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.config.learning_rate, eps=1e-5)

    def update(self, rollout, last_value=0.0):
        config = self.config
        device = next(self.actor.parameters()).device
        batch = rollout.tensors(device, last_value=last_value, gamma=config.gamma, gae_lambda=config.gae_lambda)
        advantages = batch["advantages"].clone()
        learned = batch["policy_mask"]
        if learned.any() and config.normalize_advantages:
            selected = advantages[learned]
            advantages[learned] = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
        metrics = {"actor_updates": 0, "critic_updates": 0, "policy_loss": 0.0,
                   "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                   "clip_fraction": 0.0, "forced_fraction": (~learned).float().mean().item()}
        stop = False
        for _ in range(config.epochs):
            indices = torch.randperm(len(rollout), device=device)
            for offset in range(0, len(rollout), config.minibatch_size):
                selection = indices[offset:offset + config.minibatch_size]
                values = self.critic(batch["critic_observations"][selection])
                target = batch["returns"][selection]
                if config.value_clip is None:
                    value_loss = F.mse_loss(values, target)
                else:
                    old_values = batch["old_values"][selection]
                    clipped_values = old_values + (values - old_values).clamp(-config.value_clip, config.value_clip)
                    value_loss = torch.maximum((values - target).square(), (clipped_values - target).square()).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (config.value_coefficient * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.max_grad_norm)
                self.critic_optimizer.step()
                metrics["critic_updates"] += 1
                metrics["value_loss"] += value_loss.item()
                policy_selection = selection[learned[selection]]
                if not policy_selection.numel():
                    continue
                logits = self.actor(batch["observations"][policy_selection]).float()
                logits = logits.masked_fill(~batch["action_masks"][policy_selection], float("-inf"))
                distribution = Categorical(logits=logits)
                log_prob = distribution.log_prob(batch["actions"][policy_selection])
                log_ratio = log_prob - batch["old_log_probs"][policy_selection]
                ratio = log_ratio.exp()
                selected_advantage = advantages[policy_selection]
                clipped = ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio)
                policy_loss = -torch.minimum(ratio * selected_advantage, clipped * selected_advantage).mean()
                entropy = distribution.entropy().mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                (policy_loss - config.entropy_coefficient * entropy).backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.max_grad_norm)
                self.actor_optimizer.step()
                approximate_kl = ((ratio - 1) - log_ratio).mean().item()
                metrics["actor_updates"] += 1
                metrics["policy_loss"] += policy_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["approx_kl"] += approximate_kl
                metrics["clip_fraction"] += ((ratio - 1).abs() > config.clip_ratio).float().mean().item()
                if approximate_kl > config.target_kl:
                    stop = True
                    break
            if stop:
                break
        for key in ("policy_loss", "entropy", "approx_kl", "clip_fraction"):
            metrics[key] /= max(metrics["actor_updates"], 1)
        metrics["value_loss"] /= max(metrics["critic_updates"], 1)
        metrics["early_stopped"] = stop
        return metrics

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict()}

    def load_state_dict(self, checkpoint):
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])


PPOTrainer = PPO
