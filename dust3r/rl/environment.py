"""Sequential frozen Endo3R environment, with next-left delayed rewards."""
from __future__ import annotations

from typing import Callable

import torch

from dust3r.memory.observations import ObservationBuilder


class Environment:
    """Wrap the online runner's prepare -> commit -> prepare protocol.

    GT is passed only to RewardComputer, never the actor ObservationBuilder.
    ``reset`` decodes pair 0; ``step(a_i)`` writes it, then decodes pair i+1
    and scores its LEFT prediction for a_i. The final pair has no new action.
    B=1 and one patient per state are explicit v1 invariants.
    """

    def __init__(self, runner, reward_computer, observation_builder=None,
                 gt_provider: Callable | None = None, environment_checkpoint_id=""):
        self.runner = runner
        self.reward_computer = reward_computer
        self.observation_builder = observation_builder or ObservationBuilder()
        self.gt_provider = gt_provider
        self.environment_checkpoint_id = str(environment_checkpoint_id)
        self.current_pair = None
        self.current_packet = None
        self.episode_id = ""
        self.done = True
        self.transitions = 0

    def _target(self, index, *, terminal=False):
        if self.gt_provider is None:
            return None
        return self.gt_provider(index)

    def _observe(self, pair):
        return self.observation_builder.build(pair.res1, pair.feat_k1, self.runner.state,
                                             pair.read_diagnostics)

    @torch.no_grad()
    def reset(self, frames, episode_id="episode"):
        self.episode_id = str(episode_id)
        self.transitions = 0
        self.current_pair = self.current_packet = None
        self.done = False
        self.runner.reset(frames, episode_id=self.episode_id, training=False)
        pair = self.runner.prepare_next()
        if pair is None:
            raise ValueError("An episode requires at least two input frames.")
        self.current_pair = pair
        self.reward_computer.reset(pair.res1, self._target(pair.candidate_index))
        if pair.terminal:
            self.runner.commit(None)
            self.done = True
            return None
        self.current_packet = self._observe(pair)
        return self.current_packet

    @torch.no_grad()
    def step(self, action):
        if self.done or self.current_packet is None or self.current_pair is None:
            raise RuntimeError("Episode has no pending decision; call reset first.")
        if hasattr(action, "action"):
            action = action.action
        action_tensor = torch.as_tensor(action)
        if action_tensor.numel() != 1:
            raise ValueError("V1 environment accepts one sequence action, not broadcast batched writes.")
        action_value = int(action_tensor.item())
        if action_value not in (0, 1) or not self.current_packet.action_mask[0, action_value]:
            raise ValueError("Invalid binary action; the first eligible WRITE_LONG is forced.")
        previous_pair = self.current_pair
        kernel_diagnostics = None
        write_diagnostics = {"requested_write": action_value == 1}
        try:
            kernel_diagnostics = self.runner.commit(action_value)
            write_diagnostics = getattr(self.runner, "last_commit_info", kernel_diagnostics)
            next_pair = self.runner.prepare_next()
        except (FloatingPointError, ArithmeticError) as error:
            write_diagnostics = getattr(self.runner, "last_commit_info", write_diagnostics)
            result = self.reward_computer.failure_reward(write_diagnostics)
            self.done = True
            self.transitions += 1
            self.current_pair = self.current_packet = None
            return None, result.reward, True, self._info(previous_pair, action_value,
                    write_diagnostics, result, None, failure_reason=str(error))
        if next_pair is None:
            raise RuntimeError("Runner ended before the action's next-left prediction was produced.")
        result = self.reward_computer.compute(next_pair.res1,
                    self._target(next_pair.candidate_index, terminal=next_pair.terminal),
                    write_diagnostics, terminal=next_pair.terminal)
        self.current_pair = next_pair
        self.transitions += 1
        self.done = bool(next_pair.terminal or result.failure)
        if self.done:
            if next_pair.terminal:
                self.runner.commit(None)  # terminal ST append, no learned action/RAFT/KDA
            self.current_packet = None
        else:
            self.current_packet = self._observe(next_pair)
        return self.current_packet, result.reward, self.done, self._info(
            previous_pair, action_value, write_diagnostics, result, next_pair,
            kernel_write_diagnostics=kernel_diagnostics)

    def _info(self, previous_pair, action, write_diagnostics, result, next_pair, **extra):
        return {"episode_id": self.episode_id,
                "candidate_index": previous_pair.candidate_index,
                "arrival_index": previous_pair.arrival_index,
                "reward_prediction_index": None if next_pair is None else next_pair.candidate_index,
                "action": action, "write_diagnostics": write_diagnostics,
                "quality_error": result.quality_error, "quality_metrics": result.metrics,
                "cost_components": result.components, "failure": result.failure,
                "environment_checkpoint_id": self.environment_checkpoint_id, **extra}


Endo3REnvironment = Environment
