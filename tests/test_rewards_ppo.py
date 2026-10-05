"""CPU tensor and mock-environment tests; no Endo3R/PPO training job."""
import unittest
from types import SimpleNamespace

import torch

from dust3r.rl.environment import Environment
from dust3r.rl.networks import Actor, Critic
from dust3r.rl.normalization import ObservationNormalizer
from dust3r.rl.ppo import PPO, PPOConfig
from dust3r.rl.rewards import CostCalibration, DenseGTReward, RewardComputer, SequenceGTReward
from dust3r.rl.rollout import RolloutBuffer, Transition, compute_gae


def prediction(value=1.):
    return {"pts3d": torch.full((1, 2, 2, 3), float(value)), "conf": torch.ones(1, 2, 2)}


def target(value=1., verified=True):
    return {"pointmap": torch.full((1, 2, 2, 3), float(value)),
            "valid_mask": torch.ones(1, 2, 2, dtype=torch.bool), "verified": verified}


class ProbeReward(RewardComputer):
    def __init__(self):
        super().__init__(raft_weight=0, kda_geometry_weight=0)
        self.scored_values = []

    def reset(self, initial_prediction=None, initial_target=None):
        self.scored_values.clear()

    def compute(self, next_prediction, target=None, write_diagnostics=None, *, terminal=False):
        value = float(next_prediction["pts3d"].mean())
        self.scored_values.append(value)
        return self._result(value, write_diagnostics)


class MockRunner:
    def reset(self, frames, episode_id="", training=False):
        self.frame_count = len(frames)
        self.index = 0
        self.pending = False
        self.long_updates = 0
        self.short_updates = 0
        self.last_commit_info = {}
        self.state = SimpleNamespace(has_long_memory=False, current_step=0, write_count=0,
            processed_candidate_count=0, last_write_step=-1, short_occupancy=0,
            recent_key_descriptors=torch.empty(1, 0, 1024), recent_long_descriptors=torch.empty(1, 0, 1024),
            last_long_descriptor=None, previous_uncertainty_stats=torch.zeros(1, 3),
            has_previous_raft_quality=False, predicted_radius_scale=None, has_predicted_radius_scale=False)

    def prepare_next(self):
        if self.pending:
            raise RuntimeError("Commit before preparing the next pair.")
        if self.index >= self.frame_count - 1:
            return None
        self.pending = True
        self.state.current_step = self.index
        return SimpleNamespace(res1=prediction(1 + self.long_updates), res2=prediction(999),
            feat_k1=torch.ones(1, 4, 1024), read_diagnostics=None, candidate_index=self.index,
            arrival_index=self.index + 1, terminal=self.index == self.frame_count - 2)

    def commit(self, action):
        if not self.pending:
            raise RuntimeError("No prepared pair.")
        self.short_updates += 1
        self.state.short_occupancy = min(4, self.short_updates)
        self.state.processed_candidate_count += 1
        if action == 1:
            self.long_updates += 1
            self.state.write_count += 1
            self.state.has_long_memory = True
            self.state.last_write_step = self.index
        self.last_commit_info = {"requested_write": action == 1, "successful_write": action == 1,
                                "raft_called": False, "kda_updated": action == 1, "geometry_called": False}
        self.index += 1
        self.pending = False
        return SimpleNamespace(success=action == 1, update_called=action == 1)


class RewardPPOTests(unittest.TestCase):
    def test_delayed_next_left_reward_terminal_and_reset(self):
        runner, reward = MockRunner(), ProbeReward()
        environment = Environment(runner, reward)
        packet = environment.reset([0, 1, 2])
        self.assertEqual(packet.action_mask.tolist(), [[False, True]])
        _, value, done, info = environment.step(1)
        self.assertTrue(done)
        self.assertEqual(value, -2.)
        self.assertEqual(reward.scored_values, [2.])
        self.assertEqual(info["candidate_index"], 0)
        self.assertEqual(info["reward_prediction_index"], 1)
        self.assertEqual(runner.long_updates, 1)
        self.assertEqual(runner.short_updates, 2)
        environment.reset([0, 1, 2])
        self.assertEqual(runner.long_updates, 0)
        self.assertFalse(runner.state.has_long_memory)

    def test_temp_preserves_long_state_and_no_invalid_first_skip(self):
        runner, reward = MockRunner(), ProbeReward()
        environment = Environment(runner, reward)
        environment.reset([0, 1, 2, 3])
        with self.assertRaises(ValueError):
            environment.step(0)
        _, _, done, _ = environment.step(1)
        self.assertFalse(done)
        _, _, done, _ = environment.step(0)
        self.assertTrue(done)
        self.assertEqual(runner.long_updates, 1)
        self.assertEqual(runner.short_updates, 3)

    def test_two_frames_have_no_action(self):
        environment = Environment(MockRunner(), ProbeReward())
        self.assertIsNone(environment.reset([0, 1]))
        self.assertTrue(environment.done)

    def test_fixed_episode_scale_and_verified_gt(self):
        reward = DenseGTReward(raft_weight=0, kda_geometry_weight=0)
        with self.assertRaises(ValueError):
            reward.reset(prediction(), target(2, verified=False))
        reward.reset(prediction(), target(2))
        first = reward.compute(prediction(2), target(2))
        self.assertGreater(first.quality_error, 0)  # no per-frame realignment
        second = reward.compute(prediction(), target(2))
        self.assertEqual(second.quality_error, 0)
        bad = prediction()
        bad["pts3d"][0, 0, 0] = float("nan")
        self.assertTrue(reward.compute(bad, target(2)).failure)

    def test_real_action_costs_and_no_occupancy_penalty(self):
        calibration = CostCalibration(baseline=10, raft=20, kda=5, geometry=1)
        reward = DenseGTReward(calibration=calibration)
        reward.reset(prediction(), target())
        result = reward.compute(prediction(), target(), {"requested_write": True,
                    "raft_called": True, "kda_updated": True, "geometry_called": True})
        self.assertAlmostEqual(result.reward, -.026, places=6)
        with self.assertRaises(ValueError):
            DenseGTReward()

    def test_sparse_sequence_reward_has_no_fake_dense_quality(self):
        reward = SequenceGTReward(raft_weight=0, kda_geometry_weight=0)
        self.assertEqual(reward.compute(terminal=False).reward, 0)
        self.assertEqual(reward.compute(target={"verified": True, "metrics": {"reconstruction": 2}}, terminal=True).reward, -2)

    def test_gae_terminal_and_truncation(self):
        advantages, returns = compute_gae(torch.tensor([1., 2.]), torch.tensor([.5, .7]),
                                          torch.tensor([False, True]), gamma=1, gae_lambda=1)
        self.assertTrue(torch.allclose(returns, torch.tensor([3., 2.])))
        self.assertTrue(torch.allclose(advantages, torch.tensor([2.5, 1.3])))
        _, returns = compute_gae(torch.tensor([1.]), torch.tensor([.5]), torch.tensor([False]),
                                last_value=.7, gamma=1, gae_lambda=1, truncated=torch.tensor([True]))
        self.assertTrue(torch.allclose(returns, torch.tensor([1.7])))

    def test_normalizer_frozen_version_and_discrete_features(self):
        normalizer = ObservationNormalizer()
        data = torch.arange(60, dtype=torch.float32).reshape(2, 30)
        normalizer.update(data)
        version = normalizer.freeze()
        with self.assertRaises(RuntimeError):
            normalizer.update(data)
        first, second = normalizer(data), normalizer(data)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[:, 11], data[:, 11]))
        self.assertEqual(int(normalizer.version), version)

    def test_rollout_version_and_forced_actor_mask(self):
        buffer = RolloutBuffer(4)
        buffer.add(Transition(torch.zeros(30), torch.zeros(30), 1, 0, 0, -1, True,
                              torch.tensor([False, True]), forced=True))
        with self.assertRaises(ValueError):
            buffer.add(Transition(torch.zeros(30), torch.zeros(30), 1, 0, 0, -1, True,
                                  torch.tensor([False, True]), normalization_version=1))
        actor, critic = Actor(), Critic()
        before = {key: value.clone() for key, value in actor.state_dict().items()}
        metrics = PPO(actor, critic, PPOConfig(epochs=1, minibatch_size=1)).update(buffer)
        self.assertEqual(metrics["actor_updates"], 0)
        self.assertEqual(metrics["critic_updates"], 1)
        self.assertTrue(all(torch.equal(before[key], value) for key, value in actor.state_dict().items()))

    def test_numerical_commit_failure_closes_current_action(self):
        class FailingRunner(MockRunner):
            def commit(self, action):
                if action is not None:
                    self.last_commit_info = {"requested_write": True, "raft_called": True,
                                             "geometry_called": True, "kda_updated": False}
                    raise FloatingPointError("mock invalid state")
                return super().commit(action)
        environment = Environment(FailingRunner(), ProbeReward())
        environment.reset([0, 1, 2])
        _, reward, done, info = environment.step(1)
        self.assertTrue(done)
        self.assertTrue(info["failure"])
        self.assertEqual(reward, -5)
        self.assertTrue(info["write_diagnostics"]["raft_called"])

    def test_learned_ppo_update_is_finite_and_uses_stored_small_vectors(self):
        actor, critic = Actor(), Critic()
        buffer = RolloutBuffer(4)
        mask = torch.tensor([True, True])
        with torch.no_grad():
            for index in range(4):
                observation = torch.full((30,), float(index) / 3)
                distribution = torch.distributions.Categorical(logits=actor(observation))
                action = index % 2
                buffer.add(Transition(observation, observation, action,
                    distribution.log_prob(torch.tensor(action)).item(), critic(observation).item(),
                    float(index - 2), index == 3, mask))
        metrics = PPO(actor, critic, PPOConfig(epochs=1, minibatch_size=4)).update(buffer)
        self.assertEqual(metrics["actor_updates"], 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["policy_loss"])))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in actor.parameters()))
        self.assertFalse(any(t.observation.requires_grad for t in buffer.transitions))


if __name__ == "__main__":
    unittest.main()
