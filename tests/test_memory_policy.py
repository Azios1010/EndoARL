"""CPU tests for the cheap actor boundary and binary policy masks."""
import unittest
from types import SimpleNamespace

import torch

from dust3r.memory.observations import ObservationBuilder, FEATURE_NAMES
from dust3r.memory.policies import (AlwaysWritePolicy, HeuristicWritePolicy,
                                    NeverWritePolicy, PPOWritePolicy)
from dust3r.memory.types import MemoryCandidate
from dust3r.rl.networks import Actor, Critic


def state_fixture(has_long=False):
    return SimpleNamespace(has_long_memory=has_long, current_step=0, write_count=int(has_long),
        processed_candidate_count=int(has_long), last_write_step=0, short_occupancy=0,
        recent_key_descriptors=torch.empty(1, 0, 1024), recent_long_descriptors=torch.empty(1, 0, 1024),
        last_long_descriptor=None, previous_uncertainty_stats=torch.zeros(1, 3),
        has_previous_raft_quality=False, predicted_radius_scale=None, has_predicted_radius_scale=False)


def observation_fixture(state=None, key=None):
    state = state or state_fixture()
    key = key if key is not None else torch.ones(1, 4, 1024, requires_grad=True)
    prediction = {"pts3d": torch.ones(1, 4, 4, 3, requires_grad=True),
                  "conf": torch.ones(1, 4, 4, requires_grad=True)}
    packet = ObservationBuilder().build(prediction, key, state)
    return packet, state, prediction, key


class MemoryPolicyTests(unittest.TestCase):
    def test_exact_state_schema_and_detach(self):
        packet, state, _, _ = observation_fixture()
        self.assertEqual(packet.observation.shape, (1, 30))
        self.assertEqual(len(FEATURE_NAMES), 30)
        self.assertFalse(packet.observation.requires_grad)
        self.assertTrue(torch.isfinite(packet.observation).all())
        self.assertTrue(state.has_predicted_radius_scale)
        self.assertEqual(state.write_count, 0)

    def test_forced_first_write_all_controllers(self):
        packet, _, _, _ = observation_fixture()
        for policy in (HeuristicWritePolicy(), AlwaysWritePolicy(), NeverWritePolicy(),
                       PPOWritePolicy(Actor(), Critic())):
            decision = policy.act(packet, deterministic=True)
            self.assertEqual(decision.action.item(), 1)
            self.assertTrue(decision.forced.item())
            self.assertEqual(decision.log_prob.item(), 0)

    def test_always_never_and_similarity(self):
        packet, _, _, _ = observation_fixture(state_fixture(True))
        packet.heuristic_similarity.fill_(0.95)
        self.assertEqual(HeuristicWritePolicy().act(packet).action.item(), 0)
        packet.heuristic_similarity.fill_(0.92)
        self.assertEqual(HeuristicWritePolicy().act(packet).action.item(), 1)
        self.assertEqual(AlwaysWritePolicy().act(packet).action.item(), 1)
        self.assertEqual(NeverWritePolicy().act(packet).action.item(), 0)

    def test_observation_cannot_accept_gt_future_or_current_raft(self):
        packet, state, prediction, key = observation_fixture()
        for forbidden in ("gt", "future_frames", "current_raft_uncertainty"):
            with self.assertRaises(TypeError):
                ObservationBuilder().build(prediction, key, state, **{forbidden: torch.rand(1)})
        self.assertEqual(packet.observation.shape[-1], 30)

    def test_historical_uncertainty_only_when_present(self):
        state = state_fixture(True)
        state.previous_uncertainty_stats[:] = torch.tensor([0.4, 0.1, 0.7])
        packet, _, _, _ = observation_fixture(state)
        self.assertTrue(torch.equal(packet.observation[0, 18:22], torch.zeros(4)))
        state.has_previous_raft_quality = True
        packet, _, _, _ = observation_fixture(state)
        self.assertTrue(torch.allclose(packet.observation[0, 18:22], torch.tensor([0.4, 0.2, 0.7, 1.])))

    def test_policy_observe_candidate_and_determinism(self):
        _, state, prediction, key = observation_fixture(state_fixture(True))
        candidate = MemoryCandidate(key, key, pointmap=prediction["pts3d"], confidence=prediction["conf"])
        policy = PPOWritePolicy(Actor())
        packet = policy.observe(candidate, state)
        first = policy.act(packet, deterministic=True)
        second = policy.act(packet, deterministic=True)
        self.assertTrue(torch.equal(first.action, second.action))

    def test_invalid_mask_and_entirely_invalid_prediction(self):
        packet, state, prediction, key = observation_fixture()
        with self.assertRaises(ValueError):
            AlwaysWritePolicy().act(packet, action_mask=torch.zeros(1, 2, dtype=torch.bool))
        prediction["pts3d"] = torch.full((1, 4, 4, 3), float("nan"))
        with self.assertRaises(ValueError):
            ObservationBuilder().build(prediction, key, state)

    def test_first_write_mask_cannot_be_weakened(self):
        packet, _, _, _ = observation_fixture()
        decision = NeverWritePolicy().act(packet, action_mask=torch.ones(1, 2, dtype=torch.bool))
        self.assertEqual(decision.action.item(), 1)

    def test_privileged_critic_separate_channel(self):
        packet, _, _, _ = observation_fixture(state_fixture(True))
        policy = PPOWritePolicy(Actor(), Critic(34))
        with self.assertRaises(ValueError):
            policy.act(packet)
        decision = policy.act(packet, deterministic=True, critic_observation=torch.zeros(1, 34))
        self.assertEqual(decision.value.shape, (1,))
        self.assertEqual(packet.observation.shape[-1], 30)


if __name__ == "__main__":
    unittest.main()
