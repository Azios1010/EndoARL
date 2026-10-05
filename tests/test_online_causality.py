import unittest
from copy import deepcopy

import torch

from dust3r.memory.policies import NeverWritePolicy
from dust3r.online.runner import ReconstructionFailure
from dust3r.rl.environment import Environment
from dust3r.rl.rewards import DenseGTReward
from tests.hybrid_helpers import frames, make_runner
from tests.test_uncertainty import StubFlow
from dust3r.memory.config import UncertaintyConfig
from dust3r.memory.uncertainty import UncertaintyEstimator


class OnlineCausalityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)

    def test_forced_write_then_skip_preserves_long_state(self):
        runner = make_runner(NeverWritePolicy())
        runner.reset(frames())
        first = runner.prepare_next()
        with self.assertRaisesRegex(ValueError, 'forced'):
            runner.commit(0)
        runner.commit(1)
        long_state = runner.state.kda_state.clone()
        reference = runner.state.last_long_reference_pointmap.clone()
        second = runner.prepare_next()
        runner.commit(0)
        self.assertTrue(torch.equal(runner.state.kda_state, long_state))
        self.assertTrue(torch.equal(runner.state.last_long_reference_pointmap, reference))
        self.assertEqual(runner.state.write_count, 1)
        self.assertEqual(runner.state.short_occupancy, 2)
        self.assertFalse(runner.last_commit_info['raft_called'])

    def test_arrival_prefix_does_not_depend_on_future_frames(self):
        runner1 = make_runner()
        runner2 = make_runner()
        runner2.model.load_state_dict(runner1.model.state_dict())
        runner2.memory.load_state_dict(runner1.memory.state_dict())
        inputs = frames()
        modified = deepcopy(inputs)
        modified[2]['img'].fill_(100)
        modified[3]['img'].fill_(-100)
        runner1.reset(inputs)
        runner2.reset(modified)
        pair1, pair2 = runner1.prepare_next(), runner2.prepare_next()
        self.assertTrue(torch.equal(pair1.res1['pts3d'], pair2.res1['pts3d']))
        self.assertTrue(torch.equal(runner1.observation().observation, runner2.observation().observation))
        runner1.commit(1)
        runner2.commit(1)
        self.assertTrue(torch.equal(runner1.state.kda_state, runner2.state.kda_state))

    def test_action_changes_future_not_already_decoded_outputs(self):
        first = make_runner()
        second = make_runner()
        second.model.load_state_dict(first.model.state_dict())
        second.memory.load_state_dict(first.memory.state_dict())
        for runner in (first, second):
            runner.reset(frames())
            runner.prepare_next()
            runner.commit(1)
        current1, current2 = first.prepare_next(), second.prepare_next()
        before = current1.res1['pts3d'].clone()
        first.commit(1)
        second.commit(0)
        self.assertTrue(torch.equal(current1.res1['pts3d'], before))
        self.assertTrue(torch.equal(current1.res1['pts3d'], current2.res1['pts3d']))
        future1, future2 = first.prepare_next(), second.prepare_next()
        self.assertFalse(torch.equal(future1.res1['pts3d'], future2.res1['pts3d']))

    def test_terminal_pair_short_only_and_reset(self):
        runner = make_runner()
        runner.reset(frames(3), episode_id='patient-a')
        runner.prepare_next()
        runner.commit(1)
        last = runner.prepare_next()
        self.assertTrue(last.terminal)
        with self.assertRaisesRegex(ValueError, 'Terminal'):
            runner.commit(1)
        runner.commit(None)
        self.assertEqual(runner.state.write_count, 1)
        self.assertEqual(runner.state.short_occupancy, 2)
        self.assertIsNone(runner.prepare_next())
        runner.reset(frames(2), episode_id='patient-b')
        pair = runner.prepare_next()
        self.assertFalse(runner.state.has_long_memory)
        self.assertEqual(runner.state.write_count, 0)
        self.assertEqual(runner.state.short_occupancy, 0)
        runner.commit(None)
        self.assertEqual(runner.state.short_occupancy, 1)

    def test_raft_only_write_and_last_accepted_reference(self):
        flow = StubFlow()
        estimator = UncertaintyEstimator(UncertaintyConfig(), flow_model=flow, patch_hw=(2, 2))
        runner = make_runner(uncertainty=estimator)
        runner.reset(frames(5))
        runner.prepare_next()
        runner.commit(1)
        self.assertEqual(flow.calls, 0)
        seed_reference = runner.state.last_long_reference_index
        runner.prepare_next()
        runner.commit(0)
        self.assertEqual(flow.calls, 0)
        self.assertEqual(runner.state.last_long_reference_index, seed_reference)
        runner.prepare_next()
        runner.commit(1)
        self.assertEqual(flow.calls, 1)
        self.assertEqual(runner.state.raft_call_count, 1)
        self.assertEqual(runner.state.last_long_reference_index, 2)
        runner.prepare_next()
        runner.commit(None)
        self.assertEqual(flow.calls, 1)

    def test_nonfinite_maps_fail_before_value_encoder(self):
        runner = make_runner()
        original = runner.model.downstream_head
        def invalid(*args):
            value = original(*args)
            value['pts3d'][0, 0, 0, 0] = float('nan')
            return value
        runner.model.downstream_head = invalid
        runner.reset(frames())
        with self.assertRaises(ReconstructionFailure):
            runner.prepare_next()
        self.assertEqual(runner.state.short_occupancy, 0)
        self.assertFalse(runner.state.has_long_memory)

    def test_failed_update_retains_incurred_cost_without_committing_reference(self):
        from unittest.mock import patch
        runner = make_runner()
        runner.reset(frames())
        runner.prepare_next()
        with patch.object(runner.memory.long.adapter, 'update', side_effect=FloatingPointError('kernel overflow')):
            with self.assertRaises(FloatingPointError):
                runner.commit(1)
        self.assertTrue(runner.last_commit_info['kda_updated'])
        self.assertFalse(runner.state.has_long_memory)
        self.assertEqual(runner.state.write_count, 0)
        self.assertIsNone(runner.state.last_long_reference_pointmap)

    def test_real_runner_environment_delayed_reward(self):
        runner = make_runner()
        targets = {'pointmap': torch.ones(1, 4, 4, 3), 'valid_mask': torch.ones(1, 4, 4, dtype=torch.bool),
                   'provenance_verified': True}
        env = Environment(runner, DenseGTReward(raft_weight=0, kda_geometry_weight=0),
                          gt_provider=lambda index: targets)
        packet = env.reset(frames(4))
        self.assertEqual(packet.observation.shape, (1, 30))
        packet, reward, done, info = env.step(1)
        self.assertEqual(info['candidate_index'], 0)
        self.assertEqual(info['reward_prediction_index'], 1)
        self.assertFalse(done)
        packet, reward, done, info = env.step(0)
        self.assertEqual(info['candidate_index'], 1)
        self.assertEqual(info['reward_prediction_index'], 2)
        self.assertTrue(done)
        self.assertIsNone(packet)
        env.reset(frames(4), episode_id='another-patient')
        self.assertEqual(runner.state.write_count, 0)


if __name__ == '__main__':
    unittest.main()
