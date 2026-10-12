"""Bounded CPU streaming and official-forward ordering, without legacy imports."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from dust3r.online.streaming import DeviceSequence, LegacyOfficialStream, stream_predictions
from tests.hybrid_helpers import TinyEndoModel, frames, make_runner


class FakeOfficialModel(TinyEndoModel):
    def __init__(self):
        super().__init__()
        self.events = []

    def encode_frames(self, *args):
        self.events.append('encode')
        return super().encode_frames(*args)

    def decode(self, *args):
        self.events.append('decode')
        return super().decode(*args)

    def encode_feat_key(self, raw, decoded, number):
        self.events.append('key' + str(number))
        return super().encode_feat_key(raw, decoded, number)

    def downstream_head(self, dec, shape, number):
        self.events.append('head' + str(number))
        return super().downstream_head(dec, shape, number)

    def encode_cur_value(self, *args):
        self.events.append('value')
        return super().encode_cur_value(*args)


class FakeOfficialMemory:
    def __init__(self, model, accepted):
        self.model, self.accepted = model, set(accepted)
        self.spatial_buffer = []
        self.temporal_buffer = []
        self.calls = []
        self.interval = None

    def memory_read(self, key, res):
        self.model.events.append('read')
        if not res:
            raise AssertionError('Legacy official read must request its residual.')
        return key + .1

    def add_mem_check(self, key, value, **kwargs):
        self.model.events.append('write')
        self.calls.append(kwargs)
        index = kwargs['index']
        self.temporal_buffer.append(index)
        self.temporal_buffer = self.temporal_buffer[-4:]
        if index in self.accepted:
            self.spatial_buffer.append(index)


class FakeSample:
    def __init__(self, count=6):
        self.frame_count = count
        self.calls = []

    def get_frame(self, index):
        self.calls.append(index)
        frame = dict(img=torch.full((1, 3, 4, 4), (index + 1) / 10.),
                     origin_img=torch.full((1, 4, 4, 3), index, dtype=torch.uint8),
                     frame_index=index)
        target = dict(pointmap=torch.full((1, 4, 4, 3), float(index)),
                      valid_mask=torch.ones(1, 4, 4, dtype=torch.bool), index=index)
        return frame, target


class StreamingRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def official(self, count=6, accepted=(0, 2, 4), uncertainty_check=True):
        model = FakeOfficialModel().eval()
        memory = FakeOfficialMemory(model, accepted)
        runner = LegacyOfficialStream(model, count, uncertainty_check=uncertainty_check,
                                      memory_factory=lambda *args: memory)
        return runner, model, memory

    def test_released_operation_order_and_terminal_write(self):
        runner, model, memory = self.official(count=3, accepted=(0, 1))
        inputs = frames(3)
        self.assertIsNone(runner.step(inputs[0]))
        self.assertEqual(model.events, [])
        first = runner.step(inputs[1])
        self.assertEqual(model.events, ['encode', 'decode', 'key1', 'key2', 'head1', 'head2', 'value', 'write'])
        model.events.clear()
        terminal = runner.step(inputs[2], terminal=True)
        self.assertEqual(model.events, ['encode', 'read', 'decode', 'key1', 'key2', 'head1', 'head2', 'value', 'write'])
        self.assertFalse(first.terminal)
        self.assertTrue(terminal.terminal)
        self.assertEqual([item['index'] for item in memory.calls], [0, 1])
        self.assertEqual(runner.write_attempt_count, 2)
        self.assertEqual(runner.write_count, 2)
        self.assertEqual(runner.raft_call_count, 1)
        self.assertTrue(runner.last_commit_info['terminal_legacy_write'])
        self.assertIs(runner.reference_points, terminal.res1['pts3d'])

    def test_last_accepted_reference_survives_rejected_candidates(self):
        runner, _, memory = self.official()
        sample = FakeSample()
        outputs = []
        for index in range(sample.frame_count):
            frame, _ = sample.get_frame(index)
            pair = runner.step(frame, terminal=index == sample.frame_count - 1)
            if pair is not None:
                outputs.append(pair)
        refs = [item.get('ref_view', {}).get('frame_index') for item in memory.calls]
        self.assertEqual(refs, [None, 0, 0, 2, 2])
        self.assertIs(memory.calls[1]['ref_pointmap'], outputs[0].res1['pts3d'])
        self.assertIs(memory.calls[3]['ref_pointmap'], outputs[2].res1['pts3d'])
        self.assertEqual(runner.reference_index, 4)
        self.assertEqual(runner.write_count, 3)
        self.assertEqual(runner.raft_call_count, 2)
        self.assertEqual(memory.temporal_buffer, [1, 2, 3, 4])
        self.assertEqual(len(runner.cache), 6)
        self.assertFalse(any(isinstance(value, list) for value in vars(runner).values()))

    def test_disabled_uncertainty_uses_legacy_flag_without_raft(self):
        runner, _, memory = self.official(count=4, accepted=(0, 1, 2), uncertainty_check=False)
        for index, frame in enumerate(frames(4)):
            runner.step(frame, terminal=index == 3)
        self.assertEqual(runner.raft_call_count, 0)
        self.assertTrue(all(not item['check_uncertainty'] for item in memory.calls))
        self.assertTrue(all('ref_view' not in item for item in memory.calls))

    def test_legacy_runner_requires_eval_and_exact_terminal_marker(self):
        model = FakeOfficialModel()
        with self.assertRaisesRegex(ValueError, 'inference only'):
            LegacyOfficialStream(model, 3, memory_factory=lambda *args: None)
        runner, _, _ = self.official(count=3)
        with self.assertRaisesRegex(ValueError, 'At least two'):
            LegacyOfficialStream(model.eval(), 1, memory_factory=lambda *args: None)
        inputs = frames(3)
        runner.step(inputs[0])
        with self.assertRaisesRegex(ValueError, 'terminal marker'):
            runner.step(inputs[1], terminal=True)
        runner.step(inputs[1])
        with self.assertRaisesRegex(ValueError, 'terminal marker'):
            runner.step(inputs[2], terminal=False)

    def test_episode_recreation_resets_legacy_reference_and_counters(self):
        runner, model, memory = self.official(count=3, accepted=(0, 1))
        for index, frame in enumerate(frames(3)):
            runner.step(frame, terminal=index == 2)
        clean_memory = FakeOfficialMemory(model, (0,))
        next_episode = LegacyOfficialStream(model, 3, episode_id='patient_b',
            memory_factory=lambda *args: clean_memory)
        self.assertEqual(next_episode.write_count, 0)
        self.assertEqual(next_episode.raft_call_count, 0)
        self.assertIsNone(next_episode.reference_points)
        self.assertIsNone(next_episode.left)
        self.assertEqual(next_episode.cache, (None,) * 6)
        self.assertEqual(clean_memory.spatial_buffer, [])
        self.assertIsNot(next_episode.state, memory)

    def test_device_sequence_residency_cache_hits_and_eviction(self):
        sample = FakeSample(count=20)
        sequence = DeviceSequence(sample, 'cpu')
        for index in range(len(sequence)):
            value = sequence[index]
            self.assertEqual(value['frame_index'], index)
            self.assertLessEqual(len(sequence.cache), 2)
        self.assertEqual(list(sequence.cache), [18, 19])
        self.assertIs(sequence[-1], sequence[19])
        self.assertEqual(sample.calls.count(19), 1)
        sequence[18]
        sequence[17]
        self.assertEqual(list(sequence.cache), [18, 17])
        targets = DeviceSequence(sample, 'cpu', field='target', max_cached=1)
        self.assertEqual(targets[0]['index'], 0)
        self.assertEqual(targets[1]['index'], 1)
        self.assertEqual(len(targets.cache), 1)
        with self.assertRaises(TypeError):
            sequence[:2]
        with self.assertRaises(IndexError):
            sequence[20]
        with self.assertRaises(IndexError):
            sequence[-21]

    def test_device_sequence_rejects_unbounded_requested_cache(self):
        sample = FakeSample()
        for size in (0, 3):
            with self.assertRaises(ValueError):
                DeviceSequence(sample, 'cpu', max_cached=size)

    def test_stream_yields_each_final_frame_once_with_cpu_targets(self):
        sample = FakeSample(count=5)
        runner, _, _ = self.official(count=5, accepted=(0, 2))
        seen = []
        for index, prediction, target in stream_predictions(runner, sample, 'cpu'):
            seen.append(index)
            self.assertEqual(target['index'], index)
            self.assertEqual(prediction['pts3d'].shape, (1, 4, 4, 3))
            self.assertEqual(target['pointmap'].device.type, 'cpu')
        self.assertEqual(seen, [0, 1, 2, 3, 4])

    def test_hybrid_reset_clears_uncertainty_camera_cache(self):
        runner = make_runner()
        runner.uncertainty._predicted_intrinsics = torch.ones(1, 3, 3)
        runner.uncertainty._reference_pointmap = torch.ones(1, 4, 4, 3)
        runner.uncertainty.last_diagnostics = {'old_episode': True}
        runner.reset(episode_id='another_patient')
        self.assertIsNone(runner.uncertainty._predicted_intrinsics)
        self.assertIsNone(runner.uncertainty._reference_pointmap)
        self.assertEqual(runner.uncertainty.last_diagnostics, {})
        self.assertIsNone(runner.state)


if __name__ == '__main__':
    unittest.main()
