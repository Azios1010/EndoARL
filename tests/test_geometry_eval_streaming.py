"""Synthetic streamed GT geometry metrics match the preserved list evaluator."""
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import torch

from eval.eval_hybrid import evaluate_predictions, evaluate_stream, main, subset_fingerprint


def predicted_points(index):
    y, x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
    z = 2. + .1 * index + .03 * x + .02 * y
    return torch.stack(((x - 2.) * .1, (y - 2.) * .1, z), -1).float().unsqueeze(0)


class SyntheticSample:
    def __init__(self, count=7):
        self.frame_count = count
        self.episode_id = 'synthetic_patient:0'
        self.calls = []

    def get_frame(self, index):
        self.calls.append(index)
        image = torch.full((1, 3, 4, 4), float(index))
        frame = {'img': image, 'origin_img': torch.zeros(1, 4, 4, 3, dtype=torch.uint8),
                 'idx': index, 'instance': 'synthetic:' + str(index)}
        gt = predicted_points(index) * (2. if index == 0 else 2. + .01 * index)
        mask = torch.ones(1, 4, 4, dtype=torch.bool)
        if index % 2:
            mask[:, 0, 0] = False
        angle = torch.tensor(.02 * index)
        c, s = torch.cos(angle), torch.sin(angle)
        pose = torch.eye(4).unsqueeze(0)
        pose[0, :3, :3] = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pose[0, 0, 3] = .01 * index
        inverse = torch.linalg.inv(pose)
        camera_gt = torch.einsum('bhwc,bdc->bhwd', gt, inverse[:, :3, :3]) + inverse[:, None, None, :3, 3]
        target = {'pointmap': gt, 'valid_mask': mask,
                  'depth': camera_gt[..., 2] if index % 2 else None,
                  'intrinsics': torch.tensor([[[10., 0, 2.], [0, 10., 2.], [0, 0, 1.]]]),
                  'camera_to_world': pose, 'provenance_verified': True}
        return frame, target


class SyntheticRunner:
    """Online runner stub accepts image-only inputs, never any target tensor."""
    def __init__(self):
        self.left = None
        self.seen_input_keys = []
        self.maximum_resident_frames = 0

    def step(self, frame, terminal=False):
        self.seen_input_keys.append(set(frame))
        if self.left is None:
            self.left = frame
            self.maximum_resident_frames = 1
            return None
        self.maximum_resident_frames = max(self.maximum_resident_frames, 2)
        left, right = int(self.left['idx']), int(frame['idx'])
        self.left = frame
        return SimpleNamespace(res1={'pts3d': predicted_points(left)},
                               res2={'pts3d': predicted_points(right)},
                               candidate_index=left, arrival_index=right, terminal=terminal)


class GeometryStreamingTests(unittest.TestCase):
    def reference(self, sample, pose=False):
        targets = [sample.get_frame(i)[1] for i in range(sample.frame_count)]
        predictions = [{'pts3d': predicted_points(i)} for i in range(sample.frame_count)]
        return evaluate_predictions(predictions, targets, pose=pose)

    def test_identical_scalar_metrics_to_list_api(self):
        sample = SyntheticSample()
        expected = self.reference(sample)
        runner = SyntheticRunner()
        actual = evaluate_stream(sample, runner)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual['frame_metrics']), sample.frame_count)
        self.assertEqual(runner.maximum_resident_frames, 2)
        self.assertAlmostEqual(actual['first_scale'], 2., places=6)
        self.assertGreater(actual['frame_metrics'][-1]['point_l1_normalized'], 0)

    def test_gt_calibration_and_future_targets_never_enter_runner_inputs(self):
        sample = SyntheticSample(count=20)
        runner = SyntheticRunner()
        result = evaluate_stream(sample, runner)
        self.assertEqual(len(result['frame_metrics']), 20)
        for keys in runner.seen_input_keys:
            self.assertEqual(keys, {'img', 'origin_img', 'idx', 'instance'})
            self.assertFalse(keys & {'pointmap', 'valid_mask', 'depth', 'intrinsics', 'camera_to_world'})
        # Only Python scalar metrics remain in the output; no sequence GPU tensors.
        json.dumps(result, allow_nan=False)

    def test_pose_option_matches_list_api_without_runtime_opencv(self):
        sample = SyntheticSample(count=3)
        fake_pose = {'pose_success': 1., 'pose_rotation_degrees': .5, 'pose_translation_m': .01}
        with patch('eval.eval_hybrid.evaluate_pose', return_value=fake_pose):
            expected = self.reference(sample, pose=True)
            actual = evaluate_stream(sample, SyntheticRunner(), pose=True)
        self.assertEqual(actual, expected)
        self.assertTrue(actual['pose_uses_verified_intrinsics'])

    def test_nonfinite_prediction_is_failure_not_masked_out(self):
        runner = SyntheticRunner()
        original = runner.step
        def invalid(frame, terminal=False):
            pair = original(frame, terminal)
            if pair is not None:
                pair.res1['pts3d'][0, 0, 0, 0] = float('nan')
            return pair
        runner.step = invalid
        with self.assertRaises(FloatingPointError):
            evaluate_stream(SyntheticSample(), runner)

    def test_unverified_provenance_is_rejected(self):
        sample = SyntheticSample()
        original = sample.get_frame
        def invalid(index):
            frame, target = original(index)
            target['provenance_verified'] = False
            return frame, target
        sample.get_frame = invalid
        with self.assertRaisesRegex(ValueError, 'provenance'):
            evaluate_stream(sample, SyntheticRunner())

    def test_subset_fingerprint_binds_order_and_frame_ids(self):
        full = [{'episode_id': 'patient1:0', 'source_frame_ids': [0, 10, 20]}]
        prefix = [{'episode_id': 'patient1:0', 'source_frame_ids': [0, 10]}]
        reordered = [{'episode_id': 'patient1:0', 'source_frame_ids': [0, 20, 10]}]
        self.assertEqual(subset_fingerprint(full), subset_fingerprint(full))
        self.assertNotEqual(subset_fingerprint(full), subset_fingerprint(prefix))
        self.assertNotEqual(subset_fingerprint(full), subset_fingerprint(reordered))

    def test_max_frames_less_than_two_is_rejected_before_model_load(self):
        with self.assertRaises(SystemExit) as exc:
            main(['--max-frames', '1'])
        self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
