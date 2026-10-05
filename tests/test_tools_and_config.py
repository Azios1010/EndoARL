import json
from pathlib import Path
import tempfile
import unittest

import torch

from demo_hybrid import load_frames
from eval.eval_hybrid import evaluate_predictions
from dust3r.memory.config import load_config
from tools.check_memory_backends import probe
from tests.hybrid_helpers import tiny_config, make_runner, frames


class ToolsTests(unittest.TestCase):
    def test_config_rejects_nonfinite_and_ignored_precision(self):
        for values in ({'rl': {'entropy_coef': float('nan')}},
                       {'memory': {'long': {'initial_frame_decay_rate': float('inf')}}},
                       {'training': {'pretrained_compute_dtype': 'bfloat16'}},
                       {'rl': {'actor_hidden': [64, -1]}}, {'rl': {'reward_raft_weight': -1}}):
            with self.assertRaises(ValueError):
                load_config(values)

    def test_reference_conformance_probe(self):
        report = probe(tiny_config(), reference=True)
        self.assertTrue(report['passed'])
        self.assertEqual(report['actual_short_backend'], 'math_reference')
        self.assertEqual(report['actual_long_backend'], 'torch_reference')
        self.assertEqual(report['output_shape'], [1, 4, 8])

    def test_real_gt_evaluation_has_fixed_scale_and_no_new_mask(self):
        targets = [dict(pointmap=torch.ones(1, 4, 4, 3),
                        valid_mask=torch.ones(1, 4, 4, dtype=torch.bool), depth=torch.ones(1, 4, 4),
                        camera_to_world=torch.eye(4)[None], intrinsics=torch.eye(3)[None],
                        provenance_verified=True) for _ in range(2)]
        outputs = [{'pts3d': torch.ones(1, 4, 4, 3) * 2}, {'pts3d_in_other_view': torch.ones(1, 4, 4, 3) * 4}]
        report = evaluate_predictions(outputs, targets)
        self.assertEqual(report['first_scale'], .5)
        self.assertEqual(report['frame_metrics'][0]['point_l1_normalized'], 0)
        self.assertGreater(report['frame_metrics'][1]['point_l1_normalized'], 0)
        outputs[1]['pts3d_in_other_view'][0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            evaluate_predictions(outputs, targets)

    def test_local_rgb_loader_produces_runner_contract(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            for i in range(3):
                Image.new('RGB', (8, 4), color=(10 + i, 20, 30)).save(Path(directory) / f'{i:03}.png')
            loaded = load_frames(directory, (4, 4), 3)
            self.assertEqual(loaded[0]['origin_img'].shape, (1, 4, 4, 3))
            self.assertEqual(loaded[0]['origin_img'].dtype, torch.uint8)
            self.assertEqual(loaded[0]['img'].shape, (1, 3, 4, 4))
            self.assertTrue((loaded[0]['img'] >= -1).all() and (loaded[0]['img'] <= 1).all())
            runner = make_runner()
            runner.reset(loaded)
            runner.prepare_next()
            runner.commit(1)


if __name__ == '__main__':
    unittest.main()
