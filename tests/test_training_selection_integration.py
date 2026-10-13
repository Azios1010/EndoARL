"""CPU synthetic tests: selected TRAIN targets must not rebind held-out GT."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dust3r.memory.config import load_config
from dust3r.training.data import make_training_dataset
from dust3r.training.teacher import file_sha256
from dust3r.training.provenance import environment_contract


class TrainingSelectionIntegrationTests(unittest.TestCase):
    def fixture(self, root):
        sequences = []
        identity = np.eye(4).tolist()
        for split, dataset_id in [('train', 1), ('validation', 7), ('test', 8)]:
            frames = []
            for index in range(5):
                rgb, depth = root / f'{split}_rgb{index}.npy', root / f'{split}_depth{index}.npy'
                np.save(rgb, np.zeros((4, 6, 3), dtype=np.uint8))
                np.save(depth, np.full((4, 6), 1 if index != 2 or split != 'train' else 0, dtype=np.float32))
                frames.append({'frame_id': index, 'rgb': rgb.name, 'depth': depth.name,
                               'depth_scale': 1, 'valid_mask_rule': 'depth_positive_finite',
                               'intrinsics': [[10, 0, 2], [0, 10, 1], [0, 0, 1]],
                               'camera_to_world': identity})
            sequences.append({'id': split, 'patient_id': f'patient{dataset_id}', 'split': split,
                              'provenance': {'verified': True, 'source': 'synthetic test',
                                             'depth_or_pointmap_gt': True, 'calibration_gt': True,
                                             'pose_gt': True, 'units': 'm', 'dataset_id': dataset_id},
                              'frames': frames})
        source = root / 'full.json'
        source.write_text(json.dumps({'schema_version': 1, 'sequences': sequences}), encoding='utf-8')
        return source

    def test_train_filter_preserves_full_manifest_certification_and_actor_contract(self):
        from dust3r.data.supervision import audit_supervision, build_training_manifest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            original_hash = file_sha256(source)
            inventory = audit_supervision(source, image_hw=(4, 6), clip_raw_frames=2)
            selected = root / 'selected.json'
            selected.write_text(json.dumps(build_training_manifest(source, inventory)), encoding='utf-8')
            config = load_config({'training': {'data_manifest': str(source),
                                              'image_height': 4, 'image_width': 6,
                                              'sequence_raw_frames': 2, 'tbptt_pairs': 1}})
            before = environment_contract(config, {'endo3r': 'released'}, 'frozen')
            raw = config.to_dict()
            raw['training']['supervision_manifest'] = str(selected)
            config = load_config(raw)
            dataset, contract = make_training_dataset(config, sequence_raw_frames=2)
            self.assertEqual(config.training.data_manifest, str(source))
            self.assertEqual(contract['source_manifest_sha256'], original_hash)
            self.assertNotEqual(contract['supervision_manifest_sha256'], original_hash)
            self.assertEqual(file_sha256(source), original_hash)
            self.assertEqual(environment_contract(config, {'endo3r': 'released'}, 'frozen'), before)
            self.assertTrue(all(sequence['split'] == 'train' for sequence, _, _ in dataset.samples))
            for index in range(len(dataset)):
                self.assertTrue(all(target['valid_mask'].any() for target in dataset[index].targets))

    def test_direct_training_entrypoints_do_not_choose_smoke_budgets(self):
        from train_hybrid_memory import main as hybrid_main
        from train_rl_controller import main as ppo_main
        for entrypoint, flags in [(hybrid_main, []), (ppo_main, [])]:
            with self.subTest(entrypoint=entrypoint.__module__):
                output = io.StringIO()
                with redirect_stdout(output):
                    entrypoint(['--reference', '--dry-run', *flags])
                report = json.loads(output.getvalue())
                self.assertTrue(report['dry_run'])
                self.assertTrue(report['missing_training_budgets'])
                self.assertTrue(all(value is None for value in report['training_budgets'].values()))
                with self.assertRaisesRegex(ValueError, 'budget is unselected'):
                    entrypoint(['--reference', '--run-training', *flags])

    def test_study_keeps_supervision_out_of_official_evaluation_inputs(self):
        from dust3r.experiments.study import plan_study
        plan = plan_study('configs/experiments/scared_study.json', seeds=[0],
                          manifest='full_source.json', supervision_manifest='train_selected.json')
        for config in plan.configs.values():
            self.assertTrue(config['training']['data_manifest'].endswith('full_source.json'))
            if config['memory']['backend'] != 'legacy':
                self.assertTrue(config['training']['supervision_manifest'].endswith('train_selected.json'))
        for command in plan.commands:
            selected_required = any(str(path).endswith('train_selected.json') for path in command.required_files)
            if command.stage in ('baseline', 'certify', 'evaluate', 'benchmark'):
                self.assertFalse(selected_required)
            if command.stage in ('fit', 'ppo'):
                self.assertTrue(selected_required)


if __name__ == '__main__':
    unittest.main()
