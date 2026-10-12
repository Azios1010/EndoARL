"""CPU provenance/RNG/config tests; no training or dataset/checkpoint downloads."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dust3r.memory.config import load_config
from dust3r.training.provenance import (
    canonical_sha256, capture_rng_state, environment_contract, environment_contract_sha256,
    restore_rng_state, runtime_backend_provenance, seed_loaded_opencv, source_provenance,
    verify_actor_environment,
)
from tests.hybrid_helpers import tiny_config, TinyEndoModel


HASHES = {'dust3r': 'a' * 64, 'endo3r': 'b' * 64, 'raft': 'c' * 64}
HYBRID_HASH = 'd' * 64


def actor_payload(config):
    from dust3r.rl.networks import Actor
    from dust3r.rl.normalization import ObservationNormalizer
    contract = environment_contract(config, HASHES, HYBRID_HASH)
    digest = canonical_sha256(contract)
    return {'schema': 'endo3r_ppo_write_v1', 'observation_schema': 'actor30_v1',
            'actor': Actor(hidden=config.rl.actor_hidden).state_dict(),
            'normalizer': ObservationNormalizer().state_dict(),
            'original_checkpoint_hashes': HASHES,
            'environment_checkpoint_id': digest, 'environment_contract': contract,
            'environment_contract_sha256': digest}


class EnvironmentContractTests(unittest.TestCase):
    def test_heuristic_and_ppo_share_environment_despite_training_metadata(self):
        original = tiny_config()
        edited = original.to_dict()
        edited['policy']['long_write_policy'] = 'ppo'
        edited['policy']['heuristic_similarity'] = .5
        edited['training'].update(seed=17, data_manifest='/server/different/location.json', lr_new=.0002)
        edited['checkpoints']['endo3r'] = '/server/checkpoints/same_content.pth'
        edited['rl'].update(actor_hidden=[32], actor_lr=.001, reward_write_weight=.1)
        changed = load_config(edited)
        self.assertEqual(environment_contract_sha256(original, HASHES, HYBRID_HASH),
                         environment_contract_sha256(changed, HASHES, HYBRID_HASH))

    def test_shape_precision_stride_and_uncertainty_change_identity(self):
        original = tiny_config()
        expected = environment_contract_sha256(original, HASHES, HYBRID_HASH)
        variants = []
        for group, key, value in [('training', 'image_width', 640), ('training', 'stride', 2),
                                  ('training', 'new_memory_dtype', 'bfloat16'),
                                  ('uncertainty', 'metric', 'sampson')]:
            changed = original.to_dict()
            changed[group][key] = value
            variants.append(load_config(changed))
        changed = original.to_dict()
        changed['memory']['short'].update(model_dim=16, num_heads=2, head_dim=8)
        variants.append(load_config(changed))
        for variant in variants:
            self.assertNotEqual(expected, environment_contract_sha256(variant, HASHES, HYBRID_HASH))
        altered_hashes = dict(HASHES, endo3r='e' * 64)
        self.assertNotEqual(expected, environment_contract_sha256(original, altered_hashes, HYBRID_HASH))
        self.assertNotEqual(expected, environment_contract_sha256(original, HASHES, 'f' * 64))

    def test_missing_mismatched_and_tampered_contracts_rejected(self):
        config = tiny_config()
        payload = actor_payload(config)
        self.assertEqual(verify_actor_environment(payload, config, HASHES, HYBRID_HASH),
                         payload['environment_contract_sha256'])
        old = deepcopy(payload)
        old.pop('environment_contract_sha256')
        with self.assertRaisesRegex(ValueError, 'Retrain'):
            verify_actor_environment(old, config, HASHES, HYBRID_HASH)
        sampson = config.to_dict()
        sampson['uncertainty']['metric'] = 'sampson'
        with self.assertRaisesRegex(ValueError, 'uncertainty.metric changed'):
            verify_actor_environment(payload, load_config(sampson), HASHES, HYBRID_HASH)
        tampered = deepcopy(payload)
        tampered['environment_contract']['input']['frame_stride'] = 2
        with self.assertRaisesRegex(ValueError, 'internally inconsistent'):
            verify_actor_environment(tampered, config, HASHES, HYBRID_HASH)

    def test_inference_loader_enforces_environment_before_policy_deployment(self):
        from demo_hybrid import load_inference_model
        cfg = tiny_config().to_dict()
        cfg['policy']['long_write_policy'] = 'ppo'
        config = load_config(cfg)
        payload = actor_payload(config)
        model = TinyEndoModel()
        def initialize(effective, *args):
            model.memory_config = effective
            return model, HASHES
        with patch('pathlib.Path.is_file', return_value=True), \
                patch('dust3r.training.hybrid_trainer.load_initialized_model', side_effect=initialize), \
                patch('dust3r.training.hybrid_trainer.load_training_checkpoint'), \
                patch('dust3r.training.teacher.file_sha256', return_value=HYBRID_HASH), \
                patch('torch.load', return_value=payload):
            loaded, _ = load_inference_model(config, hybrid_checkpoint='hybrid.pt', actor_checkpoint='actor.pt')
            self.assertEqual(loaded.actor_environment_contract_sha256, payload['environment_contract_sha256'])
            cfg['uncertainty']['metric'] = 'sampson'
            with self.assertRaisesRegex(ValueError, 'uncertainty.metric changed'):
                load_inference_model(load_config(cfg), hybrid_checkpoint='hybrid.pt', actor_checkpoint='actor.pt')


class RNGAndSourceTests(unittest.TestCase):
    def test_rng_weights_only_roundtrip_and_exact_generator_draws(self):
        previous = capture_rng_state()
        try:
            random.seed(31)
            np.random.seed(31)
            torch.manual_seed(31)
            random.gauss(0, 1)  # exercise cached Gaussian state
            np.random.normal()
            saved = capture_rng_state()
            expected = (random.random(), random.gauss(0, 1), np.random.normal(size=3), torch.rand(4))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'rng.pt'
                torch.save(saved, path)
                loaded = torch.load(path, map_location='cpu', weights_only=True)
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            metadata = restore_rng_state(loaded)
            actual = (random.random(), random.gauss(0, 1), np.random.normal(size=3), torch.rand(4))
            self.assertEqual(expected[0:2], actual[0:2])
            self.assertTrue(np.array_equal(expected[2], actual[2]))
            self.assertTrue(torch.equal(expected[3], actual[3]))
            self.assertIn('no_video_or_rollout_cursor', metadata['resume_semantics'])
        finally:
            restore_rng_state(previous)

    def test_cuda_rng_paths_use_saved_generators_without_real_cuda(self):
        fake = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.get_rng_state_all', return_value=[fake]):
            saved = capture_rng_state()
        self.assertTrue(torch.equal(saved['torch_cuda'][0], fake))
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=1), \
                patch('torch.cuda.set_rng_state_all') as restore:
            result = restore_rng_state(saved)
            self.assertEqual(result['cuda_rng_devices_restored'], 1)
            self.assertTrue(torch.equal(restore.call_args.args[0][0], fake))
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'CUDA RNG device count'):
                restore_rng_state(saved)

    def test_incomplete_old_rng_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'complete RNG'):
            restore_rng_state({'torch_rng': torch.get_rng_state()})

    def test_source_archive_fingerprint_changes_only_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'demo_hybrid.py').write_text('answer = 1\n', encoding='utf-8')
            (root / 'configs').mkdir()
            (root / 'configs' / 'study.yaml').write_text('enabled: true\n', encoding='utf-8')
            with patch('subprocess.run', side_effect=FileNotFoundError):
                first = source_provenance(root)
                (root / 'dataset').mkdir()
                (root / 'dataset' / 'manifest.json').write_text('{}', encoding='utf-8')
                (root / 'results.json').write_text('{}', encoding='utf-8')
                (root / 'checkpoints').mkdir()
                (root / 'checkpoints' / 'model.pt').write_bytes(b'not_a_model')
                second = source_provenance(root)
                self.assertEqual(first['source_tree_sha256'], second['source_tree_sha256'])
                (root / 'demo_hybrid.py').write_text('answer = 2\n', encoding='utf-8')
                third = source_provenance(root)
            self.assertNotEqual(first['source_tree_sha256'], third['source_tree_sha256'])
            self.assertIsNone(first['git_commit'])
            self.assertEqual(first['source_file_count'], 2)

    def test_runtime_versions_do_not_import_optional_kernels(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'fla', 'flash_attn', 'triton'}:
                raise AssertionError('Optional kernel import attempted')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            result = runtime_backend_provenance()
        self.assertIsInstance(result['torch'], str)
        json.dumps(result, allow_nan=False)

    def test_opencv_reseed_is_explicit_without_importing_backend(self):
        fake = SimpleNamespace(setRNGSeed=lambda value: setattr(fake, 'seed', value))
        with patch.dict(sys.modules, {'cv2': fake}):
            seed_loaded_opencv(19)
        self.assertEqual(fake.seed, 19)


class CostAndEntryPointTests(unittest.TestCase):
    def calibration(self, config):
        return {'schema': 'endo3r_cost_calibration_v1', 'units': 'milliseconds',
                'baseline': 10., 'raft': 5., 'geometry': 1., 'kda': .5,
                'source_profile_sha256': 'a' * 64, 'hardware_id': 'synthetic_cpu',
                'image_hw': [config.training.image_height, config.training.image_width],
                'timing_scope': 'synchronized_exclusive_components_not_headline_throughput',
                'uncertainty_metric': config.uncertainty.metric,
                'uncertainty_enabled': config.uncertainty.enabled,
                'memory_config': config.to_dict()['memory'],
                'precision_config': {'new_memory_dtype': config.training.new_memory_dtype,
                                     'pretrained_compute_dtype': config.training.pretrained_compute_dtype},
                'hybrid_checkpoint_sha256': 'source_seed_0_audit_only',
                'checkpoint_hashes': HASHES, 'disclosure': 'synthetic test costs, never measured benchmark'}

    def test_cost_metadata_filtered_and_hardware_mode_bound_not_seed_weights(self):
        from train_rl_controller import load_cost_calibration
        config = tiny_config()
        payload = self.calibration(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cost.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            cost, metadata = load_cost_calibration(path, config, 'cpu', runtime={'hardware_id': 'synthetic_cpu'})
            self.assertEqual(cost.baseline, 10.)
            self.assertEqual(cost.image_hw, (256, 320))
            self.assertEqual(metadata['hybrid_checkpoint_sha256'], 'source_seed_0_audit_only')
            self.assertIn('reusable_across_seeds', metadata['source_weight_binding'])
            with self.assertRaisesRegex(ValueError, 'hardware/runtime'):
                load_cost_calibration(path, config, 'cpu', runtime={'hardware_id': 'different_cpu'})
            for field, value, message in [('image_hw', [512, 640], 'resolution'),
                                         ('uncertainty_metric', 'sampson', 'uncertainty metric'),
                                         ('uncertainty_enabled', True, 'enabled/disabled'),
                                         ('memory_config', {}, 'architecture'),
                                         ('precision_config', {}, 'precision')]:
                altered = dict(payload, **{field: value})
                path.write_text(json.dumps(altered), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, message):
                    load_cost_calibration(path, config, 'cpu', runtime={'hardware_id': 'synthetic_cpu'})

    def test_bare_or_unidentified_cost_json_is_rejected(self):
        from train_rl_controller import load_cost_calibration
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cost.json'
            path.write_text(json.dumps({'baseline': 10., 'raft': 5.}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Regenerate'):
                load_cost_calibration(path, tiny_config(), 'cpu', runtime={'hardware_id': 'synthetic_cpu'})

    def test_ppo_default_is_dry_without_loading_reconstruction(self):
        from train_rl_controller import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(tiny_config().to_dict()), encoding='utf-8')
            output = io.StringIO()
            with patch('dust3r.training.hybrid_trainer.load_initialized_model') as initialize, redirect_stdout(output):
                self.assertEqual(main(['--config', str(path)]), 0)
                initialize.assert_not_called()
            self.assertTrue(json.loads(output.getvalue())['dry_run'])
        from train_rl_controller import build_parser
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(['--dry-run', '--run-training'])


if __name__ == '__main__':
    unittest.main()
