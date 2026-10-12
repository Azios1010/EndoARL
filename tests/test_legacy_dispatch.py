"""Legacy identity tests without importing its unavailable heavyweight deps."""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
from types import MethodType
import unittest
from unittest.mock import patch

import torch

from tests.hybrid_helpers import TinyEndoModel, frames, make_runner


ROOT = Path(__file__).resolve().parents[1]
AUDITED_LEGACY_COMMIT = '8145c8e5535d2111974561e7d1cc06f1d008fe5e'


def source_trees():
    current = ast.parse((ROOT / 'dust3r/model.py').read_text(encoding='utf-8'))
    original = ast.parse(subprocess.run(['git', 'show', AUDITED_LEGACY_COMMIT + ':dust3r/model.py'], cwd=ROOT,
                        check=True, capture_output=True, text=True).stdout)
    return current, original


def node(tree, name):
    return next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)


def method(tree, name):
    return next(item for item in node(tree, 'Endo3R').body if isinstance(item, ast.FunctionDef) and item.name == name)


class StubLegacyMemory:
    def __init__(self, *args, **kwargs):
        self.spatial_buffer = []
        self.reads = 0

    def memory_read(self, query, res=True):
        self.reads += 1
        return query

    def add_mem_check(self, *args, index, **kwargs):
        self.spatial_buffer.append(index)


def compiled_forward(tree):
    scope = {'torch': torch, 'SpatialTempralMemory': StubLegacyMemory}
    definition = deepcopy(method(tree, 'forward'))
    code = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(code, '<verified-forward>', 'exec'), scope)
    return scope['forward']


class LegacyDispatchTests(unittest.TestCase):
    def test_legacy_ppo_loader_constructs_runner_dependencies(self):
        from demo_hybrid import load_inference_model
        from dust3r.memory.config import load_config
        from dust3r.rl.networks import Actor
        from dust3r.rl.normalization import ObservationNormalizer
        from dust3r.online.runner import OnlineRunner
        current, _ = source_trees()
        scope = {}
        definition = deepcopy(method(current, 'configure_memory'))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
                     '<configure-memory>', 'exec'), scope)
        model = TinyEndoModel()
        model.configure_memory = MethodType(scope['configure_memory'], model)
        config = load_config({'memory': {'backend': 'legacy'},
                              'baseline_identity': 'LEGACY_POLICY',
                              'policy': {'long_write_policy': 'ppo'}})
        def initialize(effective, *args):
            model.configure_memory(effective)
            return model, {}
        payload = dict(schema='endo3r_ppo_write_v1', observation_schema='actor30_v1',
                       actor=Actor().state_dict(), normalizer=ObservationNormalizer().state_dict(),
                       original_checkpoint_hashes={})
        from dust3r.training.provenance import environment_contract_sha256
        payload['environment_contract_sha256'] = environment_contract_sha256(config, {}, None)
        with patch('pathlib.Path.is_file', return_value=True), \
             patch('dust3r.training.hybrid_trainer.load_initialized_model', side_effect=initialize), \
             patch('torch.load', return_value=payload):
            loaded, _ = load_inference_model(config, actor_checkpoint='local-actor.pt')
        runner = OnlineRunner(loaded)
        self.assertFalse(runner.uncertainty.enabled)
        self.assertEqual(runner.config.policy.long_write_policy, 'ppo')

    def test_original_memory_class_is_unchanged(self):
        current, original = source_trees()
        self.assertEqual(ast.dump(node(current, 'SpatialTempralMemory')),
                         ast.dump(node(original, 'SpatialTempralMemory')))

    def test_legacy_forward_body_is_unchanged_after_dispatch(self):
        current, original = source_trees()
        modified = method(current, 'forward')
        baseline = method(original, 'forward')
        self.assertEqual([ast.dump(item) for item in modified.body[2:]],
                         [ast.dump(item) for item in baseline.body])

    def test_default_legacy_result_and_final_write_match_original(self):
        current, original = source_trees()
        model = TinyEndoModel().eval()
        forward_old = MethodType(compiled_forward(original), model)
        forward_new = MethodType(compiled_forward(current), model)
        old_pred, old_pairs, old_state = forward_old(frames(4), return_memory=True)
        new_pred, new_pairs, new_state = forward_new(frames(4), return_memory=True)
        self.assertEqual(old_state.spatial_buffer, [0, 1, 2])
        self.assertEqual(new_state.spatial_buffer, [0, 1, 2])
        self.assertEqual(len(old_pairs), len(new_pairs))
        for before, after in zip(old_pred, new_pred):
            self.assertEqual(set(before), set(after))
            for key in before:
                self.assertTrue(torch.equal(before[key], after[key]))

    def test_new_forward_dispatch_preserves_output_contract(self):
        current, _ = source_trees()
        runner = make_runner()
        model = runner.model
        model.memory_config = runner.config
        model.hybrid_memory, model.hybrid_uncertainty, model.long_write_policy = runner.memory, runner.uncertainty, runner.policy
        forward = MethodType(compiled_forward(current), model)
        with torch.no_grad():
            predictions, pairs, state = forward(frames(4), return_memory=True)
        self.assertEqual(len(predictions), 4)
        self.assertEqual(len(pairs), 3)
        self.assertIn('pts3d', predictions[0])
        self.assertTrue(all('pts3d_in_other_view' in value for value in predictions[1:]))
        self.assertEqual(state.write_count, 2)  # terminal write omitted only in new mode

    def test_optional_dependencies_are_not_imported_on_construction(self):
        import builtins
        original_import = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.split('.')[0] in {'fla', 'flash_attn', 'triton'}:
                raise AssertionError('Optional kernel import attempted during construction: ' + name)
            return original_import(name, *args, **kwargs)
        from dust3r.memory.config import MemoryConfig
        from dust3r.memory.hybrid import HybridMemory
        with patch('builtins.__import__', side_effect=guard):
            import dust3r
            memory = HybridMemory(MemoryConfig())
            state = memory.reset(image_hw=(32, 32), patch_hw=(2, 2))
            self.assertFalse(state.has_long_memory)
            self.assertEqual(state.short_occupancy, 0)


if __name__ == '__main__':
    unittest.main()
