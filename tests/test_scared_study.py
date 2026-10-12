"""Read-only command planning and synthetic path checks; no jobs are launched."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dust3r.experiments.study import (execute_plan, materialize_configs, plan_study,
                                     validate_execution_inputs, _validate_kernel_proof,
                                     StudyCommand, StudyPlan)
from tools.run_scared_study import main


REPOSITORY = Path(__file__).resolve().parents[1]
STUDY = REPOSITORY / "configs" / "experiments" / "scared_study.json"


class StudyTests(unittest.TestCase):
    def test_kernel_contract_matches_hybrid_ppo_but_binds_dimensions_and_dtype(self):
        from tools.check_memory_backends import kernel_config_sha256
        from tests.hybrid_helpers import tiny_config
        config = tiny_config()
        policy_variant = config.to_dict()
        policy_variant['policy']['long_write_policy'] = 'ppo'
        policy_variant['training']['seed'] = 99
        policy_variant['uncertainty']['metric'] = 'sampson'
        self.assertEqual(kernel_config_sha256(config), kernel_config_sha256(policy_variant))
        shape_variant = config.to_dict()
        shape_variant['memory']['short'].update(model_dim=16, head_dim=8)
        self.assertNotEqual(kernel_config_sha256(config), kernel_config_sha256(shape_variant))
        precision_variant = config.to_dict()
        precision_variant['training']['new_memory_dtype'] = 'float16'
        self.assertNotEqual(kernel_config_sha256(config), kernel_config_sha256(precision_variant))

    def test_probe_records_requested_config_before_reference_mutation(self):
        from tools.check_memory_backends import kernel_config_sha256, probe
        from tests.hybrid_helpers import tiny_config
        from dust3r.memory.config import load_config
        from dust3r.training.provenance import canonical_sha256
        config = tiny_config().to_dict()
        config['uncertainty']['enabled'] = True
        config = load_config(config)
        proof = probe(config, reference=True, device='cpu')
        self.assertTrue(proof['passed'])
        self.assertEqual(proof['requested_config_sha256'], canonical_sha256(config.to_dict()))
        self.assertEqual(proof['kernel_config_sha256'], kernel_config_sha256(config))
        self.assertNotEqual(proof['requested_config_sha256'], proof['effective_probe_config_sha256'])
        self.assertTrue(config.uncertainty.enabled)
        self.assertEqual(proof['device'], 'cpu')

    def test_persisted_kernel_proof_rejects_stale_shapes_and_device(self):
        from tools.check_memory_backends import kernel_config_sha256
        from tests.hybrid_helpers import tiny_config
        from dust3r.training.provenance import canonical_sha256
        config = tiny_config().to_dict()
        proof = {'passed': True, 'reference_mode': False, 'actual_short_backend': 'sdpa_flash',
                 'actual_long_backend': 'fla_kda_chunk', 'device': 'cuda:0',
                 'requested_config_sha256': canonical_sha256(config),
                 'kernel_config_sha256': kernel_config_sha256(config)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'SYNTHETIC_kernel_conformance.json'
            path.write_text(json.dumps(proof), encoding='utf-8')
            _validate_kernel_proof(path, expected_config=config, expected_device='cuda:0')
            policy_variant = json.loads(json.dumps(config))
            policy_variant['policy']['long_write_policy'] = 'ppo'
            _validate_kernel_proof(path, expected_config=policy_variant, expected_device='cuda:0')
            with self.assertRaisesRegex(ValueError, 'CUDA device'):
                _validate_kernel_proof(path, expected_config=config, expected_device='cuda:1')
            policy_variant['memory']['long']['num_heads'] = 4
            with self.assertRaisesRegex(ValueError, 'dimensions/backend/precision'):
                _validate_kernel_proof(path, expected_config=policy_variant, expected_device='cuda:0')
            proof.pop('requested_config_sha256')
            path.write_text(json.dumps(proof), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'requested config provenance'):
                _validate_kernel_proof(path, expected_config=config)
            # The optional check preserves basic direct-test use while the
            # real executor always supplies materialized config and device.
            _validate_kernel_proof(path)

    def test_executor_binds_proof_to_actual_materialized_file_before_process(self):
        from tools.check_memory_backends import kernel_config_sha256
        from tests.hybrid_helpers import tiny_config
        from dust3r.training.provenance import canonical_sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = tiny_config().to_dict()
            config = root / 'materialized.json'
            drifted = json.loads(json.dumps(old))
            drifted['memory']['short']['head_dim'] = 8
            config.write_text(json.dumps(drifted), encoding='utf-8')
            proof = root / 'kernel_conformance.json'
            proof.write_text(json.dumps({'passed': True, 'reference_mode': False,
                'actual_short_backend': 'sdpa_flash', 'actual_long_backend': 'fla_kda_chunk',
                'device': 'cuda:0', 'requested_config_sha256': canonical_sha256(old),
                'kernel_config_sha256': kernel_config_sha256(old)}), encoding='utf-8')
            command = StudyCommand('fit', 0, 'synthetic_guard_only',
                ['python', str(REPOSITORY / 'train_hybrid_memory.py'), '--config', str(config)],
                required_files=[str(config), str(proof)])
            plan = StudyPlan(REPOSITORY, root / 'output', {}, {str(config): old},
                             [command], ('fit',), 'cuda:0', None, [])
            with patch('dust3r.experiments.study.subprocess.run') as run:
                with self.assertRaisesRegex(ValueError, 'materialized command config'):
                    execute_plan(plan)
                run.assert_not_called()

    def _ready_path_fixture(self, directory, stage, manifest, profile_input=None):
        """Path-only synthetic artifacts; never passed to a model/checkpoint loader."""
        root = Path(directory)
        checkpoint_args = {key: root / f"SYNTHETIC_{key}.pt" for key in
                           ("dust3r_checkpoint", "endo3r_checkpoint", "raft_checkpoint")}
        plan = plan_study(STUDY, output_root=root / "output", stages=(stage,), seeds=[0],
                          manifest=manifest, profile_input=profile_input,
                          cost_calibration=root / "SYNTHETIC_costs.json", **checkpoint_args)
        materialize_configs(plan)
        produced = set()
        for command in plan.commands:
            for name in command.required_files:
                path = Path(name)
                if name not in produced and not path.exists():
                    # Never create files in the real repository/checkpoints.
                    self.assertTrue(path.resolve().is_relative_to(root.resolve()))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("SYNTHETIC PATH-ONLY FIXTURE", encoding="utf-8")
            produced.update(command.outputs)
        return plan

    def test_default_cli_never_writes_or_runs_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_exist"
            with patch("dust3r.experiments.study.subprocess.run", side_effect=AssertionError("Process started in dry run")):
                with redirect_stdout(io.StringIO()) as captured:
                    self.assertEqual(main(["--study", str(STUDY), "--output-root", str(output)]), 0)
            result = json.loads(captured.getvalue())
            self.assertTrue(result["dry_run"])
            self.assertFalse(output.exists())
            self.assertEqual(len(result["config_overrides"]), 9)
            self.assertTrue(result["no_downloads"])
            self.assertTrue(result["no_installations"])

    def test_stages_and_splits_are_causal_and_test_not_used_to_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = plan_study(STUDY, output_root=directory, seeds=[0], profile_input=directory)
            stages = [command.stage for command in plan.commands]
            self.assertLess(stages.index("certify"), stages.index("profile"))
            self.assertLess(stages.index("profile"), stages.index("ppo"))
            self.assertLess(stages.index("ppo"), stages.index("evaluate"))
            for command in plan.commands:
                if command.stage in ("baseline", "certify"):
                    self.assertEqual(command.argv[command.argv.index("--split") + 1], "validation")
                    self.assertEqual(command.argv[command.argv.index("--max-frames") + 1], "64")
                if command.stage in ("evaluate", "benchmark") and Path(command.argv[1]).name != 'profile_memory.py':
                    self.assertEqual(command.argv[command.argv.index("--split") + 1], "test")
                if command.stage in ("fit", "ppo"):
                    self.assertIn("--run-training", command.argv)
                    self.assertNotIn("--split", command.argv)
                self.assertTrue(Path(command.argv[1]).is_absolute())

    def test_heuristic_and_ppo_reuse_exact_certified_attention_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = plan_study(STUDY, output_root=directory, stages=("evaluate", "benchmark"), seeds=[1])
            hybrid_weights = []
            actors = []
            for command in plan.commands:
                if command.experiment_id in ("hybrid_sampson", "hybrid_ppo"):
                    hybrid_weights.append(command.argv[command.argv.index("--hybrid-checkpoint") + 1])
                if command.experiment_id == "hybrid_ppo":
                    actors.append(command.argv[command.argv.index("--actor-checkpoint") + 1])
            self.assertEqual(len(hybrid_weights), 4)
            self.assertEqual(len(set(hybrid_weights)), 1)
            self.assertEqual(Path(hybrid_weights[0]).name, "hybrid_certified.pt")
            self.assertEqual(len(set(actors)), 1)

    def test_materialization_is_explicit_absolute_and_refuses_config_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "configs"
            manifest = Path(directory) / "SYNTHETIC_manifest.json"
            plan = plan_study(STUDY, output_root=output, stages=("fit",), seeds=[2], manifest=manifest,
                              dust3r_checkpoint=Path(directory) / "dust3r.pt", endo3r_checkpoint=Path(directory) / "endo3r.pt",
                              raft_checkpoint=Path(directory) / "raft.pt")
            self.assertFalse(output.exists())
            with patch("dust3r.experiments.study.subprocess.run", side_effect=AssertionError("Job started")):
                written = materialize_configs(plan)
            self.assertEqual(len(written), 3)
            for path in written:
                config = json.loads(Path(path).read_text())
                self.assertEqual(config["training"]["seed"], 2)
                self.assertEqual(config["training"]["data_manifest"], str(manifest.resolve()))
                self.assertTrue(all(Path(value).is_absolute() for value in config["checkpoints"].values()))
            first = Path(written[0])
            first.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                materialize_configs(plan)

    def test_missing_prerequisites_prevent_any_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = plan_study(STUDY, output_root=directory, stages=("ppo",), seeds=[0])
            with patch("dust3r.experiments.study.subprocess.run") as subprocess_run:
                with self.assertRaises(FileNotFoundError):
                    execute_plan(plan)
                self.assertFalse(subprocess_run.called)

    def test_official_legacy_raft_remains_explicit_fixed_path(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = plan_study(STUDY, output_root=directory, stages=("baseline",), seeds=[0],
                              raft_checkpoint=Path(directory) / "custom_raft.pt")
            command = plan.commands[0]
            self.assertIn(str(REPOSITORY / "checkpoints" / "raft-things.pth"), command.required_files)
            self.assertNotIn("copy", command.argv)

    def test_profile_is_optional_and_costs_are_shared_across_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = plan_study(STUDY, output_root=directory, seeds=[0, 1])
            self.assertFalse(any(command.stage == "profile" for command in plan.commands))
            with self.assertRaisesRegex(ValueError, "profile-input"):
                plan_study(STUDY, output_root=directory, stages=("profile",), seeds=[0])
            plan = plan_study(STUDY, output_root=directory, seeds=[0, 1], profile_input=directory)
            calibrations = [command for command in plan.commands if Path(command.argv[1]).name == "calibrate_costs.py"]
            self.assertEqual(len(calibrations), 1)
            self.assertEqual(len({config["rl"]["cost_calibration_path"] for config in plan.configs.values()}), 1)

    def test_existing_cost_calibration_is_not_overwritten_by_profile_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            costs = Path(directory) / "existing_costs.json"
            plan = plan_study(STUDY, output_root=directory, stages=("profile",), seeds=[0],
                              profile_input=directory, cost_calibration=costs)
            self.assertFalse(any(Path(command.argv[1]).name == "calibrate_costs.py" for command in plan.commands))
            self.assertEqual(next(iter(plan.configs.values()))["rl"]["cost_calibration_path"], str(costs.resolve()))

    def test_actor_profile_is_separate_from_headline_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = plan_study(STUDY, output_root=directory, stages=('benchmark',), seeds=[0], profile_input=directory)
            profiles = [command for command in plan.commands if Path(command.argv[1]).name == 'profile_memory.py']
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0].experiment_id, 'hybrid_ppo')
            headline = [command for command in plan.commands if Path(command.argv[1]).name == 'benchmark_experiment.py'
                        and command.experiment_id == 'hybrid_ppo'][0]
            component = headline.argv[headline.argv.index('--component-profile') + 1]
            self.assertEqual(profiles[0].outputs, [component])
            self.assertIn(component, headline.required_files)

    def test_cpu_execution_rejected_without_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = plan_study(STUDY, output_root=directory, stages=("fit",), seeds=[0], device="cpu")
            with self.assertRaisesRegex(ValueError, "CUDA"):
                validate_execution_inputs(plan)

    def test_official_legacy_device_assumption_not_silently_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = plan_study(STUDY, output_root=directory, stages=('baseline',), seeds=[0], device='cuda:1')
            with self.assertRaisesRegex(ValueError, 'visible CUDA device 0'):
                validate_execution_inputs(plan)

    def test_manifest_dataset_split_mismatch_blocks_training_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "SYNTHETIC_manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "sequences": [
                {"id": "synthetic_dataset8", "patient_id": "synthetic_patient", "split": "train",
                 "provenance": {"dataset_id": 8}, "frames": []}]}), encoding="utf-8")
            plan = self._ready_path_fixture(directory, "fit", manifest)
            with patch("dust3r.experiments.study.subprocess.run") as subprocess_run:
                with self.assertRaisesRegex(ValueError, "contradicts"):
                    execute_plan(plan)
                self.assertFalse(subprocess_run.called)

    def test_test_rgb_is_forbidden_as_cost_profile_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_input = root / "synthetic_test_rgb"
            profile_input.mkdir()
            rgb = profile_input / "fake.npy"
            rgb.write_bytes(b"SYNTHETIC PATH-ONLY FIXTURE")
            manifest = root / "SYNTHETIC_manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "sequences": [
                {"id": "synthetic_dataset8", "patient_id": "synthetic_patient", "split": "test",
                 "provenance": {"dataset_id": 8}, "frames": [{"rgb": str(rgb)}]}]}), encoding="utf-8")
            plan = self._ready_path_fixture(directory, "profile", manifest, profile_input)
            with self.assertRaisesRegex(ValueError, "test RGB"):
                validate_execution_inputs(plan)


if __name__ == "__main__":
    unittest.main()
