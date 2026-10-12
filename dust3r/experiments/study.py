"""Concrete SCARED server commands; planning is read-only by default.

Executing a stage requires a separate explicit server-run request. Commands are
argument lists (shell=False); there are no installation/acquisition commands.
Training always requires the entrypoint's explicit --run-training flag.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .results import load_json


STAGES = ("preflight", "baseline", "fit", "certify", "profile", "ppo", "evaluate", "benchmark", "summarize")


@dataclass
class StudyCommand:
    stage: str
    seed: int | None
    experiment_id: str
    argv: list
    required_files: list = field(default_factory=list)
    required_directories: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    kernel_proof_output: str | None = None

    def to_dict(self):
        return vars(self).copy()


@dataclass
class StudyPlan:
    repository: Path
    output_root: Path
    study: dict
    configs: dict
    commands: list
    selected_stages: tuple
    device: str
    manifest: str | None
    external_files: list

    def to_dict(self):
        return {"schema": "endo3r_scared_command_plan_v1", "dry_run": True,
                "repository": str(self.repository), "output_root": str(self.output_root),
                "device": self.device, "stages": list(self.selected_stages),
                "budgets_are_smoke_defaults": True, "no_downloads": True, "no_installations": True,
                "config_overrides": self.configs, "commands": [command.to_dict() for command in self.commands],
                "external_files": self.external_files,
                "notes": self.study.get("notes", [])}


def _absolute(value, root):
    if value is None:
        return None
    path = Path(value)
    return str((path if path.is_absolute() else root / path).resolve())


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Study {name} must be a positive integer")
    return value


def plan_study(study_path, *, repository=None, output_root="outputs/scared_study", stages=("all",),
               manifest=None, dust3r_checkpoint=None, endo3r_checkpoint=None, raft_checkpoint=None,
               cost_calibration=None, profile_input=None, device="cuda:0", seeds=None,
               python_executable=None, trusted_checkpoints=False):
    """Read configurations and produce commands; writes/executes nothing."""
    repository = Path(repository or Path(__file__).resolve().parents[2]).resolve()
    study_path = Path(_absolute(study_path, repository))
    study = load_json(study_path)
    if study.get("schema") != "endo3r_scared_study_v1" or study.get("dataset") != "SCARED":
        raise ValueError("Expected endo3r_scared_study_v1 catalog for SCARED")
    if study.get("train_datasets") != [1, 2, 3, 4, 5, 6] or study.get("validation_datasets") != [7] or study.get("test_datasets") != [8, 9]:
        raise ValueError("v1 study locks train1-6/validation7/test8-9; do not tune on the test data")
    for key in ("attention_steps", "attention_epochs", "ppo_updates", "benchmark_repeats", "warmup_frames", "certification_max_frames", "stride"):
        _positive_integer(study.get(key), key)
    if study["warmup_frames"] < 2 or study["certification_max_frames"] < 2:
        raise ValueError("Warmup/certification need at least two frames")
    selected_seeds = list(study.get("seeds", []) if seeds is None else seeds)
    if not selected_seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in selected_seeds) or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Unique nonnegative integer seeds are required")
    if "all" in stages and len(stages) != 1:
        raise ValueError("Choose all or explicit stages, never combine them")
    if "all" in stages:
        selected = list(STAGES)
        if profile_input is None:
            selected.remove("profile")
    else:
        if not stages or set(stages) - set(STAGES):
            raise ValueError("Unknown or empty study stages")
        selected = [stage for stage in STAGES if stage in stages]
    output_root = Path(_absolute(output_root, repository))
    manifest = _absolute(manifest, repository)
    profile_input = _absolute(profile_input, repository)
    supplied_cost = _absolute(cost_calibration, repository)
    cost_path = supplied_cost or str(output_root / "cost_calibration.json")
    executable = python_executable or sys.executable
    trust = ["--trusted-checkpoints"] if trusted_checkpoints else []
    configs, commands, external_files = {}, [], []
    experiments = study.get("experiments", [])
    kinds = {e.get("kind"): e for e in experiments if e.get("enabled", True) is not False}
    if set(kinds) != {"legacy", "hybrid", "ppo"} or len(kinds) != len(experiments):
        raise ValueError("Study requires one unique legacy, hybrid and PPO experiment")
    image_hw = study.get("image_hw")
    if not isinstance(image_hw, list) or len(image_hw) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in image_hw):
        raise ValueError("Study image_hw must be [H,W] positive integers")
    from dust3r.memory.config import load_config
    paths_by_seed, common_inputs = {}, {}
    for seed in selected_seeds:
        seed_root = output_root / f"seed_{seed}"
        paths = {"raw_hybrid": str(seed_root / "hybrid_t1.pt"),
                 "certified_hybrid": str(seed_root / "hybrid_certified.pt"),
                 "actor": str(seed_root / "actor.pt"), "ppo": str(seed_root / "ppo_training.pt"),
                 "baseline_validation": str(seed_root / "legacy_validation.json"),
                 "certification_report": str(seed_root / "hybrid_validation.json"),
                 "preflight_proof": str(seed_root / "kernel_conformance.json")}
        paths_by_seed[seed] = paths
        for kind, experiment in kinds.items():
            template = load_config(_absolute(experiment["config"], repository)).to_dict()
            checkpoint_overrides = {"dust3r": dust3r_checkpoint, "endo3r": endo3r_checkpoint, "raft": raft_checkpoint}
            for key, override in checkpoint_overrides.items():
                template["checkpoints"][key] = _absolute(override or template["checkpoints"][key], repository)
            template["training"].update(seed=seed, data_manifest=manifest or _absolute(template["training"].get("data_manifest"), repository),
                                         image_height=image_hw[0], image_width=image_hw[1], stride=study["stride"])
            template["rl"]["cost_calibration_path"] = cost_path
            template = load_config(template).to_dict()
            config_path = str(seed_root / f"config_{experiment['id']}.json")
            configs[config_path] = template
            paths[kind + "_config"] = config_path
        hybrid_inputs = configs[paths["hybrid_config"]]
        common_inputs[seed] = [hybrid_inputs["training"]["data_manifest"], hybrid_inputs["checkpoints"]["dust3r"], hybrid_inputs["checkpoints"]["endo3r"], hybrid_inputs["checkpoints"]["raft"]]
        external_files += [path for path in common_inputs[seed] if path is not None]

    def command(stage, seed, experiment_id, script, args, required=(), outputs=(), directories=(), kernel_proof=None):
        commands.append(StudyCommand(stage, seed, experiment_id,
                                     [executable, str(repository / script), *args],
                                     list(required), list(directories), list(outputs), kernel_proof))

    quality_reports, timing_reports = [], []
    official_raft = str(repository / "checkpoints" / "raft-things.pth")
    for seed in selected_seeds:
        p = paths_by_seed[seed]
        inputs = common_inputs[seed]
        hybrid_id, ppo_id, legacy_id = kinds["hybrid"]["id"], kinds["ppo"]["id"], kinds["legacy"]["id"]
        manifest_flags = ["--manifest", inputs[0]] if inputs[0] is not None else []
        base = ["--device", device, *manifest_flags, *trust]
        if "preflight" in selected:
            command("preflight", seed, hybrid_id, "tools/check_memory_backends.py",
                    ["--config", p["hybrid_config"], "--server-kernels", "--device", device],
                    required=[p["hybrid_config"]], outputs=[p["preflight_proof"]], kernel_proof=p["preflight_proof"])
            command("preflight", seed, hybrid_id, "train_hybrid_memory.py",
                    ["--config", p["hybrid_config"], "--dry-run", "--validate-data", "--validate-modules", "--device", device],
                    required=[p["hybrid_config"], *inputs])
        if "baseline" in selected:
            command("baseline", seed, legacy_id, "eval/eval_hybrid.py",
                    ["--config", p["legacy_config"], *base, "--split", "validation", "--max-frames", str(study["certification_max_frames"]),
                     "--run-evaluation", "--output", p["baseline_validation"]],
                    required=[p["legacy_config"], *inputs[:3], official_raft], outputs=[p["baseline_validation"]])
        if "fit" in selected:
            command("fit", seed, hybrid_id, "train_hybrid_memory.py",
                    ["--config", p["hybrid_config"], *base, "--seed", str(seed), "--stage", "t1", "--run-training",
                     "--total-steps", str(study["attention_steps"]), "--epochs", str(study["attention_epochs"]),
                     "--log-jsonl", str(output_root / f"seed_{seed}" / "attention_training.jsonl"), "--output", p["raw_hybrid"]],
                    required=[p["hybrid_config"], *inputs, p["preflight_proof"]], outputs=[p["raw_hybrid"]])
        if "certify" in selected:
            command("certify", seed, hybrid_id, "eval/eval_hybrid.py",
                    ["--config", p["hybrid_config"], *base, "--split", "validation", "--max-frames", str(study["certification_max_frames"]),
                     "--hybrid-checkpoint", p["raw_hybrid"], "--baseline-report", p["baseline_validation"],
                     "--certified-checkpoint-output", p["certified_hybrid"], "--run-evaluation", "--output", p["certification_report"]],
                    required=[p["hybrid_config"], *inputs, p["raw_hybrid"], p["baseline_validation"], p["preflight_proof"]],
                    outputs=[p["certification_report"], p["certified_hybrid"]])
        if "ppo" in selected:
            command("ppo", seed, ppo_id, "train_rl_controller.py",
                    ["--config", p["ppo_config"], *base, "--seed", str(seed), "--hybrid-checkpoint", p["certified_hybrid"],
                     "--run-training", "--updates", str(study["ppo_updates"]), "--output", p["ppo"], "--actor-output", p["actor"],
                     "--log-jsonl", str(output_root / f"seed_{seed}" / "ppo_training.jsonl")],
                    required=[p["ppo_config"], *inputs, p["certified_hybrid"], p["certification_report"], p["preflight_proof"], cost_path],
                    outputs=[p["ppo"], p["actor"]])
        for kind, experiment in kinds.items():
            identifier = experiment["id"]
            weights = [] if kind == "legacy" else ["--hybrid-checkpoint", p["certified_hybrid"]]
            weights += ["--actor-checkpoint", p["actor"]] if kind == "ppo" else []
            required = [p[kind + "_config"], *inputs[:3]]
            required += [official_raft] if kind == "legacy" else [inputs[3], p["certified_hybrid"], p["preflight_proof"]]
            required += [p["actor"]] if kind == "ppo" else []
            quality = str(output_root / f"seed_{seed}" / f"quality_{identifier}.json")
            timing = str(output_root / f"seed_{seed}" / f"performance_{identifier}.json")
            quality_reports.append(quality)
            timing_reports.append(timing)
            if "evaluate" in selected:
                command("evaluate", seed, identifier, "eval/eval_benchmark.py",
                        ["--config", p[kind + "_config"], *base, "--split", "test", "--seed", str(seed), "--experiment-id", identifier,
                         *weights, "--run-evaluation", "--output", quality], required=required, outputs=[quality])
            if "benchmark" in selected:
                component_flags, component_required = [], []
                if kind == 'ppo' and profile_input is not None:
                    actor_profile = str(output_root / f'seed_{seed}' / 'actor_component_profile.json')
                    command('benchmark', seed, identifier, 'tools/profile_memory.py',
                            ['--config', p[kind + '_config'], '--input', profile_input, *weights,
                             '--device', device, '--seed', str(seed),
                             '--num-frames', str(study['certification_max_frames']),
                             '--warmup-frames', str(study['warmup_frames']), *trust,
                             '--server-run', '--output', actor_profile],
                            required=required, directories=[profile_input], outputs=[actor_profile])
                    component_flags = ['--component-profile', actor_profile]
                    component_required = [actor_profile]
                command("benchmark", seed, identifier, "tools/benchmark_experiment.py",
                        ["--config", p[kind + "_config"], *base, "--split", "test", "--seed", str(seed), "--experiment-id", identifier,
                         *weights, "--repeats", str(study["benchmark_repeats"]), "--warmup-frames", str(study["warmup_frames"]),
                         *component_flags, "--server-run", "--output", timing],
                        required=[*required, *component_required], outputs=[timing])
    if "profile" in selected:
        if profile_input is None:
            raise ValueError("Profile stage needs --profile-input pointing to existing train/validation RGB, never test8/9")
        seed = selected_seeds[0]
        p = paths_by_seed[seed]
        profile = str(output_root / "component_profile.json")
        command("profile", seed, kinds["hybrid"]["id"], "tools/profile_memory.py",
                ["--config", p["hybrid_config"], "--input", profile_input, "--hybrid-checkpoint", p["certified_hybrid"],
                 "--device", device, "--num-frames", str(study["certification_max_frames"]), "--warmup-frames", str(study["warmup_frames"]),
                 '--seed', str(seed), *trust, "--server-run", "--output", profile],
                required=[p["hybrid_config"], *common_inputs[seed][1:], p["certified_hybrid"], p["preflight_proof"]],
                directories=[profile_input], outputs=[profile])
        if supplied_cost is None:
            command("profile", seed, kinds["hybrid"]["id"], "tools/calibrate_costs.py",
                    ["--profile", profile, "--output", cost_path], required=[profile], outputs=[cost_path])
    if "summarize" in selected:
        paper_reference = _absolute(study["paper_reference"], repository)
        command("summarize", None, "all", "tools/render_benchmark.py",
                ["--reports", *quality_reports, *timing_reports, "--catalog", str(study_path),
                 "--paper-reference", paper_reference, "--output", str(output_root / "benchmark")],
                required=[*quality_reports, *timing_reports, paper_reference, str(study_path)],
                outputs=[str(output_root / "benchmark" / "benchmark.json"), str(output_root / "benchmark" / "benchmark.html")])
    # Run complete stages across all seeds before later stages consume their
    # artifacts. This also guarantees profile/cost calibration precedes PPO.
    commands.sort(key=lambda c: STAGES.index(c.stage))
    resolved_manifest = manifest or common_inputs[selected_seeds[0]][0]
    return StudyPlan(repository, output_root, study, configs, commands, tuple(selected), device,
                     resolved_manifest, sorted(set(external_files)))


def materialize_configs(plan):
    """Explicit reversible config generation, without subprocesses or training."""
    written, resolved = [], []
    for name, payload in plan.configs.items():
        path = Path(name)
        exact = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != exact:
            raise FileExistsError(f"Refusing to overwrite a different experiment config: {path}; use another output root")
        resolved.append((path, exact))
    for path, exact in resolved:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(exact, encoding="utf-8")
        written.append(str(path))
    return written


def validate_execution_inputs(plan):
    """Fail before any command for unavailable external/unproduced prerequisites."""
    if not plan.device.startswith("cuda") and any(stage != "summarize" for stage in plan.selected_stages):
        raise ValueError("Server execution requires CUDA; CPU reference runs are not this benchmark protocol")
    if plan.device not in {'cuda', 'cuda:0'}:
        for command in plan.commands:
            if '--config' in command.argv:
                name = command.argv[command.argv.index('--config') + 1]
                if plan.configs.get(name, {}).get('baseline_identity') == 'LEGACY_OFFICIAL':
                    raise ValueError('LEGACY_OFFICIAL assumes visible CUDA device 0; choose cuda:0 and set CUDA_VISIBLE_DEVICES externally')
    produced = set()
    missing = []
    for command in plan.commands:
        if not Path(command.argv[1]).is_file():
            missing.append(f"Entrypoint missing: {command.argv[1]}")
        for path in command.required_files:
            if path is None:
                missing.append(f"{command.stage}: required manifest/weight path was not supplied")
            elif path not in produced and not Path(path).is_file():
                missing.append(f"{command.stage}: required local file missing: {path}")
        for path in command.required_directories:
            if not Path(path).is_dir():
                missing.append(f"{command.stage}: existing RGB directory required: {path}")
        produced.update(command.outputs)
    if missing:
        raise FileNotFoundError("Server study prerequisites are not ready:\n" + "\n".join(sorted(set(missing))))
    if plan.manifest and Path(plan.manifest).is_file() and any(stage != "summarize" for stage in plan.selected_stages):
        manifest = load_json(plan.manifest)
        if manifest.get("schema_version") != 1 or not manifest.get("sequences"):
            raise ValueError("An actual prepared SCARED sequence manifest is required")
        expected = {**{i: "train" for i in plan.study["train_datasets"]},
                    **{i: "validation" for i in plan.study["validation_datasets"]},
                    **{i: "test" for i in plan.study["test_datasets"]}}
        present, patients = set(), {}
        for sequence in manifest["sequences"]:
            dataset_id = sequence.get("provenance", {}).get("dataset_id")
            if isinstance(dataset_id, bool) or not isinstance(dataset_id, int) or dataset_id not in expected or sequence.get("split") != expected[dataset_id]:
                raise ValueError("Manifest SCARED dataset-id/split provenance contradicts train1-6/validation7/test8-9")
            present.add(dataset_id)
            patient = sequence.get("patient_id")
            if not patient or patient in patients and patients[patient] != sequence["split"]:
                raise ValueError("Manifest patient identifiers are missing or leak across study splits")
            patients[patient] = sequence["split"]
        required_dataset_ids = set()
        if any(stage in plan.selected_stages for stage in ("fit", "ppo")):
            required_dataset_ids.update(plan.study["train_datasets"])
        if any(stage in plan.selected_stages for stage in ("baseline", "certify")):
            required_dataset_ids.update(plan.study["validation_datasets"])
        if any(stage in plan.selected_stages for stage in ("evaluate", "benchmark")):
            required_dataset_ids.update(plan.study["test_datasets"])
        if not required_dataset_ids <= present:
            raise ValueError(f"Full study data is incomplete; missing SCARED dataset ids {sorted(required_dataset_ids - present)}")
        for command in plan.commands:
            if Path(command.argv[1]).name != 'profile_memory.py' or not command.required_directories:
                continue
            directory = Path(command.required_directories[0]).resolve()
            training_match = False
            for sequence in manifest["sequences"]:
                for frame in sequence.get("frames", []):
                    rgb = Path(frame["rgb"])
                    rgb = rgb if rgb.is_absolute() else Path(plan.manifest).parent / rgb
                    if rgb.resolve().is_relative_to(directory):
                        if sequence["split"] == "test":
                            raise ValueError("Cost profiling input includes datasets8/9 test RGB; use train/validation only")
                        training_match = True
            if not training_match:
                raise ValueError("Profile RGB directory has no verified train/validation frames in the supplied manifest")


def _validate_kernel_proof(path, expected_config=None, expected_device=None):
    proof = load_json(path)
    if (proof.get("passed") is not True or proof.get("reference_mode") is not False or
            proof.get("actual_short_backend") not in {"sdpa_flash", "flash_attn"} or
            not str(proof.get("actual_long_backend", "")).startswith("fla_kda_")):
        raise ValueError(f"Actual Flash/FLA conformance proof is required, not CPU reference tests: {path}")
    if expected_config is not None:
        from tools.check_memory_backends import kernel_config_sha256
        if isinstance(expected_config, (str, Path)):
            expected_config = load_json(expected_config)
        requested = proof.get('requested_config_sha256')
        if not isinstance(requested, str) or len(requested) != 64:
            raise ValueError(f'Kernel proof lacks requested config provenance; rerun server preflight: {path}')
        if proof.get('kernel_config_sha256') != kernel_config_sha256(expected_config):
            raise ValueError(f'Kernel proof memory dimensions/backend/precision differ from the materialized command config: {path}')
    if expected_device is not None and proof.get('device') != str(expected_device):
        raise ValueError(f'Kernel proof CUDA device differs from the requested server command: {path}')


def execute_plan(plan):
    """Explicit server-only executor; no shell interpolation, no hidden retries."""
    validate_execution_inputs(plan)
    log_dir = plan.output_root / "command_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = plan.output_root / "command_status.jsonl"
    records = []
    for index, command in enumerate(plan.commands):
        # Artifacts promised by earlier commands must really exist now.
        for name in command.required_files:
            if name is None or not Path(name).is_file():
                raise FileNotFoundError(f"{command.stage}: prerequisite absent before execution: {name}")
            if Path(name).name == "kernel_conformance.json":
                config_path = (command.argv[command.argv.index('--config') + 1]
                               if '--config' in command.argv else None)
                _validate_kernel_proof(name, expected_config=config_path, expected_device=plan.device)
        if command.stage == "ppo":
            report = next(path for path in command.required_files if Path(path).name == "hybrid_validation.json")
            payload = load_json(report)
            if payload.get("split") != "validation" or payload.get("ppo_certification", {}).get("validated_for_ppo") is not True:
                raise ValueError("PPO requires a successful validation7 certification report; test data cannot certify the policy")
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(command.argv, cwd=plan.repository, shell=False, capture_output=True, text=True)
        stem = f"{index:03}_{command.stage}_{command.seed if command.seed is not None else 'all'}_{command.experiment_id}"
        stdout_path, stderr_path = log_dir / f"{stem}.stdout.log", log_dir / f"{stem}.stderr.log"
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")
        record = {"stage": command.stage, "seed": command.seed, "experiment_id": command.experiment_id,
                  "argv": command.argv, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "return_code": result.returncode, "stdout": str(stdout_path), "stderr": str(stderr_path)}
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        records.append(record)
        if result.returncode:
            raise RuntimeError(f"Study stage {command.stage} failed with code {result.returncode}; see {stderr_path}")
        if command.kernel_proof_output:
            proof = json.loads(result.stdout)
            path = Path(command.kernel_proof_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(proof, indent=2, allow_nan=False), encoding="utf-8")
            config_path = (command.argv[command.argv.index('--config') + 1]
                           if '--config' in command.argv else None)
            _validate_kernel_proof(path, expected_config=config_path, expected_device=plan.device)
        for output in command.outputs:
            if not Path(output).is_file():
                raise FileNotFoundError(f"Successful process did not produce its declared artifact: {output}")
    return records
