"""Explicit supervised hybrid trainer; --dry-run never starts training.

SERVER_VALIDATION_REQUIRED: supply verified local model/RAFT checkpoints and
GT manifest, validate Flash/FLA forward/backward on the server before using
efficient backends. No entrypoint downloads or installs anything.
"""
import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hybrid_memory.yaml")
    parser.add_argument("--manifest", help="Override strict local GT manifest path")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/requirements only; no optimization")
    parser.add_argument("--validate-data", action="store_true", help="Read and validate all local GT arrays")
    parser.add_argument("--validate-modules", action="store_true", help="Construct standalone memory, without Endo3R checkpoints")
    parser.add_argument("--reference", action="store_true", help="Explicit CPU correctness backends; never advertised as efficient")
    parser.add_argument("--stage", choices=("t1", "t2", "r1_supervised"), default="t1")
    parser.add_argument('--resume-checkpoint', help='Existing hybrid weights; required for T2/R1 adaptation')
    parser.add_argument("--unfreeze-decoder", action="store_true")
    parser.add_argument("--unfreeze-heads", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", default="checkpoints/hybrid_memory_t1.pt")
    parser.add_argument("--teacher-cache-dir", help="Optional prefix-aligned local teacher trace cache")
    parser.add_argument("--teacher-metadata", help="JSON containing verified teacher checkpoint/preprocessing/action hashes")
    parser.add_argument("--trusted-checkpoints", action="store_true", help="Explicit trust for local released pickle metadata")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from dust3r.memory.config import load_config
    from dust3r.training.hybrid_trainer import (primitive_config, config_value, validate_required_paths,
                                               configure_trainable_parameters, build_optimizer_groups,
                                               optimizer_group_summary)
    config = load_config(args.config)
    raw = primitive_config(config)
    if args.reference:
        raw["memory"]["short"]["backend"] = "math_reference"
        raw["memory"]["short"]["require_flash"] = False
        raw["memory"]["long"]["backend"] = "torch_reference"
        raw["uncertainty"]["enabled"] = False
    if args.manifest:
        raw["training"]["data_manifest"] = args.manifest
    config = load_config(raw)
    if config_value(config.memory, "backend") != "hybrid_flash_kda":
        raise ValueError("Explicit hybrid training requires memory.backend=hybrid_flash_kda; legacy training stays untouched")
    if config_value(config.policy, "long_write_policy") == "ppo":
        raise ValueError("Supervised attention training must disable PPO and use a deterministic write policy")
    requirements = validate_required_paths(config)
    if args.stage != 't1' and not args.resume_checkpoint:
        requirements.append('T2/R1 adaptation requires --resume-checkpoint from a trained hybrid stage')
    if args.resume_checkpoint and not Path(args.resume_checkpoint).is_file():
        requirements.append(f'Local resume checkpoint missing: {args.resume_checkpoint}')
    report = {"dry_run": args.dry_run, "config_valid": True,
              "backend": {"short": config.memory.short.backend, "long": config.memory.long.backend},
              "stage": args.stage, "requirements_ready": not requirements, "blocking_requirements": requirements,
              "no_downloads": True, "no_installations": True,
              "server_validation": "SERVER_VALIDATION_REQUIRED: real kernel/model gradients, GT provenance, training fit"}
    teacher_metadata = None
    if bool(args.teacher_cache_dir) != bool(args.teacher_metadata):
        raise ValueError("Optional teacher warm start needs both --teacher-cache-dir and --teacher-metadata")
    if args.teacher_metadata:
        if not Path(args.teacher_cache_dir).is_dir():
            raise FileNotFoundError("Optional teacher cache directory must already exist locally")
        with open(args.teacher_metadata, encoding="utf-8") as handle:
            teacher_metadata = json.load(handle)
        records = teacher_metadata if isinstance(teacher_metadata, list) else [teacher_metadata]
        for record in records:
            for key in ("checkpoint_hash", "preprocessing_hash", "action_trajectory_hash"):
                if not record.get(key):
                    raise ValueError(f"Teacher metadata requires {key}")
        if isinstance(teacher_metadata, list):
            teacher_metadata = {record['episode_id']: record for record in records}
        report["teacher_distillation"] = "optional_prefix_aligned_warm_start"
    if args.validate_data:
        from dust3r.training.sequence_dataset import SequenceDataset
        dataset = SequenceDataset(config.training.data_manifest,
                                  image_hw=(config.training.image_height, config.training.image_width))
        report["dataset"] = dataset.validate_all()
    if args.validate_modules:
        import torch
        from dust3r.memory.hybrid import HybridMemory
        memory = HybridMemory(config.memory).to(args.device)
        report["module_parameters"] = sum(parameter.numel() for parameter in memory.parameters())
        # Simulate model registration exactly for freeze/optimizer validation.
        holder = torch.nn.Module()
        holder.hybrid_memory = memory
        report["parameter_status"] = configure_trainable_parameters(holder, args.stage)
        report["optimizer_groups"] = optimizer_group_summary(build_optimizer_groups(holder, config.training, args.stage))
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    if requirements:
        raise FileNotFoundError("Training readiness failed:\n" + "\n".join(requirements))
    if args.total_steps < 1 or args.epochs < 1:
        raise ValueError("Explicit positive --total-steps and --epochs are required")
    from dust3r.training.hybrid_trainer import HybridTrainer, load_initialized_model
    from dust3r.training.sequence_dataset import SequenceDataset
    model, hashes = load_initialized_model(config, args.device, args.trusted_checkpoints)
    if args.resume_checkpoint:
        from dust3r.training.hybrid_trainer import load_training_checkpoint
        load_training_checkpoint(args.resume_checkpoint, model)
    status = configure_trainable_parameters(model, args.stage, args.unfreeze_decoder, args.unfreeze_heads)
    dataset = SequenceDataset(config.training.data_manifest, split="train",
                              image_hw=(config.training.image_height, config.training.image_width),
                              sequence_raw_frames=config.training.sequence_raw_frames, stride=config.training.stride)
    runner = model.make_online_runner(training=True)
    teacher_factory = None
    if teacher_metadata is not None:
        from dust3r.training.teacher import TeacherTraceCache, TeacherTraceMetadata
        def teacher_factory(sample):
            record = (teacher_metadata if 'checkpoint_hash' in teacher_metadata
                      else teacher_metadata.get(sample.episode_id))
            if record is None:
                raise FileNotFoundError(f'No causal teacher trace metadata for training clip {sample.episode_id}')
            metadata = {key: value for key, value in record.items() if key in TeacherTraceMetadata.__dataclass_fields__}
            metadata['episode_id'] = sample.episode_id
            return TeacherTraceCache(args.teacher_cache_dir, TeacherTraceMetadata(**metadata))
    trainer = HybridTrainer(model, runner, config, stage=args.stage, total_steps=args.total_steps,
                            teacher_cache_factory=teacher_factory)
    print(json.dumps({**report, "parameter_status": status,
                      "optimizer_groups": optimizer_group_summary(trainer.optimizer.param_groups)}, indent=2))
    for epoch in range(args.epochs):
        for sample_index in range(len(dataset)):
            logs = trainer.train_sequence(dataset[sample_index])
            print(json.dumps({"epoch": epoch, "sample": sample_index, "windows": logs}))
            if trainer.global_step >= args.total_steps:
                break
        if trainer.global_step >= args.total_steps:
            break
    # Unvalidated checkpoints are deliberately labelled; not PPO-eligible yet.
    trainer.save_checkpoint(args.output, {"validated_for_ppo": False, "reason": "run held-out server evaluation"}, hashes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
