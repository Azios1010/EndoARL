"""SERVER_VALIDATION_REQUIRED: explicit, streamed, uninstrumented throughput.

Default invocation is dry. This script never installs, downloads or trains.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scared/hybrid_sampson.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hybrid-checkpoint")
    parser.add_argument("--actor-checkpoint")
    parser.add_argument("--component-profile", help="Optional existing synchronized PPO policy.act profile JSON; latency is a separate validation proxy.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--max-frames", type=int, help="Explicit bounded subset; recorded in protocol")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--server-run", action="store_true")
    parser.add_argument("--trusted-checkpoints", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    from demo_hybrid import prepare_config
    config = prepare_config(args.config, args.reference)
    if args.repeats < 1 or args.warmup_frames < 2 or args.seed < 0:
        raise ValueError("Positive repeats, >=2 warmup frames and nonnegative seed required")
    if args.component_profile and config.policy.long_write_policy != "ppo":
        raise ValueError("--component-profile applies only to PPO; heuristic/legacy runs have no actor proxy.")
    if not args.server_run:
        print(json.dumps({"dry_run": True, "experiment_id": args.experiment_id,
                          "config_valid": True, "SERVER_VALIDATION_REQUIRED": True,
                          "scope": "online_pointmap_predicted_camera_depth_memory_policy_no_component_hooks",
                          "manifest": args.manifest or config.training.data_manifest,
                          "component_profile": args.component_profile,
                          "actor_latency_source": "separate_validation_component_profile_proxy" if args.component_profile else None,
                          "no_downloads": True, "no_installations": True}, indent=2))
        return 0
    if not args.output:
        raise ValueError("--output is mandatory for a traceable benchmark")
    import torch
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official throughput requires a prepared CUDA server; CPU references are unit tests only")
    from demo_hybrid import load_inference_model
    from dust3r.training.sequence_dataset import SequenceDataset
    from dust3r.training.teacher import file_sha256
    from dust3r.training.reproducibility import seed_everything
    from dust3r.experiments.benchmark import (benchmark_sequence, aggregate_performance,
                                             runtime_metadata, canonical_hash,
                                             validate_actor_component_profile)
    from dust3r.evaluation.streaming import CAMERA_SOURCE, DEPTH_PROTOCOL, POSE_PROTOCOL, SCALE_PROTOCOL
    seed_everything(args.seed)
    manifest = args.manifest or config.training.data_manifest
    dataset = SequenceDataset(manifest, split=args.split,
                              image_hw=(config.training.image_height, config.training.image_width), stride=config.training.stride)
    model, hashes = load_inference_model(config, device=args.device, hybrid_checkpoint=args.hybrid_checkpoint,
                                        actor_checkpoint=args.actor_checkpoint, trusted_checkpoints=args.trusted_checkpoints)
    checkpoint_hashes = dict(hashes)
    if args.hybrid_checkpoint:
        checkpoint_hashes['hybrid'] = file_sha256(args.hybrid_checkpoint)
    if args.actor_checkpoint:
        checkpoint_hashes['actor'] = file_sha256(args.actor_checkpoint)
    hardware = runtime_metadata(args.device)
    from dust3r.training.provenance import source_provenance, runtime_backend_provenance
    source_identity = source_provenance()
    actor_profile_timing = {}
    component_profile_identity = None
    if args.component_profile:
        def reject_constant(value):
            raise ValueError(f"Nonfinite component-profile JSON value: {value}")
        with Path(args.component_profile).open(encoding="utf-8") as handle:
            component_profile = json.load(handle, parse_constant=reject_constant)
        actor_profile_timing = validate_actor_component_profile(component_profile,
            config=config.to_dict(), hardware_id=hardware["hardware_id"],
            original_checkpoint_hashes=hashes,
            hybrid_checkpoint_sha256=checkpoint_hashes.get("hybrid"),
            actor_checkpoint_sha256=checkpoint_hashes.get("actor"),
            source_tree_sha256=source_identity.get("source_tree_sha256"))
        component_profile_identity = {"path": str(Path(args.component_profile).resolve()),
                                      "sha256": file_sha256(args.component_profile),
                                      "source": "separate_validation_component_profile_proxy"}
    sequences = []
    peaks_allocated, peaks_reserved = [], []
    selections = []
    warmup_reports = []
    frame_fingerprints, provenance_fingerprints = [], []
    for sequence_index in range(len(dataset)):
        sample = dataset.stream(sequence_index)
        if args.max_frames is not None:
            if args.max_frames < 2:
                raise ValueError("--max-frames must be >=2")
            sample = sample.prefix(min(sample.frame_count, args.max_frames))
        selections.append({"episode_id": sample.episode_id, "frames": sample.frame_count,
                           "source_indices": [sample.source_frame_id(i) for i in range(sample.frame_count)]})
        identities = [f"{sample.sequence['id']}:{sample.source_frame_id(i)}" for i in range(sample.frame_count)]
        frame_fingerprints.append([sample.episode_id, canonical_hash(identities)])
        provenance_fingerprints.append([sample.episode_id, canonical_hash(sample.provenance)])
        warmup_count = min(sample.frame_count, args.warmup_frames)
        warmup = benchmark_sequence(model, sample.prefix(warmup_count), config, args.device,
                                    pnp_seed=args.seed)
        if warmup["status"] != "completed":
            raise RuntimeError(f"Warmup reconstruction/camera failed for {sample.episode_id}: {warmup['failure']}")
        warmup_reports.append({"episode_id": sample.episode_id, "frames": warmup_count,
                               "camera_depth_warmed": True, "raft_calls": warmup["total_raft_calls"],
                               "scope": "same_online_camera_depth_memory_controller_pipeline_prefix_discarded"})
        del warmup
        for repeat in range(args.repeats):
            seed_everything(args.seed)
            torch.cuda.reset_peak_memory_stats(args.device)
            result = benchmark_sequence(model, sample, config, args.device, pnp_seed=args.seed)
            result["repeat"] = repeat
            sequences.append(result)
            peaks_allocated.append(torch.cuda.max_memory_allocated(args.device))
            peaks_reserved.append(torch.cuda.max_memory_reserved(args.device))
    aggregate = aggregate_performance(sequences)
    aggregate.update(peak_allocated_bytes=max(peaks_allocated, default=0),
                     peak_reserved_bytes=max(peaks_reserved, default=0))
    timing = {"fps": aggregate["fps"], "mean_latency_ms": aggregate["mean_ms"],
              "p50_latency_ms": aggregate["p50_ms"], "p95_latency_ms": aggregate["p95_ms"],
              "latency_unit": "decoded_pair_including_terminal_two_outputs",
              "pair_ms": aggregate["pair_ms"], "amortized_output_ms": aggregate["amortized_output_ms"],
              "peak_vram_mib": aggregate['peak_allocated_bytes'] / (1024 ** 2),
              "actor_mean_latency_ms": aggregate.get('actor_mean_ms'),
              "actor_latency_source": None, "actor_latency_is_headline_per_frame_measurement": False,
              "write_rate": aggregate['write_rate'], "raft_call_rate": aggregate['raft_call_rate'],
              "kda_update_rate": aggregate['kda_update_rate'],
              "coverage_complete": aggregate['coverage_complete'], "hardware_id": hardware['hardware_id'],
              "dtype": config.training.new_memory_dtype, "pretrained_dtype": config.training.pretrained_compute_dtype,
              "warmup_frames": args.warmup_frames, "timing_method": "cuda_sync_pair_boundaries",
              "timing_scope": "online_pointmap_predicted_camera_depth_memory_policy_no_component_hooks"}
    timing.update(actor_profile_timing)
    report = {"schema": "endo3r_performance_v1", "experiment_id": args.experiment_id,
              "baseline_identity": config.baseline_identity, "seed": args.seed,
              "manifest_sha256": file_sha256(manifest), "split": args.split,
              "checkpoint_hashes": checkpoint_hashes,
              "frame_indices_sha256": canonical_hash(frame_fingerprints),
              "provenance_sha256": canonical_hash(provenance_fingerprints),
              "protocol": DEPTH_PROTOCOL, "scale_protocol": SCALE_PROTOCOL,
              "prediction_camera_source": CAMERA_SOURCE, "pose_protocol": POSE_PROTOCOL,
              "image_hw": [config.training.image_height, config.training.image_width],
              "config": config.to_dict(), "config_sha256": canonical_hash(config.to_dict()),
              "original_checkpoint_hashes": hashes,
              "hybrid_checkpoint_sha256": file_sha256(args.hybrid_checkpoint) if args.hybrid_checkpoint else None,
              "actor_checkpoint_sha256": file_sha256(args.actor_checkpoint) if args.actor_checkpoint else None,
              "sequence_selection": selections, "sequence_selection_sha256": canonical_hash(selections),
              "warmup_frames": args.warmup_frames, "warmup": warmup_reports,
              "hardware": hardware, "timing": timing, "repeats": args.repeats,
              "timing_scope": "online_pointmap_predicted_camera_depth_memory_policy_no_component_hooks",
              "fps_denominator": "completed_output_frames", "io_in_headline_fps": False,
              "input_preparation_scope": "streaming_rgb_gt_loader_plus_transfer_reported_separately",
              "state_reset": "every_sequence_and_repeat", "sequences": sequences,
              "aggregate": aggregate, "paper_fps_hardware_match": "UNKNOWN",
              "actual_backends": sequences[-1]["actual_backends"] if sequences else {},
              "uncertainty_metric": config.uncertainty.metric,
              "reference_mode": args.reference}
    report['source_provenance'] = source_identity
    report['actor_component_profile'] = component_profile_identity
    report['runtime_backend_provenance'] = runtime_backend_provenance(model)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
