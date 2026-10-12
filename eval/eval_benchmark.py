"""Versioned predicted-camera Endo3R benchmark, local files only.

Default is dry validation; --run-evaluation explicitly enables server inference
or scoring precomputed local pointmaps. No install/download/hub code exists.
SERVER_VALIDATION_REQUIRED: real data alignment, checkpoints, Flash/FLA/PnP.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/hybrid_memory.yaml")
    result.add_argument("--manifest")
    result.add_argument("--split", default="test")
    result.add_argument("--device", default="cpu")
    result.add_argument("--hybrid-checkpoint")
    result.add_argument("--actor-checkpoint")
    result.add_argument("--trusted-checkpoints", action="store_true")
    result.add_argument("--reference", action="store_true")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--experiment-id")
    result.add_argument("--skip-pose", action="store_true")
    result.add_argument("--max-sequences", type=int)
    result.add_argument("--max-frames", type=int, help="Explicit prefix per sequence; fingerprints record the selected frames.")
    result.add_argument("--prediction-manifest", help="Local schema1 {sequences:{episode_id:[.npy pointmap paths]},checkpoint_hashes:{...}}")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--run-evaluation", action="store_true")
    result.add_argument("--output", help="Required strict benchmark JSON path for evaluation")
    return result


def _prediction_entries(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("sequences"), dict):
        raise ValueError("Prediction manifest requires schema_version1 and episode_id->ordered .npy paths mapping.")
    if not payload.get("checkpoint_hashes"):
        raise ValueError("Precomputed predictions must identify checkpoint SHA256 hashes.")
    for digest in payload["checkpoint_hashes"].values():
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Checkpoint hashes must be lowercase SHA256 fingerprints.")
    return payload


def evaluate_stream(sample, *, runner=None, prediction_paths=None, seed=0, pose=True):
    """Public integration API for orchestration; samples expose get_frame(i).

    Runner consumes only frame dict, never target. CPU GT/current prediction
    are submitted immediately; final-frame output uses terminal pair res2.
    """
    if (runner is None) == (prediction_paths is None):
        raise ValueError("Supply exactly one streaming runner or precomputed prediction path list.")
    from dust3r.evaluation import StreamingSequenceEvaluator
    evaluator = StreamingSequenceEvaluator(sample.episode_id, expected_frames=sample.frame_count,
                 provenance=sample.provenance, pnp_seed=seed, pose=pose)
    if prediction_paths is not None:
        import numpy as np
        if len(prediction_paths) != sample.frame_count:
            raise ValueError("Precomputed predictions must cover exactly every selected sequence frame.")
        for index, path in enumerate(prediction_paths):
            frame, target = sample.get_frame(index)
            evaluator.add_frame(index, np.load(path, allow_pickle=False), target,
                                frame_identity=frame.get("instance", str(index)))
    else:
        import torch
        from dust3r.training.hybrid_trainer import move_tensors
        parameter = next(runner.model.parameters())
        previous_target = previous_identity = None
        with torch.no_grad():
            for index in range(sample.frame_count):
                frame, target = sample.get_frame(index)
                pair = runner.step(move_tensors(frame, parameter.device), terminal=index == sample.frame_count - 1)
                if pair is not None:
                    evaluator.add_frame(index - 1, pair.res1, previous_target,
                                        frame_identity=previous_identity)
                    if pair.terminal:
                        evaluator.add_frame(index, pair.res2, target, frame_identity=frame.get("instance", str(index)))
                    del pair
                previous_target, previous_identity = target, frame.get("instance", str(index))
    result = evaluator.finalize()
    if runner is not None:
        state = runner.state
        result["controller_counts"] = {name: getattr(state, name, getattr(runner, name, None)) for name in (
            "write_count", "write_attempt_count", "raft_call_count", "kda_update_count")}
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    from demo_hybrid import prepare_config
    from dust3r.evaluation.streaming import CAMERA_SOURCE, DEPTH_PROTOCOL, POSE_PROTOCOL, SCALE_PROTOCOL
    config = prepare_config(args.config, args.reference)
    manifest = args.manifest or config.training.data_manifest
    dry_report = {"dry_run": True, "no_downloads": True, "SERVER_VALIDATION_REQUIRED": True,
                  "manifest": manifest, "split": args.split, "protocol": DEPTH_PROTOCOL,
                  "prediction_camera_source": CAMERA_SOURCE, "scale_protocol": SCALE_PROTOCOL,
                  "pose_protocol": None if args.skip_pose else POSE_PROTOCOL,
                  "memory_backend": config.memory.backend, "policy": config.policy.long_write_policy,
                  "required": ["local calibrated manifest", "identified local checkpoints or precomputed predictions",
                               "local OpenCV for predicted-camera PnP", "Flash/FLA for efficient hybrid backend"],
                  "will_run_inference": False}
    if args.dry_run or not args.run_evaluation:
        print(json.dumps(dry_report, indent=2, allow_nan=False))
        return 0
    if not manifest or not args.output:
        raise ValueError("--manifest and --output are required for explicit server evaluation.")
    if args.max_sequences is not None and args.max_sequences < 1:
        raise ValueError("--max-sequences must be positive.")
    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError("--max-frames must be >=2.")
    import torch
    import numpy as np
    from dust3r.training.sequence_dataset import SequenceDataset
    from dust3r.evaluation.reports import file_sha256, make_report, save_report
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = SequenceDataset(manifest, split=args.split,
                              image_hw=(config.training.image_height, config.training.image_width),
                              stride=config.training.stride)
    if not hasattr(dataset, "stream"):
        raise RuntimeError("Streaming dataset API SequenceDataset.stream is required; no whole-sequence GPU fallback.")
    predictions = _prediction_entries(args.prediction_manifest) if args.prediction_manifest else None
    model = None
    if predictions is None:
        from demo_hybrid import load_inference_model
        from dust3r.online.streaming import create_stream_runner
        model, hashes = load_inference_model(config, device=args.device,
            hybrid_checkpoint=args.hybrid_checkpoint, actor_checkpoint=args.actor_checkpoint,
            trusted_checkpoints=args.trusted_checkpoints)
        if args.hybrid_checkpoint:
            hashes["hybrid"] = file_sha256(args.hybrid_checkpoint)
        if args.actor_checkpoint:
            hashes["actor"] = file_sha256(args.actor_checkpoint)
    else:
        hashes = predictions["checkpoint_hashes"]
    sequences = []
    for index in range(min(len(dataset), args.max_sequences or len(dataset))):
        sample = dataset.stream(index)
        if args.max_frames is not None:
            sample = sample.prefix(min(args.max_frames, sample.frame_count))
        if predictions is None:
            runner = create_stream_runner(model, config, frame_count=sample.frame_count, episode_id=sample.episode_id)
            result = evaluate_stream(sample, runner=runner, seed=args.seed, pose=not args.skip_pose)
            del runner
        else:
            listed = predictions["sequences"].get(sample.episode_id)
            if listed is None:
                raise ValueError(f"Missing prediction sequence {sample.episode_id}")
            if args.max_frames is not None:
                listed = listed[:sample.frame_count]
            prediction_root = Path(args.prediction_manifest).resolve().parent
            paths = [Path(path) if Path(path).is_absolute() else prediction_root / path for path in listed]
            result = evaluate_stream(sample, prediction_paths=paths, seed=args.seed, pose=not args.skip_pose)
        sequences.append(result)
    actual = {}
    if model is not None:
        memory = getattr(model, "hybrid_memory", None)
        actual = {"short": memory.short.last_backend if memory is not None else "legacy_dense",
                  "long": memory.long.adapter.last_backend if memory is not None else "legacy_dense"}
    report = make_report(sequences, config=config.to_dict(), manifest_sha256=file_sha256(manifest),
        checkpoint_hashes=hashes, split=args.split,
        image_hw=(config.training.image_height, config.training.image_width), seed=args.seed,
        experiment_id=args.experiment_id, actual_backends=actual,
        additional_metadata={"preprocessing_protocol": "manifest_joint_center_crop_resize_pixel_center_K",
                             "prediction_manifest_sha256": file_sha256(args.prediction_manifest) if args.prediction_manifest else None,
                             "max_sequences": args.max_sequences, "max_frames": args.max_frames,
                             "timing_measured": False})
    from dust3r.training.provenance import source_provenance, runtime_backend_provenance
    report['source_provenance'] = source_provenance()
    report['runtime_backend_provenance'] = runtime_backend_provenance(model)
    save_report(report, args.output)
    print(json.dumps({"output": args.output, "aggregate": report["aggregate"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
