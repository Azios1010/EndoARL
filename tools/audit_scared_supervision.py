"""Inventory server-local GT coverage; optionally create separate TRAIN-only runs.

No dataset, checkpoint, package, model, training job, or GPU is acquired or run.
The original manifest and full validation/test/evaluation frame lists are never
changed. A 1-pixel threshold prevents crashes, not insufficient supervision.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dust3r.data.supervision import audit_supervision, build_training_manifest, write_json_new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Original verified all-split manifest")
    parser.add_argument("--report", required=True, help="Fresh JSON inventory path; never overwritten")
    parser.add_argument("--training-manifest", help="Optional fresh train-only supervision manifest; never use for evaluation")
    parser.add_argument("--config", help="Existing hybrid config supplying image resolution, stride and clip length")
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--stride", type=int, help="Stride on the SOURCE manifest; derived manifest loader uses stride=1")
    parser.add_argument("--clip-raw-frames", type=int)
    parser.add_argument("--min-valid-pixels", type=int, default=1,
                        help="Minimum final valid GT pixels; default 1 is crash-prevention only")
    parser.add_argument("--min-valid-fraction", type=float, default=0.)
    parser.add_argument("--expected-frame-id-step", type=int, default=1,
                        help="Expected raw numeric frame-ID increment; gaps split runs")
    args = parser.parse_args(argv)
    config_training = {}
    if args.config:
        from dust3r.memory.config import load_config
        config_training = vars(load_config(args.config).training)
    image_hw = (args.image_height if args.image_height is not None else config_training.get("image_height", 256),
                args.image_width if args.image_width is not None else config_training.get("image_width", 320))
    stride = args.stride if args.stride is not None else config_training.get("stride", 1)
    clip = args.clip_raw_frames if args.clip_raw_frames is not None else config_training.get("sequence_raw_frames", 9)
    outputs = [Path(path).resolve() for path in (args.report, args.training_manifest) if path]
    if len(set(outputs)) != len(outputs) or Path(args.manifest).resolve() in outputs:
        parser.error("Output paths must be distinct from each other and the source manifest")
    for output in outputs:
        if output.exists():
            parser.error(f"Refusing to overwrite existing output: {output}")
    inventory = audit_supervision(args.manifest, image_hw=image_hw, source_stride=stride,
                                 clip_raw_frames=clip, min_valid_pixels=args.min_valid_pixels,
                                 min_valid_fraction=args.min_valid_fraction,
                                 expected_frame_id_step=args.expected_frame_id_step)
    training_manifest, error = None, None
    if args.training_manifest:
        try:
            training_manifest = build_training_manifest(args.manifest, inventory)
        except ValueError as exc:
            error = str(exc)
            inventory["training_manifest_error"] = error
    report_path = write_json_new(args.report, inventory)
    manifest_path = write_json_new(args.training_manifest, training_manifest) if training_manifest else None
    print(json.dumps({"inventory": report_path, "training_manifest": manifest_path,
                      "frames_audited": len(inventory["frames"]),
                      "failures": len(inventory["failures"]),
                      "sequence_failures": len(inventory["sequence_failures"]),
                      "train_clips": inventory["train_clip_count"],
                      "split_counts": inventory["split_counts"],
                      "threshold_is_quality_certification": False,
                      "source_manifest_unchanged": True,
                      "training_manifest_error": error}, indent=2))
    return 2 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
