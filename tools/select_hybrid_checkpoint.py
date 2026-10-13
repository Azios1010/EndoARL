"""Rank existing validation reports; never runs inference/training/downloads.

Default is read-only. Use --write-selection with two new output paths to call
the existing held-out certification gate and save a separate PPO checkpoint.
SERVER_VALIDATION_REQUIRED: reports must identify real Flash/FLA backends,
released initialization weights, exact held-out frames and source fingerprints.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', required=True, help='Identified LEGACY_OFFICIAL validation JSON')
    parser.add_argument('--candidate', action='append', nargs=2, required=True,
                        metavar=('CHECKPOINT', 'VALIDATION_JSON'), help='Repeat for each attention checkpoint/report pair')
    parser.add_argument('--quality-tolerance', type=float, default=.05, help='Existing per-sequence relative gate; default 5 percent')
    parser.add_argument('--min-training-steps', type=int, help='Operator-specified budget, not a convergence threshold')
    parser.add_argument('--ppo-manifest', help='Optional exact manifest the current PPO trainer will use; mismatches block selection')
    parser.add_argument('--allow-bounded-validation', action='store_true', help='Explicitly acknowledge prefix-only selection; report preserves bounded coverage')
    parser.add_argument('--write-selection', action='store_true', help='Explicitly certify selected checkpoint and write two new artifacts')
    parser.add_argument('--output-checkpoint', help='NEW separately named certified checkpoint; never overwritten')
    parser.add_argument('--output-report', help='NEW machine-readable selection JSON; never overwritten')
    args = parser.parse_args(argv)
    if args.write_selection and (not args.output_checkpoint or not args.output_report):
        parser.error('--write-selection requires --output-checkpoint and --output-report')
    if not args.write_selection and (args.output_checkpoint or args.output_report):
        parser.error('Outputs require explicit --write-selection; default command is read-only')
    from dust3r.training.selection import select_hybrid_checkpoint, write_checkpoint_selection
    result = select_hybrid_checkpoint(
        [{'checkpoint': pair[0], 'report': pair[1]} for pair in args.candidate], args.baseline_report,
        tolerance=args.quality_tolerance, min_training_steps=args.min_training_steps,
        ppo_manifest=args.ppo_manifest, allow_bounded_validation=args.allow_bounded_validation)
    if args.write_selection and result['selected_candidate'] is not None:
        result = write_checkpoint_selection(result, args.output_checkpoint, args.output_report)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['selected_candidate'] is not None else 2


if __name__ == '__main__':
    raise SystemExit(main())
