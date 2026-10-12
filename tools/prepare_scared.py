"""Prepare only data already present on a server; never acquires any resource."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dust3r.data.scared import build_scared_manifest, discover_source_index, manifest_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Existing server-local SCARED root")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-index", help="Verified explicit sequence/frame index JSON")
    group.add_argument("--mapping", help="Verified native-layout globs and JSON fields mapping")
    parser.add_argument("--output", required=True, help="Fresh output manifest path; source dataset is never modified")
    parser.add_argument("--split-map", help="Optional JSON assigning every dataset id 1..9 to a split")
    parser.add_argument("--dry-run", action="store_true", help="Validate declarations/paths only; no writes, copies, extraction or image decoding")
    parser.add_argument("--extract-video", action="store_true", help="Extract declared left RGB frames from local videos to output area; lazy OpenCV import")
    parser.add_argument("--verify-arrays", action="store_true", help="After preparation validate RGB/GT pixel alignment and finite coverage one frame at a time")
    args = parser.parse_args(argv)
    split_map = json.loads(Path(args.split_map).read_text(encoding="utf-8")) if args.split_map else None
    source = args.source_index or discover_source_index(args.data_root, args.mapping)
    result = build_scared_manifest(args.data_root, source, args.output, dry_run=args.dry_run,
                                  extract_video=args.extract_video, verify_arrays=args.verify_arrays,
                                  splits=split_map)
    print(json.dumps({"dry_run": args.dry_run, "manifest_written": None if args.dry_run else str(Path(args.output).resolve()),
                      **manifest_summary(result)}, indent=2))


if __name__ == "__main__":
    main()
