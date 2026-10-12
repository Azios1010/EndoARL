"""Render supplied server JSON results only; never performs inference/training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dust3r.experiments.results import render_benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="*", default=[], help="endo3r_benchmark_v1 JSON quality/performance artifacts")
    parser.add_argument("--output", required=True, help="Output directory for CSV/JSON/Markdown/HTML/SVG")
    parser.add_argument("--paper-reference", help="Optional explicitly reference-only published numbers JSON")
    parser.add_argument("--catalog", help="Optional study catalog; pending experiments are shown as N/A, never fake measurements")
    args = parser.parse_args(argv)
    print(json.dumps(render_benchmark(args.reports, args.output, paper_reference=args.paper_reference, catalog=args.catalog), indent=2))


if __name__ == "__main__":
    main()
