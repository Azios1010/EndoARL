"""Plan/write/execute reproducible SCARED server stages; default is read-only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dust3r.experiments.study import STAGES, execute_plan, materialize_configs, plan_study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="configs/experiments/scared_study.json")
    parser.add_argument("--stage", nargs="+", choices=("all", *STAGES), default=["all"])
    parser.add_argument("--manifest")
    parser.add_argument("--dust3r-checkpoint")
    parser.add_argument("--endo3r-checkpoint")
    parser.add_argument("--raft-checkpoint", help="Hybrid RAFT path; official legacy additionally requires checkpoints/raft-things.pth")
    parser.add_argument("--cost-calibration", help="Existing measured calibration; otherwise profile stage produces output-root/cost_calibration.json")
    parser.add_argument("--profile-input", help="Existing train/validation left RGB directory; no datasets8/9 tuning")
    parser.add_argument("--output-root", default="outputs/scared_study")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--attention-steps", type=int, help="Explicit supervised optimizer-step budget; no default fitting recipe")
    parser.add_argument("--attention-epochs", type=int, help="Explicit supervised epoch limit")
    parser.add_argument("--ppo-updates", type=int, help="Explicit PPO update budget")
    parser.add_argument("--python", dest="python_executable")
    parser.add_argument("--trusted-checkpoints", action="store_true")
    parser.add_argument("--write-configs", action="store_true", help="Explicit config materialization only; never launches jobs by itself")
    parser.add_argument("--server-run", action="store_true", help="Execute selected stages on a provisioned server; requires --write-configs or already materialized matching configs")
    args = parser.parse_args(argv)
    plan = plan_study(args.study, output_root=args.output_root, stages=tuple(args.stage), manifest=args.manifest,
                      dust3r_checkpoint=args.dust3r_checkpoint, endo3r_checkpoint=args.endo3r_checkpoint,
                      raft_checkpoint=args.raft_checkpoint, cost_calibration=args.cost_calibration,
                      profile_input=args.profile_input, device=args.device, seeds=args.seeds,
                      python_executable=args.python_executable, trusted_checkpoints=args.trusted_checkpoints,
                      attention_steps=args.attention_steps, attention_epochs=args.attention_epochs, ppo_updates=args.ppo_updates)
    report = plan.to_dict()
    if args.write_configs:
        report["written_configs"] = materialize_configs(plan)
    if args.server_run:
        if not args.write_configs:
            required_configs = {name for command in plan.commands for name in command.required_files if name in plan.configs}
            for name in required_configs:
                payload = plan.configs[name]
                path = Path(name)
                if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != payload:
                    raise ValueError("Server-run requires matching materialized configs; review and use --write-configs")
        report["dry_run"] = False
        report["execution_records"] = execute_plan(plan)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
