# Local experiment implementation verification

This is the verification snapshot captured before publication. Git HEAD/status and the no-commit/no-push statement below describe that local verification phase, not subsequent publishing.

Status: READY_FOR_SERVER_VALIDATION. Local code/reference verification is complete; no real-data training, CUDA benchmark or reconstruction metric is claimed.

## Executed checks

- `python -B -m unittest discover -s tests`: **209 passed, 0 failed**.
- AST parsing: **46 changed/new Python files passed**.
- `git diff --check`: passed; Git CRLF normalization notices are informational.
- Ten CLI/config/help/reference checks passed: hybrid trainer reference construction, PPO dry mode, both evaluators, uninstrumented benchmark dry mode, component profiler dry mode, study dry mode, CPU memory backend conformance, SCARED preparer help and cost-calibration help.
- Default study dry mode: 37 concrete commands; no configs written or processes started.
- Benchmark preview rendered HTML/CSV/JSON/Markdown/SVG with **0 measured runs**, 3 pending experiment rows and one separate published paper reference.

Expected argparse errors in negative unit tests and deprecated autocast warnings from the preserved official forward do not indicate test failures.

## Execution constraints

- No package installation, environment modification, dataset/checkpoint download, model-hub access, GPU benchmark or real training/PPO fitting was performed.
- CPU tests use tiny synthetic/reference tensors; the numerical PPO-update unit test is not an Endo3R training run.
- SCARED data stays on the server. Native mapping/rectification/pose/synchronization declarations require operator verification.
- Released DUSt3R/Endo3R/RAFT files, CUDA Flash/FLA/Triton environment and real fitting/evaluation remain server requirements.
- New actor checkpoints bind the actual reconstruction/uncertainty environment; old unbound actors fail clearly.
- Official memory/model source was not modified in this local task. Legacy numerical streamed equivalence and actual CUDA gradients remain SERVER_VALIDATION_REQUIRED.
- No commit or push was performed.

## Git state

- Branch: `main`, tracking `endoarl/main`.
- HEAD: `c8694c13730a7517bebc768006957298e5dd2253`.
- Working tree: **16 modified files, 46 new/untracked files**, including this record and benchmark preview.

## Modified files

- `README.md`
- `demo_hybrid.py`
- `dust3r/memory/config.py`
- `dust3r/memory/types.py`
- `dust3r/memory/uncertainty.py`
- `dust3r/online/runner.py`
- `dust3r/training/hybrid_trainer.py`
- `dust3r/training/sequence_dataset.py`
- `dust3r/training/validation.py`
- `eval/eval_hybrid.py`
- `tests/test_legacy_dispatch.py`
- `tests/test_validation.py`
- `tools/check_memory_backends.py`
- `tools/profile_memory.py`
- `train_hybrid_memory.py`
- `train_rl_controller.py`

## Added files

- `configs/experiments/paper_reference.json`
- `configs/experiments/scared_native_mapping.example.json`
- `configs/experiments/scared_study.json`
- `configs/scared/hybrid_ppo.yaml`
- `configs/scared/hybrid_ppo_no_uncertainty.yaml`
- `configs/scared/hybrid_sampson.yaml`
- `configs/scared/legacy_official.yaml`
- `docs/LOCAL_EXPERIMENT_VERIFICATION.md`
- `docs/SCARED_EXPERIMENT_PROTOCOL.md`
- `docs/benchmark_preview/benchmark.csv`
- `docs/benchmark_preview/benchmark.html`
- `docs/benchmark_preview/benchmark.json`
- `docs/benchmark_preview/benchmark.md`
- `docs/benchmark_preview/quality_vs_fps.svg`
- `docs/benchmark_preview/quality_vs_vram.svg`
- `dust3r/data/__init__.py`
- `dust3r/data/scared.py`
- `dust3r/evaluation/__init__.py`
- `dust3r/evaluation/camera.py`
- `dust3r/evaluation/metrics.py`
- `dust3r/evaluation/reports.py`
- `dust3r/evaluation/streaming.py`
- `dust3r/experiments/__init__.py`
- `dust3r/experiments/benchmark.py`
- `dust3r/experiments/results.py`
- `dust3r/experiments/study.py`
- `dust3r/memory/camera_geometry.py`
- `dust3r/online/streaming.py`
- `dust3r/training/provenance.py`
- `dust3r/training/reproducibility.py`
- `eval/eval_benchmark.py`
- `tests/test_benchmark_metrics.py`
- `tests/test_benchmark_report.py`
- `tests/test_benchmark_runner.py`
- `tests/test_cost_calibration.py`
- `tests/test_experiment_provenance.py`
- `tests/test_geometry_eval_streaming.py`
- `tests/test_sampson.py`
- `tests/test_scared_data.py`
- `tests/test_scared_study.py`
- `tests/test_streaming_runner.py`
- `tools/benchmark_experiment.py`
- `tools/calibrate_costs.py`
- `tools/prepare_scared.py`
- `tools/render_benchmark.py`
- `tools/run_scared_study.py`

## Server handoff

Read [SCARED_EXPERIMENT_PROTOCOL.md](SCARED_EXPERIMENT_PROTOCOL.md). Review `python tools/run_scared_study.py --stage all` with actual server paths. Explicit `--write-configs --server-run` is required for jobs; training scripts independently require `--run-training`.

Baseline, Flash/KDA+heuristic+Sampson and Flash/KDA+PPO+Sampson use declared splits/seeds/protocols. Both hybrid variants share the same certified attention weights per seed. Published paper values are not measurements on this machine or a validated speedup denominator.
