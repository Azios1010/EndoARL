# Endo3R benchmark

Benchmark artifacts; unavailable measurements are N/A.

| Experiment | Source | Seeds | Status | AbsRel ↓ | SqRel (m) ↓ | RMSE (m) ↓ | RMSE-log ↓ | δ₁ ↑ | ATE (mm) ↓ | RPE-t (mm) ↓ | RPE-r (°) ↓ | FPS ↑ | Decoded pair mean (ms) ↓ | Decoded pair p50 (ms) ↓ | Decoded pair p95 (ms) ↓ | Amortized output mean (ms) ↓ | Amortized output p50 (ms) ↓ | Amortized output p95 (ms) ↓ | Peak VRAM (MiB) ↓ | WRITE rate ↓ | RAFT rate ↓ | KDA rate ↓ | Actor act profile proxy (ms) ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Endo3R-paper | paper_reference_not_measured | N/A | reference | 0.124 | N/A | N/A | 0.135 | 0.839 | N/A | N/A | N/A | 19.17 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Endo3R released legacy | pending_server_run_not_measured | N/A | pending | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Flash/KDA + heuristic + Sampson | pending_server_run_not_measured | N/A | pending | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Flash/KDA + PPO + Sampson | pending_server_run_not_measured | N/A | pending | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Endo3R-paper provenance

```json
{
  "identity": {},
  "config_sha256": [],
  "seed_normalized_config_sha256": null,
  "checkpoint_hashes_by_seed": {},
  "actual_backends": {},
  "timing_context": {},
  "failure_counts": {},
  "artifact_paths": []
}
```

## legacy_official provenance

```json
{
  "identity": {},
  "config_sha256": [],
  "seed_normalized_config_sha256": null,
  "checkpoint_hashes_by_seed": {},
  "actual_backends": {},
  "timing_context": {},
  "failure_counts": {},
  "artifact_paths": []
}
```

## hybrid_sampson provenance

```json
{
  "identity": {},
  "config_sha256": [],
  "seed_normalized_config_sha256": null,
  "checkpoint_hashes_by_seed": {},
  "actual_backends": {},
  "timing_context": {},
  "failure_counts": {},
  "artifact_paths": []
}
```

## hybrid_ppo provenance

```json
{
  "identity": {},
  "config_sha256": [],
  "seed_normalized_config_sha256": null,
  "checkpoint_hashes_by_seed": {},
  "actual_backends": {},
  "timing_context": {},
  "failure_counts": {},
  "artifact_paths": []
}
```

## Published paper reference — not measured on this server

| Metric | Published value | Reported unit |
| --- | --- | --- |
| AbsRel | 0.124 | dimensionless |
| SqRel | 0.227 | unspecified_in_table |
| RMSE | 1.209 | unspecified_in_table |
| RMSE-log | 0.135 | dimensionless |
| δ₁ | 0.839 | dimensionless |
| ATE | 0.112 | mm |
| RPE-t | 0.228 | mm |
| RPE-r | 0.201 | degree |
| FPS | 19.17 | frames/s; paper hardware/scope not established |

```json
{
  "experiment_id": "Endo3R-paper",
  "source": "paper_reference_not_measured",
  "citation_url": "https://arxiv.org/html/2504.03198v1",
  "reported_units": {
    "rmse": "unspecified_in_table",
    "sq_rel": "unspecified_in_table",
    "ate": "mm",
    "rpe_translation": "mm",
    "rpe_rotation": "degree",
    "fps": "frames/s"
  },
  "depth_metrics": {
    "abs_rel": 0.124,
    "sq_rel": 0.227,
    "rmse": 1.209,
    "rmse_log": 0.135,
    "delta1": 0.839
  },
  "pose_metrics": {
    "ate": 0.112,
    "rpe_rotation": 0.201,
    "rpe_translation": 0.228
  },
  "timing": {
    "fps": 19.17
  },
  "hardware": "unspecified_in_v1",
  "notes": [
    "Values transcribed from Endo3R arXiv v1 Tables1/2, not measured locally.",
    "RMSE/SqRel units, FPS hardware/timing scope and trajectory alignment are insufficiently specified for direct numeric equivalence.",
    "No speedup ratio against this FPS is calculated."
  ]
}
```

Published metric units remain as reported; hardware/timing scope is unspecified. No paper FPS speedup is calculated.

Paper numbers are reference-only; GPU/timing scope and unspecified metric units are not inferred.
Std is sample standard deviation across available seeds; single-seed std is N/A.
N/A is never interpreted as zero; every metric records measured_n.
Complete and incomplete quality results are labelled separately.
Only matching verified training.seed values are normalized across seeds; all other config fields remain fixed.
Decoded-pair latency includes the terminal pair's two outputs; amortized output latency divides pair time by output count and is not observed frame-arrival latency.
Failed pose windows and incomplete timing mark rows incomplete and exclude them from quality/compute plots.
No FPS speedup is computed against the published paper.
No measured server results have been supplied; pending rows contain no measured metrics.
