# Paper-matched evaluation and the three-way attention comparison

The active catalog is `configs/experiments/scared_study.json`. Its measured
comparison has exactly three branches:

| ID | Attention/memory | WRITE controller | Uncertainty | Weights |
| --- | --- | --- | --- | --- |
| `hybrid_sampson` | Exact SDPA Flash short memory, recurrent FLA KDA long memory, learned fusion | Heuristic | Existing Sampson/RAFT settings | Certified attention checkpoint for that seed |
| `hybrid_ppo` | Identical attention/memory | PPO, TEMP_ONLY or WRITE_LONG | Identical Sampson/RAFT settings | Same certified attention checkpoint, plus an actor trained in that frozen environment |
| `legacy_official` | Original released Endo3R dense dual memory | Original rule | Original released geometry_3d implementation | Released Endo3R checkpoint |

"Simpson" in the experiment request refers to **Sampson distance**. Neither
hybrid branch disables or changes it. Resolved hybrid configs differ only in
`policy.long_write_policy`; the study planner rejects differences in uncertainty,
memory, precision, reconstruction training settings, preprocessing or initial
checkpoint paths. The existing study driver binds both hybrid evaluation branches
to the same `hybrid_certified.pt` per seed and loads the PPO actor through the
existing environment contract.

The paper describes Sampson uncertainty, whereas released
`SpatialTempralMemory.uncertainty_map` calls
`detect_dynamic_regions_3d_distance`. The third branch preserves that released
behavior; replacing it would create a modified baseline. Published paper numbers
remain a separately identified reference and are never labelled as a server rerun.

## Evaluation settings matched to the paper

All three branches use SCARED test datasets **8–9**, left rectified RGB,
**320×256**, stride **1**, identical source frame IDs, preprocessing, GT validity,
seeds and hardware/timing scope. Seeds are 0/1/2. The local measurement procedure
uses four warmup frames, three timing repetitions and streaming sequence resets;
these choices are explicit project protocol, not recovered unpublished paper
timing settings. Prediction cameras/depth use pointmaps; GT K/pose are scoring
inputs only. Deployment metrics remain separate from GT-camera certification.
The shared crop/resampling and explicit scale/pose aggregation are project
protocols; identical resolution/split alone does not establish `paper_exact`.
Paper rectification parameters and complete metric/timing details are not known.

The current selection split remains train **1–6**, validation **7**, test **8–9**.
This holdout permits validation-only checkpoint selection and PPO certification.
The paper trained with SCARED **1–7**. A train1–7 refit requires a separately
verified, independent validation source before attention/PPO selection; dataset7
must not simultaneously be training data and a certification holdout. Test8–9
cannot replace validation. Thus the active config matches paper evaluation
settings while explicitly retaining a different training split.
This is a holdout for the new attention/PPO fitting, not an unseen dataset for
the released backbone: the paper's released Endo3R was already trained using
SCARED1–7. That initialization is shared and disclosed for both hybrid branches.

The paper reports an eight-dataset training mixture and a combination of
dynamics-aware flow, monocular-depth and confidence regression losses. The
released repository does not provide its training code. The project's existing
supervised pointmap objective remains an identified new training procedure;
paper loss weights, epoch counts and learning rates are not invented. This is
not an exact reconstruction of the original training recipe.

## Correct data sources and target identities

The source registry is [`scared_sources.json`](../configs/experiments/scared_sources.json).

- Original SCARED: [challenge home](https://endovissub2019-scared.grand-challenge.org/Home/),
  [geometry/GT description](https://endovissub2019-scared.grand-challenge.org/About/),
  [download access entry](https://endovissub2019-scared.grand-challenge.org/Downloads/).
  The download page could not be retrieved in this session; no direct archive URL
  or present account access is asserted. The existing local release is reused.
- Paper-style stereo training-label generator:
  [OpenStereo](https://github.com/XiandaGuo/OpenStereo),
  [StereoAnything author instructions](https://github.com/XiandaGuo/OpenStereo/blob/master/docs/StereoAnything.md).
  The author instructions list `StereoAnything-NMRF_swinT.bin` and a checkpoint
  access URL. No weights are acquired automatically, and this is not evidence
  of the exact version used by Endo3R's authors.
- The other seven training-dataset links are recorded from the
  [Endo3R author README](https://github.com/wrld/Endo3R#prepare-datasets).
  They are source references, not declarations that those datasets are present
  or mapped in this SCARED-only pipeline.

Endo3R paper section3.1 rectifies SCARED/StereoMIS stereo and uses StereoAnything
depth for training. The locally prepared A00 disparity instead comes from
original `scene_points` GT. Both rectified stereo RGB views exist in A00, but its
current disparity must not be relabelled as StereoAnything pseudo-depth.
No precomputed author StereoAnything training-target download was listed in the
inspected Endo3R README. Regenerating those labels would require a verified local
stereo checkpoint, exact source IDs/calibration and a separate output/provenance
manifest. Original measured/warped GT must remain the evaluation target.

The full raw/local A00 source contains datasets1–9. Source completeness does not
guarantee valid GT after320×256 resampling. The known dataset2/keyframe4/frame1729
empty-target failure still blocks full fitting. Do not fill its GT, treat
pseudo-depth as test GT or silently remove RGB frames from FPS denominators.

## Budget selection and execution

Attention steps, attention epochs and PPO updates default to **null**, not a
chosen1000-step/one-epoch/one-update recipe. Dry plans show explicit required
budget placeholders. Before executing selected fitting stages, pass
`--attention-steps`, `--attention-epochs` and/or `--ppo-updates` with a selected
positive budget. Execution validates those budgets before starting any process.
Evaluation-only stages do not require a new training budget when the required
certified checkpoints/actor already exist.

```bash
# Read-only review: no optimization or model inference.
python tools/run_scared_study.py --stage all

# Materialize matching configs outside the repo; still no optimization.
python tools/run_scared_study.py --stage all --write-configs \
  --manifest /work/scared_prepared/verified_manifest.json \
  --dust3r-checkpoint /models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --endo3r-checkpoint /models/endo3r.pth \
  --raft-checkpoint ./checkpoints/raft-things.pth \
  --output-root /work/endoarl_paper_comparison
```

These paths are placeholders. Actual execution additionally needs `--server-run`,
verified full data/backend preflight, an explicit fitting budget and all relevant
checkpoint/certification/cost-calibration prerequisites. No new production
measurement is implied by a dry/materialized config or a rendered N/A table.

Sources: [Endo3R paper](https://arxiv.org/html/2504.03198v1),
[official released model](https://github.com/wrld/Endo3R/blob/main/dust3r/model.py),
[released depth scorer](https://github.com/wrld/Endo3R/blob/main/eval/depth_evaluation.py).
