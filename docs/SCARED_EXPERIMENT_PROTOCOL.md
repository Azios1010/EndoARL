# SCARED local-to-server experiment protocol

This implementation prepares **Flash/KDA + heuristic + Sampson** and **Flash/KDA + PPO + Sampson** comparisons against the released Endo3R baseline. No SCARED data, checkpoints or optional CUDA packages are fetched by any command. Local development uses synthetic CPU fixtures. Actual fitting and measurements are `SERVER_VALIDATION_REQUIRED`.

## Experiment identities

| ID | Reconstruction/memory | Long-write controller | WRITE uncertainty | Initialization |
|---|---|---|---|---|
| `legacy_official` | Released Endo3R, original dense memory and original risks | Original similarity rule | Original 3D/RAFT implementation | Released checkpoints |
| `hybrid_sampson` | Exact Flash short memory + fixed-state KDA + learned fusion | Recent-descriptor heuristic, threshold 0.92 | Predicted-camera Sampson residual from RAFT flow | Released network + supervised trained memory |
| `hybrid_ppo` | **Same certified hybrid checkpoint as `hybrid_sampson` for that seed** | Two-action PPO: TEMP_ONLY / WRITE_LONG | **Same Sampson estimator and constants** | Frozen certified reconstruction + trained actor |

Sampson is an uncertainty measure, not an attention kernel or a third learned action. Keeping it in both hybrid experiments separates the controller's effect. `configs/scared/hybrid_ppo_no_uncertainty.yaml` is an optional separate ablation; an actor trained with Sampson must not be loaded into that changed environment.

The original `SpatialTempralMemory` and original forward body remain unchanged. The streamed legacy adapter reproduces operation ordering, including the official terminal write and uncertainty flag behavior. Its numerical agreement with the released CUDA path must be checked on the server. Streaming input/output residency differs from the original demonstration that retains a full GPU sequence; this is disclosed and used equally for all headline benchmarks. No `LEGACY_CLEAN` result is created implicitly.

## Data and split contract

The proposed model-selection split is datasets **1–6 train, 7 validation, 8–9 test**. Patient identities must be verified and disjoint. Paper evaluation uses 8–9; holding 7 out means this fit is **not the paper's full training recipe**. Do not tune thresholds, reward weights, iteration budgets or checkpoint selection on 8–9. After selecting settings, a separately identified 1–7 refit may be reported, with no remaining validation or certification leakage.

Each sequence needs aligned left RGB, depth or camera-space XYZ, valid mask/rule, verified rectified intrinsics, verified pose direction and axes, units, source frame IDs and GT provenance. The converter changes millimetres to metres and W2C to C2W only when explicitly declared. It never invents identity poses, calibration or synchronization. Native release layouts vary, so the example mapping is deliberately **unverified**. It is not a ready-to-run claim about the user's server files.

1. Inspect the existing server release. Copy `configs/experiments/scared_native_mapping.example.json` outside the repository and replace its globs, JSON field paths, patient identities, units and verification flags. Alternatively provide an explicit `schema_version: 1` source index with `sequences`, per-sequence `root`, metadata declarations and per-frame `rgb`, `depth`/`pointmap`, `pose`, `intrinsics`, `frame_id`.
2. Confirm RGB/GT index alignment, rectification, calibration, pose direction and kinematic-warp GT limitations. File existence alone does not prove these. Missing/ambiguous index joins fail rather than silently discarding frames. For partial GT, supply a verified explicit source index; the converter does not guess a video offset.
3. Validate metadata without decoding or writing:

```bash
python tools/prepare_scared.py --data-root /data/SCARED --mapping /data/config/scared_verified.json --output /work/scared_prepared/manifest.json --dry-run
```

4. Prepare an external manifest and validate arrays on the server:

```bash
python tools/prepare_scared.py --data-root /data/SCARED --mapping /data/config/scared_verified.json --output /work/scared_prepared/manifest.json --verify-arrays
```

Use `--source-index` instead of `--mapping` for the explicit index. Video extraction is opt-in `--extract-video` and requires a declared left-image layout and frame index. Preparation refuses to overwrite an existing manifest. It writes outside the dataset source tree and leaves source files intact.

The loader resizes RGB to 320×256, uses nearest-neighbour GT/mask resizing, scales intrinsics, preserves row-major encoder patch order and expresses supervised pointmaps in the selected sequence's first camera frame. Stride is explicitly recorded (v1: 1). Full-sequence evaluation loads only a current frame/target pair; PPO device views cache at most two frames. Exact whole-sequence depth median alignment retains valid depth samples on **CPU** and is not constant-memory scoring.

## Architecture and causal uncertainty

Encoder, decoder and DPT pointmap/confidence heads are initialized from released weights. Short attention: 512 dimensions, 8 heads of 64, up to four frames, exact softmax. Long memory: 256 dimensions, 4 heads of 64, FP32 recurrent state `[B,4,64,64]`. Both produce `[B,P,1024]` and feed learned channel fusion. These are v1 configurations, not locally validated GPU speed or quality results.

For each arriving pair, read existing memory, decode and predict, construct current K/V, build the 30-scalar inference-only actor observation from **pre-update** memory, choose an action, always append temporal memory, then optionally compute uncertainty and update KDA. Candidate index t is reconstructed after frames t and t+1 have arrived, as in Endo3R; the action timestamp is arrival t+1. No unarrived frame t+2 or future reconstruction enters it. The next quality reward is the later final left output for candidate t+1 after arrival t+2. GT and current action's RAFT result never enter the actor. The first viable long write is forced. The terminal pair has no learned action or long write in the hybrid path; it still yields the final output. TEMP_ONLY leaves the long recurrent state and long reference unchanged.

On WRITE_LONG after a reference exists:

```text
RAFT(current RGB, last successful long-write RGB), 20 iterations, frozen/eval/no_grad/FP32
→ current-to-reference flow
→ cameras estimated from predicted pointmaps (no GT intrinsics/pose)
→ relative transform T_ref←cur = W2C_ref · C2W_cur
→ F = K_ref⁻ᵀ [t]× R K_cur⁻¹
→ squared Sampson residual d² in pixels²
→ u = clamp(d² / sigma_px², 0, 1)
→ per-patch valid fraction and mean u
→ c = valid_fraction · clamp(exp(-alpha · u), valid_confidence_floor, 1)
→ legal KDA beta/update-strength and decay weighting
```

The first write uses neutral finite-pixel confidence and skips RAFT/PnP. Invalid correspondences, out-of-bounds flow, failed predicted cameras, zero denominators and near-zero translation do not become high confidence. Pure rotation is a declared degenerate Sampson case and can suppress writes; evaluate its frequency. No successful write means no reference replacement. Only the last successful reference/camera is cached, and episode reset clears it. `sigma_px=1`, translation threshold `1e-6` in predicted common-frame units, confidence alpha/floor, and bounded write-PnP sampling are **provisional**; choose them on validation, not test.

Write-camera PnP is bounded/masked for uncertainty. Benchmark camera PnP uses the released dense pointmap procedure. These are distinct scopes and both are included when incurred in end-to-end timing. The former can be expensive even after removing dense long-memory retrieval.

## Training procedure implemented

All new training entrypoints default to dry/config mode; `--run-training` is required for optimization. The study driver invokes it only under explicit `--server-run`. No command installs dependencies or acquires missing files.

**T0:** load local released DUSt3R and Endo3R checkpoints and reproduce legacy validation/test inference. Legacy CUDA code assumes device 0 and loads `checkpoints/raft-things.pth` relative to the repository; providing another config path cannot repair that official behavior. Place an existing verified file at that location on the server yourself. The driver must report this restriction.

**T1:** RL disabled. Train short Q/K/V/output projections, KDA projections/gates/output projection and fusion. Freeze image encoder, decoder, DPT heads, `attn_head_1`, `attn_head_2`, `value_encoder`, shared `norm_q/norm_k/norm_v` and RAFT. Initialize pretrained modules from released weights, new projections from module initializers, fusion with initial short preference. AdamW `(0.9,0.95)`, new LR `1e-4`, weight decay `0.01` except norms/bias/gate timescales, warm-up+cosine schedule, gradient clipping `1`. Use nine raw frames (eight pairs), four-pair TBPTT; detach state **after** backward/update and refresh projected short caches from detached features. Short history is ≤4 frames. First 10% of steps use deterministic always-write warm-up, then configured similarity policy.

The primary supported objective is pair-averaged masked common-first-camera point L1. Fit one detached scale from the first prediction/GT median radius and normalize errors by first-frame GT radius. Supervise both `res1` and `res2` with true GT masks; predicted invalidity may not shrink them. Do not realign each future frame. This is a defensible new objective in `dust3r/training/losses.py`, **not a recovered authors' Endo3R recipe**. Depth/pose/temporal training weights remain zero; confidence calibration is disabled until checkpoint semantics are verified. Optional prefix-matched frozen teacher feature L1+cosine is supported but not required and must identify teacher, preprocessing and action-trajectory hashes.

**T2 optional:** resume trained memory, unfreeze existing key/value encoders and norms at `1e-5`, optionally last two decoder blocks and heads at `5e-6`, while new memory remains `1e-4`. This is a separate supervised run, not enabled by the default study. The full image encoder stays frozen. Select checkpoint/budget using validation quality.

**Certification:** evaluate both official and hybrid using identical source frames, resolution, manifest, released initialization and detached first-frame geometry-scale protocol. Default validation prefix is 64 frames. Certification checks per-sequence normalized point L1 and depth AbsRel within relative tolerance 5% and requires actual Flash/FLA backend identities; it saves a **new** checkpoint. This is a safety gate, not a test-set paper metric or a claim of convergence. Failure stops PPO. Report all failures; do not silently repair the official baseline.

**R0:** freeze encoder/decoder/heads, key/value/norms, ST/LT/fusion and RAFT. Train only separate 30→64→64→2 actor and critic with PPO. Stochastic actions in fitting; deterministic argmax in evaluation. The actor uses current confidence/validity, feature novelty, memory/counter history, historical uncertainty and cheap fusion/output-norm diagnostics. No dense attention matrices are produced for RL. The critic is non-privileged in v1.

Dense GT reward uses fixed first-frame alignment and normalized point L1 of the **next decoded left output**, associated with the **previous action**, plus calibrated action-dependent RAFT/KDA/geometry cost. Forced first-write rows are excluded from actor optimization. PPO stores only observations/action masks/actions/log probabilities/value/reward/done and diagnostics; it never retains the video autograd graph. GAE, clipping, advantage normalization, frozen observation-normalizer snapshots, KL stop and gradient clipping are implemented in `dust3r/rl`. Default gamma `.99`, lambda `.95`, clip `.2`, actor/critic LR `3e-4` are tunable. KDA has fixed state, so there is no growing occupancy penalty. Terminal current output is not assigned as quality to an action that has not yet affected memory.

Checkpoint schemas include configs, initialized checkpoint hashes, source provenance, optimizer/scheduler/scaler as relevant and RNG states. PPO actors bind the reconstruction/uncertainty/preprocessing environment and observation schema; mismatched actors fail clearly. Resume restarts at a fresh clip/episode boundary; it does **not** promise an exact in-episode cursor replay.

The study's `1000` supervised steps, one epoch and one PPO update are smoke defaults. They are not adequate evidence of fit or final experiment budgets. Set explicit larger budgets on the server after a finite forward/backward smoke run and validation learning curves. Attention success is reconstruction certification; PPO success is improved validation quality/compute trade-off before test evaluation.

## Server workflow and artifacts

`configs/experiments/scared_study.json` is the declarative catalog. `tools/run_scared_study.py` builds ordered per-seed commands, materializes resolved configs only when requested, and executes them only with `--server-run`. Start with `--help` and a default dry invocation. Required inputs are existing server manifest, existing DUSt3R/Endo3R/RAFT checkpoints and a writable output root. Profiling needs a supplied **existing validation/train RGB directory**; do not profile test data to tune the cost reward. The driver emits exact commands for:

```text
preflight → official validation → supervised fit per seed → matched validation/certification
→ bounded component profile → frozen cost calibration → PPO per seed
→ held-out predicted-camera quality evaluation → uninstrumented throughput benchmark
→ joined HTML/CSV/JSON/Markdown/SVG benchmark table
```

Seeds are 0/1/2. Each experiment/seed has stable config/report/checkpoint paths. Heuristic and PPO use exactly the same certified attention weights for that seed. The actor adds its own checkpoint identity. Logs are JSONL and only written when execution is explicitly requested. Stages stop on errors; there is no automatic installation, checkpoint acquisition or baseline repair.

Concrete full workflow command for a prepared server (first remove `--write-configs --server-run` to review it):

```bash
python tools/run_scared_study.py --stage all \
  --manifest /work/scared_prepared/manifest.json \
  --dust3r-checkpoint /models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --endo3r-checkpoint /models/endo3r.pth \
  --raft-checkpoint ./checkpoints/raft-things.pth \
  --profile-input /data/SCARED/VERIFIED_VALIDATION_LEFT_RGB_DIRECTORY \
  --output-root /work/endoarl_scared --device cuda:0 \
  --trusted-checkpoints --write-configs --server-run
```

The input/profile paths must point to actual verified files, not these placeholders. Existing-cost-only execution can pass `--cost-calibration /work/measured_costs.json` instead of generating it. Stage selection supports separate restarts, for example `--stage preflight baseline fit`, then `--stage certify profile`, then `--stage ppo evaluate benchmark summarize`; dependent artifacts must already exist. Use a new output root when changing resolved configurations. Explicit trust is needed only for verified existing released pickle metadata.

Validate actual server kernels before fitting:

```bash
python tools/check_memory_backends.py --config configs/scared/hybrid_sampson.yaml --server-kernels --device cuda:0
```

Expected standalone shapes: input/output `[1,4,1024]`, KDA state `[1,4,64,64]`, finite gradients including final-state recurrence. FLA expected revision is `9f38d24980c46d46bd38614e743cdacd21906578`; verify the package source checkout/revision, CUDA/PyTorch/Triton compatibility, BF16 support and no hidden fallback on the server. Optional `flash-attn` is required only when that backend is selected; v1 prefers a **forced** SDPA Flash kernel. FP32 KDA CPU reference is a correctness tool, not the efficient training backend. Missing dependencies/checkpoints fail rather than downloading.

`tools/profile_memory.py` reports synchronized **exclusive** components: encoding, decoding, K/V, heads, ST/LT read/fusion, policy/observation, updates, RAFT and geometry. Its bounded compatibility run retains its bounded outputs and is not a headline FPS benchmark. `tools/calibrate_costs.py --profile PROFILE.json --output COST.json` converts measured positive per-call action costs and a per-pair common baseline into frozen PPO proxies. It requires actual Flash/FLA, hardware and image-size metadata; missing RAFT/update calls require a longer/appropriate profile, not invented zeros. Calibration can be reused across seeds only with the same architecture, precision, uncertainty metric, resolution and hardware.

## Quality comparison protocol

`eval/eval_benchmark.py` is the deployment-oriented scorer. Predictions and camera estimates receive **no GT K or pose**. It anchors intrinsics from the first predicted pointmap with Weiszfeld focal estimation (10 iterations) and uses dense predicted-pointmap PnP (100 iterations, 2-pixel reprojection threshold). Predicted local depth is derived using predicted cameras. CPU float64 focal/scoring and first-frame depth behavior are explicitly identified; numerical agreement with released scripts still requires server evidence.

Depth uses one **whole-sequence** median GT/predicted depth scale over true GT-valid pixels, clips to `1e-4..100` metres, then computes AbsRel, SqRel (m), RMSE (m), log-RMSE and δ1. Aggregation is frame-weighted. Failed predictions/cameras are counted and never silently removed from the denominator or hidden as perfect depth.

Pose uses predicted C2W, overlapping five-frame Sim(3)-aligned windows, stride one, adjacent relative-pose errors, ATE/RPE-translation in mm and RPE-rotation in degrees. Window failures and coverage are recorded. The paper does not fully specify pose alignment/aggregation; this is an explicit reproducible protocol, **not `paper_exact`**. A separate GT-camera first-frame geometry scorer (`eval/eval_hybrid.py`) is used only for training certification and must not be substituted for deployment depth/pose numbers.

The paper's table reports SCARED AbsRel `.124`, SqRel `.227`, RMSE `1.209`, log-RMSE `.135`, δ1 `.839`, FPS `19.17`; pose ATE `.112`, rotation RPE `.201`, translation RPE `.228`. RMSE/SqRel units are not explicit in the table and GPU/FPS timing scope is not sufficient to equate hardware. The renderer preserves those ambiguous values in a separate published table and never uses paper FPS as a measured speedup denominator. Primary source: [Endo3R paper](https://arxiv.org/html/2504.03198v1). The eight-dataset training mixture and unavailable original recipe prevent claiming exact paper reproduction from SCARED-only fitting.

## Headline performance protocol

`tools/benchmark_experiment.py` defaults dry. Actual execution needs `--server-run`, an identified experiment, local weights, manifest and prepared CUDA. Same GPU/process configuration, image size, stride, seeds, source frame IDs, precision and checkpoints are used across variants. Warm-up uses four frames, including predicted-camera/depth work and eligible WRITE/RAFT; reset all sequence state afterward. Repeat each full test sequence three times with synchronized timing at pair/sequence boundaries, not nested component hooks.

Headline scope includes encoder, memory reads, fusion, decoder, heads, key/value creation, observation/actor, short update, eligible RAFT/geometry/KDA update, predicted-camera PnP and depth conversion. RGB/GT loading and host-to-device preparation are timed separately and excluded from headline FPS. Sequence memory/camera construction and initial-frame setup are separate `sequence_memory_camera_setup_ms` / `initial_frame_setup_ms`; `fps_with_sequence_setup` additionally includes them. This matters because official legacy reloads RAFT at memory construction. Headline FPS describes decoded-pair steady-state work, with setup costs disclosed. Dataset preload/GT work is never counted as neural performance. No 3D renderer/Gaussian fitting is included. A sequence of N frames yields N outputs from N−1 decoded pairs; latency columns are **decoded pair latency**, with separate amortized output latency. Official terminal work is recorded. CUDA peak allocated/reserved bytes include resident model weights. Actor exclusive latency is unavailable in uninstrumented runs unless separately measured; it remains N/A rather than a guessed zero.

Measure FPS, mean/p50/p95 pair latency, amortized output latency, peak VRAM, successful WRITE rate, attempted write/RAFT/KDA call counts and rates, failed coverage, actual backend names and state bytes. With `--profile-input`, the study also produces a separate bounded PPO component profile on validation/train RGB and supplies it to the headline tool via `--component-profile`. Its synchronized `LongWritePolicy.act` mean is labeled **separate validation component profile proxy**, not test-frame actor latency. Config, model/actor hashes, source tree and hardware must match. Headline FPS still uses no component hooks. Without that artifact actor cost stays N/A. KDA state stays bounded; total live memory also includes model, four temporal frames, one reference and activation workspace. Failed coverage gives no headline FPS. Full raw repeats and protocol metadata remain in JSON.

## Table rendering and local verification

Quality/performance JSONs are joined only when experiment, seed, resolved config hash, loaded checkpoint hashes, manifest/split/resolution, exact source-frame fingerprints, provenance and scoring protocols match. Different hardware/precision/timing scopes cannot be averaged. Seed aggregation permits only the seed field to differ in the otherwise identical config, and records per-metric measured counts and sample standard deviation. Missing values remain N/A. Pending runs and synthetic tests are not measurements.

Local preview command (no inference/data):

```bash
python tools/render_benchmark.py --output docs/benchmark_preview --paper-reference configs/experiments/paper_reference.json --catalog configs/experiments/scared_study.json
```

Outputs `benchmark.html`, `.csv`, `.json`, `.md`, and SVG quality-vs-FPS/VRAM charts. Actual study summary passes matching quality/performance artifacts and creates mean±std rows for the three experiments. Paper-only rows are visually separate. Preview charts contain no invented experiment points.

Allowed local verification:

```bash
python -B -m unittest discover -s tests -v
python -B train_hybrid_memory.py --config configs/scared/hybrid_sampson.yaml --reference --validate-modules
python -B train_rl_controller.py --config configs/scared/hybrid_ppo.yaml
python -B tools/check_memory_backends.py --config configs/scared/hybrid_sampson.yaml --reference
```

These use existing Python packages and tiny synthetic/reference tensors. Unit tests cover equations, causal WRITE/SKIP, delayed rewards, resets, device caches, geometry, exact ST reference, KDA recurrence/gradients, GT masks, manifest splits, checkpoint contracts, dry orchestration and artifact joins. They do not demonstrate real-data reconstruction accuracy, efficient CUDA kernel compatibility or paper-equivalent FPS.

## Server readiness gates still outstanding

- Verify native SCARED layout, patient identities, calibration, pose direction, RGB/GT synchronization and valid masks; inspect reprojection and depth ranges before fitting.
- Supply released DUSt3R/Endo3R weights and the existing RAFT checkpoint; official legacy requires its hard-coded relative path and CUDA device 0.
- Validate actual Flash/FLA forward, recurrent final-state backward, TBPTT and memory/latency on intended hardware, including the expected FLA revision.
- Compare streamed official outputs against the original forward on a short real sequence and quantify any predicted-camera scorer numerical differences.
- Establish supervised convergence and matched validation certification before PPO; replace smoke budgets with a declared budget and validation checkpoint-selection rule.
- Measure RAFT/Sampson/KDA action costs, verify frozen calibration environment, train actors, then freeze every setting before evaluating 8–9.
- Publish failures, dataset coverage, raw metrics, hardware and timing scope along with the table. Do not claim Gaussian reconstruction, paper-exact reproduction or speed improvements before measurements exist.
