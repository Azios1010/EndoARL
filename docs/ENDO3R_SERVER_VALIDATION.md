# Hybrid Endo3R implementation and server validation

This implements `ENDO3R_FLASH_KDA_RL_PLAN.md` against source commit
`8145c8e5535d2111974561e7d1cc06f1d008fe5e`. The audit and plan are unchanged.
The only existing source file changed is `dust3r/model.py`: optional module
registration, local-checkpoint validation for the configured path, and forward
dispatch. The original `SpatialTempralMemory` class and original forward body
are preserved; their known defects have not been silently fixed.

No package installation, downloads, model-hub access, real training, PPO
training, large inference, or CUDA kernel execution occurred during coding.
CPU tests use small tensors, injected models/flow, and temporary local fixtures.
Their backward/optimizer calls test gradient and PPO arithmetic, not learning
on actual reconstruction sequences. Efficient execution and reconstruction
quality remain **SERVER_VALIDATION_REQUIRED**.

## Implemented contracts

* `HybridMemory` owns registered short/long/fusion parameters. Its sequence state
  is runner-owned data, never persistent patient state in model weights.
* Short memory stores at most four projected frame K/V caches. The v1 projections
  are 1024 -> 512, eight heads of 64; output is `[B,P,1024]`. `math_reference`
  performs exact FP32 dense softmax. Generic `sdpa` is labelled unverified.
  `flash` tries forced PyTorch Flash SDPA, then official FlashAttention; neither
  selects math silently. Explicit unavailable efficient backends raise errors.
* Long memory is FP32 `[B,4,64,64]`, independent of sequence length: 65,536 bytes
  for B=1, excluding model parameters/temporary tensors. Read is a query/state
  contraction, with no update or decay. One accepted frame writes one chunk of
  P encoder tokens in their existing deterministic order. No second long KV
  bank, token pruning, `mem_attn`, or `mem_count` is used in hybrid mode.
* `KDAAdapter` isolates optional FLA imports. `torch_reference` implements the
  channel-gated delta recurrence explicitly; it is a correctness backend, not
  an efficient training implementation. The canonical key-first state is
  isolated from the FLA API. Chunk final-state gradients and optional no-grad
  recurrent inference require server conformance before use.
* The channel fusion gate returns `Q + g*O_ST + (1-g)*O_LT`. Empty, short-only,
  long-only and both-bank cases are finite. Diagnostics are scalar summaries;
  actor observations never materialize an attention matrix.
* `OnlineRunner.prepare_next()` decodes a pair with pre-update memory, then
  exposes an actor observation. `commit()` always appends short memory and
  optionally writes long memory. Candidate i is eligible after frame i+1 has
  arrived; frames beyond that arrival are excluded. The first eligible write
  is forced. The terminal pair appends short memory only, with no policy,
  RAFT or KDA write. Two-frame videos have no learned action.
* Uncertainty is a registered, frozen, permanently eval RAFT wrapper, created
  once per model. It uses no-grad FP32 flow and safe geometry against the last
  successful long-write reference. First write and disabled uncertainty skip
  RAFT. Missing enabled weights raise a local path error before loading RAFT.
  Invalid correspondence gives zero patch write confidence, never NaN or
  fabricated high confidence.
* Patch confidence gates both beta and decay. With P patches:
  `beta = sigmoid(beta_logit) * c`; `g = clamp(g_base,-5,0) * c/P`.
  `uncertainty_write_gate=false` ablates disagreement attenuation by using
  `c=1[c>0]`: fully invalid patches still cannot overwrite or decay state.
  RAFT still runs for accepted WRITE candidates in that ablation. Use
  `uncertainty.enabled=false` to ablate RAFT calls themselves.
* Heuristic, always, never and PPO policies share observe/act/reset. Actions are
  exactly TEMP_ONLY=0 and WRITE_LONG=1. The 30-scalar actor schema uses current
  predictions/features and historical memory summaries; GT, future frames and
  current-action RAFT outputs have no actor input interface. Actor/critic have
  separate 64/64 hidden layers; optional privileged critic uses 34 features.
* Environment rewards action i using the next decoded left prediction i+1.
  Rollouts retain small observations, action masks, frozen normalizer versions,
  old log probabilities, values, costs and GT metrics, not reconstruction graphs.
  GAE distinguishes terminal and rollout truncation. Forced actions contribute
  critic targets but no actor/entropy objective. Costs use calibrated RAFT,
  geometry and KDA-update times; fixed-state occupancy is not penalized.
* Supervised training has its own streaming runner, masked common-frame pointmap
  loss, first-frame detached episode scale, optional local teacher traces,
  AdamW parameter groups, AMP scaler where appropriate, and four-pair TBPTT.
  Short raw training shadows are detached and reprojected after optimizer steps;
  the entire episode graph is not retained. No broken legacy training method
  is called.
* Legacy + external policy has a separate `LegacyMemoryAdapter`. Its original
  joint attention, RAFT lifecycle and pruning stay legacy behavior. Label this
  mode `LEGACY_POLICY`; the terminal write schedule differs from official mode.
  The PPO entrypoint can use this frozen adapter with a verified official
  held-out report, without loading hybrid weights.

## Local commands exercised

All commands below were run with the existing Python environment. `-B` suppresses
bytecode files. The server commands in the next section were **not** executed.

```text
python -B -m unittest discover -s tests -v
python -B tools/check_memory_backends.py --reference
python -B tools/check_memory_backends.py
python -B train_hybrid_memory.py --dry-run --reference --validate-modules
python -B train_rl_controller.py --dry-run --reference --validate-modules
python -B train_hybrid_memory.py --dry-run --validate-modules
python -B train_rl_controller.py --dry-run --validate-modules
python -B demo_hybrid.py --dry-run
python -B eval/eval_hybrid.py
python -B tools/cache_teacher_traces.py
python -B tools/profile_memory.py
git diff --check
git diff --stat
git status --short
```

In-memory compilation was performed for all 49 changed/new Python files without
writing `.pyc` files. Production module construction succeeded without importing
FLA, FlashAttention or Triton. Production dry checks deliberately return
`requirements_ready=false` for missing checkpoints and GT. CPU conformance with
the full 1024 input interface returned `[1,4,1024]`, state `[1,4,64,64]`, and finite
new-module gradients. These results do not establish GPU compatibility or quality.

Final local suite: **84 passed, 0 failed**, in 4.790 seconds. `git diff --check`
passed. Importing `dust3r` and hybrid memory succeeded without optional kernel
imports. Importing the complete baseline `dust3r.model` is currently blocked by
the existing missing `evo` dependency; it was not installed or substituted.

## Server dependencies and local artifacts

Prepare the server separately; this document contains no automatic provisioning.
Keep the original `requirements.txt` unchanged. `requirements-hybrid.txt` records
additional requirements without making optional kernels mandatory imports.

| Requirement | Expected version/revision | Verification gate |
| --- | --- | --- |
| PyTorch CUDA | >=2.7; record exact Torch/CUDA/driver/GPU | Forced rectangular Flash BF16 forward and backward |
| Preferred short kernel | Built-in SDPA Flash | Actual backend label `sdpa_flash`, no math fallback |
| Optional short kernel | Official flash-attn >=2.2; record exact tested version | Head dim 64 and rectangular noncausal attention pass parity |
| Long kernel | FLA 0.6.0, commit `9f38d24980c46d46bd38614e743cdacd21906578` | Exact pinned API/final-state/backprop parity |
| FLA compiler | Triton >=3.3; record exact compatible version | Successful chunk and optional recurrent kernel compilation |
| CPU/data | Existing PyTorch, NumPy, Pillow; PyYAML for YAML | Reference tests and strict manifest validation |
| Baseline model dependencies | Original requirements and bundled CroCo/RAFT | `from dust3r.model import Endo3R` succeeds locally |
| DUSt3R initialization | `checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` | Local released checkpoint metadata/strict weights validated |
| Endo3R initialization | `checkpoints/endo3r.pth` | Local released model weights; SHA-256 recorded |
| RAFT | `checkpoints/raft-things.pth` | Enabled hybrid estimator loads strict weights on selected device |
| Training GT | Local JSON manifest + images/depth or pointmaps/masks/calibration/poses | Genuine source provenance and patient-disjoint splits |
| PPO environment | Certified supervised hybrid checkpoint, or official legacy report | Reconstruction gate before frozen-environment PPO |
| PPO costs | Local calibrated timing JSON | Same hardware, resolution and precision as environment |
| Deployment actor | `endo3r_ppo_write_v1` actor checkpoint + normalizer | Observation schema and environment checkpoint identity match |

The legacy adapter retains its fixed `checkpoints/raft-things.pth` location even
when uncertainty is disabled, because the original constructor always loads it.
The configured hybrid estimator can use another explicitly supplied local path.
`--trusted-checkpoints` is an explicit choice for the released pickle metadata;
new checkpoint/teacher/actor loaders use `weights_only=True`. No loader downloads
missing artifacts. RAFT is always frozen. Crop size must be divisible by the
actual checkpoint patch size; the runner validates this rather than guessing.

## SERVER_VALIDATION_REQUIRED commands and success conditions

Use these only after the environment and all named local artifacts are prepared.
Run the CPU suite first. Keep actual backend labels and package/commit/GPU metadata
with the results. The short and KDA adapters must match their reference equations
within measured BF16/FP16 tolerances; do not treat speed or final-state gradients
as established by a successful package import.

```text
python -B tools/check_memory_backends.py --server-kernels --device cuda
python -B train_hybrid_memory.py --dry-run --validate-modules --validate-data --manifest /data/verified_sequences.json
python -B train_rl_controller.py --dry-run --validate-modules --manifest /data/verified_sequences.json --hybrid-checkpoint /checkpoints/hybrid_certified.pt
```

Expected conformance probe: output `[1,4,1024]`, FP32 state `[1,4,64,64]`, finite
final-state-derived gradients for all new parameters, actual short kernel
`sdpa_flash` or `flash_attn`, actual long kernel `fla_kda_chunk`. Also compare
reference and server state/output/gradient tensors at P=4, P=320 and odd P=317;
include mixed zero confidence, all-zero confidence identity, multiple frames and
carried initial-state gradients. Check optional fused-recurrent no-grad output
against chunk inference. A kernel failure blocks efficient training; select a
reference backend explicitly only for diagnosis, never claim it is efficient.

After a supervised checkpoint exists, run bounded local inference and held-out
evaluation. A configured legacy official comparison must set backend `legacy`,
identity `LEGACY_OFFICIAL`, policy `heuristic`; it executes the untouched path.
For legacy policy ablations use identity `LEGACY_POLICY` and an external policy.

```text
python -B demo_hybrid.py --run-inference --device cuda --input /data/rgb --max-frames 8 --hybrid-checkpoint /checkpoints/hybrid_t1.pt --trusted-checkpoints
python -B eval/eval_hybrid.py --run-evaluation --config /configs/legacy_official.yaml --manifest /data/verified_sequences.json --split validation --device cuda --trusted-checkpoints --output /results/legacy.json
python -B eval/eval_hybrid.py --run-evaluation --manifest /data/verified_sequences.json --split validation --device cuda --hybrid-checkpoint /checkpoints/hybrid_t1.pt --trusted-checkpoints --baseline-report /results/legacy.json --certified-checkpoint-output /checkpoints/hybrid_certified.pt --output /results/hybrid.json
python -B tools/profile_memory.py --server-run --device cuda --input /data/rgb --num-frames 8 --hybrid-checkpoint /checkpoints/hybrid_certified.pt --trusted-checkpoints --output /results/profile_short.json
python -B tools/profile_memory.py --server-run --device cuda --input /data/rgb --num-frames 64 --hybrid-checkpoint /checkpoints/hybrid_certified.pt --trusted-checkpoints --output /results/profile_long.json
```

Certification creates a separate checkpoint; it never overwrites the source.
It checks held-out split, identical manifest/crop/scale protocol and sequence
coverage, actual efficient backends, checkpoint hashes, and per-sequence pointmap
and depth absolute-relative error within the configured 5% relative tolerance.
Failure leaves PPO blocked. The source checkpoint, report and baseline hashes
are recorded. Finite results alone do not certify teacher quality or PPO policy
deployment. With `--pose`, record OpenCV PnP success coverage and errors, including
failed estimates; this optional metric does not enter the v1 reconstruction loss.

Profiling synchronizes CUDA around scopes, reports exclusive component timings,
sequence wall time/FPS, pair p50/p95, peak allocated/reserved memory and counters.
All predictions remain on device for the compatibility forward; distinguish
this output retention from bounded memory-state storage. Compare write/RAFT rates,
short/long read, KDA writes, decoder/value encoder and RAFT; do not assume memory
attention dominates. Legacy RAFT timing remains included in its original
uncertainty scope. Calibrate PPO costs from measured component times, not a GPU
timer inside every rollout.

Cost file schema (replace every value with measurements, no guessed values):
`baseline`, `raft`, `kda`, `geometry` are times in the same units; `hardware_id`
identifies the measured server and `image_hw` is `[256,320]`. Baseline must be
positive; component costs finite and nonnegative. Configure
`rl.cost_calibration_path` to the local JSON file. No long occupancy term exists.

## GT manifest and training workflow

The released training recipe is incomplete; the new selected objective is
explicitly the minimum GT-supported masked common-frame pointmap L1, rather than
a claimed reproduction of an unreleased loss. Depth/pose/calibration produce
common-frame GT. Confidence calibration is optional and disabled by default;
there is no fabricated pose, temporal, uncertainty or memory-selection target.

The local manifest schema is defined in `dust3r/training/sequence_dataset.py`.
Each sequence provides unique `id`, `patient_id`, `split`, and provenance with
`verified=true`, documented `source`, `depth_or_pointmap_gt=true`,
`calibration_gt=true`, `pose_gt=true`, `units=m`. Frames are ordered and provide
local `rgb`, `valid_mask`, finite 3x3 `intrinsics`, proper 4x4 `camera_to_world`,
and exactly one of `depth` plus metric `depth_scale`, or `pointmap` plus explicit
`pointmap_space` (`camera`, `world`, `first_camera`). Paths are relative to the
manifest unless absolute. Real provenance must be established by the dataset
owner; setting Boolean metadata does not create ground truth. Patient leakage,
missing files, empty GT coverage and placeholder transforms are rejected.

Joint center crop/resize uses RGB bilinear, GT/mask nearest and adjusted pixel
center intrinsics. RGB reaches the model as `[1,3,H,W]` in [-1,1] and the flow
estimator as matching uint8 RGB. For a 16-pixel checkpoint, 256x320 gives P=320;
short projected K/V has at most `[1,4,320,8,64]` each. Geometry losses use true GT
masks, never predicted masks. Both decoded views are supervised with a fixed
detached first-frame episode scale; nonfinite predicted points count as failures.

Future training stages are implemented, but none were run:

| Stage | Trainable modules | Initialization / objective | Gradient/checkpoint behavior |
| --- | --- | --- | --- |
| T0 | None | Released local DUSt3R + Endo3R baseline | Unchanged inference; optional finite teacher traces |
| T1 | ST/LT projections, KDA gates, fusion | Released reconstruction weights frozen; new Xavier QKV/small output and gate initialization; GT pointmap L1; optional teacher L1+cos warm start | AdamW LR 1e-4, 9-frame clips, TBPTT 4 pairs, max grad norm 1; new hybrid checkpoint |
| T2 | T1 + key/value/shared norms; optional final two decoder blocks and heads | Load T1 checkpoint, same GT objective | Existing feature modules LR 1e-5, decoder/heads 5e-6; encoder remains frozen |
| R0 | Actor and critic only | Certified hybrid or separately verified legacy environment frozen; delayed GT/cost reward | PPO 128 transitions, 64 minibatch, 4 epochs; separate actor/critic Adam; actor + normalizer export |
| R1 supervised | Fusion and long-memory parameters only | Load stable hybrid, supervised GT adaptation | Explicit `r1_supervised` trainer stage; subsequent actor retraining needed if environment weights change |

The uncertainty mapping is analytic and has no trainable parameters. RAFT stays
frozen/eval in every stage. The initial key heads, value encoder and shared norms
are reused from released weights and frozen in T1, then eligible in T2. T2/R1
requires `--resume-checkpoint`. That CLI option loads prior weights; full optimizer,
scheduler, scaler and RNG restoration is supported by `load_training_checkpoint`
for an explicit exact-resume caller. Optional teacher traces are local, detached,
prefix aligned and hashed; `tools/cache_teacher_traces.py` is dry unless
`--run-cache` is explicitly supplied. Teacher and student use their own causal
histories; distillation does not force equal memory decisions. The student has
no permanent dependency on the legacy bank.

PPO initially freezes the whole reconstruction environment. GAE gamma=.99 and
lambda=.95, policy clip=.2, entropy=.01, value weight=.5, separate Adam LR=3e-4,
max grad norm=.5 are configurable defaults. Observation normalization changes
between rollouts, never within one rollout/PPO update. The optional privileged
critic receives only a separate training-only GT feature vector. DenseGTReward
is the supported v1 trainer mode; sparse/teacher reward interfaces reject missing
protocols instead of inventing dense supervision. No end-to-end reconstruction
PPO or token action space is included.

## File inventory

Modified existing file: `dust3r/model.py` only.

Added implementation files:

```text
configs/hybrid_memory.yaml
requirements-hybrid.txt
demo_hybrid.py
train_hybrid_memory.py
train_rl_controller.py
dust3r/memory/__init__.py
dust3r/memory/config.py
dust3r/memory/types.py
dust3r/memory/hybrid.py
dust3r/memory/short_flash.py
dust3r/memory/kda_adapter.py
dust3r/memory/long_kda.py
dust3r/memory/fusion.py
dust3r/memory/uncertainty.py
dust3r/memory/observations.py
dust3r/memory/policies.py
dust3r/memory/legacy_adapter.py
dust3r/online/__init__.py
dust3r/online/runner.py
dust3r/rl/__init__.py
dust3r/rl/networks.py
dust3r/rl/normalization.py
dust3r/rl/rewards.py
dust3r/rl/rollout.py
dust3r/rl/ppo.py
dust3r/rl/environment.py
dust3r/training/__init__.py
dust3r/training/sequence_dataset.py
dust3r/training/losses.py
dust3r/training/teacher.py
dust3r/training/hybrid_trainer.py
dust3r/training/validation.py
eval/eval_hybrid.py
tools/check_memory_backends.py
tools/cache_teacher_traces.py
tools/profile_memory.py
tests/__init__.py
tests/hybrid_helpers.py
tests/test_fusion.py
tests/test_hybrid_gradients.py
tests/test_legacy_dispatch.py
tests/test_long_kda.py
tests/test_memory_policy.py
tests/test_online_causality.py
tests/test_rewards_ppo.py
tests/test_short_flash.py
tests/test_tools_and_config.py
tests/test_training.py
tests/test_uncertainty.py
tests/test_validation.py
docs/ENDO3R_SERVER_VALIDATION.md
```

The two earlier untracked audit/plan documents are preserved separately; they
are not implementation additions. Git changes remain unstaged/uncommitted.
