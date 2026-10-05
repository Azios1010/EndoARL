# 1. Design Goals and Constraints

Planning date: 2026-10-02. Source commit: `8145c8e5535d2111974561e7d1cc06f1d008fe5e`. Primary source: [completed implementation audit](END3R_ATTENTION_RL_AUDIT.md). This document specifies proposed implementation; it does not report an implemented architecture or measured reconstruction improvement. Planning added this document only. No existing source, dependency, configuration, checkpoint, or dataset was changed; no training was run.

**Decision labels:** **LOCKED** is a requirement or an interface invariant for implementation. **PROVISIONAL** is a concrete v1 choice that the coding agent can implement, but experiments may revise. **NEEDS_RUNTIME_VALIDATION** is a claim or gate that cannot be established by source inspection. Numerical defaults below remain PROVISIONAL even where a configuration gives a single value. Code facts are labeled VERIFIED FROM CODE; local kernel probes are labeled VERIFIED LOCAL PROBE.

| ID | LOCKED direction / invariant |
|---|---|
| L01 | Short memory uses exact softmax attention through a verified FlashAttention-capable kernel; dense math is a correctness fallback only. |
| L02 | Long memory uses bounded recurrent Kimi Delta Attention state, not a growing long KV bank or the full Kimi Linear model. |
| L03 | Short and long outputs have a learned lightweight fusion and preserve the decoder input interface `[B,P,1024]`. |
| L04 | RL controls the long-memory write decision only. |
| L05 | The two learned actions are exactly `0=TEMP_ONLY`, `1=WRITE_LONG`. |
| L06 | Every decoded candidate enters short memory, independent of the action. |
| L07 | Only WRITE_LONG can update KDA; TEMP_ONLY leaves long state and long-reference metadata unchanged. |
| L08 | Actor inputs exclude GT, frames after the decision arrival time, current-action RAFT results, and dense attention maps. |
| L09 | RAFT executes only on WRITE_LONG candidates needing a reference comparison; the first valid long write needs no RAFT. |
| L10 | Uncertainty controls long WRITE strength, not a nonexistent KDA attention matrix. |
| L11 | Hybrid attention/reconstruction is trained and validated before PPO. |
| L12 | Initial PPO trains actor and critic only, with all reconstruction and memory parameters frozen. |
| L13 | LEGACY_OFFICIAL stays separately runnable and unchanged in behavior; corrections cannot be silently attributed to it. |

**LOCKED non-goals:** replacing the DUSt3R image encoder or decoder; RL controlling decoder layers, arbitrary token attention, or read modes; architecture search; delete/replace/merge actions; PPO from random reconstruction initialization; actor GT/future leakage; or a second giant KV bank beside the production KDA state. An explicit long-KV bank is allowed only in a separately selected teacher/reference experiment, never as an unnoticed part of the KDA backend.

**PROVISIONAL v1 operating envelope:** one patient/video per state, `B=1`, constant cropped `H,W` per episode, patch size verified from the checkpoint, 320x256 images initially (`P=320` for patch 16). Also test 224x224 (`P=196`) and 512x384 (`P=768`) when memory permits. General-shaped interfaces use B for clarity; heterogeneous batched writes are not implemented by broadcasting one action over a batch. Future vectorization must own independent states per episode.

**NEEDS_RUNTIME_VALIDATION:** the local 4 GiB RTX 2050 is not an established training platform for this model. A compatible Linux/WSL CUDA environment and a training GPU with approximately 24 GiB are provisional requirements, not a guarantee of fit. No Endo3R speedup is claimed from Kimi language-model results.

# 2. Legacy Code Facts Used

The audit remains the detailed evidence ledger. Line numbers refer to the audited commit, before proposed edits.

| VERIFIED FROM CODE fact | File / symbol / approximate lines | Design consequence |
|---|---|---|
| The online sequence loop encodes adjacent pairs and reuses the previous right image features | `dust3r/model.py::Endo3R.encode_frames` 630–638; `forward` 735–823 | Preserve feature caching and pair traversal. |
| At later pairs the memory query is the previous pair's `feat_k2`, not freshly encoded left features | `Endo3R.forward` 754–760 | Keep the actual query source. |
| Both streams exchange information in decoder cross-attention | `AsymmetricCroCo3DStereo._decoder` 200–220; `croco/models/blocks.py::DecoderBlock` 171–191 | The left candidate incorporates the right frame; use arrival-time causality, not a claim of zero-lookahead left-frame inference. |
| Memory read and residual output use 1024 channels and dense `[B,P,M]` scores | `SpatialTempralMemory.memory_read` 456–509 | Hybrid must return exactly `[B,P,1024]`. |
| Key construction concatenates encoder 1024 and final decoder 768 channels, then projects to 1024 | `Endo3R.set_attn_head/encode_feat_key` 591–602, 640–644 | Reuse pretrained key MLPs initially. |
| Default value encoder embeds the current predicted 3D pointmap, uses six 1024-channel transformer blocks, and forms stored value `cur_v + feat_k1` | `set_memory_encoder`, `encode_cur_value`, `forward` 569–589, 653–661, 782–801 | Preserve geometry-valued memory and its residual. |
| Temporal insertion occurs before the spatial similarity decision, with four-frame capacity | `SpatialTempralMemory.add_mem_check` 402–428 | New short insertion is unconditional, but actor observes pre-insertion statistics. |
| `check_sim` compares aligned patch keys against recent presumed spatial frames and skips writes above 0.92 | `check_sim/add_mem_check` 378–428 | A descriptor heuristic is an explicit approximation, not exact baseline equivalence. |
| Uncertainty runs after spatial acceptance, with 20 RAFT iterations; first accepted candidate has neutral uncertainty | `uncertainty_map/add_mem_check` 417–453 | Preserve conditional expense and first-write initialization, replace lifecycle and write semantics. |
| Combined retrieval has one softmax over temporal plus spatial banks when uncertainty metadata exists | `memory_read` 464–503 | Splitting branches and learning fusion changes the attention function and requires training. |
| `mem_attn` accumulation is commented out; pruning reorders individual tokens; threshold renormalization can divide by zero | `memory_read/memory_prune` 494–528 | These mechanisms are not imported into hybrid. |
| Training requests undefined `sp_mem.add_mem` | `Endo3R.forward` 735–739, 784–785 | New explicit trainer/runner must bypass this branch. |
| Original Endo3R geometric training recipe is not released; CroCo RGB pretraining and scene-parameter optimization are not it | `README.md` 24–26; `croco/pretrain.py`; `dust3r/cloud_opt/base_opt.py::BasePCOptimizer.forward` 372–399 | Do not claim to reproduce the original loss. |
| Demo datasets can fabricate depth=1, identity pose, and unit intrinsics | `dust3r/datasets/demo.py::Demo._get_views` 39–82 | GT availability requires provenance, not the presence of `pts3d` fields. |
| Baseline checkpoint metadata determines actual encoder/decoder depth, heads, patch/head modes | `dust3r/model.py::load_model` 53–69; `Endo3R.__init__` 550–566 | Validate metadata before assumptions about weights or dimensions. |

**VERIFIED LOCAL PROBE:** GPU is RTX 2050, SM 8.6, 4,294,443,008 bytes VRAM; installed Windows Python 3.14.3, Torch 2.9.0+cu128. A forced SDPA FLASH_ATTENTION probe with Q `[1,8,196,64]` and K/V `[1,8,784,64]` failed for FP16/BF16/FP32 with “Torch was not compiled with flash attention” / “No available kernel”; math output was finite. `is_flash_attention_available()` was false. This is a small backend probe, not Endo3R inference or profiling. There are no required checkpoints or verified depth/pose GT sequences in the clone. Local example RGB videos do not establish GT. Missing import/dependency details are in audit section 7.

# 3. Proposed End-to-End Architecture

**LOCKED interfaces and causal clock:** for pair index `i`, the latest acquired frame is `arrival_index=i+1`, and the candidate written is `candidate_index=i`. The decision is made after both images through arrival i+1 have been processed. It may use left features that depend on the acquired right frame. No actor/state dependency on frame i+2 or later is allowed. Calling this an action at left frame i without this one-frame delay would be misleading. Prefix-invariance tests enforce this convention.

```text
acquire I[i+1]; pair (I[i], I[i+1])
  -> existing encode_frames, with previous right feature reuse
  -> if i>0: query = previous feat_k2; else: fused_feat1 = feat1
  -> read projected <=4-frame ST cache with exact Flash softmax
  -> read the PRE-UPDATE KDA state by a query/state contraction
  -> availability-aware learned fusion + query residual [B,P,1024]
  -> existing decode and prediction heads -> dec1/dec2, res1/res2
  -> existing key/value encoders -> feat_k1, feat_k2, cur_v
  -> observation from current reconstruction and PRE-UPDATE state
  -> binary policy action (first long write forced when a decision exists)
  -> unconditional short insertion of (feat_k1, cur_v+feat_k1)
  -> WRITE only: optional RAFT -> sanitized confidence -> frame-chunk KDA update
  -> cache accepted long reference and counters -> next arrival
```

**PROVISIONAL terminal rule:** process all `N-1` pairs and preserve all N prediction outputs. The final pair produces the delayed reward for the preceding action and appends its left candidate to ST, but makes no new learned decision and performs no RAFT/KDA write: there is no subsequent memory read to benefit. The final right frame is an output, not a memory candidate. Thus an episode has `max(N-2,0)` decision transitions. For N=2, there is no action or forced initialization; for N>=3 the first decision forces WRITE_LONG. NeverWritePolicy means “initial seed only,” not an empty long branch forever. Report write rates using this eligible-decision denominator. Terminal omission is not a third learned action.

**LOCKED TEMP_ONLY invariant:** no long decay, no KDA update call, no RAFT, no long-reference change. A non-mutating long read still happens on every subsequent pair when long state exists. RL therefore reduces write/RAFT costs, not fixed KDA read cost.

**PROVISIONAL safety exception:** a requested WRITE with no valid patch correspondence is recorded as an unsuccessful write attempt; it pays actual incurred compute cost but does not corrupt state/reference. This is deterministic write validation, not another policy action. Initial candidate with no finite valid pointmap cannot seed long memory and fails the episode explicitly. Counters distinguish requested and successful writes.

# 4. New Module/File Structure

**LOCKED separation:** do not turn `SpatialTempralMemory` into a hybrid class. Future implementation changes the existing `dust3r/model.py` only to register optional modules/config and dispatch to a new runner. Its default remains LEGACY_OFFICIAL, and the existing forward body and class remain intact. No external KDA import occurs in legacy mode.

**PROVISIONAL file allocation:** the exact manifest below fixes where the coding agent should put v1 responsibilities; it does not authorize creating these files during planning. Directory-relative names in the responsibility table resolve under their stated package.

| Proposed file | Responsibility |
|---|---|
| `dust3r/memory/__init__.py` | Public internal interfaces; no eager optional-kernel imports. |
| `dust3r/memory/types.py` | HybridMemoryState, ReadDiagnostics, WriteDiagnostics, ObservationPacket, PolicyDecision, candidate/reference structures. |
| `dust3r/memory/config.py` | Validated typed config; reject incompatible dimensions, unknown backends, and misleading fallback claims. |
| `dust3r/memory/hybrid.py` | HybridMemory read, short append, validated long write; no PPO optimizer logic. |
| `dust3r/memory/short_flash.py` | Projected short ring, exact SDPA/Flash/math backends. |
| `dust3r/memory/long_kda.py` | Visual projections, gates, pure read, frame update; internal state contract. |
| `dust3r/memory/kda_adapter.py` | Pinned FLA API, state conversion, reference recurrence and backend tests. |
| `dust3r/memory/fusion.py` | Availability-aware channel gate and output diagnostics. |
| `dust3r/memory/uncertainty.py` | Single registered, frozen RAFT; correspondence validation and confidence mapping. |
| `dust3r/memory/observations.py` | Strict 30-scalar actor features; no GT access. |
| `dust3r/memory/policies.py` | Shared policy interface, heuristic/always/never/PPO wrappers. |
| `dust3r/memory/legacy_adapter.py` | Optional LEGACY_POLICY action boundary; preserves official class implementation. |
| `dust3r/online/__init__.py`, `dust3r/online/runner.py` | New stepwise pair runner, episode reset, terminal handling, feature caching and output compatibility. |
| `dust3r/training/__init__.py`, `sequence_dataset.py`, `losses.py`, `teacher.py`, `hybrid_trainer.py` | Provenance-checked sequence loader, new reconstruction objective, optional frozen teacher traces and gradient training. |
| `dust3r/rl/__init__.py`, `environment.py`, `networks.py`, `normalization.py`, `rewards.py`, `rollout.py`, `ppo.py` | Sequential environment, separate actor/critic, observation normalization, causal rewards, storage and PPO. |
| `train_hybrid_memory.py`, `train_rl_controller.py`, `demo_hybrid.py` | New explicit entrypoints; never invoke the undefined legacy training branch. |
| `tools/check_memory_backends.py`, `cache_teacher_traces.py`, `profile_memory.py` | Bounded readiness probes, optional teacher extraction and exclusive timing. |
| `eval/eval_hybrid.py` | New validated evaluator; do not silently repair the original wrapper. |
| `configs/hybrid_memory.yaml`, `requirements-hybrid.txt` | Separate hybrid config/environment specification; retain old defaults and requirements. |
| `tests/test_short_flash.py`, `test_long_kda.py`, `test_fusion.py`, `test_uncertainty.py`, `test_memory_policy.py`, `test_online_causality.py`, `test_rewards_ppo.py`, `test_hybrid_gradients.py`, `test_legacy_dispatch.py` | Tests described in section 17. |

**PROVISIONAL optional experiment files, not required production modules:** `dust3r/memory/legacy_clean.py` plus `docs/LEGACY_CLEAN_FIXES.md` for an explicitly corrected baseline; `dust3r/memory/long_dense_reference.py` for a separately selected explicit long-KV reference. Neither imports or runs in the KDA production path. All directories/files in this table are proposed, not created during planning.

Exact future v1 implementation manifest (one existing file modified; 44 proposed new files):

```text
MODIFY dust3r/model.py
ADD dust3r/memory/__init__.py
ADD dust3r/memory/types.py
ADD dust3r/memory/config.py
ADD dust3r/memory/hybrid.py
ADD dust3r/memory/short_flash.py
ADD dust3r/memory/long_kda.py
ADD dust3r/memory/kda_adapter.py
ADD dust3r/memory/fusion.py
ADD dust3r/memory/uncertainty.py
ADD dust3r/memory/observations.py
ADD dust3r/memory/policies.py
ADD dust3r/memory/legacy_adapter.py
ADD dust3r/online/__init__.py
ADD dust3r/online/runner.py
ADD dust3r/training/__init__.py
ADD dust3r/training/sequence_dataset.py
ADD dust3r/training/losses.py
ADD dust3r/training/teacher.py
ADD dust3r/training/hybrid_trainer.py
ADD dust3r/rl/__init__.py
ADD dust3r/rl/environment.py
ADD dust3r/rl/networks.py
ADD dust3r/rl/normalization.py
ADD dust3r/rl/rewards.py
ADD dust3r/rl/rollout.py
ADD dust3r/rl/ppo.py
ADD train_hybrid_memory.py
ADD train_rl_controller.py
ADD demo_hybrid.py
ADD tools/check_memory_backends.py
ADD tools/cache_teacher_traces.py
ADD tools/profile_memory.py
ADD eval/eval_hybrid.py
ADD configs/hybrid_memory.yaml
ADD requirements-hybrid.txt
ADD tests/test_short_flash.py
ADD tests/test_long_kda.py
ADD tests/test_fusion.py
ADD tests/test_uncertainty.py
ADD tests/test_memory_policy.py
ADD tests/test_online_causality.py
ADD tests/test_rewards_ppo.py
ADD tests/test_hybrid_gradients.py
ADD tests/test_legacy_dispatch.py
```

**LOCKED public contracts** (internal APIs designed here, not claims about external KDA):

```python
HybridMemory.reset(batch_size, image_hw, patch_hw, device, training) -> HybridMemoryState
HybridMemory.read(query, state) -> (fused_feature, ReadDiagnostics)
HybridMemory.append_short(k, v, descriptor, state) -> HybridMemoryState
HybridMemory.write_long(candidate, write_confidence, state) -> (HybridMemoryState, WriteDiagnostics)
ShortTermFlashMemory.read(query, state) -> output_1024
LongTermKDAMemory.read(query, state) -> output_1024
LongTermKDAMemory.update(k, v, update_weight, state) -> (new_state, WriteDiagnostics)
LongTermKDAMemory.reset(batch_size, device) -> fp32_state
LongWritePolicy.observe(candidate, pre_update_state, read_diagnostics) -> ObservationPacket
LongWritePolicy.act(packet, action_mask, deterministic) -> PolicyDecision
UncertaintyEstimator.estimate(candidate, accepted_reference) -> WriteConfidence
OnlineRunner.reset(sequence_metadata)
OnlineRunner.step(next_frame) -> PairResult  # also exposes decision/diagnostic hooks
RewardComputer.evaluate_next(previous_transition, next_pair_result, gt_context) -> RewardResult
```

State is episode-owned and returned explicitly. Trainable parameters belong to registered modules, not state. Reads must not mutate tensors, counters, or references. `PolicyDecision` contains action, optional log_prob/value, forced flag and valid mask; the memory module receives the action enum, not PPO internals.

# 5. HybridMemoryState Definition

**LOCKED ownership:** state lives on the runner's device, is separate from `state_dict()`, and is never shared between videos/patients. Reset allocates empty ST buffers, zero FP32 KDA state, false presence/quality bits, empty references/descriptors and zero counters. Device changes require an explicit state migration or reset, not implicit `.cuda()`. Resolution/patch-layout change requires reset or a hard error. V1 asserts B=1.

Notation: T<=4 retained ST frames; L<=4 recent accepted-long descriptors; Hs=8, ds=64; Hl=4, kl=vl=64; D=1024. BF16 below is the provisional efficient-kernel activation dtype after validation. Module output to the legacy decoder is cast to its expected dtype (initially FP32).

| Field | Shape / dtype | Ownership, reset, autograd |
|---|---|---|
| `short_k`, `short_v` | `[B,T,P,8,64]`, BF16 | Projected K/V only in inference. Empty at reset. Functional append during training, graph retained within TBPTT window; inference ring can use preallocated slots. |
| `short_raw_k`, `short_raw_v` | Training-only `[B,T,P,1024]`, source activation dtype | Raw shadows for reprojecting the four retained frames after optimizer boundaries; not stored in deployment. Within-window graph; detach at boundary. |
| `short_frame_indices`, `short_cursor`, `short_occupancy` | `[T]` int64; scalar integers | Oldest-first logical order irrespective of physical ring positions; no autograd; zero/empty reset. |
| `kda_state` | `[B,4,64,64]`, FP32 | Canonical key-first state, 16,384 floats/64 KiB per B. Zero reset; differentiable within TBPTT during attention training, detached/inference no_grad otherwise. |
| `has_long_memory` | bool | False until a valid accepted write. Never infer presence from a state norm. |
| `recent_key_descriptors` | `[B,T,1024]`, FP32 | Detached L2-normalized pooled candidate keys, ST-aligned, at most four. No autograd. |
| `recent_long_descriptors` | `[B,L,1024]`, FP32 | Detached accepted-write descriptors for temporary heuristic; at most four. No patch KV bank. |
| `last_long_descriptor` | `[B,1024]`, FP32 plus validity bit | Detached; unavailable reset. |
| `last_long_reference_index` | Candidate index int64 or -1 | The accepted delayed left candidate, not immediate prior input. No autograd. |
| `last_long_reference_arrival_index` | Arrival index int64 or -1 | Documents what frames were known at write time. |
| `last_long_reference_image` | `[B,3,H,W]`, uint8 | Detached cropped RGB in RAFT input units 0..255. Exactly one accepted reference, unavailable reset. |
| `last_long_reference_pointmap` | `[B,H,W,3]`, FP32 | Detached predicted points in first-camera coordinates, exactly one reference. No GT. |
| `last_long_reference_valid_mask` | `[B,H,W]`, bool | Finite reference points/image domain; detached. |
| `previous_uncertainty_stats` | `[B,3]`, FP32: mean U, std U, valid-flow fraction | Most recent successful reference-comparison write only; zero reset. No current-action results enter its preceding observation. |
| `has_previous_raft_quality` | bool | False for initial seed/no previous comparison. Explicitly distinguishes zeros from missing history. |
| `last_write_step` | int64 or -1 | Last successful accepted candidate decision index. |
| `write_count` | int64 | Successful KDA update count (includes forced seed). |
| `write_attempt_count` | int64 | Requested WRITE_LONG count, including validation rejection. |
| `raft_call_count`, `kda_update_count` | int64 each | Actual calls/updates, not inferred from actions. Reset zero. |
| `current_step`, `processed_candidate_count` | int64 each | Decision/candidate position and previous completed ST insertions; independent of unknown future episode length. |
| `candidate_index`, `arrival_index` | int64 | Set by runner; i and i+1. Metadata, not learned tensors. |
| `predicted_radius_scale` | `[B,1]`, FP32 with validity bit | Detached first valid predicted median radius for actor's scale summaries; never a GT-derived value. |
| `sequence_id`, `image_hw`, `patch_hw`, `patch_layout_id` | Immutable metadata | Prevent cross-patient reuse; patch order tied to encoder output. |

`RunnerState` separately owns the previous `feat2,pos2,shape2,feat_k2` and the current left image, next-pair cache, pending reward transition and output list. These caches also obey TBPTT detachment. Stream inference may emit outputs rather than retaining all maps on GPU; compatibility `forward` may collect them. Output retention is not counted as recurrent-memory size.

**LOCKED removal from hybrid:** no `mem_k`, `mem_v` growing spatial bank; no `mem_attn`, `mem_count`, `lm`, `wm`, `top_k`, or `memory_prune`. Short occupancy is in frames, KDA state dimensions are in channels, and write_count is accepted frames. No token-age score or patch-level pruning is needed. `mem_c` and per-history `mem_uncert` are replaced by current write confidence and three historical quality scalars. Configuration rejects old long-size/top-k fields for hybrid rather than ignoring them silently.

**PROVISIONAL training cache choice:** projected K/V minimizes inference storage and repeat projection work; raw shadows add bounded training memory but solve stale projected caches after weight updates. At an optimizer boundary, detach raw shadows and reproject them with updated parameters. Do not reuse a retained old autograd graph after an optimizer step. Inference stores no raw shadows. See section 15 for the precise boundary algorithm.

# 6. Short-Term FlashAttention Design

**PROVISIONAL concrete v1:** D=1024 input/output, projected dimension 512, eight heads, 64 channels/head, four frames, attention dropout 0. Bias-free Q/K/V/output linears. Reuse checkpoint `norm_q/norm_k/norm_v`, initially frozen. These remain registered ONLY on Endo3R; pass a non-Module reference container (weak references) into HybridMemory, so its state_dict does not register copies/aliases. Endo3R owns their device and freeze status. HybridMemory applies normalization once before branch projections; its public API accepts raw 1024-D features. Branch-internal helpers accept normalized features, while standalone public read/update wrappers use the same reference container. Sharing normalized Q/K/V does not change the public shapes.

For T retained frames, S=TP:

\[
Q_s=\operatorname{split}_8(W_q^s LN_q(Q)),\quad
K_s=\operatorname{split}_8(W_k^s LN_k(K_{ST})),\quad
V_s=\operatorname{split}_8(W_v^s LN_v(V_{ST})).
\]
\[
A_s=\operatorname{softmax}(Q_sK_s^\top/\sqrt{64}),\quad
O_{ST}=W_o^s\operatorname{merge}_8(A_sV_s)\in\mathbb R^{B\times P\times1024}.
\]

Q layout for SDPA: `[B,8,P,64]`; cached K/V flattened oldest-to-newest to `[B,8,TP,64]`. Official flash-attn layout is `[B,P,8,64]` and `[B,TP,8,64]`; adapter handles contiguous layout. All cached frames precede the candidate decision/read. No causal triangular mask is used: patch indices are spatial tokens, and all memory tokens are available history. Use `is_causal=False`, no attention mask in v1, explicit `dropout_p=0.0` even in module eval. T=0 bypasses kernels and returns an unavailable branch, not an empty softmax.

**LOCKED projection timing:** the equations for K_s/V_s execute only at short append or training-cache refresh from raw 1024-D K/V. Read consumes cached projected 512-D tensors and does NOT reapply norm_k/norm_v or Wk/Wv. Q is normalized/projected for each read. Reuse a candidate's normalized raw K/V for a same-step LT update where practical; do not cache normalization across a parameter update.

**LOCKED exactness / backend priority:**

1. Force and verify PyTorch SDPA FLASH_ATTENTION on the actual Q/K/V shapes, hardware and dtype. Do not count an automatic math fallback as Flash.
2. If SDPA is unsuitable, validate official `flash_attn_func` for rectangular cross-attention and backward. Do not use the inference-only KV-cache API for training.
3. Dense PyTorch FP32 softmax/math is the mandatory numerical reference and emergency correctness mode; logs/config/results must say `math_reference`, never “efficient Flash.”

PyTorch documents multiple SDPA implementations and backend controls; kernel availability depends on build and inputs. Source: [PyTorch 2.9 SDPA](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html). Official FlashAttention-2 documents Ampere/Ada/Hopper, FP16/BF16, head dimension <=256, CUDA/PyTorch/Linux requirements, with Windows build caveats. This supports trying 64-channel heads on SM86, not a verified local execution claim. Source: [official FlashAttention requirements](https://github.com/Dao-AILab/flash-attention#installation-and-features).

**NEEDS_RUNTIME_VALIDATION before enabling `require_flash=true`:** record GPU/SM, Torch/CUDA/driver versions, installed package revision, activation dtype, layout/strides, dropout, forward AND backward, rectangular P vs 4P, all three planned resolutions, kernel actually selected, and dense-reference tolerance. Current native Windows Torch fails this gate. BF16 is the provisional default on a compatible build; FP16 needs finite-gradient/scaler validation. FP32 remains the source model's initial arithmetic and correctness reference.

Projected cache storage is `2 B T P 512` activations: at B=1/T=4/P=320/BF16, 2.5 MiB; at P=768, 6 MiB. Optional training raw shadows add `2 B T P 1024` source activations plus their graphs. **STATIC ESTIMATE:** Flash avoids explicitly materializing `[B,8,P,4P]`, but exact attention remains O(B P TP 512); projections and output cost O(BP*1024*512), including new K/V on writes. No asymptotic exact-attention speedup is claimed.

# 7. Long-Term KDA Design

## Official evidence and adapter boundary

**VERIFIED EXTERNAL SOURCE:** KDA is the channel-gated delta-rule component of Kimi Linear, not the entire architecture. The official model points to FLA KDA. Sources: [Kimi Linear paper](https://arxiv.org/html/2510.26692v1#S3), [Moonshot Kimi Linear repository](https://github.com/MoonshotAI/Kimi-Linear). The implementation target is **PROVISIONAL pinned FLA revision** `9f38d24980c46d46bd38614e743cdacd21906578` (package version 0.6.0). Never install an unpinned latest revision and assume identical semantics.

| Inspected official/recommended behavior | Source / implication |
|---|---|
| `chunk_kda(q,k,v,g,beta,...)`, q/k `[B,T,H,K]`, v `[B,T,HV,V]`, g `[B,T,HV,K]`, beta `[B,T,HV]`, initial/final state key-first `[B,HV,K,V]` | [FLA chunk API](https://github.com/fla-org/flash-linear-attention/blob/9f38d24980c46d46bd38614e743cdacd21906578/fla/ops/kda/chunk.py#L197). Use H=HV to avoid GQA complexity. |
| Chunk op has backward for q/k/v/g/beta and initial-state gradients | [FLA chunk autograd](https://github.com/fla-org/flash-linear-attention/blob/9f38d24980c46d46bd38614e743cdacd21906578/fla/ops/kda/chunk.py#L118). Final-state-only loss propagation must be separately tested. |
| FP16/BF16 inputs, FP32 log gates/state and 64-channel heads are covered in tests; zero log decay is also exercised | [FLA KDA tests](https://github.com/fla-org/flash-linear-attention/blob/9f38d24980c46d46bd38614e743cdacd21906578/tests/ops/test_kda.py#L543). Testing is evidence of API targets, not proof on RTX2050. |
| `fused_recurrent_kda` carries state and produces recurrent inference outputs; it is not the training backward path | [FLA recurrent API](https://github.com/fla-org/flash-linear-attention/blob/9f38d24980c46d46bd38614e743cdacd21906578/fla/ops/kda/fused_recurrent.py#L339). Use for inference parity/benchmark only. |
| Trainable per-head decay scale and per-channel time bias exist in the official layer | [FLA KDA layer gates](https://github.com/fla-org/flash-linear-attention/blob/9f38d24980c46d46bd38614e743cdacd21906578/fla/layers/kda.py#L142). Adapt gates without importing LLM convolutions or model blocks. |

**NEEDS_RUNTIME_VALIDATION environment:** FLA CUDA extras specify Torch>=2.7 and Triton>=3.3, Python>=3.10; Triton lists Linux and NVIDIA compute capability >=8.0. The local SM86 meets that family threshold, but native Windows/Python3.14/package compatibility and kernel behavior are unverified. Sources: [pinned FLA package metadata](https://github.com/fla-org/flash-linear-attention/blob/9f38d24980c46d46bd38614e743cdacd21906578/pyproject.toml), [Triton compatibility](https://github.com/triton-lang/triton#compatibility). Use a separately validated Linux/WSL environment. The separate Moonshot FlashKDA implementation targets SM90+ and CUDA12.9+ at inspected revision `7afb9f454f160a6c4bbc0999beca0a8c40a38934`; it is not the RTX2050 backend. Source: [FlashKDA requirements](https://github.com/MoonshotAI/FlashKDA#requirements).

## Concrete state, read and update

**PROVISIONAL v1:** projected long dimension 256, four heads, key dimension 64, value dimension 64, state `[B,4,64,64]` FP32. Q/K/V/out projections are learned bias-free 1024<->256 linears. Q and K are independently per-head L2-normalized in FP32 with epsilon 1e-6, then cast for kernels. Values are projected but not L2-normalized. Input shared checkpoint LayerNorms are applied once as in section 6. No Kimi LLM weights, depthwise convolution, MoE, MLA, artificial KV history, or extra attention map is imported.

**LOCKED non-mutating read:** external KDA combines read and update; Endo3R requires a query against an existing state before the candidate K/V exists. The adapter implements the existing-state contraction explicitly:

\[
q_{t,p,h}=\operatorname{L2Norm}(W_q^{LT}LN_q(Q_t))_{p,h},\quad
o_{t,p,h}=\frac{1}{\sqrt{64}}q_{t,p,h}^{\top}S_{t-1,h},\quad
O_{LT}=W_o^{LT}\operatorname{merge}_h(o_{t,p,h}).
\]

Compute contraction in FP32, then cast output to the decoder dtype. Read never applies decay or increments state. Do not simulate a read by calling the external recurrent update with beta=0: a nonzero decay would still mutate memory. `has_long_memory=false` bypasses it.

**LOCKED frame-chunk interpretation:** one accepted VIDEO frame is one logical update containing P spatial patch tokens. Their sequence is the encoder's deterministic flatten order, verified against `pos1`/patch embedding; do not reorder by salience, uncertainty, or a top-k. Internal kernel chunks of 64 tokens are a numerical implementation detail, not 64 video frames. P need not be a multiple of 64; validate API behavior without adding synthetic updates. Write_count advances once per successful frame, not P times.

**PROVISIONAL position choice:** no new 2D/time embedding in v1. Keys already combine positional transformer outputs and cross-view geometry, values include global pointmaps. This does not prove perfect spatial representation; compare a minimal 2D-position ablation only if the recurrent scan shows spatial-order sensitivity. No time gap embedding is required to keep causality: age is a policy feature; a TEMP_ONLY does not advance KDA decay.

Let k/v be projected candidate tokens, h a head, p the patch scan index; let confidence c in [0,1] be from section 9. Define learned base gates:

\[
\beta^{base}_{p,h}=\sigma((W_\beta LN_k(K_t))_{p,h}+b_\beta),
\]
\[
g^{base}_{p,h,c}=-\exp(A_{log,h})\operatorname{softplus}((W_g LN_k(K_t))_{p,h,c}+dt_{bias,h,c}),
\]
\[
\beta^{eff}_{p,h}=c_{p}\beta^{base}_{p,h},\qquad
g^{eff}_{p,h,c}=\frac{c_p}{P}g^{base}_{p,h,c}.
\]

The **PROVISIONAL 1/P factor** is an Endo3R frame adaptation: it normalizes the nominal aggregate decay per accepted frame instead of forgetting P times faster at a larger resolution. It is not claimed to be an unmodified paper equation. The official legal inputs remain nonpositive channel log decay and beta in [0,1]. Bound learned log-scale/gates for finite arithmetic (v1 `A_log` clamped to [-5,5] before exp, base log decay clamped to [-5,0]); report saturation rates. Initial A_log=0, Wg=0 and `dt_bias=inv_softplus(0.1)` give nominal exp(-0.1) aggregate frame decay at c=1. Beta bias `logit(0.1)`, small Wbeta initializes gentle correction. These are tunable initializations.

For each patch:

\[
\bar S_{p,h}=\operatorname{diag}(\exp(g^{eff}_{p,h}))S_{p-1,h},\quad
S_{p,h}=\bar S_{p,h}+\beta^{eff}_{p,h}k_{p,h}(v_{p,h}-k_{p,h}^{\top}\bar S_{p,h})^{\top}.
\]

State is key-first. The scanned final state becomes the next long state. This orientation and the external post-update output contraction are verified in [FLA reference recurrence](https://github.com/fla-org/flash-linear-attention/blob/9f38d24980c46d46bd38614e743cdacd21906578/fla/ops/kda/naive.py#L10). Multiplying ONLY beta by confidence would leave invalid patches decaying old memory. Here c=0 makes BOTH gates zero, giving an exact mathematical no-op patch; mixed-precision kernel parity needs a tolerance test. All-zero-confidence candidates and TEMP_ONLY bypass the kernel entirely for bitwise preservation. No value scaling is used to imitate old logit weighting.

**PROVISIONAL adapter invocation:** use `chunk_kda` for differentiable frame updates and initially for frame-level inference, carrying `initial_state`, requesting `output_final_state=True`, with q=zeros of `[B,P,4,64]` because candidate update needs no output read. Use external normalized keys, explicit FP32 log gates, post-sigmoid beta cast as required. Set `use_qk_l2norm_in_kernel=False`, `use_gate_in_kernel=False`, `use_beta_sigmoid_in_kernel=False`, `allow_neg_eigval=False`, `safe_gate=False`, `state_v_first=False`, `return_intermediate_states=False`; internal chunk size 64, recomputation enabled by default. Discard the external per-patch output; only final state is semantically consumed. At this revision FLA does not disable gradient materialization, and PyTorch defaults unused output adjoints to zero tensors; no artificial output anchor is required. Sources: [pinned autograd forward](https://github.com/fla-org/flash-linear-attention/blob/9f38d24980c46d46bd38614e743cdacd21906578/fla/ops/kda/chunk.py#L24), [PyTorch gradient-materialization contract](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html). **NEEDS_RUNTIME_VALIDATION:** explicitly test final-state-only gradients to K/V, beta/decay parameters and initial state through a later pure read. Reject a revision change that breaks this path.

`fused_recurrent_kda` is optional no_grad inference parity/latency comparison, never the training path. A small FP32 PyTorch recurrence is the correctness fallback; it is not the claimed efficient KDA implementation. The rest of Endo3R never sees FLA-specific cache objects or state orientation switches.

**STATIC ESTIMATE:** state storage is O(B H k v), independent of accepted history. Long read is O(B P H k v), update is O(B P H k v) plus projections/gates; each frame's dense projection cost still depends on 1024*256. For H=4/k=v=64 this is a fixed 16,384 state scalars and approximately P*16,384 contraction work, not O(PM1024) with growing M. Kernel chunk scratch/backward graphs are additional transient memory and must be measured. Fixed state capacity may forget or mix geometry; bounded storage is not evidence of sufficient reconstruction quality.

# 8. Short/Long Fusion Design

**PROVISIONAL channel gate:** pool each query/output over patches, concatenate three 1024-vectors, use `Linear(3072,64)->GELU->Linear(64,1024)->sigmoid`. Gate is `[B,1,1024]`, shared across patches but channel-specific:

\[
g=\sigma(MLP([\operatorname{mean}_P Q,\operatorname{mean}_P O_{ST},\operatorname{mean}_P O_{LT}])),\quad
F=Q+g\odot O_{ST}+(1-g)\odot O_{LT}.
\]

Input pooling may accumulate FP32; final output is the decoder's dtype. Gate output bias `logit(0.75)` initially favors short memory; small nonzero output weights avoid a fully dead upstream gradient. Approximately 262k weights are evaluated once per frame, not per patch. No attention matrix is exposed for RL.

| Availability | LOCKED behavior |
|---|---|
| Neither branch | Return Q without evaluating softmax or gate; initial first pair uses the original raw feat1 shortcut. |
| Short only | Force g=1 and return Q+O_ST; do not learn a gate against a fabricated long output. |
| Long only | Force g=0 and return Q+O_LT. |
| Both | Learned channel gate and query residual as above. |

**LOCKED cheap detached diagnostics:** availability bits, gate mean/std, RMS short/long outputs, and ratio `||g*O_ST||/(||g*O_ST||+||(1-g)*O_LT||+eps)`. These come from already computed tensors and small reductions. No entropy, full attention weights, or artificial KDA attention matrix. Norm denominators have epsilon; unavailable outputs/diagnostics use zeros and explicit presence bits. Nonfinite fused output fails the step; it is not silently converted into a valid reconstruction.

# 9. Uncertainty Redesign

**LOCKED role:** per-patch WRITE confidence affects the legal KDA beta and decay gates in section 7. It does not weight dense long attention, alter short Flash softmax, or enter the actor before its action.

**LOCKED module lifecycle:** register one UncertaintyEstimator under Endo3R for the hybrid mode. Construct/load RAFT once at model setup (or once lazily on the first enabled WRITE), using the configured checkpoint and device. RAFT parameters `requires_grad=False`, eval mode remains enforced when the parent is trained, and calls are under `torch.no_grad()`. No unconditional `.cuda()` or DataParallel assumption. Explicitly validate checkpoint key prefixes when loading. Initial v1 runs RAFT in FP32 with its mixed_precision option disabled and outer autocast disabled, at cropped H,W, 20 iterations, `test_mode=True`. Future lower iteration counts require accuracy experiments, not a silent default change. Source retained for network: `dust3r/core/raft.py::RAFT.forward` 86–144.

**LOCKED reference:** last successful accepted WRITE_LONG candidate. Cache only its RGB crop, predicted common-coordinate pointmap and validity mask. On TEMP_ONLY, preserve them. A first valid long write uses neutral confidence on valid patches and no RAFT. “Missing reference” after state initialization is an invariant failure, not permission to treat the candidate as high confidence.

**PROVISIONAL sanitized geometry algorithm:**

1. Verify candidate/reference image and pointmap dimensions, coordinate convention and finite values. RGB input is cropped uint8 converted to FP32 0..255 for RAFT, not encoder-normalized [-1,1].
2. Compute current-to-reference flow once. Add flow to the current pixel grid. Mark nonfinite flow and endpoints outside `[0,W-1] x [0,H-1]` invalid.
3. Bilinearly sample reference pointmap and a finite-reference mask at endpoints. Require all contributing reference pixels valid (sampled mask >=1-1e-6). Current point must also be finite. No correspondence is rescued with border replication.
4. Measure common-frame disagreement `d(x)=||X_cur(x)-X_ref(x+flow(x))||_2`. Let `s_ref=max(median(||X_ref|| over valid reference pixels),1e-6)`. Set valid-pixel `u=clamp(d/(0.1*s_ref+eps),0,1)`; invalid pixels have u=1. This predicted-scale normalization replaces unstable old inf/min-max operations; it is a new calibrated uncertainty proxy, not Sampson distance or GT uncertainty.
5. Pool to the exact patch grid. For patch p, compute `f_p=valid_pixel_fraction` and `u_p=mean(u over valid pixels)`; if none valid, u_p=1. Define `c_p=f_p*clamp(exp(-5*u_p),0.05,1)` in FP32, then pass `[B,P,1]` to KDA gates. The 0.05 floor never turns invalid data confident because it is multiplied by f_p. Validate [0,1], finite values and deterministic pooling order.
6. Record mean/std uncertainty over valid correspondences and valid-flow fraction only after the write completes. If none valid, record failure diagnostics separately; do not replace the last successful historical quality/reference.

The relative scale denominator and alpha=5 are **PROVISIONAL**, to calibrate on actual motion/depth distributions. They are not learned in v1. Reconstruction gradients do not pass through RAFT, flow, sanitization, or confidence; gradients do pass through the KDA gates/projections with c treated as a detached input.

| Edge case | LOCKED safe behavior |
|---|---|
| First write / no prior long state | No RAFT; patch finite-point fraction is confidence (neutral 1 for fully valid patches). Invalid candidate cannot seed; fail episode if all invalid. |
| Zero valid flow correspondences | c=0 everywhere; no KDA call/commit, no reference change; count attempted WRITE and actual RAFT call. |
| Partial out-of-bounds flow / NaN pointmap pixels | Invalidate those correspondences; zero their contribution; retain valid regions. Never use `inf-inf`, NaN min/max, or unconditional nan_to_num as confidence. |
| Entire nonfinite reconstruction | Runner raises an explicit failure; evaluation counts failure, reward uses configured failure penalty/termination. Do not drop failed pixels from GT metrics to improve scores. |
| Degenerate/zero motion | Valid zero flow is allowed; confidence comes from 3D agreement, not motion magnitude. |
| Reference unavailable with has_long_memory=true | Error/reset the episode with diagnostic; no hidden high-confidence write. |
| Successful partial write | Commit final state and reference plus reference validity mask once; increment successful counts once. |

**LOCKED finite-feature guard:** correspondence masks alone do not sanitize NaN keys/values (`0*NaN` is NaN). In v1 the runner checks predicted maps BEFORE `encode_cur_value`, and checks feat_k1/feat_k2/cur_v/current_v BEFORE observation, ST append or any KDA projection. Any nonfinite current predicted map or memory feature fails the step/episode explicitly; no NaN is allowed to enter the value encoder's global self-attention. Evaluation records the failure; training rejects the step and records coverage. The estimator still handles partial invalid cached/test correspondences safely, but that robustness is not permission to feed a partially NaN pointmap through the value encoder. A future sanitized-value-input mode would require a separate explicit implementation and evaluation mask policy; it is not v1.

The old uncertainty detector's mask/normalization is replaced only in NEW_HYBRID. Original `SpatialTempralMemory.uncertainty_map` and RAFT mode remain LEGACY_OFFICIAL behavior. No legacy uncertainty flag is used to switch off ST retrieval in hybrid.

# 10. RL MDP Definition

## Episode, transition and reset

**LOCKED:** episode is one ordered video from one patient; reset clears all episodic buffers, KDA state, references, feature caches, counters and pending reward. Actor is feed-forward in v1. A global learned observation-normalizer checkpoint is not patient state and is not reset each episode. Underlying environment state includes memory, cached images/features and video progression; the compact actor observation is a partial observation, so the actor technically operates in a POMDP rather than observing the full Markov state.

Observation s_i is constructed after pair(i,i+1) prediction, before this candidate's ST insertion or action-dependent work. Action a_i updates memory for pair(i+1,i+2). The next decoded LEFT prediction supplies the quality component for a_i. It is incorrect to reward a_i using current res1 or current res2, both produced before a_i. No GT enters observe/act; RewardComputer is a separate training-only consumer.

Actions: TEMP_ONLY always appends ST and preserves long tensors/reference; WRITE_LONG appends ST then optionally estimates confidence and updates long. Mask first eligible action to `[TEMP=0,WRITE=1]`. Once long exists use `[1,1]`. Forced actions have log_prob=0 and are excluded from actor/entropy PPO losses, but may train the critic. Terminal pair has no new transition; it closes the previous pending reward and ends the episode. External rollout truncation is not video done and must bootstrap rather than reset.

## Final v1 actor observation

**PROVISIONAL exactly 30 scalars**, detached FP32 `[B,30]`. Descriptor is L2-normalized mean over P of raw `feat_k1` (1024 channels), epsilon 1e-6. Current pointmap/confidence refers to res1 in common first-camera coordinates. All memory counts/descriptors/quality are PRE-UPDATE snapshots. Confidence statistics use finite positive values, `z=log(clamp(conf,1e-4,1e4))`. Point radius is `||X||`; no pose/PnP is required.

| Index | Feature | Range / normalization / missing handling |
|---|---|---|
| 1–5 | Mean, std, p10, p50, p90 of z | Continuous running normalization; zero if no valid confidence, with validity ratio 6. |
| 6 | Joint finite pointmap / positive finite confidence ratio | [0,1] over all pixels; no GT mask. |
| 7 | Cosine to last long descriptor | [-1,1], zero if absent; bit 18. |
| 8–9 | Max and mean cosine to recent <=4 ST descriptors | [-1,1], zero if empty; bit 28. |
| 10 | L2 difference between unit current and last long descriptor, divided by 2 | [0,1], zero if absent. |
| 11 | log1p RMS of raw pooled key | Continuous running normalization. |
| 12 | has_long_memory | Binary. |
| 13 | clip(log1p(write_count)/log(1025),0,1) | Accepted writes, bounded logarithmic count. |
| 14 | write_count/max(processed_candidate_count,1) | [0,1], previous completed candidates only; first observation zero. |
| 15 | clip(log1p(steps since accepted long write)/log(257),0,1) | Zero if absent; bit 12. |
| 16 | clip(log1p(current_step)/log(1025),0,1) | No known total sequence length or future progress input. |
| 17 | short occupancy/4 | [0,1]. |
| 18 | last long descriptor valid | Binary. |
| 19–20 | Previous successful RAFT uncertainty mean and clip(2*std,0,1) | [0,1]; zero if unavailable, bit 22. |
| 21 | Previous successful valid-flow fraction | [0,1], zero if unavailable. |
| 22 | has_previous_raft_quality | Binary; false on initial neutral seed. |
| 23–24 | Gate mean and clip(2*gate std,0,1) | [0,1]; forced g=1 short-only, g=0 long-only, zeros if neither. |
| 25–26 | log1p RMS of O_ST and O_LT | Continuous running normalization, zero unavailable, bits 28/12. |
| 27 | Weighted ST contribution ratio | [0,1], zero if neither; epsilon denominator. |
| 28 | has_short_memory | Binary. |
| 29 | log1p(median predicted radius / initial predicted radius scale) | Continuous running normalization; missing zero. Scale is detached prediction only. |
| 30 | log1p((radius p90-p10)/initial predicted radius scale) | Continuous running normalization; missing zero. |

**PROVISIONAL cheap percentiles:** deterministic <=16x16 grid of image pixels for confidence/radius quantiles (deduplicate integer sample indices); full-map reductions for mean/std/validity. Do not sort every image pixel or materialize attention. Initial predicted-radius scale is registered before the first observation from that first candidate, and thereafter frozen within the episode. This is not GT-derived normalization. If the current entire output is invalid, use explicit failure handling rather than submitting a plausible all-zero observation.

Running standardization applies only to indices 1–5,11,25,26,29,30; retain cosines, ratios and bits in their defined ranges. Fit mean/variance on training rollouts; freeze one snapshot throughout collection plus its PPO epochs; update between rollout batches only. Epsilon 1e-6, standardized values clipped [-5,5]. Checkpoint the normalizer and freeze it at deployment. Store both raw and normalized observations/version so old log probabilities use identical inputs. **LOCKED exclusions:** current RAFT uncertainty, future errors, future frames, total episode quality, GT, post-sequence PnP, attention entropy/matrices, and current post-write counters.

## Reward availability and exact timing

**VERIFIED FROM CODE / BLOCKER:** no recovered Endo3R training criterion or verified GT sequences are present. Demo's fabricated depth/pose/intrinsics cannot support PPO quality rewards. The loss/reward modes below are implementable contracts, not claims that local supervision exists. Release RGB examples alone permit teacher-proxy experiments only after a valid checkpoint is obtained.

| RewardComputer mode | Required evidence/data | v1 use |
|---|---|---|
| `DenseGTReward` | Verified metric depth/pointmaps, calibration, common-coordinate pose, validity masks at each scored frame | Preferred once section 14's strict manifest passes; currently BLOCKED. |
| `SequenceGTReward` | Verified end-of-sequence depth/reconstruction and optional poses | Sparse terminal quality only; no invented dense GT reward. Disabled by default. |
| `TeacherProxyReward` | Frozen, identified, finite-output legacy teacher and matched causal input prefix | Optional warm-up/proxy ablation; measures teacher agreement, not actual GT quality. |

For DenseGTReward use the same normalized masked common-frame point error as section 14, evaluated on NEXT pair's left prediction. The episode's scale is aligned once from the first prediction/GT, detached and fixed; never independently median-align each future frame. Denote this error E_(i+1). PROVISIONAL reward:

\[
r_i=-w_q E_{i+1}-\lambda_w I_{write\_attempt}
-\lambda_r I_{RAFT}\frac{\tau_{RAFT}}{\tau_{base}}
-\lambda_k\left(I_{KDA}\frac{\tau_{KDA}}{\tau_{base}}+I_{unc\_geom}\frac{\tau_{geom}}{\tau_{base}}\right).
\]

Defaults: wq=1, lambda_w=0, lambda_r=lambda_k=0.01, terminal weight=0. These are tunable starting points, not reward optimization results. Setting lambda_w=0 avoids double-penalizing a write already charged by real component costs; a nonzero count penalty is an ablation. The actor cost is essentially present for both actions and is reported rather than hidden. No long-state occupancy penalty: state size is fixed. Tau values come from isolated measurements on the target hardware/resolution; store calibrated medians in config/checkpoint. Do not insert per-step CUDA synchronization into training merely to generate a noisy reward. State rejection after RAFT still pays RAFT/geometry cost; no KDA cost if update was skipped.

Optional terminal metrics may include depth/3D and pose only when GT/protocol exists, with predeclared weights/scales. Full-sequence median-aligned depth metrics from `eval/depth_evaluation.py::depth_evaluation` 34–79 are evaluation metrics, not a causal dense reward definition. TeacherProxy substitutes a fixed matched teacher next-frame pointmap error; keep it labeled proxy. **PROVISIONAL finite reward bound:** clip normalized quality error to [0,5] for PPO only; reconstruction loss/evaluation remain unclipped. A numerical reconstruction/state failure uses quality error=5, pays already incurred action costs and terminates with an explicit failure flag. The environment's failure handler closes the pending prior transition when next decode fails; if failure precedes any decision, record an unscorable failed episode rather than inventing an action. A nonfinite state caused during the current action closes that action as an absorbing failure. Missing GT instead discards the affected episode as described in section 14; it is not a model failure reward. Never filter failed predictions out of the GT mask.

**LOCKED reward association:** store the write/RAFT/KDA flags at a_i; delay its quality component until arrival i+2 produces next res1. Before training, demonstrate that editing a_i can alter that next prediction while current res1/res2 remain unchanged. This is a sequential write problem even though actor inference is feed-forward.

## Shared policy interface and ablations

`observe` uses the common ObservationBuilder; an ObservationPacket may also carry cheap detached recent-long cosine scores for the heuristic, never full attention. `act` applies the same validity mask for all controllers.

**PROVISIONAL HeuristicWritePolicy:** compare current pooled unit descriptor with up to four recent accepted-long descriptors; if maximum cosine >0.92 choose TEMP_ONLY, else WRITE_LONG. The first is forced WRITE. This approximates the legacy aligned-patch rule but is not mathematically identical. Maintain no growing bank. AlwaysWrite chooses WRITE at every eligible decision; NeverWrite chooses TEMP after forced seed; PPO chooses its logits. The memory module implements neither threshold policy nor PPO training.

# 11. PPO Actor/Critic Design

**PROVISIONAL v1:** actor `30->Linear64->Tanh->Linear64->Tanh->Linear2`; separate critic `30->64->Tanh->64->Tanh->1`, with independent parameters. Orthogonal hidden initialization gain sqrt(2), actor output gain 0.01, critic output gain 1, zero biases. Apply mask before Categorical construction. Sample stochastically during PPO; evaluate deterministic argmax, tie goes to action 0. Export only actor, normalizer and config at deployment.

**PROVISIONAL optional privileged critic, default OFF:** append four training-only features: current normalized GT point error mean, p90, valid-GT ratio, GT-available bit, yielding 34 inputs. Critic data is supplied by a separate gt_context interface after actor observation construction; actor stays exactly 30-D and has no reference to that context. V1 does not use future quality even in the critic. Privilege is optional and blocked without real GT. Discard the critic at inference.

| PPO setting | PROVISIONAL default / discipline |
|---|---|
| Optimizers | Separate Adam actor and critic, lr=3e-4, betas=(0.9,0.999), eps=1e-5, no weight decay. |
| Rollout/minibatch/epochs | 128 eligible transitions across complete or partial episodes; minibatch 64; four PPO epochs. |
| Discount/GAE | gamma=0.99, lambda=0.95. |
| Policy clipping / entropy | clip ratio=0.2; entropy coefficient=0.01. |
| Value loss / clipping | MSE coefficient=0.5; optional value clipping=0.2, default on. |
| Gradient norm | Clip actor/critic each to 0.5. |
| Advantage normalization | Mean/std over trainable, non-forced action samples in each rollout; epsilon 1e-8. |
| Reward normalization | Fixed GT scale/cost ratios described above; no adaptive reward centering in v1. Log each component separately. |
| Stopping diagnostics | Approximate KL, clip fraction, entropy, value error, forced/rejected write fractions; stop PPO epoch at KL>0.03 provisionally. |

GAE: delta_i = r_i + gamma*(1-done_i)*V(s_(i+1)) - V(s_i); recurse with gamma*lambda and true video-terminal masking. At rollout-only truncation, bootstrap the next observation value and preserve the episode state. Forced transitions train value targets but are omitted from policy/entropy loss. A rollout with no learned-action samples skips actor updates rather than dividing by zero.

**LOCKED rollout schema:** raw/normalized actor observation, optional critic-only features, action, old log_prob, old value, reward, true done, rollout-truncated flag, action_mask, forced/policy-loss mask, requested/successful write flags, actual RAFT/KDA flags, quality metric/mask availability, cost components, episode ID/candidate/arrival indices, normalization version, frozen environment checkpoint ID. The rollout holds detached scalars/small vectors, not KDA graphs, full attention, image histories, or growing KV banks. Environment memory stays live across a rollout cut; reset at patient boundary.

R0 runs the frozen reconstruction/HybridMemory in eval/no_grad; actor/critic training is ordinary PPO over stored observations. Never call backward through Endo3R from a PPO loss in this stage. A change to reconstruction or normalization invalidates old on-policy samples. Save actor, critic, normalizer, optimizer state, RNG and environment/config/checkpoint IDs for reproducibility.

# 12. Exact New Forward Pseudocode

**LOCKED API ordering; PROVISIONAL terminal/TBPTT policy.** `run_episode` is used by the new inference compatibility forward and by explicit training/RL runners. For long videos, the equivalent `step()` emits PairResult. Tensor names follow the original code. `candidate.pointmap` is a canonical alias made BEFORE legacy-compatible dictionary renaming.

```python
def run_episode(model, frames, memory, policy, mode, gt_context=None):
    assert len(frames) >= 2
    if mode == "attention_train":
        assert len(frames) >= 3  # Need a subsequent read of seeded memory.
    assert all_equal_cropped_hw(frames)
    # Explicit new state, never SpatialTempralMemory for hybrid.
    state = memory.reset(batch_size=1, image_hw=HW, patch_hw=checkpoint_patch_hw,
                         device=model_device, training=(mode == "attention_train"))
    runner_cache = empty_feature_cache()
    pending_transition = None
    preds, pair_preds, diagnostics = [], [], []

    for i in range(len(frames) - 1):
        # At this point only the input prefix through I[i+1] may be exposed
        # to prediction, policy and memory. GT is a reward/loss side channel.
        state.current_step = i
        state.candidate_index, state.arrival_index = i, i + 1
        feat1, feat2, pos1, pos2, shape1, shape2 = model.encode_frames(
            frames[i], frames[i+1], runner_cache.feat1, runner_cache.feat2,
            runner_cache.pos1, runner_cache.pos2,
            runner_cache.shape1, runner_cache.shape2)

        if i == 0:
            fused_feat1 = feat1       # Preserve pretrained cold-start shortcut.
            read_diag = unavailable_branch_diagnostics()
        else:
            query = runner_cache.feat_k2
            fused_feat1, read_diag = memory.read(query, state)  # PRE-WRITE state.

        dec1, dec2 = model.decode(fused_feat1, pos1, feat2, pos2)
        # Use the actual downstream_head wrapper and layer-list conventions.
        res1 = model.downstream_head(dec1, shape1, 1)
        res2 = model.downstream_head(dec2, shape2, 2)
        assert_finite_prediction_maps_before_value_encoding(res1, res2)
        feat_k1 = model.encode_feat_key(feat1, dec1[-1], 1)
        feat_k2 = model.encode_feat_key(feat2, dec2[-1], 2)
        cur_v = model.encode_cur_value(res1, dec1, pos1, shape1)
        current_k, current_v = feat_k1, cur_v + feat_k1
        assert_finite_memory_features(feat_k1, feat_k2, cur_v, current_v)

        candidate = Candidate(
            key=current_k, value=current_v,
            descriptor=detach(l2norm(mean_P(current_k))),
            pointmap=common_first_camera_alias(res1, pair_index=i),
            confidence=res1["conf"],
            cropped_rgb=frames[i]["origin_img"].permute(0, 3, 1, 2),
            candidate_index=i, arrival_index=i+1)
        assert_valid_output_or_explicit_failure(candidate, res2)

        # Prior action's future-quality reward comes from NOW's left output.
        # Never use current res2 as a reward for the action still to be chosen.
        if pending_transition is not None:
            complete_delayed_reward(pending_transition, candidate,
                                    gt_context=gt_context, mode=mode)

        # Reconstruction loss sees both outputs through a separate GT API.
        if mode == "attention_train":
            accumulate_reconstruction_loss(res1, res2, gt_context, i)
            optionally_accumulate_teacher_memory_loss(fused_feat1, prefix=i+1)

        terminal_pair = (i == len(frames)-2)
        if not terminal_pair:
            # Packet snapshots PRE-UPDATE memory and historical quality only.
            initialize_predicted_scale_if_missing(candidate, state)
            packet = policy.observe(candidate, state, read_diag)
            action_mask = [0, 1] if not state.has_long_memory else [1, 1]
            decision = policy.act(packet, action_mask,
                                  deterministic=(mode == "eval"))
            # Actor/critic bookkeeping is outside HybridMemory.
            pending_transition = make_transition_if_rl(packet, decision)
        else:
            packet, decision = None, None  # No unused last action.

        # Every decoded left candidate enters ST, including the terminal pair.
        state = memory.append_short(current_k, current_v,
                                    candidate.descriptor, state)
        state.processed_candidate_count += 1

        write_diag = no_write_diagnostics()
        if decision is not None and decision.action == WRITE_LONG:
            state.write_attempt_count += 1
            if not state.has_long_memory:
                write_conf = neutral_confidence_on_valid_candidate(candidate)
                # No RAFT; require at least one valid patch to seed.
            elif uncertainty_enabled:
                write_conf = model.uncertainty_estimator.estimate(
                    candidate, state.accepted_long_reference())
                state.raft_call_count += int(write_conf.raft_called)
            else:
                write_conf = finite_candidate_confidence(candidate)

            if write_conf.has_any_positive_patch:
                state, write_diag = memory.write_long(candidate, write_conf, state)
                # Commit reference/statistics ONCE, only after finite state success.
                commit_detached_long_reference(candidate, write_conf, state)
                state.write_count += 1
                state.kda_update_count += 1
                state.last_write_step = i
            else:
                write_diag = rejected_zero_valid_write(write_conf)
                # No state/reference commit; RAFT work, if any, is still charged.
            attach_actual_cost_flags(pending_transition, write_conf, write_diag)
        # TEMP_ONLY: no uncertainty call, no long update, no reference change.

        # Do not mutate cached candidate references by later output renaming.
        compatible_res1, compatible_res2 = legacy_output_names(res1, res2, i)
        if mode != "attention_train":
            pair_preds.append((compatible_res1, compatible_res2))
            preds.append(compatible_res1)
        else:
            record_detached_cpu_training_metrics(compatible_res1, compatible_res2)
        runner_cache = cache_for_encode_frames(feat1, feat2, pos1, pos2,
                                              shape1, shape2, feat_k2)
        diagnostics.append(detached_step_log(read_diag, decision, write_diag, state))

        if mode == "attention_train" and tbptt_boundary_or_final(i):
            if window_has_trainable_connected_loss():
                backward_accumulated_window_loss()
                clip_and_optimizer_step()
            state, runner_cache = detach_all_cross_window_graphs(state, runner_cache)
            state = reproject_detached_short_raw_cache_with_current_weights(state)

        if terminal_pair:
            if mode != "attention_train":
                preds.append(compatible_res2)  # Final right output, no write.
            finalize_episode_done_and_optional_terminal_reward()
            pending_transition = None

    return compatible_original_outputs(preds, pair_preds), diagnostics
```

The calls to existing Endo3R methods above use its actual signatures (`encode_frames` returns features first, then positions, then shapes). Candidate, reset, reference/diagnostic and cache helpers are proposed new contracts. Initialize all six cached encode_frames fields to None. `origin_img` is `[B,H,W,3]`; convert the permuted crop to the documented uint8 RAFT representation without changing RGB values. Copy the original feature reuse semantics into the runner rather than invoking legacy forward recursively. The explicit attention trainer consumes losses/metrics stepwise and does not collect live prediction graphs across windows; its return is a training summary rather than compatibility prediction lists. Inference compatibility forward collects the original outputs. During PPO, run_episode's gradient-training branch is off; incremental Environment.step pauses at action points and closes each reward at the next decode. Do not precompute the whole sequence with one fixed memory trajectory and then pretend PPO actions changed it.

**LOCKED return compatibility:** preserve original output dictionary keys and ordering from `Endo3R.forward` 803–823, including first `pts3d`, later `pts3d_in_other_view`, and final right prediction. Diagnostics are opt-in through the new runner or explicit flag; default legacy return signature is unchanged. Export evaluation pose/depth via the new evaluator after inference as appropriate; do not feed those post-sequence estimates into actor state.

# 13. Legacy-to-New Code Mapping

| Existing file/function | Keep / replace / wrap | Proposed module / exact interface boundary | Why |
|---|---|---|---|
| `dust3r/model.py::Endo3R.__init__` 550–566 | Keep default; add optional config and lazy hybrid registration | `memory/config.py`, `hybrid.py`, `uncertainty.py`, policy registration | Load original pretrained DUSt3R then Endo3R weights; new params absent in legacy state_dict. Log missing/unexpected keys for hybrid loading. |
| `Endo3R.forward` 735–823 | Keep original body; add early dispatch before fresh legacy memory construction | `online/runner.py::run_episode`; default `memory_backend=legacy` falls through | New training bypasses undefined add_mem; official mode behavior preserved. |
| `SpatialTempralMemory.memory_read` 456–509 | LEGACY-ONLY | `HybridMemory.read` at original 759–760 boundary | New read returns decoder-ready `[B,P,1024]` without threshold/prune inheritance. |
| `SpatialTempralMemory.add_mem_check` 402–440 | LEGACY-ONLY; optional external-policy subclass | `HybridMemory.append_short/write_long` after original cur_v line 782; `legacy_adapter.py` | Separate unconditional ST from action/uncertainty/long commit. |
| `SpatialTempralMemory.uncertainty_map` 444–453 | LEGACY-ONLY | `UncertaintyEstimator.estimate` only after WRITE | Proper lifecycle, finite correspondences and update gates. |
| `Endo3R.encode_feat_key` 640–644 | KEEP weights/interface | runner calls attn_head_1/2 wrappers on feat1/dec1 and feat2/dec2 | Pretrained feature space remains the input to new memory. |
| `Endo3R.encode_cur_value` 653–661 | KEEP default use_feat=False | runner forms cur_v+feat_k1 exactly once | Preserve geometric values; initially freeze weights but keep differentiable inputs. |
| `Endo3R.decode` 663–666 | KEEP | same signature using fused_feat1 [B,P,1024], pos1, feat2, pos2 | No DUSt3R decoder redesign. |
| `Endo3R.downstream_head` 668–672 | KEEP | original two heads and FP32 wrapper | Preserve prediction interfaces and geometry conventions. |
| `demo.py` 85–227 / original eval wrappers | KEEP | new `demo_hybrid.py`, `eval/eval_hybrid.py` | New config/diagnostics/evaluation without repairing original demo behavior in place. |

**LOCKED baseline concepts:**

* `LEGACY_OFFICIAL`: untouched original class/forward behavior at the cited source commit, original default heuristic. Even known faults are reported rather than silently repaired.
* `NEW_HYBRID`: new Flash/KDA/fusion/uncertainty architecture, its own runner and trainers.
* `LEGACY_POLICY`: original dense memory math/pruning/lifecycle plus externally supplied write decisions; a controlled RL ablation, not identical to official heuristic execution.
* `LEGACY_CLEAN` is optional, separately named, with an explicit fix manifest and comparisons for every correction; not a prerequisite to rewriting the official baseline.

**PROVISIONAL minimum legacy+RL adapter:** instantiate a new subclass of SpatialTempralMemory in `legacy_adapter.py`; override only `check_sim` to return the supplied action's skip/write result. Set the pending decision immediately before invoking original add_mem_check. This preserves original temporal-first insertion, confidence/uncertainty work and pruning, and avoids modifying the class. First write uses the same forced mask; subsequent valid actions replace the gate. The common runner builds actor signals before this call. Official long-bank diagnostics are unavailable: use availability bits, fused-memory output norm as the long proxy and zeros for unavailable separate-ST/fusion diagnostics, recorded as LEGACY_POLICY observation semantics. Train a separate actor checkpoint; do not transplant hybrid actor weights and claim equivalent states. Legacy bugs may prevent this ablation; mark that failure instead of fixing it silently.

The LEGACY_POLICY runner has a distinct update branch: call original `add_mem_check` EXACTLY ONCE after setting the action, not HybridMemory.append_short followed by another legacy long write. Supply the original kwargs `index=i`, `check_uncertainty=...`, `cur_conf=res1['conf']` and, for i>0 when uncertainty enabled, `ref_view`, `cur_view`, `ref_pointmap`, `cur_pts3d` using the reference-selection logic at original lines789–799. The terminal pair's action-free ST-only insert calls the adapter once with a pending deterministic skip (not a learned third action); no RAFT/spatial append occurs. LEGACY_OFFICIAL retains its original N-1 candidate update calls, including the last pair. Record this terminal compute difference and compare an additional legacy-policy heuristic schedule when isolating pure policy savings.

`LegacyAdapterState` wraps the original memory object plus the detached descriptor/counter/reference metadata needed by the common ObservationBuilder; it is NOT HybridMemoryState with a dummy KDA tensor. Derive physical short occupancy from `temp_mem_k.shape[1]//P` and long presence from nonempty spatial `mem_k`. Track accepted writes/references after the original call, retaining the last accepted candidate map/image needed for its reference rather than inventing KDA metrics. A common runner dispatches read/update to backend-specific helpers; the shared contract is prediction/diagnostic/policy behavior, not identical storage fields. Enforce B=1 and report legacy nonfinite results as a blocked/failing experiment.

| Legacy mechanism | Hybrid disposition | Reason / replacement |
|---|---|---|
| `check_sim` | REPLACE; original LEGACY-ONLY | Temporary bounded descriptor heuristic; PPO shares same action interface. |
| `mem_attn`, `mem_count` | LEGACY-ONLY | No importance accumulation or age pruning in recurrent state. |
| `memory_prune`, `top_k`, growing 20k bank | REMOVE from hybrid / LEGACY-ONLY | KDA state fixed-size; no patch top-k. |
| `attn_thresh` | LEGACY-ONLY | Exact ST softmax and pure LT read; no threshold/zero-renormalization path. |
| `mem_uncert` | REPLACE | Current patch write confidence and bounded historical quality stats, not history-sized logits metadata. |
| `mem_c` | REMOVE from hybrid | Current confidence summaries used for actor/loss; do not retain unbounded dense confidence maps. |
| `wm`, `lm` | REMOVE from hybrid | Typed short-frame occupancy and accepted-write count avoid mixed frame/token units. |
| `spatial_buffer` | REMOVE from hybrid | Fixed recurrent state plus <=4 descriptors, one reference. |
| `temporal_buffer` | REPLACE | Clean four-frame projected K/V ring, raw shadows training-only. |
| RAFT | KEEP network, REPLACE lifecycle | One registered frozen/eval optional WRITE estimator. |
| `uncertainty_map` | REPLACE | Sanitized disagreement/confidence; original implementation legacy-only. |

# 14. Training Strategy

## Verified supervision and minimum defensible objective

**BLOCKER / VERIFIED FROM CODE:** original Endo3R weight-training code and loss recipe cannot be recovered from this release. `README.md` 24–26 states training is unreleased. `croco/models/criterion.py::MaskedMSE` 13–49 supervises RGB masked patches, not geometric reconstruction. `dust3r/cloud_opt/optimizer.py::PointCloudOptimizer.forward` 489–546 optimizes scene parameters, not this network. Do not describe those as Endo3R's reconstruction training objective. Existing `Endo3R.forward` training calls undefined add_mem. A new explicit supervised trainer is required.

**LOCKED data gate:** before T1/PPO, produce a sequence manifest whose source/provenance verifies each target, units, calibration and masks. The loader must fail rather than create Demo-style placeholder GT. Required minimum for the selected v1 objective: ordered RGB; real depth (or real dense pointmap); real camera intrinsics matching the crop/resize; real camera-to-world poses for common-coordinate cross-frame points; valid GT pixels; patient/split IDs. If only single-frame depth exists without poses, that does not establish the selected global pointmap objective. That case requires a separately validated supervision mode, not fabricated identity poses.

| Target | Evidence in release | Selected use |
|---|---|---|
| Depth GT | Demo optionally loads `_depth.png` but otherwise uses ones; `eval/depth_evaluation.py::eval_single_scene` 98–115 expects separate GT depth /1000 | Use only genuine, documented units; loaders differ, so inspect dataset format. Local clone has no verified GT. |
| Pose GT / calibration | Demo can read sidecars but defaults identity / unit intrinsics | Require real calibrated C2W poses and intrinsics for pointmap targets; optional pose evaluation, no v1 pose loss. |
| Pointmap GT / masks | `BaseStereoViewDataset.__getitem__` 67–125 computes pts3d and valid_mask even with fabricated inputs | Recompute from provenance-verified D,K,T, not blindly trust field presence. |
| Confidence target | No recovered Endo3R network-training criterion | No confidence target or direct confidence loss selected in default v1. |
| Temporal/memory labels | None established | No invented temporal-consistency or memory-selection supervision. Optional teacher feature loss is explicitly new. |

**PROVISIONAL primary objective: new scale-aligned, masked common-frame pointmap L1, not the unreleased authors' criterion.** For frame j, use true backprojection and common first-camera transform:

\[
X_j^*=T_0^{-1}T_j\,\operatorname{backproject}(D_j^*,K_j^*).
\]

The geometry helper for C2W conversion is `dust3r/utils/geometry.py::depthmap_to_absolute_camera_coordinates` 401–423. Strict loader applies the inverse-first-camera transform and crop-adjusts intrinsics; verify transform convention with a synthetic calibrated scene. GT mask is supplied validity AND finite positive GT depth AND valid calibration/pose, plus documented GT occlusion mask if available. Prediction validity must not remove hard pixels from this mask. Nonfinite predictions are explicit training/evaluation failures.

Let R0 be median norm of first-frame GT points over the true GT mask M0. Fit one detached episode scale `a0 = median_M0(||X_0^*||)/median_M0(||pred_X_0||)` on the first prediction only, using the SAME true mask in both medians. Reject episodes with empty M0, nonfinite predictions or a degenerate GT/predicted median <=1e-6; do not hide them with an enormous epsilon-derived scale. Do not realign future predictions independently or learn this scale as a network parameter. Keep it fixed for the episode and outside actor/state. With R0>epsilon:

\[
e_j(x)=\frac{\operatorname{mean}_{xyz}|a_0\hat X_j(x)-X_j^*(x)|}{R_0},\quad
L_{pts,j}=\operatorname{mean}_{x\in M_j^*}e_j(x).
\]

Train both res1 and res2 against their corresponding frames in first-camera coordinates. For window W pairs, use `L_recon = mean_W(0.5*L_pts,left + 0.5*L_pts,right)`. Points come from the original head output aliases (`pts3d` / `pts3d_in_other_view`), GT from strict SequenceDataset, with masks/scale as above. Adjacent pairs repeat frames, so log unique-frame metrics separately from this balanced pair loss. Default weights: lambda_pts=1; lambda_depth=lambda_conf=lambda_pose=lambda_temp=0. This supplies gradients through the same reconstruction outputs without pretending additional labels exist.

**LOCKED empty/missing GT handling:** strict dataset validation requires a nonempty real GT mask for every scored frame of a sample/episode. If a later decode reveals missing/invalid supervision, supervised training rejects that sample/window with coverage logged; DenseGTReward discards the affected episode's rollout transitions rather than silently assigning only compute penalties. Never call mean on an empty mask or treat missing quality as zero error. Model numerical failure is a separate recorded failure/penalty; data-unavailability is not a policy outcome. RGB-only data must explicitly choose TeacherProxy, not run DenseGTReward with fabricated fields.

**PROVISIONAL optional confidence calibration, disabled:** only after checkpoint postprocess/confidence mode is verified, use `c=clamp(pred_conf,1,100)` and `L_conf=mean_M(c*e - 0.2*log(c))`; this is a new confidence-weighted geometric term, not an original recovered baseline loss or GT confidence map. Its tuning and interaction with point loss require a separate experiment. The default pointmap loss does not supervise confidence directly; actor confidence usefulness depends on pretrained calibration and later measurements. No differentiable PnP, pose, direct memory-output, or direct uncertainty supervision is selected for v1.

GT poses may later permit pose metrics, but demo PnP is downstream and non-differentiable as used (`demo.py::main` 179–196). It is not added to training simply because a CLI prints pose results. Missing GT prevents completion of supervised T1 and GT-quality PPO; optional teacher-only warm-up cannot satisfy this gate.

## Stage T0 / A0 — released initialization, baseline and optional teacher

**LOCKED:** load released DUSt3R and Endo3R checkpoints; no reconstruction training from scratch. Required paths are `checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`, `checkpoints/endo3r.pth`, `checkpoints/raft-things.pth`, matching `Endo3R.__init__`, `demo.py` and legacy memory construction. Log hashes, checkpoint `args.model`, missing/unexpected keys, head mode, patch size, coordinate/normalization conventions. All are currently absent. Loading pickled checkpoint metadata on modern Torch must be resolved explicitly for trusted released files; never silently ignore mismatches.

Run unchanged LEGACY_OFFICIAL on bounded videos and evaluate with a separately validated evaluation protocol. Save `baseline_manifest.json`, timing/quality/failure tables and output hashes; do not train this stage. A frozen official legacy model can be an optional teacher only if its outputs are finite and reproducible. Preserve its audited behavior; if it fails, record the failure, do not silently fix it to manufacture teacher outputs.

## Stage T1 / A1–A2 — supervised hybrid memory training, RL off

**PROVISIONAL deterministic write curriculum:** first 10% of scheduled optimization steps use AlwaysWrite for controlled initialization, remaining steps use the bounded descriptor HeuristicWritePolicy (threshold 0.92). Alternatively use one policy throughout as an ablation. The forced first seed still applies. No stochastic actor or PPO loss is used. All WRITE gates are differentiable except detached uncertainty; the selection action itself is discrete/no-grad.

| Component | Initialization | T1 status / optimizer group | Loss and gradient route |
|---|---|---|---|
| Image encoder | Released DUSt3R/Endo3R weights | FROZEN; compute raw image encoder under no_grad when safe | No parameter gradient; preserves pretrained visual features. |
| Decoder, both streams | Released weights | FROZEN parameters, but NOT wrapped in no_grad | L_pts through decoder inputs to new memory. |
| DPT/prediction heads | Released weights | FROZEN parameters, keep differentiable input operations | L_pts through pointmaps to decoder/memory. |
| `attn_head_1`, `attn_head_2` | Released Endo3R 1792->1024 MLPs | FROZEN initially | Differentiable decoder-derived key inputs; no parameter updates. |
| `value_encoder`, `value_norm`, `value_out`, `pos_patch_embed` | Released Endo3R weights, default use_feat=False | FROZEN initially | Keep graph through pointmap patch embedding/value transformer and future writes; do not detach cur_v inside window. |
| `norm_q`, `norm_k`, `norm_v` | Reuse released LN1024 parameters | FROZEN initially | Single normalization boundary shared by new branches; derivatives to input still flow. |
| ST Q/K/V linears | New Xavier-uniform; no bias | TRAIN, lr=1e-4 | L_pts from future reads; optional feature teacher warm-up. |
| ST output linear | New small normal std=1e-3 | TRAIN, lr=1e-4 | Same; nonzero initialization permits upstream gradients. |
| LT Q/K/V linears | New Xavier-uniform | TRAIN, lr=1e-4 | Query gradients via pure read; K/V via accepted update/future state reads. |
| LT output linear | New small normal std=1e-3 | TRAIN, lr=1e-4 | Reconstruction/optional distillation. |
| LT beta/decay projections, A_log, dt_bias | Wbeta small normal, beta bias logit(0.1); Wg=0, A_log=0, dt_bias inv_softplus(0.1) | TRAIN, lr=1e-4; no weight decay for scalar gate/bias parameters | Future reconstruction through chunk_kda final state. Initial state itself is fixed zero, not a trained patient embedding. |
| Fusion MLP | Xavier hidden, small normal final, final bias logit(0.75) | TRAIN, lr=1e-4 | Reconstruction and optional feature distillation. |
| Uncertainty mapping alpha, floor, relative distance scale | Fixed alpha=5, floor=.05, scale=.1 | NOT trainable in v1 | Detached confidence controls beta/decay; no uncertainty labels invented. |
| RAFT | Released raft-things checkpoint | FROZEN, permanently eval, no_grad | No loss or optimizer group. |
| Actor / critic | Not needed / not executed | OFF | No PPO gradients. |

Optimizer: AdamW, base lr=1e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=.01 except biases/LayerNorm/gate scalar parameters. Clip total trainable gradient norm=1. Warm-up 5% of planned optimizer steps then cosine to 1e-6. Exact total step budget depends on available GT and is a runtime/data decision; do not invent a recovered schedule. Check gradients for every trainable group and log recurrent/gate saturation. New memory branches use validated BF16 autocast; state/gate reductions/loss are FP32. Keep existing encoder/decoder/head arithmetic FP32 initially to isolate the memory change; more AMP is a separate fit/performance experiment.

**PROVISIONAL samples:** B=1, nine consecutive raw frames produce eight pairs, cropped 320x256, TBPTT four pairs. Use patient-level train/validation splits, fixed stride=1 initially, no color/geometry augmentation until baseline preprocessing is reproduced. For a subsequence, reset memory and use its first camera as common origin; whole-video validation resets only at video start. First overfit a few genuine GT clips to prove gradients and declining loss; then train on the available training split. A 24 GiB target is provisional; measure before setting dataset workers or longer sequence lengths. Do not train on all local RGB videos and call teacher agreement GT reconstruction quality.

**Success criteria, PROVISIONAL:** finite states/outputs/gradients throughout held-out sequences; each trainable group receives nonzero finite gradient in a WRITE/read test; lower held-out pointmap error than untrained hybrid; normalized held-out quality within 5% relative of identified legacy baseline before PPO eligibility; state storage constant across sequence length; measured backend selection correct. If quality falls outside this provisional tolerance, stop RL progression and diagnose dimension/gates/teacher supervision; do not solve it by jointly PPO-training the whole reconstruction network. Produce `hybrid_memory_t1.pt` with config, original checkpoint hashes, source/external revisions, optimizer/scaler/RNG and validation summary, excluding patient states.

## Optional teacher distillation — useful warm start, not mandatory

**PROVISIONAL choice:** useful optional first-10%-steps warm start, not necessary for v1 if real GT training is stable. Frozen teacher feature is original memory_read output at the same read prefix before candidate write; student is new fused feature. Skip first pair (there is no teacher memory read). Match the candidate/arrival clock and patch indices, not a full-sequence teacher that saw future frames; queries follow each model's own causal trajectory as specified below. Proposed loss:

\[
L_{mem}=\operatorname{mean}|F_{hybrid}-F_{legacy}|+
0.1\operatorname{mean}_{p}(1-\cos(F_{hybrid,p},F_{legacy,p})),\quad
L=L_{recon}+\lambda_{mem}L_{mem}.
\]

Initialize lambda_mem=.1 and linearly anneal to zero over warm-up; reconstruction stays primary. Teacher uses its OWN query/cache and official heuristic trajectory at the same candidate/arrival prefix and patch ordering; numerical teacher/student queries may differ. This is aligned feature-target distillation, not teacher evaluation of the student's exact numeric query. Cached teacher targets therefore remain valid for the defined procedure. An alternative shared-query teacher read would require a separate live procedure and is deferred. Do not force a giant bank into deployment. Optional decoder/pointmap distillation is deferred. On limited VRAM, `tools/cache_teacher_traces.py` runs a separate frozen teacher pass and saves detached CPU traces keyed by video/prefix/checkpoint/preprocessing/action-trajectory hashes. Do not run teacher and student giant banks together on the 4 GiB GPU. Nonfinite teacher traces are rejected with coverage reported. TeacherProxy-only training is labeled experimental pretraining and does not fulfill GT quality gates.

## Stage T2 / A3 — optional reconstruction fine-tuning

**PROVISIONAL, gated by T1:** initially unfreeze key MLPs, value encoder/norm/out/pointmap patch embed and shared norm_q/k/v at lr=1e-5 if T1 quality needs it. Then, if necessary, unfreeze the last two decoder blocks of EACH stream at lr=5e-6; optional DPT heads also lr=5e-6 only when supported supervision requires adaptation. New memory stays lr=1e-4. Image encoder remains frozen in v1; full encoder unfreezing is excluded unless a later plan authorizes it. Use the same masked pointmap objective, sequence/TBPTT strategy and patient split. Checkpoint `hybrid_memory_ft.pt`; log exactly which groups were unfrozen. Success is improved held-out reconstruction without state/latency regressions; failures revert to the last validated T1 checkpoint, not silently change baseline.

## Stage R0 — frozen-environment PPO

**LOCKED:** freeze encoder, both decoder streams, heads, key/value encoders, normalization, ST/LT projections/gates, fusion and RAFT; eval/no_grad all environment inference. Actor and critic start from section 11 initialization and train with their separate PPO Adam optimizers and losses, not L_pts gradients. Required training episodes and reward mode are section 10's verified data contract. Use full eligible video episodes where practical; 128-transition rollout cuts may span episodes without truncating memory history. No TBPTT reconstruction graph is retained. Produce `ppo_write_actor.pt`, `ppo_write_training.pt` (critic/optimizer/RNG/normalizer) and a frozen-environment hash. Success: held-out quality meets predeclared tolerance and measured write/RAFT/time trade-off improves over heuristic with actor overhead reported; a lower write rate alone is not sufficient.

**PROVISIONAL PPO sample eligibility:** use videos with at least four raw frames so at least one non-forced learned decision exists. Three-frame episodes can test initialization/value targets but cannot count as actor-training progress; two-frame episodes are inference-only. No fixed truncation of the recurrent video state at the 128-transition rollout boundary.

## Stage R1 — optional limited adaptation after PPO

**PROVISIONAL safest schedule:** alternate a short GT-supervised adaptation phase (only fusion and LT gates/projections, lr=1e-6 initially, same reconstruction loss) with refrozen environment and freshly collected PPO rollouts. Decoder/encoder remain frozen. Discard stale rollouts whenever the environment changes. Do not backpropagate PPO through KDA or jointly optimize a changing reconstruction environment within old PPO minibatches. Save versioned `hybrid_memory_r1.pt` and a policy retrained/evaluated for that exact environment. This phase is optional and cannot repair missing GT.

**LOCKED checkpoint compatibility:** save schema version, source commit, official checkpoint hashes, memory backend/projection config, FLA/Flash package revisions, observation schema ID, normalizer version and reward calibration. Never save patient KDA state/references as model weights; reset on deployment video boundary. Loading hybrid params into LEGACY_OFFICIAL is rejected; hybrid initialization from official weights explicitly reports newly initialized keys.

# 15. TBPTT / Gradient Strategy

**PROVISIONAL v1 TBPTT=4 decoded pairs**, eight-pair training samples. The four-frame ST cache may carry gradients from previous frames inside the current window; KDA retains a differentiable state chain only across those four pairs. Discrete heuristic decisions, descriptors, RAFT/confidence, references and actor diagnostics are detached. Existing frozen decoder/heads/value/key modules still differentiate their inputs. Only the frozen image encoder can safely be no_grad in T1; freezing a decoder parameter does not justify detaching its memory input.

At a window boundary:

1. Sum/normalize that window's reconstruction and optional distillation losses; call backward once and clip/step once. Do not update parameters while an old graph will be reused.
2. Detach KDA state, all retained short raw/projected features, cached `feat_k2` and any decoder-derived runner cache. Detach pointmaps/references at storage time. Clear accumulated loss references.
3. Reproject the retained detached four-frame raw K/V using the updated ST projections/LayerNorm. Those new projections can receive gradients from next-window reads; their old source features intentionally have no earlier-window gradients. Do not carry old projected graphs or stale weights as if recomputed.
4. Carry numerical KDA state forward detached, like conventional TBPTT; do not replay the whole accepted history after every optimizer step. Its numerical state was made with earlier parameters, an expected approximation to quantify. Reset it at episode boundary, never leave state graph attached across patients.

If norm/key/value modules are later unfrozen, the detached raw-cache representation still intentionally reflects its originating step; only projection/normalization is refreshed at boundary. Report this approximation. Within-window append must be functional (`cat`/new tensor or safe equivalent); inference-only ring mutation must not overwrite tensors saved for backward.

**NEEDS_RUNTIME_VALIDATION critical gradient test:** discard chunk_kda output, use final state in a future pure query read and point loss, then verify gradients to initial state, K/V projections, beta/decay and LT output/Q projection. A no-long-write test should yield no LT-update parameter gradient, without claiming a bug. Demonstrate graph depth stops after four pairs and peak training memory plateaus with additional windows. If chunk backward fails, block T1; the fused recurrent inference operator is not an autograd substitute.

Memory includes frozen-network saved activations needed for input gradients, short raw shadows, KDA chunk scratch and the four-pair graph. Frozen weights alone do not remove this cost. Before increasing TBPTT/resolution, profile peak allocation; reduce TBPTT to two as an explicitly named experimental fallback if necessary, never silently retain a full-video graph.

# 16. Configuration Schema

**PROVISIONAL concrete v1 config**, except LOCKED semantic fields. Typed validation rejects inconsistent `model_dim != num_heads*head_dim`, invalid recurrent layout, legacy-only pruning options in hybrid, or efficient labels on math fallbacks. Required paths/data/cost calibration start null so missing assets fail readiness rather than select placeholders.

```yaml
schema_version: 1
baseline_identity: NEW_HYBRID   # LEGACY_OFFICIAL | LEGACY_POLICY | optional LEGACY_CLEAN
checkpoints:
  dust3r: checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
  endo3r: checkpoints/endo3r.pth
  raft: checkpoints/raft-things.pth
memory:
  backend: hybrid_flash_kda    # legacy | hybrid_flash_kda; optional experiment backends explicit
  input_dim: 1024
  output_dim: 1024
  short:
    max_frames: 4
    model_dim: 512
    num_heads: 8
    head_dim: 64
    backend: sdpa_flash       # sdpa_flash | flash_attn | math_reference
    require_flash: true
    dropout: 0.0
    store_projected: true
    raw_shadow_training_only: true
  long:
    backend: fla_kda          # fla_kda | torch_reference
    implementation_revision: 9f38d24980c46d46bd38614e743cdacd21906578
    model_dim: 256
    num_heads: 4
    key_dim: 64
    value_dim: 64
    state_dtype: float32
    state_v_first: false
    chunk_by_frame: true
    kernel_chunk_size: 64
    inference_mode: chunk     # optional fused_recurrent after parity/latency validation
    qk_l2_normalize: true
    uncertainty_write_gate: true
    confidence_gates_beta_and_decay: true
    decay_normalization: divide_by_frame_patches
    initial_beta: 0.1
    initial_frame_decay_rate: 0.1
  fusion:
    type: channel_gate
    hidden_dim: 64
    output_dim: 1024
    initial_short_preference: 0.75
uncertainty:
  enabled: true
  estimator: raft
  run_on: write_long_only
  reference: last_successful_long_write
  raft_iters: 20
  raft_dtype: float32
  raft_mixed_precision: false
  distance_scale_fraction: 0.1
  confidence_alpha: 5.0
  confidence_floor_on_valid: 0.05
  invalid_as_high_uncertainty: true
  reject_all_invalid_write: true
policy:
  long_write_policy: heuristic  # heuristic | ppo | always | never
  actions: [temp_only, write_long]
  force_first_write: true
  terminal_pair_action: none
  observation_schema: actor30_v1
  heuristic_similarity: 0.92
  heuristic_recent_long_descriptors: 4
  deterministic_eval: true
training:
  entrypoint: train_hybrid_memory.py
  data_manifest: null         # mandatory provenance-checked path for supervised training
  image_width: 320
  image_height: 256
  batch_size: 1
  sequence_raw_frames: 9
  tbptt_pairs: 4
  stride: 1
  new_memory_dtype: bfloat16
  pretrained_compute_dtype: float32
  freeze_image_encoder: true
  freeze_decoder_t1: true
  freeze_heads_t1: true
  freeze_key_value_t1: true
  freeze_shared_norms_t1: true
  optimizer: adamw
  lr_new: 0.0001
  lr_key_value_norm_ft: 0.00001
  lr_last_decoder_heads_ft: 0.000005
  weight_decay: 0.01
  gradient_clip: 1.0
  objective: masked_common_frame_pointmap_l1
  scale_alignment: first_frame_detached_episode_scale
  loss_weights: {pointmap: 1.0, depth: 0.0, confidence: 0.0, pose: 0.0, temporal: 0.0}
  teacher_distillation: optional_warm_start
  teacher_weight_initial: 0.1
  teacher_cosine_weight: 0.1
  teacher_anneal_fraction: 0.1
rl:
  entrypoint: train_rl_controller.py
  algorithm: ppo
  actor_state_dim: 30
  actor_hidden: [64, 64]
  critic_hidden: [64, 64]
  privileged_critic: false
  frozen_environment: true
  reward_mode: dense_gt       # blocked without verified GT; sequence_gt | teacher_proxy explicit
  reward_quality_weight: 1.0
  reward_quality_error_cap: 5.0
  failure_quality_error: 5.0
  reward_write_weight: 0.0
  reward_raft_weight: 0.01
  reward_kda_geometry_weight: 0.01
  terminal_quality_weight: 0.0
  cost_calibration_path: null # mandatory before compute-sensitive PPO
  rollout_transitions: 128
  minibatch_size: 64
  ppo_epochs: 4
  actor_lr: 0.0003
  critic_lr: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  entropy_coef: 0.01
  value_coef: 0.5
  value_clip: 0.2
  max_grad_norm: 0.5
  target_kl: 0.03
  advantage_normalization: true
  reward_normalization: fixed_gt_scale_and_cost_ratios
  observation_normalizer: continuous_indices_only
evaluation:
  include_terminal_write_in_rates: false
  export_state_between_patients: false
  quality_relative_tolerance: 0.05
  diagnostics: scalar_only
```

**LOCKED legacy configuration behavior:** absent memory config or `backend: legacy` with heuristic executes official branch and instantiates no hybrid parameters. `legacy+ppo` explicitly selects LEGACY_POLICY and its runner/adapter. LEGACY_CLEAN cannot be selected without a fix manifest. A correctness-only profile can set `require_flash:false`, `short.backend:math_reference`, `long.backend:torch_reference`, but result labels must expose that change and no efficient-runtime success gate is satisfied. Store all overrides in experiment artifacts.

# 17. Unit and Integration Tests

Tests are an implementation specification; none of these architecture tests were written/run during planning. **LOCKED** release gates test behavior, numerical correctness and causal isolation, not merely mirror implementations. **PROVISIONAL tolerances:** BF16 exact-attention comparison relative/absolute around 2e-2/2e-2 on well-scaled small tensors, FP32 reference around 1e-5; select final tolerances from official kernel tests and measured output distributions, with error histograms, not only a permissive allclose.

| Test file | Required tests / failure criteria |
|---|---|
| `tests/test_short_flash.py` | Correct [B,P,1024] output; zero-memory bypass; five frame appends retain exactly newest four in chronological order; projected K/V equal direct projection; SDPA/flash_attn match exact dense softmax under no mask/dropout; rectangular shapes/resolutions and backward; forced flash fails clearly if unavailable; math fallback finite and honestly labeled. |
| `tests/test_long_kda.py` | Zero reset reproducible; state element count constant after 1/10/100 writes; SKIP uses no kernel and preserves tensor values/reference; WRITE changes state; pure read leaves state bitwise unchanged; deterministic patch ordering; chunk final state matches FP32 reference and recurrent inference within tolerance; P not multiple of64; q dummy output discarded backward; zero-confidence candidate bypasses kernel; mixed c=0 tokens legal/no NaNs; no cross-patient leakage. |
| `tests/test_fusion.py` | Neither/short-only/long-only/both cases, exact bypass rules and [B,P,1024]; finite gate/output/ratio with zero branch norms; channel gate gradients; scalar diagnostics only. |
| `tests/test_uncertainty.py` | First write no RAFT; TEMP_ONLY no RAFT; one call per later WRITE; correct last-accepted reference after skips; all-invalid rejection; OOB/nonfinite flow and NaN point pixels never produce NaNs/high confidence; degenerate zero flow; patch pooling order; parent.train() cannot enable RAFT train; CPU/test-stub and CUDA device behavior; no grads to RAFT. |
| `tests/test_memory_policy.py` | Two actions only; first forced WRITE; Never policy seeds then skips; TEMP preserves long state/reference/counters; WRITE increments attempts vs successful/RAFT counts correctly; heuristic >.92 boundary and <=4 descriptors; rejected write is not a new action; observation length30 and missing-value bits; deterministic argmax. |
| `tests/test_online_causality.py` | Prefix through arrival i+1 produces identical actor packet/action/state despite changes in later frames; no GT affects actor packet; current action cannot change already-produced pair outputs; it can change next pair; final pair closes pending reward but no new action/RAFT/KDA; N=2 behavior; all N output names/order match original convention. |
| `tests/test_rewards_ppo.py` | Reward a_i uses next res1 only; fixed episode scale; fabricated GT rejected; GT/critic paths absent actor; forced-action loss masking; GAE terminal vs rollout truncation; normalizer snapshot stable across PPO epochs; no policy-update batch with all forced samples; PPO clipping/logprob calculation against small analytical cases. |
| `tests/test_hybrid_gradients.py` | Finite nonzero gradients for new QKV/out/gates/fusion through future reconstruction; frozen weights unchanged; input gradients through frozen decoder/value modules; graph cut at four-pair boundary incl cached feat_k2; no in-place saved-buffer overwrite; raw-cache reprojection after optimizer update; many TBPTT windows do not grow saved graph/memory without bound. |
| `tests/test_legacy_dispatch.py` | Default/explicit legacy no optional FLA import/new params, exact original return shape/keys and fixed-prefix outputs; official class unchanged; adapter action only controls spatial gate; legacy failures recorded; no baseline bug correction hidden in dispatch. |

**NEEDS_RUNTIME_VALIDATION integration gates:** real checkpoint finite three-frame and >=32-frame runs; crop/patch/coordinate convention; strict GT loader synthetic calibrated geometry; actual Flash and FLA kernels with training backward; resource accounting includes output retention; policy observations contain no tensors capable of linking actor to future/GT. Run simulated/stub tests without checkpoints, but do not count them as model quality evidence.

# 18. Performance/Profile Plan

**NOT PROFILED:** planning ran no Endo3R sequence inference, training or architecture benchmark. Bounded forced-SDPA probes were performed; current environment lacks required checkpoints/imports, compiled Flash backend and FLA. Audit section 7 lists prior static/algebra checks. Sources inspected here are evidence for interfaces, not speed claims.

**LOCKED accounting:** compare identical videos, crops, stride, warm-up, checkpoint versions, GPU, dtype and output-retention policy. Measure cold model/RAFT load separately. Do not repair baseline modes or cast original attention to BF16 without labeling a distinct baseline. Kernel-only comparisons may use controlled common dtype, labeled separately from full-model results.

Proposed `tools/profile_memory.py` accepts `--backend legacy|hybrid_flash_kda|...`, `--policy heuristic|always|never|ppo`, `--num-frames`, `--resolution`, `--seed`, `--warmup`, `--output-json`. Short run: 8 frames; long run: 64 frames (then 128 only if bounded resources permit). Legacy must grow enough accepted spatial memory to exercise its bank; report actual token count, not assumed growth. Include an always-write controlled sequence and a natural heuristic sequence, each finite and bounded.

Synchronize CUDA before/after measured GPU regions or use CUDA events with final synchronization; warm up selected kernels before steady-state measurement. Profile exclusive scopes to avoid double counting:

| Scope | Boundary |
|---|---|
| Image loading / transfer / encode | Dataset preprocessing separately; encode_frames new image encoder; separate first-pair two-image case. |
| Memory read | ST projections/read, LT projections/contraction, fusion; legacy memory_read with actual M. |
| Decoder | Both decoder streams, excluding heads. |
| Key/value encoding | attn_head_1/2, pointmap patch embed and value_encoder separately. |
| Prediction heads | Both downstream heads. |
| Observation + policy | Reductions, normalizer, actor; GPU/CPU transfer if any. |
| ST update | Projection and ring append/reprojection where applicable. |
| Heuristic similarity | Descriptor pool/cosine only; compare with legacy aligned-patch check. |
| RAFT | Actual calls only, 20 iterations, no cold load in amortized inference scope. |
| Uncertainty geometry | Flow sampling, validity, disagreement/pooling/confidence excluding RAFT. |
| KDA update / legacy prune | Kernel+gates vs spatial append/prune; record state size/M and invocation flag. |
| Downstream pose / reconstruction | Validated post-sequence PnP/evaluation separately; excluded actor and core per-step FPS unless explicitly included. |

Reset peak memory counters after load/warm-up; log allocated/reserved/peak GPU memory and CPU/device state/cache bytes. Report per-pair time, per-output-frame convention, total wall time, FPS only with denominator defined, p50/p95, conditional RAFT/write latency and amortized contributions. State storage, training graph, output-list storage, RAFT workspace and kernel scratch are separate categories. Record failures/OOMs; do not average them away.

**STATIC targets, not promises:** Flash preserves O(P*4P*512) exact attention arithmetic but lowers score materialization/IO; KDA changes history-dependent O(PM1024) retrieval to bounded O(P*4*64*64) read/update plus projections; RL targets fewer RAFT/geometry/KDA UPDATE calls. Encoder, decoder and six-block value encoder remain substantial fixed costs. Actor at most a small MLP plus reductions, but actual overhead is a measured gate.

Plot legacy read latency versus M and hybrid read latency versus number of accepted writes (should not grow structurally). Compare conditional RAFT latency with ST/LT read on the same workload. Measure cumulative write/gate saturation and reconstruction drift with increasing sequence length. Calibrate tau_base/tau_RAFT/tau_geometry/tau_KDA for reward only after correct kernels and finite outputs. Do not transfer Kimi throughput numbers to this vision model.

**PROVISIONAL success gates:** long-state storage plateaus; actual Flash verified; stateful update/read numerics pass; hybrid total p95 and peak VRAM improve versus a clearly named baseline at matched quality. PPO actor+observation overhead ideally <1% of total pair latency, but this is a target to measure. If RAFT/value encoder dominates, report the limited total gain rather than inflate attention-only gains into end-to-end claims.

# 19. Ablation Matrix

**LOCKED separability:** select attention backend and write policy independently through one runner/policy interface. Every experiment records baseline identity, normalization/uncertainty semantics, checkpoint/training budget, observation schema and output dtype.

| ID | Attention / memory | Write policy | Purpose / feasibility |
|---|---|---|---|
| B0 | LEGACY_OFFICIAL combined dense memory | Original heuristic | Required reference; unchanged defects/failures reported. |
| B1 | Optional LEGACY_CLEAN | Explicit clean heuristic | Only with fix manifest; never substituted for B0. |
| A1 | Flash ST + KDA LT + fusion | Bounded descriptor heuristic | Attention-only primary. |
| A2 | Same hybrid | Always write | Controlled write frequency; no learned controller. |
| R1 | LEGACY_POLICY dense memory via adapter | PPO write | RL-only feasibility ablation; baseline faults may block it. |
| AR | Same trained hybrid | PPO write | Combined research direction, after A1 validation. |

**PROVISIONAL attention comparisons:** B0 joint full memory; Flash ST + explicit legacy-style LT KV reference; Flash ST + KDA LT; KDA dimensions 128 (2x64), 256 (4x64), 512 (8x64) with equal kernel-valid heads; ST dimensions 256 (4x64) /512 (8x64); learned channel gate vs fixed .5 blend vs short/long-only diagnostic; uncertainty-to-write gates on/off; per-frame decay normalization 1/P vs calibrated alternative. Dimension changes require retraining, not swapping incompatible weights.

The optional Flash-ST/explicit-LT reference is **not official joint-softmax equivalence**. `long_dense_reference.py` uses the same 256-D projections/output interface and exact softmax over an explicit accepted-frame KV bank, whole-frame FIFO capped to `floor(20000/P)` frames, no token reorder/prune. Select `hybrid_flash_dense_reference` exclusively instead of KDA; no parallel production bank. Run uncertainty off initially to isolate attention representation, and label any clean FIFO/normalized uncertainty changes. A separate verified LEGACY_CLEAN split-bank variant may be added later; do not call this reference a source-preserving B0.

**PROVISIONAL RL comparisons on the SAME frozen hybrid checkpoint:** always; never-after-seed; descriptor threshold .92 (also optional threshold sweep); PPO; PPO with compute coefficients zero; PPO with WRITE uncertainty disabled. Retrain/freeze consistently for uncertainty architecture changes rather than conflating them with policy effects. Teacher distillation on/off is a training ablation, not an inference dependency. Include actor-free heuristic timing.

**LOCKED metrics:** validated depth/reconstruction errors, pointmap error, pose where genuine GT/protocol exists; FPS/denominator, per-pair p50/p95 and peak VRAM; requested/accepted WRITE rates, RAFT calls, KDA updates, quality rejections; actor/observation overhead; finite-output failure rate; state bytes vs sequence length. Match patient-level train/test splits, preprocessing and seeds. Report at least three training seeds when resources allow, otherwise disclose single-seed limits. Quality–compute Pareto comparisons matter more than write rate alone.

# 20. Implementation Phases

All file changes below are FUTURE implementation tasks. No phase was executed in this planning task. Tests and success/failure gates are LOCKED; durations/learning budgets/numeric criteria are PROVISIONAL.

| Phase | Files expected to be touched | Dependencies / concrete tasks | Tests / success gate | Failure gate / artifact |
|---|---|---|---|---|
| P0 — environment / baseline | `requirements-hybrid.txt`, `tools/check_memory_backends.py`, `eval/eval_hybrid.py` (new); no baseline repairs | Obtain identified checkpoints/data separately; inspect metadata, import readiness, compatible Linux/WSL/Torch/Triton; validate official B0 finite inference and GT provenance | Forced Flash forward/backward, FLA chunk/recurrent/reference, metadata/GT/crop/coordinate tests; known baseline outputs/failures archived | Missing assets/GT/kernels => block performance/training, allow interface work only; artifacts environment lock, checkpoint hashes, GT manifest, baseline_manifest. |
| P1 — scaffolding / dispatch | `dust3r/model.py` minimal registration/dispatch; new memory/types/config/init, online/init/runner, policies/observations; config YAML | Keep default legacy body/class; define state/read/write/reset and causal indices; introduce stubs that cannot masquerade as efficient backend | Legacy default return/state_dict/import isolation; episode reset/cold/terminal behavior; config rejects mismatches | Default behavior drift => stop; artifact scaffolding config/schema and dispatch tests, no trained checkpoint. |
| P2 — exact short Flash | `short_flash.py`, `hybrid.py`, `test_short_flash.py` | Projected ring and raw training shadows; forced SDPA first, official flash_attn fallback, math reference | Exact dense comparison/backward, four-frame eviction, backend identification, no empty softmax | No efficient kernel => correctness-only artifact, do not claim Flash achieved; artifact ST test/benchmark log and initialized module. |
| P3 — KDA adapter/state | `kda_adapter.py`, `long_kda.py`, `test_long_kda.py` | Pin FLA; state orientation; pure read; frame-chunk beta/decay; dtype/layout adapter; reference recurrence | State size constant, chunk/recurrent/reference parity, discarded-output/final-state gradient, SKIP identity | Invalid kernels/gradients => block T1; artifact KDA conformance report / initialized weights. |
| P4 — learned fusion / full forward | `fusion.py`, `hybrid.py`, `runner.py`, `demo_hybrid.py`; fusion/causality/gradient tests | Add availability gate; preserve key/value/decoder/head output contracts; scalar diagnostics | Finite cold/ST/LT/both branches; prefix invariance; actual checkpoint short/long inference | Nonfinite outputs/coordinate drift => stop; artifact `hybrid_init.pt` and forward trace, not a trained-quality result. |
| P5 — write uncertainty | `uncertainty.py`, `hybrid.py`, `runner.py`, uncertainty tests | Single RAFT, frozen/eval/device, accepted reference, sanitized correspondences, confidence both gates, reject invalid writes | First/TEMP no calls, later WRITE one call; no NaNs; reference/counter semantics | Missing RAFT => enabled mode blocked; explicit uncertainty-off experiment allowed; artifact U calibration/latency report. |
| P6 — attention training | training/init/sequence_dataset/losses/teacher/hybrid_trainer; `train_hybrid_memory.py`, `tools/cache_teacher_traces.py`, gradient tests | Strict GT; T1 deterministic writes; exact freeze/lr groups; optional teacher warm-up; TBPTT4; optional T2 | All new gradients; short clip overfit then patient holdout; finite constant-state long inference; provisional <=5% relative quality gap | Missing GT/original recipe claimed => stop claim; quality regression => tune attention before RL; artifacts `hybrid_memory_t1.pt`, optional `hybrid_memory_ft.pt`, manifests. |
| P7 — RL environment | rl/init/environment/rewards/rollout/normalization; policies/observations; reward tests | 30-D observation; forced mask; delayed next-left reward; measured cost calibration; episode/truncation semantics | No GT/future/currentRAFT observation; action can affect next outputs; terminal GAE; honest reward mode | Unsupported GT/reward or leakage => block PPO; artifacts observation schema, calibrated costs, sample validated rollouts. |
| P8 — PPO | rl/networks/ppo; `train_rl_controller.py`, RL tests | Freeze environment; actor/critic initialization, separate optimizers; rollout128/GAE/clipped PPO; deterministic eval | Actor mask/logprob/normalizer consistency; holdout quality/compute vs always/heuristic; actor overhead measured | Lower writes without acceptable quality is not success; artifacts actor/normalizer and training checkpoint tied to frozen hybrid hash. |
| P9 — optional limited adaptation | hybrid_trainer/ppo entrypoints with explicit stage flags | Alternate low-LR supervised fusion/LT adaptation and refrozen fresh PPO, never stale rollout training | Versioned environment, quality retained, no whole-network PPO | Missing GT/nonstationary batch => skip; artifacts versioned R1 hybrid+policy, optional only. |
| P10 — ablate / profile | `tools/profile_memory.py`, `eval/eval_hybrid.py`; optional legacy_adapter/legacy_clean/long_dense_reference; all integration tests | B0/A/R/AR, dimensions/fusion/U/cost penalties; synchronized timing and memory; named clean fixes only if needed | Matched quality/crops/checkpoints, p50/p95/FPS/VRAM/components/write/RAFT/KDA; source identity documented | Failed B0/unsupported ablation disclosed rather than hidden; artifacts reproducible result tables, fix manifest if B1, final deployment checkpoint/config. |

**LOCKED progression:** P1–P4 reference/scaffolding can proceed before all training assets; efficient-runtime claims require P0/P2/P3 kernel gates. P6 cannot complete without actual reconstruction supervision or an explicitly downgraded teacher-only experiment. P8 cannot begin before a stable validated P6 checkpoint and valid reward/cost evidence. Optional P9 is not a way around these dependencies.

# 21. Risks and Fallbacks

| Risk / classification | Evidence or reason | Planned response / label |
|---|---|---|
| BLOCKER: efficient local backends unavailable | Forced SDPA Flash fails; FLA/Triton absent on native Windows | Separate validated Linux/WSL environment; math/recurrence only correctness fallback, no claimed efficient implementation. |
| BLOCKER: checkpoints, real GT and original training recipe absent | Audit readiness and README | Obtain/check actual assets; use new explicit pointmap objective with provenance. Teacher-only warm-up stays proxy; cannot claim recovered Endo3R training. |
| HIGH: legacy correctness affects benchmark/teacher/R1 | Audited mem_attn, token pruning, zero threshold rows, RAFT train, missing method | Preserve B0; count failures. Optional separately named B1 with fix manifest; never silently overwrite B0. |
| HIGH: KDA cannot retain all spatial geometry in small state | Fixed low-dimensional summary replaces explicit retrievable keys | Train attention first, measure long-video drift; dimension/fusion/write-gate ablations, optional teacher warm-up. No secret fallback giant bank. |
| HIGH: delta-rule patch scan and decay over P tokens | KDA originated as token recurrence; frame spatial scan adaptation is new | Deterministic order, 1/P decay calibration, state/norm/saturation tests; positional/order experiments only if needed. |
| HIGH: state backward or numerical precision fails | Final-state-only use differs from usual full output training; Triton not run locally | Mandatory chunk backward/reference tests; block training on failure, do not substitute inference recurrent operator. |
| HIGH: PPO reward exploits uncalibrated quality or masks | Missing GT, per-frame scale realignment, prediction-based masking could hide errors | Strict GT masks, one episode scale, failures counted, next-left timing, verified evaluation separate actor. |
| HIGH: accidental future leakage | Left prediction incorporates acquired right frame | Arrival-time indexing and prefix tests; actor sees only prefix through i+1, GT API separated. |
| MEDIUM: all-invalid write uncertainty | Occlusions/OOB/NaNs/motion failure | Zero confidence => no update/reference commit; actual costs counted; first invalid seed fails clearly. |
| MEDIUM: retained autograd/stale short projections | Recurrent/short caches across optimizer boundaries | TBPTT4, all caches detached, raw shadows reprojected, no step until window graph consumed. |
| MEDIUM: training memory still large | Frozen decoder/value modules need saved input-gradient activations | Measure on larger training GPU; explicit TBPTT2/resolution fallback; never retain full-video graph. |
| MEDIUM: output retention makes apparent memory growth | Compatibility return collects all N pointmaps | Measure memory states separately; stream outputs/CPU retention for deployment profile; match baseline retention in comparisons. |
| MEDIUM: policy does not reduce dominant cost | Encoder/decoder/value encoder remain fixed; KDA reads constant | End-to-end component timing; charge actual RAFT/update costs, report limited gain. |
| MEDIUM: heuristic descriptor mismatch / actor partial observation | New bounded pooled keys differ from aligned-patch baseline | Label approximation; baseline and heuristic ablations; don't claim threshold .92 equivalent behavior. |
| LOW: backend checkpoint/state schema drift | External APIs and dtype conventions evolve | Pin versions/revisions; load validation; conversion only in adapter; save schema IDs. |

**LOCKED fallback discipline:** a fallback is explicit in configuration, artifact name and metrics. Exact math satisfies correctness but not Flash efficiency; teacher reward satisfies proxy agreement but not GT quality; LEGACY_CLEAN satisfies corrected baseline behavior but not B0 identity. An efficiency/quality claim cannot borrow a success gate from a different mode.

# 22. Open Questions Requiring Runtime Evidence

All entries are **NEEDS_RUNTIME_VALIDATION**, not unresolved core interface choices:

1. What are the released checkpoint's actual patch size, encoder/decoder/head configuration, point/confidence postprocessing and load-state mismatches? Do its outputs follow common first-camera coordinates across the full online sequence?
2. Which separately provisioned environment runs both exact Flash forward/backward and the pinned FLA KDA kernels on the selected GPU? Does it support BF16 with P=196/320/768, rectangular short memory and final-state-only KDA gradients?
3. What verified sequence datasets expose real calibrated depth/points and camera poses, valid masks/units and patient split IDs? Does their crop/resize reproduce model preprocessing? Original loss remains unavailable; can the selected new pointmap objective match baseline quality?
4. Can the original B0 run finite without bug fixes on short/long videos? Which defects actually trigger, and is an optional separately named B1 needed for interpretable results/teacher traces/R1?
5. Does 4x64x64 KDA state preserve reconstruction at longer sequences? How do state norms, write gates, 1/P decay and scan order behave? Which dimension/fusion ablation, if any, is necessary?
6. How well does predicted-scale 3D disagreement identify poor writes? What are valid-flow fractions and uncertainty distributions; are v1 alpha=5, distance fraction=.1 and confidence floor=.05 effective? Calibrate these exact defaults instead of treating them as learned facts.
7. Is RAFT or six-block value encoding more costly than memory read at matched resolution/dtype? What fraction of total latency is action-dependent after KDA removes growing retrieval?
8. Does TBPTT4 fit and train stably on the actual training GPU? Do raw short-cache refresh and detached carried KDA state bias quality? Does optional T2 improve quality without destabilizing memory?
9. Are actor confidence/novelty/history signals predictive of beneficial writes? Is a 30-scalar feed-forward partial observation enough, and is policy overhead below its provisional target?
10. With delayed next-left reward and fixed episode scale, can PPO improve held-out quality–cost over heuristic/always? Which cost multipliers yield a real Pareto trade-off rather than underwriting/forgetting? Are optional terminal metrics supported by GT?
11. Does the legacy-policy adapter remain executable after spatial pruning, and can it be evaluated honestly given baseline faults? If not, R1 is reported infeasible rather than replaced silently.

**PROVISIONAL choices already made for implementation:** ST512/8x64; LT256/4x64 state; channel gate; confidence gating BOTH beta and 1/P decay; 30 actor scalars; TBPTT4; masked common-frame point L1; explicit freeze/lr hierarchy; PPO defaults in config; terminal no-action. Runtime may revise them through named experiments, not by redesigning core read/update/policy interfaces during initial implementation.

# 23. Final Implementation Readiness Verdict

**READY_TO_IMPLEMENT_WITH_BLOCKERS**

The module boundaries, state layout, pre-update reads, frame-chunk updates, causal decision/reward clock, uncertainty mapping, exact actor vector, forward ordering, legacy isolation, losses, gradient windows and staged optimizers are specified sufficiently for scaffolding and correctness implementation. The research directions remain LOCKED; numerical choices are explicitly PROVISIONAL.

Efficient deployment and training are blocked by absent identified checkpoints/verified GT, an uncompiled local Flash backend and missing compatible FLA/Triton environment. The original Endo3R weight-training criterion is unreleased, so the proposed pointmap objective is a new minimum defensible objective requiring real calibrated data, not a recovered baseline recipe. KDA final-state gradients, numerical/quality behavior, training-memory fit and total runtime benefit still need real kernel/model evidence. PPO is additionally gated by a validated trained hybrid checkpoint and quality/cost reward calibration. These blockers do not authorize source fixes during planning or silent baseline correction during implementation.

**Planning provenance:** repository commit verified with git; audited memory/model/decoder/value/RAFT/dataset/loss/evaluation boundaries used with cited symbols; official Kimi paper, Moonshot repositories, pinned FLA ops/layer/tests/package metadata, PyTorch SDPA/autograd and Triton/FlashAttention requirements inspected. Runtime commands during planning were read-only git/rg/file queries, environment/dependency/GPU checks and tiny forced-SDPA correctness probes. Document checks passed: 23 consecutively numbered required sections, balanced fences, YAML parse, short/long dimension equations, binary action count, actor dimension30, TBPTT4, and a unique 44-new/1-existing future file manifest. No Endo3R inference/training/download/install was performed. RTK was unavailable, so a session-local `rtk` pass-through honored command prefixes without installing anything. Final `git diff --stat` is empty; `git status --short --untracked-files=all` shows only the prior audit and this new plan as untracked documentation. The prior audit SHA256 is `A8F9E3F792E97584B9C8BF263C1110E3BB10E675F9F3EF7D26CC216905BC6258`; this task never wrote it.

```text
?? docs/END3R_ATTENTION_RL_AUDIT.md
?? docs/ENDO3R_FLASH_KDA_RL_PLAN.md
```
