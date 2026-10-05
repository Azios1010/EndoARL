# 1. Executive Summary

Audit date: 2026-10-02. Repository commit: `8145c8e5535d2111974561e7d1cc06f1d008fe5e`. Scope: the freshly cloned implementation, principally the normal online `demo.py` path. No source, configuration, dependency, checkpoint, or dataset was changed. No training, downloads, commits, or pushes were performed. This report is the only added repository file.

Evidence labels used throughout:

- **VERIFIED FROM CODE**: a statement directly established by the inspected implementation; line ranges refer to this commit.
- **INFERRED**: a consequence, complexity calculation, or insertion feasibility derived from code, without baseline execution.
- **PAPER INTENT / COMMENT ONLY**: a paper, README, or comment statement that does not establish executable behavior.
- **POTENTIAL BUG**: suspicious behavior with its trigger and consequence identified; runtime confirmation is specified where needed.
- **UNKNOWN / NEEDS RUNTIME VERIFICATION**: information not established by the supplied code/assets or measurements.

**VERIFIED FROM CODE — architecture.** Endo3R processes consecutive pairs, reuses the previous right image's encoder features, reads memory using that image's previously decoded key, runs two DUSt3R decoder streams, predicts dense pointmaps/confidence, encodes a pointmap into memory values, and updates memory. The latest frame needs one-frame lookahead to become a memory candidate; the final frame is predicted but never written. The demo estimates focal length and camera poses after the sequence using focal estimation and OpenCV PnP, then converts global points to local depth. Sources: `dust3r/model.py::Endo3R.encode_frames/forward` (~630–638, 735–823); `demo.py::main` (~152–227).

**INFERRED — compute bottlenecks.** Growing dense memory retrieval, the dual decoder, image encoder, six-block value encoder, and conditional 20-iteration RAFT are all material candidates. A latency winner cannot be declared: **NOT PROFILED**. All transformer attention is explicit dense PyTorch attention; the optional CUDA extension accelerates RoPE. Sources: `dust3r/model.py::SpatialTempralMemory.memory_read/uncertainty_map` (~444–509), `Endo3R.set_memory_encoder` (~569–589); `croco/models/blocks.py::Attention/CrossAttention` (~81–169); `croco/models/pos_embed.py::RoPE2D` (~106–159).

**INFERRED — memory bottlenecks.** Default uncertainty-enabled retrieval has a completed-update upper bound of `M <= 20,000 + 8P`, not merely 20,000 tokens. It materializes a dense `[B,P,M]` matrix and concatenated/normalized K/V. Independent of that budget, all GPU input frames and pairwise predictions remain resident; `mem_c` and `spatial_buffer` grow with accepted writes. Sources: `dust3r/model.py::SpatialTempralMemory.add_mem_check/memory_read/memory_prune` (~402–528), `add_mem_c` (~308–315), `Endo3R.forward` (~744–821); `demo.py::main` (~108–112).

**INFERRED — top attention insertion point.** `SpatialTempralMemory.memory_read` is the most isolated retrieval boundary: `[B,P,1024] -> [B,P,1024]`, without replacing CroCo projections. A long-only replacement is structurally narrower than changing the pretrained backbone, but is not a direct module swap: temporal and spatial tokens share one softmax and threshold denominator. Quality safety remains unmeasured. Source: `dust3r/model.py::SpatialTempralMemory.memory_read` (~464–501).

**INFERRED — top RL insertion point.** The spatial-write gate immediately after unconditional temporal insertion and around `check_sim` is the cleanest sequential decision boundary. A future choice between spatial write and temporal-only retention would affect subsequent retrieval and conditional RAFT work. This is an insertion candidate, not a selected controller or architecture. Source: `dust3r/model.py::SpatialTempralMemory.add_mem_check` (~406–428).

**Major correctness risks:** attention importance is never accumulated; pruning reorders individual tokens and invalidates the similarity check's assumed recent-frame layout; threshold renormalization can yield NaNs; RAFT remains in training mode; disabling uncertainty also disables temporal retrieval; training and offline demo branches call undefined methods. Sources: `dust3r/model.py::SpatialTempralMemory.memory_read/memory_prune` (~494–528), `check_sim` (~383–394), `__init__` (~259–262), `Endo3R.forward` (~735–739, 784–785); `demo.py::main` (~119–133). Detailed severities appear in section 13.

# 2. Exact Inference Call Graph

**VERIFIED FROM CODE — normal online path:**

```text
sorted RGB filenames
  -> Demo._get_views
  -> sample_frame_idx: filenames[::kf_every]
  -> imread_cv2 / PIL -> crop + resize RGB/depth/intrinsics
  -> BaseStereoViewDataset.__getitem__: ImgNorm, true_shape, auxiliary GT fields
  -> DataLoader(batch_size=1): list of frame dictionaries
  -> demo.main: move ALL normalized images to device; model.eval()
  -> Endo3R.forward: construct a fresh SpatialTempralMemory, load RAFT checkpoint
  -> for i = 0 ... N-2, pair (frame_i, frame_i+1):
       encode_frames
         first pair: encode_image_pairs -> DUSt3R._encode_image
         later: reuse previous right features; encode_image(new right frame)
       i == 0: fused reference = raw left encoder features
       i > 0: memory_read(previous right key, res=True) -> fused reference
       decode -> DUSt3R._decoder -> two self/cross-attention decoder streams
       encode_feat_key(raw left, decoded left, head 1) -> current write key
       encode_feat_key(raw right, decoded right, head 2) -> next-step query
       downstream_head 1 / 2 -> pointmaps + confidence [optional learned pose]
       encode_cur_value(left pointmap) -> pointmap patch embed -> value_encoder
       stored value = encoded value + current write key
       if uncertainty enabled and i > 0: select last accepted reference/index/pointmap
       add_mem_check:
         append temporal -> trim temporal keys/values to four frames
         check_sim -> similar: return (no spatial write, no RAFT)
         otherwise:
           first candidate: zero uncertainty
           later candidate: RAFT(current, already selected reference)
             -> flow -> 3D disagreement uncertainty -> patch pool
           append spatial key/value, confidence, uncertainty
           update wm/lm -> prune individual spatial tokens if lm > 20,000
       retain left prediction and both pair predictions
  -> append final right prediction; return preds, preds_all [optional memory]
  -> demo.main: focal estimate from frame 0 -> shared 3x3 intrinsics
  -> PnP RANSAC for each frame -> world-to-camera, inverse camera-to-world pose
  -> transform global pointmap -> local z-depth
  -> if --save_result: depth PNG/NPY and result.npy reconstruction arrays
```

Execution citations:

| Boundary | Exact implementation and approximate lines |
|---|---|
| Loading/subsampling | `dust3r/datasets/demo.py::Demo._get_views` (30–98); `dust3r/datasets/base_many_view_dataset.py::BaseManyViewDataset.sample_frame_idx` (49–55); `dust3r/utils/image.py::imread_cv2/ImgNorm` (26, 80–90) |
| Resize and batch | `dust3r/datasets/base/base_stereo_view_dataset.py::BaseStereoViewDataset._crop_resize_if_necessary/__getitem__` (145–203, 67–126); `demo.py::main` (98–112) |
| Model/checkpoint and memory construction | `demo.py::main` (91–95); `dust3r/model.py::Endo3R.__init__/forward` (550–566, 735–747); `SpatialTempralMemory.__init__` (243–263) |
| Encoder caching | `dust3r/model.py::Endo3R.encode_image/encode_image_pairs/encode_frames` (604–638); `AsymmetricCroCo3DStereo._encode_image` (153–167) |
| Memory read before decoder | `dust3r/model.py::Endo3R.forward` (758–767); `SpatialTempralMemory.memory_read` (456–509) |
| Decoder | `dust3r/model.py::Endo3R.decode` (663–666); `AsymmetricCroCo3DStereo._decoder` (200–220); `croco/models/blocks.py::DecoderBlock.forward` (186–191) |
| Keys, heads, values | `dust3r/model.py::Endo3R.encode_feat_key/downstream_head/encode_cur_value/encode_value` (640–661, 668–672); `forward` (769–782) |
| Updates/uncertainty/pruning | `dust3r/model.py::Endo3R.forward` (784–801); `SpatialTempralMemory.add_mem_check/uncertainty_map/memory_prune` (402–453, 511–537) |
| Pointmap output | `dust3r/model.py::Endo3R.forward` (803–823); `dust3r/heads/dpt_head.py::PixelwiseTaskWithDPT.forward` (244–253); `dust3r/heads/postprocess.py::postprocess` (63–72) |
| Pose/depth/reconstruction output | `demo.py::main` (152–227); `dust3r/post_process.py::estimate_focal_knowing_depth` (12–60); `dust3r/utils/geometry.py::geotrf/inv` (238–310) |

**VERIFIED FROM CODE:** normal online inference does not call `dust3r.inference.inference`; that helper is used by `--offline` on `model.dust3r`. The offline branch subsequently calls missing `Endo3R.offline_reconstruction`, so it is not a working alternate Endo3R path. Sources: `demo.py::main` (~119–138); `dust3r/inference.py::inference` (~83–98); `dust3r/model.py::Endo3R` (~550–856, complete method inventory).

**VERIFIED FROM CODE:** memory is recreated for every `forward(frames)` call. The implementation receives a whole sequence and returns at its end; no public persistent single-frame streaming interface is supplied. For `N` frames it performs `N-1` pair steps, `N-2` memory reads, and `N-1` candidate updates. Sources: `dust3r/model.py::Endo3R.forward` (~735–823).

# 3. Tensor Shape Trace

Notation: `B` batch size; `H,W` cropped image dimensions; `P=(H/16)(W/16)` for the expected 16-pixel patches; `T <= 4` retained temporal frames; `S` physical spatial token count; `M=TP+S` with uncertainty metadata, otherwise `M=S`; `L_d` checkpoint-defined decoder depth. Actual released checkpoint contents are absent. Hardcoded Endo3R interfaces expect encoder dimension 1024 and decoder dimension 768. Sources: `dust3r/model.py::Endo3R.set_memory_encoder/set_attn_head` (~565–601), `SpatialTempralMemory.add_mem_c/add_mem_uncertainty` (~308–339).

| Stage | File / Function | Input Shape | Output Shape | Notes |
|---|---|---|---|---|
| File decode | `dust3r/datasets/demo.py::Demo._get_views` (30–98); `dust3r/utils/image.py::imread_cv2` (80–90) | RGB file | `[H0,W0,3]` | RGB uint8; optional depth `[H0,W0]`, intrinsics `[3,3]`, pose `[4,4]` |
| Crop/resize | `dust3r/datasets/base/base_stereo_view_dataset.py::_crop_resize_if_necessary` (145–203) | RGB/depth/intrinsics | RGB `[H,W,3]`, depth `[H,W]` | Principal-point crop, rescale, final crop; `origin_img` is cropped RGB |
| Normalization | `dust3r/datasets/base/base_stereo_view_dataset.py::__getitem__` (92–110); `dust3r/utils/image.py::ImgNorm` (26) | PIL RGB | `[3,H,W]`, `true_shape [2]` | Mean/std 0.5, approximately `[-1,1]`; auxiliary pointmap `[H,W,3]` |
| Collate/device | `demo.py::main` (105–112) | `N` view dicts | `N` images `[B,3,H,W]` | `B=1`; `origin_img [B,H,W,3]`, `true_shape [B,2]` |
| Image patches | `dust3r/patch_embed.py::PatchEmbedDust3R.forward` (19–29); `croco/models/blocks.py::PatchEmbed` (209–247) | `[B,3,H,W]` | `[B,P,1024]`, positions `[B,P,2]` | 1024 required by Endo3R; patch size needs checkpoint validation |
| Encoder | `dust3r/model.py::AsymmetricCroCo3DStereo._encode_image` (153–167) | Patch tokens, positions | `[B,P,1024]` | First pair batches two images as `[2B,3,H,W]`; later encodes one |
| Query carried across pair steps | `dust3r/model.py::Endo3R.forward` (758–774) | Previous right encoder `[B,P,1024]` + right decoder `[B,P,768]` | `feat_k2 [B,P,1024]` | Query is a learned key from the previous pair, not fresh raw encoder features |
| Memory K/V assembly | `dust3r/model.py::SpatialTempralMemory.memory_read` (464–476) | Temporal `[B,TP,1024]`, spatial `[B,S,1024]` | K/V `[B,M,1024]`, U `[1,1,M]` | Temporal prefix added only if `mem_uncert` exists |
| Memory attention | `dust3r/model.py::SpatialTempralMemory.memory_read` (470–501) | Q `[B,P,1024]`, K/V `[B,M,1024]` | Matrix `[B,P,M]`, fused `[B,P,1024]` | One full-channel attention; query residual |
| Decoder projection | `dust3r/model.py::AsymmetricCroCo3DStereo._decoder` (200–208) | Fused reference/new right `[B,P,1024]` | Both `[B,P,768]` | Shared decoder embedding |
| Decoder layers | `dust3r/model.py::AsymmetricCroCo3DStereo._decoder` (209–220) | Two decoder streams | Each list: encoder tokens plus `L_d` decoder outputs | Last decoder output `[B,P,768]`; DPT consumes several layer outputs |
| Two key MLPs | `dust3r/model.py::Endo3R.encode_feat_key/set_attn_head` (591–602, 640–644) | Concatenation `[B,P,1792]` | `[B,P,1024]` each | 1792→1792→1024 with GELU; no attention operation |
| Dense heads | `dust3r/model.py::Endo3R.downstream_head` (668–672); `dust3r/heads/dpt_head.py::PixelwiseTaskWithDPT.forward` (244–253) | Decoder layer lists | Points `[B,H,W,3]`, confidence `[B,H,W]` | Optional pose `[B,7]` depends on checkpoint head configuration |
| Pointmap patch embedding | `dust3r/model.py::Endo3R.encode_cur_value` (653–661) | Points permuted `[B,3,H,W]` | `[B,P,1024]` | Default `use_feat=False`; independent copy of image patch embed |
| Value encoding | `dust3r/model.py::Endo3R.encode_value` (646–651) | `[B,P,1024]` | `[B,P,1024]` | Six transformer blocks, LayerNorm, output linear; `use_feat=True` input dimension 768 |
| Actual stored value | `dust3r/model.py::Endo3R.forward` (782–801) | `cur_v`, `feat_k1`, both `[B,P,1024]` | `cur_v+feat_k1 [B,P,1024]` | Written with current left key |
| Temporal append | `dust3r/model.py::SpatialTempralMemory.add_temp_mem/add_mem_check` (341–351, 406–413) | One frame K/V `[B,P,1024]` | `[B,TP,1024]` | Append then retain last `4P` tokens |
| Similarity | `dust3r/model.py::SpatialTempralMemory.check_sim` (378–399) | Current `[B,P,C]`, spatial tail `[B,wm*P,C]` | Correlation `[B,wm,P]`, means `[B,wm]`, one Boolean | Corresponding patch positions only, not all-pairs matching |
| Optical flow | `dust3r/model.py::SpatialTempralMemory.uncertainty_map` (444–453) | Cropped RGB pair `[B,3,H,W]`, padded to /8 | Flow `[B,H,W,2]` | Current→reference, 20 iterations |
| Uncertainty | `dust3r/utils/geometry.py::detect_dynamic_regions_3d_distance` (16–67) | Two points `[B,H,W,3]`, flow `[B,H,W,2]` | `[B,H,W]` | 3D disagreement after flow sampling; not Sampson distance |
| Spatial metadata pool | `dust3r/model.py::SpatialTempralMemory.add_mem_c/add_mem_uncertainty` (308–339) | Confidence/uncertainty `[B,H,W]` | Per write `[1,1,P]` when `B=1` | Confidence accumulates along frame axis; uncertainty along token axis |
| Spatial append/prune | `dust3r/model.py::SpatialTempralMemory.add_spatial_mem/memory_prune` (354–375, 511–537) | Existing S tokens + P tokens | On prune: K/V `[B,20000,1024]`, age/attention `[B,20000,1]`, U `[1,1,20000]` | Removes individual tokens, not complete frames |
| Focal/intrinsics | `dust3r/post_process.py::estimate_focal_knowing_depth` (12–60); `demo.py::main` (152–159) | First points `[1,H,W,3]`, center `[2]` | Focal `[1]`, intrinsic `[3,3]` | Ten Weiszfeld iterations; one focal shared across sequence |
| PnP and depth | `demo.py::main` (176–199) | Flattened global points `[HW,3]`, image grid `[HW,2]` | W2C/C2W `[4,4]`, depth `[1,H,W]` | All pixels passed to PnP; confidence/valid mask not used there |
| Final reconstruction | `demo.py::main` (211–227) | Per-frame arrays | Points/images `[N,H,W,3]`, confidence `[N,H,W]`, poses `[N,4,4]`, depths `[N,1,H,W]` | `result.npy` contains these arrays plus intrinsic |

**VERIFIED FROM CODE — resolution:** `--resolution 224` gives 224×224 (`P=196`); 320 gives width 320/height 256 (`P=320`); 512 gives width 512/height 384 (`P=768`). Other requested resolution values fall back to 224. Source: `demo.py::get_args_parser/main` (~47, 98–105). These P values assume the expected patch size 16; pooling and pose code independently hardcode 16.

# 4. Memory Lifecycle

The actual class name is **`SpatialTempralMemory`**, not `SpatialTemporalMemory`. It is a plain Python class, not an `nn.Module`. Source: `dust3r/model.py::SpatialTempralMemory` (~242–549).

## Short-term temporal memory

**VERIFIED FROM CODE:** every successfully processed candidate left frame is appended to `temporal_buffer` and `temp_mem_k/v` before any similarity decision. It retains the latest four candidates by tensor/list slicing. A rejected spatial candidate remains in temporal memory. The final right frame has no candidate update; therefore “every input frame is inserted” is false literally. The training branch does not use this working update path. Sources: `dust3r/model.py::SpatialTempralMemory.add_mem_check` (~406–416); `Endo3R.forward` (~748–752, 784–801, 816).

Temporal entries contain K and `cur_v+K`. Normal `forward` does not pass `pts_cur` or `img_cur`, so optional point/image buffers stay `None`. Temporal confidence is never populated. Source: `dust3r/model.py::Endo3R.forward` (~782–801); `SpatialTempralMemory.add_temp_mem/init_mem` (~265–286, 341–351).

## Spatial/long memory

**VERIFIED FROM CODE — acceptance:** after temporal insertion, reject spatial write if `check_sim(feat_k, thresh=0.92)` is true. Otherwise write immediately to spatial memory. With uncertainty enabled, frame 0 receives zero uncertainty; later accepted candidates receive computed uncertainty. There is no uncertainty cutoff or confidence cutoff for acceptance. Sources: `dust3r/model.py::SpatialTempralMemory.add_mem_check/check_sim` (~378–440).

For current key patch `p` and accepted recent spatial frame `t`, the redundancy signal is:

```text
r[b,t] = mean_p cosine(current_key[b,p], spatial_tail_key[b,t,p])
skip spatial write iff max over ALL b,t of r[b,t] > 0.92
```

**VERIFIED FROM CODE:** the comparison uses at most the last `wm*P` spatial tokens, reshaped as `wm` complete frames; `wm <= 4`. It does not compare against the whole long history, and compares corresponding patch positions only. `mem_k is None` or a threshold exactly 1.0 bypasses the check. Sources: `dust3r/model.py::SpatialTempralMemory.check_sim` (~380–399).

**INFERRED:** the filter can suppress recent redundancy before pruning, but neither guarantees global uniqueness nor compares motion-aligned content. After token pruning, the assumed “recent frames with matching patch order” layout is no longer established. This matters to both the baseline and a future controller. Sources: `dust3r/model.py::SpatialTempralMemory.check_sim/memory_prune` (~383–394, 519–528).

## Variable inventory

| Variable | Actual role, unit, and lifecycle | Code citation |
|---|---|---|
| `temporal_buffer` | Candidate frame indices; append every left candidate; retain last 4 | `dust3r/model.py::init_mem/add_mem_check` (266, 406–413) |
| `spatial_buffer` | All accepted spatial frame indices; append; never pruned; last index chooses uncertainty reference | `dust3r/model.py::init_mem/add_mem_check/Endo3R.forward` (267, 418, 427, 790–793) |
| `temp_mem_k`, `temp_mem_v` | Separate temporal tensors `[B,TP,1024]`; append and trim | `dust3r/model.py::add_temp_mem/add_mem_check` (341–348, 411–412) |
| `spat_mem_k`, `spat_mem_v`, `spat_mem_c` | Initialized `None`; never assigned or read afterwards | `dust3r/model.py::init_mem` (274–276); repository identifier search |
| `mem_k`, `mem_v` | Actual spatial tensors `[B,S,1024]`, including newest admitted frames and older tokens | `dust3r/model.py::add_spatial_mem/memory_prune` (366–367, 523–524) |
| `temp_mem_c` | Initialized `None`; unused | `dust3r/model.py::init_mem` (273); identifier search |
| `mem_c` | Average-pooled prediction confidence `[1,F,P]`, F accepted writes; unused by attention/pruning; never trimmed | `dust3r/model.py::add_mem_c/add_spatial_mem` (308–315, 372) |
| `mem_uncert` | Pooled uncertainty `[1,1,S]`; append token axis; gather on prune; read for logit scaling | `dust3r/model.py::add_mem_uncertainty/memory_read/memory_prune` (333–339, 473–482, 527–528) |
| `mem_count` | Per-token age in subsequent accepted spatial writes `[B,S,1]`; new tokens start at zero | `dust3r/model.py::add_spatial_mem/memory_prune` (358–364, 513–526) |
| `mem_attn` | Zero per-token importance tensor; append zeros; accumulation commented; gather on prune | `dust3r/model.py::add_spatial_mem/memory_read/memory_prune` (360, 364, 504–507, 513, 525) |
| `mem_pts`, `mem_imgs` | Optional payloads; default online path leaves them `None`; conditional gather branch is suspicious | `dust3r/model.py::init_mem/add_spatial_mem/memory_prune` (281–286, 368–369, 530–533) |
| `temp_mem_pts`, `temp_mem_imgs` | Optional temporal payloads; append if supplied but never shortened with temporal K/V | `dust3r/model.py::add_temp_mem/add_mem_check` (347–348, 409–413) |
| `spat_mem_pts`, `spat_mem_imgs` | Initialized only; dead aliases | `dust3r/model.py::init_mem` (283, 286); identifier search |
| `wm` | Count of recent accepted spatial frames; saturates at 4; used to slice similarity tail | `dust3r/model.py::check_sim/add_mem_check` (383, 432–435) |
| `lm` | Token count attributed to old spatial memory; increments by P after wm is full | `dust3r/model.py::add_mem_check` (435, 439–441) |
| `num_patches` | P tokens/frame, captured from first key; implicitly assumes fixed sequence shape | `dust3r/model.py::add_mem_check/add_temp_mem` (404–405, 342–343) |
| `long_mem_size` | 20,000 old-token trigger threshold; not a frame count and not a strict total spatial cap | `dust3r/model.py::__init__/add_mem_check` (244, 252, 439) |
| `short_mem_size` | 4 frames; separately bounds temporal list and wm; also participates in pruning age cutoff | `dust3r/model.py::__init__/add_mem_check/memory_prune` (244, 253, 409–413, 433, 514) |
| `top_k` | 20,000 total spatial tokens retained after pruning, including recent tokens | `dust3r/model.py::__init__/memory_prune` (254, 519) |
| `sim_thresh`, `attn_thresh` | 0.92 frame similarity gate; 0.001 post-softmax probability cutoff in eval | `dust3r/model.py::__init__/add_mem_check/memory_read` (245–256, 414, 494–496) |
| `interval` | Set to `max(10,N//10)` in eval, but its only proposed condition is commented out | `dust3r/model.py::Endo3R.forward/add_mem_check` (746–747, 414) |
| `save_attn`, `attn_vis` | Optional pre-threshold attention recording; flatten/append without bound; separate from mem_attn | `dust3r/model.py::__init__/init_mem/memory_read` (246, 289–290, 486–490) |

The inventory is **VERIFIED FROM CODE**. “Spatial” and “long” refer to `mem_k/v`, not the unused `spat_mem_*` fields.

## Age, importance, and pruning

**VERIFIED FROM CODE:** on the first spatial write, all ages are zero. Each subsequent accepted spatial write increments every retained old token by 1, then appends P tokens with age 0. Skipped candidates and reads do not increment age. Thus ages are accepted-write counts, not elapsed frames, wall time, or read counts. Source: `dust3r/model.py::SpatialTempralMemory.add_spatial_mem` (~358–364).

**VERIFIED FROM CODE:** importance is never accumulated. The complete `mem_attn` write inventory is initialization to zero, concatenation of new zeros, and gathering already zero values at prune. The intended `self.mem_attn += total_attn[..., None]` is commented at line 507. `total_attn` is computed every read and discarded. Sources: `dust3r/model.py::SpatialTempralMemory.add_spatial_mem/memory_read/memory_prune` (~358–364, 504–507, 513–526).

Pruning does:

```text
weights = mem_attn / mem_count
weights[mem_count < 4 + 5] = 1e8
indices = topk(weights, 20000, token axis)
gather mem_k, mem_v, mem_attn, mem_count, and mem_uncert using indices
lm = top_k - wm*P
```

**POTENTIAL BUG:** `0/0` occurs for new tokens in the first expression; the subsequent age mask overwrites their NaNs under default finite ages. Older tokens have `0/positive_age = 0`. Scores therefore encode only a nine-write recency preference, followed by arbitrary ties, rather than usage importance. Top-k tie order is not a FIFO guarantee. If more than the budget share the protected score, even recent tokens can be dropped. Sources: `dust3r/model.py::SpatialTempralMemory.memory_prune` (~513–526). A controlled toy probe confirmed zero importance after reads and removal of newest entries under tied protected scores; section 7 documents its scope.

**VERIFIED FROM CODE:** pruning removes individual spatial key/value tokens and matching age/attention/uncertainty entries. It does not remove temporal K/V, `spatial_buffer` indices, or `mem_c`. Optional point/image gathering is attempted only if `mem_pts` exists. Source: `dust3r/model.py::SpatialTempralMemory.memory_prune` (~523–533).

**POTENTIAL BUG:** `torch.topk` gathers tokens in score order, without restoring chronological/patch order. `check_sim` nevertheless interprets the last `4P` tokens as recent whole frames. `top_k % P != 0` is not itself a reshape error here because the similarity check slices exactly `wm*P` first; the problem is semantic misalignment. `wm/lm` remain logical counters after physical frame boundaries are destroyed. Sources: `dust3r/model.py::SpatialTempralMemory.memory_prune/check_sim` (~519–528, 383–391).

**VERIFIED FROM CODE:** temporal and spatial K/V are physically separate, but an admitted recent frame is in both. There is no transfer of temporal eviction into spatial storage: spatial insertion happens immediately after the similarity gate; temporal trimming simply drops its oldest K/V. The spatial tensor internally has a logical recent/old split before pruning. Source: `dust3r/model.py::SpatialTempralMemory.add_mem_check` (~406–441).

# 5. Current Memory Attention

**VERIFIED FROM CODE — equation.** Let the input `F` be the carried query key, `Q=LN_q(F)`, `K=LN_k(concat(K_temp,K_spatial))`, and `V=LN_v(concat(V_temp,V_spatial))`. With uncertainty disabled/not populated, K/V contain only spatial tokens. These learned LayerNorms have dimension C=1024. Sources: `dust3r/model.py::Endo3R.set_memory_encoder` (~585–589); `SpatialTempralMemory.memory_read` (~464–498).

```math
S_{b,p,m}=\frac{\sum_{c=1}^{C} Q_{b,p,c}K_{b,m,c}}{\sqrt C}
```

If stored uncertainty has a positive maximum, pad its temporal prefix with zeros and use:

```math
g_m=\operatorname{clamp}(\exp(-20U_m),0.1,1),\qquad S'_{b,p,m}=g_m S_{b,p,m}.
```

Otherwise `S'=S`. Then:

```math
A=\operatorname{softmax}_{m}(S'),\qquad A_d=\operatorname{Dropout}(A)\text{ if supplied}.
```

For eval threshold `tau=0.001`:

```math
\widetilde A_{b,p,m}=A_{d,b,p,m}\,\mathbf1[A_{d,b,p,m}\ge\tau],\qquad
\widehat A=\frac{\widetilde A}{\sum_m\widetilde A},\qquad
O_{b,p,c}=\sum_m\widehat A_{b,p,m} V_{b,m,c}+F_{b,p,c}.
```

No threshold means `A_d` is used directly; `res=False` omits the residual. This is one full-channel scaled dot-product attention, with additional logit weighting/thresholding, not a learned multihead QKV layer. Sources: `dust3r/model.py::SpatialTempralMemory.memory_read` (~470–509).

| Tensor | Shape / token meaning |
|---|---|
| Query | `[B,P,1024]`, P current reference-frame patch queries |
| K / V | `[B,M,1024]`, M temporal+spatial tokens or spatial-only tokens |
| Logits / probabilities | `[B,P,M]`, explicitly materialized dense matrices |
| Uncertainty | `[1,1,M]` in supported B=1 path; temporal entries zero |
| Return | `[B,P,1024]`, query residual added by default |

**VERIFIED FROM CODE:** eval memory has no dropout object; training constructs memory with dropout 0.15 and threshold zero. `model.eval()` controls this construction; the `forward(eval=...)` keyword is unused. Sources: `dust3r/model.py::Endo3R.__init__/set_memory_encoder/forward` (~555, 589, 735–739).

**POTENTIAL BUG:** thresholding has no epsilon or zero-row fallback. For uniform attention and M>1000 every probability is <0.001, leaving a zero denominator. A tiny tensor probe reproduced nonfinite renormalization; occurrence with trained features is **UNKNOWN / NEEDS RUNTIME VERIFICATION**. Source: `dust3r/model.py::SpatialTempralMemory.memory_read` (~494–498).

**INFERRED:** uncertainty multiplication is not a guaranteed probability penalty. Multiplying a negative logit by a number below one increases that logit toward zero and can increase its probability. Temporal zero uncertainty also exempts duplicate recent temporal tokens from the spatial penalty. Source: `dust3r/model.py::SpatialTempralMemory.memory_read` (~474–484).

**INFERRED — complexity:** QK and AV each require `O(BPMC)` time; approximately `2BPMC` multiply-accumulates / `4BPMC` FLOPs combined. Attention-matrix memory is `O(BPM)`. K/V storage, their concatenation/normalization, and LayerNorm work add `O(BMC)` storage/work. Post-softmax zeroing does not save dense QK/AV arithmetic: both einsums still use the full shapes. Source: `dust3r/model.py::SpatialTempralMemory.memory_read` (~465–498).

## Exact default budget accounting

**VERIFIED FROM CODE / INFERRED BOUND:** let L=`long_mem_size=20000`, s=`short_mem_size=4`, K=`top_k=20000`:

- Before first prune, after F accepted writes: `wm=min(F,s)`, `lm=max(F-s,0)*P`, `S=F*P`.
- Prune occurs when `lm>L`; it retains **K total spatial tokens**, then sets `lm=K-wm*P`.
- Later completed updates can have `S <= L+sP`; temporal contributes at most sP, so **`M <= L+2sP = 20000+8P`**.
- Immediately before pruning, the newly appended spatial frame can temporarily raise spatial tokens to `L+(s+1)P`; with temporal tokens, conceptual total is bounded by `L+(2s+1)P`. The read itself happens before that current update.

Sources: `dust3r/model.py::SpatialTempralMemory.__init__/add_mem_check/memory_prune` (~244–254, 432–441, 519–526). Actual sequence occupancy depends on accepted writes and successful execution, not input length alone.

| Resolution H×W | P | Max completed-state M | One FP32 `[1,P,M]` matrix | K+V at M, FP32 | QK+AV FLOPs |
|---|---:|---:|---:|---:|---:|
| 224×224 | 196 | 21,568 | 16.126 MiB | 168.5 MiB | 17.315 G |
| 256×320 | 320 | 22,560 | 27.539 MiB | 176.25 MiB | 29.570 G |
| 384×512 | 768 | 26,144 | 76.594 MiB | 204.25 MiB | 82.242 G |

These are **STATIC ESTIMATE ONLY**, B=1/C=1024/FP32, not peak allocation measurements. Several logits/probabilities/copies may coexist. The M-column includes temporal/spatial duplication. First-prune accepted-write counts are 107, 67, and 31 respectively; this needs at least one additional input frame because only left frames are written. Source arithmetic: `floor(L/P)+s+1` from `add_mem_check` (~432–440).

# 6. Attention Inventory

**UNKNOWN / NEEDS RUNTIME VERIFICATION:** actual encoder/decoder layer counts, head counts, head mode, RoPE mode, and patch size are defined by `ckpt['args'].model`, evaluated in `dust3r/model.py::load_model` (~53–66). The expected checkpoint is absent. The filename is not sufficient evidence. Endo3R hardcoded interfaces constrain channels to 1024/768; DPT requires `dec_depth>9`. Sources: `dust3r/model.py::Endo3R.set_memory_encoder/set_attn_head` (~565–601); `dust3r/heads/dpt_head.py::create_dpt_head` (~256–277).

| Component | Type | Tokens | Dim | Layers / heads | Complexity | Likely Cost |
|---|---|---|---|---|---|---|
| Image encoder | Pre-LN self-attention + MLP | P, no CLS token | 1024 expected | `L_e/H_e` checkpoint-defined | `O(L_e BP²C_e)` attention; projections/MLPs `O(L_e BPC_e²)` | Large fixed frame cost |
| Decoder self-attention | Pre-LN self-attention on each stream | P each | 768 expected | `2L_d` calls / `H_d` heads | `O(2L_d BP²C_d)` plus projections | Large aggregate cost |
| Decoder cross-attention | Learned Q/K/V; other stream normalized | Q=P, K/V=P | 768 expected | `2L_d` calls / `H_d` heads | `O(2L_d BP²C_d)` plus projections | Large aggregate cost |
| Value encoder | Pre-LN self-attention + MLP | P pointmap patches | 1024 default; 768 with use_feat | **6 layers / 16 heads verified** | `O(6BP²C_v)` plus `O(6BPC_v²)` | Material; runs before spatial rejection |
| Memory read | Full-channel cross-attention | Q=P, K/V=M | 1024 | 1 read / effectively 1 head | `O(BPMC)`; matrix `O(BPM)` | Grows with accepted memory |
| Key “attn heads” | MLP, **not attention** | P | 1792→1792→1024 | Two MLPs | `O(BP(1792²+1792*1024))` each | Potential sizable non-attention cost |
| DPT / optional pose head | Convolution/refinement; no transformer attention | Dense spatial maps | DPT feature 256, last 128 | Head checkpoint-defined | Spatial convolution cost | Must measure separately |
| RAFT | All-pairs feature correlation + recurrent conv updates; not memory attention | n=(H/8)(W/8) | Feature 256 | 20 updates, 4 correlation levels | Correlation `O(Bn²*256)` plus recurrent work | Could be major conditional cost |

Inventory citations: `dust3r/model.py::AsymmetricCroCo3DStereo._encode_image/_decoder` (~153–167, 200–220); `Endo3R.set_memory_encoder/encode_cur_value/set_attn_head` (~569–602, 653–661); `croco/models/blocks.py::Attention/Block/CrossAttention/DecoderBlock` (~81–191); `dust3r/heads/dpt_head.py::DPTOutputAdapter_fix/PoseDecoder/create_dpt_head` (~71–164, 256–277); `dust3r/core/corr.py::CorrBlock` (~12–60); `dust3r/core/raft.py::RAFT.forward` (~98–142).

**VERIFIED FROM CODE:** reconstruction attention uses explicit matmul/einsum → softmax → dense matmul/einsum. Repository searches found no FlashAttention, SDPA, or xFormers calls. Transformer self/cross matrices are `[B,heads,P,P]`. The cuRoPE extension applies rotary position transforms; a PyTorch fallback exists. Sources: `croco/models/blocks.py::Attention.forward/CrossAttention.forward` (~94–111, 149–168); `dust3r/model.py::SpatialTempralMemory.memory_read` (~470–498); `croco/models/pos_embed.py::RoPE2D` (~106–159); `croco/models/curope/curope2d.py::cuRoPE2D` (~12–40).

## Conditional cost ranking — STATIC ESTIMATE ONLY

`croco/README.MD` (~115) gives an example with encoder 24 layers/16 heads at 1024 and decoder 12 layers/12 heads at 768. This is **PAPER INTENT / COMMENT ONLY / REFERENCE CONFIGURATION**, not verified metadata for the absent Endo3R checkpoint. Generic `CroCoNet` defaults are different and incompatible with Endo3R's hardcoded key dimensions. Source: `croco/models/croco.py::CroCoNet.__init__` (~23–35).

Under that conditional 24/12-layer example, one steady pair step has 24 encoder attention calls, 48 decoder self+cross calls, six value calls, and one memory read. QK+AV MACs are approximately encoder `48P²*1024`, decoder `96P²*768`, value encoder `12P²*1024`, and memory `2PM*1024`. Thus memory/decoder dot-product cost is `M/(36P)`.

| Regime | Biggest attention dot-product estimate | Second | Smaller components |
|---|---|---|---|
| Small memory, typical initial sequence | Dual decoder aggregate | Image encoder | Value encoder / retrieval, depending on M |
| Near max memory at P=196 or 320 | Memory read (about 3.06× / 1.96× decoder dot MACs) | Dual decoder | Image encoder, then value encoder |
| Near max memory at P=768 | Dual decoder, narrowly above retrieval | Memory read (~0.946× decoder) | Image encoder, then value encoder |

**INFERRED:** full-block latency can rank differently because projections/MLPs are `O(PC²)`, and dtype, bandwidth, launch overhead, and RAFT matter. There is no profiling evidence that memory read is the overall largest bottleneck. Sources: attention implementations and dimensions cited above.

# 7. Runtime / Profiling Results

**NOT PROFILED — no baseline sequence inference was run.** The local environment is partially available but does not meet baseline requirements:

| Read-only observation | Result |
|---|---|
| Python | 3.14.3, `C:\Program Files\Python314\python.exe` |
| PyTorch | 2.9.0+cu128; compiled CUDA 12.8; `torch.cuda.is_available()` true; one visible GPU |
| Hardware | NVIDIA GeForce RTX 2050, 4096 MiB; driver 592.82; `nvidia-smi` reports CUDA capability 13.1 |
| Checkpoints | No `.pth`, `.pt`, or `.ckpt` files; no checkpoint directory in clone |
| Example inputs | 66 Hamlyn RGB frames and 30 SCARED RGB frames; no associated full GT datasets |
| Missing modules | `einops`, `roma`, `timm`, `viser`, `evo`, `kornia`; optional/auxiliary `pillow_heif`, `sam2`, `third_party` also absent |
| Actual import check | `python -B -c "import dust3r.model"` fails with `ModuleNotFoundError: No module named 'evo'` through `utils.misc -> utils.vo_eval` |
| Syntax check | All 97 repository Python files parse with `ast.parse`; this does not establish import or runtime correctness |
| Inference latency / FPS / peak GPU allocation | **UNKNOWN / NEEDS RUNTIME VERIFICATION**, no measurements |

Observation evidence is the command record below; code requirements are `dust3r/model.py::Endo3R.__init__/SpatialTempralMemory.__init__` (~552–562, 259–260), `dust3r/utils/misc.py` (~7), `dust3r/utils/vo_eval.py` (~6), `requirements.txt` (~1–15), and `README.md` (~41–52). No installation or large download was attempted. The 4 GiB GPU may impose further constraints, but fit is unmeasured.

## Bounded algebra/lifecycle probes actually executed

These are **not Endo3R inference, profiling, or reconstruction quality tests**. Source class/function ASTs were compiled in an in-memory namespace; no repository imports or writes were needed. RAFT construction was bypassed, identity normalization used, and a toy P=2/C=3/budget=4/short=1 memory was created expressly to isolate bookkeeping.

| Probe | Observed result | Audited code |
|---|---|---|
| Five spatial writes with reads between writes | All mem_attn entries stayed zero; ages incremented only on writes | `dust3r/model.py::add_spatial_mem/memory_read` (358–364, 504–507) |
| Toy prune of 8 tokens to 4, tied protected weights | Retained older age-3/age-2 tokens; newest tokens were removed in this run | `dust3r/model.py::memory_prune` (513–526); tie order is not a guaranteed ordering |
| Uniform softmax with 1200 keys, cutoff 0.001 | Every weight below threshold; zero-sum renormalization nonfinite | `dust3r/model.py::memory_read` (494–496) |
| Extracted uncertainty function, zero flow | Finite zero uncertainty | `dust3r/utils/geometry.py::detect_dynamic_regions_3d_distance` (16–67) |
| Same function, every correspondence out of bounds | All uncertainty values NaN | Same function (58–64) |
| CUDA tensor divided by CPU scalar tensor sqrt(1024) | Works locally, result 0.03125 | `dust3r/model.py::memory_read` (471); CPU scalar is not a demonstrated device bug |
| Installed DataParallel constructor inspection | Single-visible-device case calls `self.module.to(self.src_device_obj)` | Installed `torch/nn/parallel/data_parallel.py::DataParallel.__init__` (~168–169) |

The last check prevents an overclaim: the absence of an explicit `.cuda()` on RAFT in repository code does **not** prove CPU placement on this single-GPU environment.

## Profiling script PLAN only — no script/module created

Once compatible imports and all three checkpoints are available, use an external or in-memory harness, run from repository root, with `python -B`, and invoke `model.forward` directly. Do not invoke `demo.main`: it creates directories before inference. Load/crop a bounded frame subset without output saving. Record checkpoint metadata and load-state incompatibilities first. Sources: `demo.py::main` (~88–108); `dust3r/model.py::load_model/Endo3R.forward` (~53–69, 735–823).

1. Measure load time separately, including the RAFT checkpoint load in every memory construction. Keep the baseline model modes/devices intact; report a blocker instead of silently changing them.
2. Use the original `model.eval()` plus `torch.no_grad()` behavior. For each GPU timed region call `torch.cuda.synchronize(device)` before the start and after completion; use CUDA events or synchronized `perf_counter`. Also measure complete-step wall time to expose CPU synchronization and data transfers.
3. Wrap these exact boundaries in the external harness:

| Required breakdown | Timing boundary |
|---|---|
| Image encoding | `dust3r/model.py::Endo3R.encode_image_pairs/encode_image` (613–628, 604–611) |
| Decoder | `dust3r/model.py::Endo3R.decode` (663–666) |
| Memory read | `dust3r/model.py::SpatialTempralMemory.memory_read` (456–509) |
| Key encoding | `dust3r/model.py::Endo3R.encode_feat_key` (640–644), both calls |
| Value encoding / pointmap patch embedding | `dust3r/model.py::Endo3R.encode_cur_value/encode_value` (646–661) |
| Prediction heads | `dust3r/model.py::Endo3R.downstream_head` (668–672), both calls |
| Similarity | `dust3r/model.py::SpatialTempralMemory.check_sim` (378–399) |
| RAFT / optical flow | `dust3r/core/raft.py::RAFT.forward` (86–144) called from uncertainty_map |
| Uncertainty geometry and pooling | `dust3r/utils/geometry.py::detect_dynamic_regions_3d_distance` (16–67); memory pooling (308–339) |
| Memory update/prune | `dust3r/model.py::SpatialTempralMemory.add_mem_check/memory_prune` (402–440, 511–537), subtract nested similarity/RAFT/uncertainty times for exclusive costs |
| Focal/pose/depth | Reproduce `demo.py::main` (152–199) in memory: time CPU focal, PnP, transform separately |

4. Use a short eight-frame subset and a longer up-to-66-frame existing Hamlyn sequence at the same resolution/stride. If later data are available, use a bounded real sequence long enough to cross pruning (at least 108/68/32 inputs for P196/320/768 **if every candidate is accepted**). Do not repeat frames to present synthetic growth as a real sequence result.
5. Warm up a separate small call, then reset measured memory state. Run a few bounded repeats, report cold initialization separately and per-step p50/p95/mean. Count N input frames, N−1 pair steps, and N−2 reads explicitly; do not equate pair-step FPS with capture FPS without specifying stride/lookahead.
6. Reset CUDA peak stats after loading. Report peak allocated **and** reserved memory, resident model/input/output baseline, and per-step allocations. Keep outputs for the faithful baseline; any output-offload diagnostic is a separately labeled experiment.
7. At each read log P, temporal tokens, spatial tokens, M, wm/lm, accepted-write count, RAFT-call flag, prune flag, zero attention rows, and nonfinite counts. Plot read latency against M, not sequence index alone. Compare RAFT conditional-call latency and amortized per-frame contribution against retrieval in the same sequence.

**UNKNOWN / NEEDS RUNTIME VERIFICATION:** whether read latency measurably increases on this workload and whether RAFT exceeds memory read. Code establishes dense `O(PMC)` scaling, not their actual relative latency.

# 8. RAFT / Uncertainty Audit

| Question | VERIFIED FROM CODE answer | Citation |
|---|---|---|
| Loaded in normal inference? | Yes, whenever a new memory object is constructed, even if uncertainty is disabled or no RAFT call follows | `dust3r/model.py::SpatialTempralMemory.__init__/Endo3R.forward` (259–263, 735–739) |
| Every frame? | No. Runs only for spatial candidates surviving similarity, with uncertainty enabled and index>0 | `dust3r/model.py::add_mem_check` (414–424) |
| When does it start? | Earliest candidate index i=1, the second input frame as the left member of the second pair; requires a third input frame | `dust3r/model.py::Endo3R.forward/add_mem_check` (748–752, 789–801, 419–420) |
| Reference selection? | Last accepted spatial-buffer index, otherwise i−1; not necessarily the previous chronological frame | `dust3r/model.py::Endo3R.forward` (790–799) |
| RAFT version/settings? | `dust3r.core.raft.RAFT`, large/basic model, small=False, mixed_precision=False, alternate_corr=False | `dust3r/model.py::create_raft_args` (539–547); `dust3r/core/raft.py::RAFT.__init__` (29–56) |
| Iterations? | 20 explicitly; RAFT's own default 12 is overridden | `dust3r/model.py::uncertainty_map` (450); `dust3r/core/raft.py::forward` (86) |
| Image resolution/range? | Cropped `origin_img` at inference H,W, not original file resolution; pad to multiples of 8; uint8 RGB scaled inside RAFT to [-1,1] | `dust3r/datasets/demo.py::_get_views` (86–91); `dust3r/model.py::InputPadder/uncertainty_map` (35–50, 445–451); `dust3r/core/raft.py::forward` (89–93) |
| Frozen? | Parameters require_grad=False; memory is not an nn.Module registered under Endo3R | `dust3r/model.py::SpatialTempralMemory.__init__` (242–262) |
| Eval mode? | No `.eval()` or `freeze_bn()` call on newly constructed RAFT; Endo3R.eval() cannot propagate to this future plain-object member | `dust3r/model.py::SpatialTempralMemory.__init__/Endo3R.forward` (259–263, 735–739); `dust3r/core/raft.py::__init__/freeze_bn` (55, 58–61) |
| Gradients? | Demo wraps all inference with no_grad. Frozen RAFT on ordinary image tensors needs no trainable-parameter gradients; uncertainty geometry has no internal no_grad guard for other callers | `demo.py::main` (85, 137–138); `dust3r/model.py::uncertainty_map` (444–453) |
| Cache? | No cross-call flow or RAFT feature cache. Pooled uncertainty is stored once per accepted write and reused for later reads | `dust3r/model.py::uncertainty_map/add_mem_uncertainty/memory_read` (333–339, 444–453, 473–482) |
| Recomputed unnecessarily? | Existing uncertainty is not recomputed each read. RAFT weights reload each forward, references can be re-encoded on multiple calls, and all 20 upsampled flow outputs are retained despite test_mode using only the final one | `dust3r/model.py::__init__` (259–260); `dust3r/core/raft.py::forward` (98–111, 121–142) |

**VERIFIED FROM CODE — active path:**

```text
RAFT(current cropped RGB, last accepted reference RGB)
 -> current-to-reference pixel flow
 -> sample reference global pointmap at current_pixel + flow
 -> Euclidean distance from current global pointmap
 -> min/max normalization over valid correspondences
 -> keep only pixels whose flow norm exceeds the image/batch mean
 -> uncertainty [B,H,W]
 -> 16x16 average pool -> spatial mem_uncert
 -> future exp(-20U), clamp[0.1,1] -> multiply attention logits
```

Sources: `dust3r/model.py::SpatialTempralMemory.uncertainty_map/add_mem_uncertainty/memory_read` (~333–339, 450–452, 474–482); `dust3r/utils/geometry.py::detect_dynamic_regions_3d_distance` (~30–67).

**VERIFIED FROM CODE:** prediction confidence `res1['conf']` is separately pooled into `mem_c` but never consumed. The exponential quantity named `confidence` in `memory_read` comes from uncertainty, not from `mem_c`. No uncertainty/confidence statistic rejects a spatial frame or removes uncertain tokens. Sources: `dust3r/model.py::SpatialTempralMemory.add_spatial_mem/add_mem_check/memory_read` (~371–373, 417–428, 480–482).

**POTENTIAL BUG:** out-of-bounds correspondences receive default occlusion uncertainty 0, rather than high uncertainty. If every correspondence is invalid, `min_distance=inf`, normalization produces NaNs, and multiplying by a zero mask does not recover finite values. The `threshold=0.1` argument and `valid_distance` temporary are unused. A NaN maximum can also make `if mem_uncert.max()>0` false and bypass weighting entirely. Sources: `dust3r/utils/geometry.py::detect_dynamic_regions_3d_distance` (~16, 57–64); `dust3r/model.py::memory_read` (~478–482). All-invalid NaNs were reproduced in the isolated probe; prevalence on real video is unknown.

**POTENTIAL BUG — device portability:** images use unconditional `.cuda()`; RAFT is briefly DataParallel then immediately unwrapped. Local single-GPU DataParallel moves the module onto its source GPU, so CPU placement is not established locally. Multi-visible-GPU behavior, non-default `--device`, CPU inference, and checkpoint tensor placement need explicit verification. Sources: `dust3r/model.py::SpatialTempralMemory.__init__/uncertainty_map` (~259–262, 445–446); `demo.py::get_args_parser` (~44); installed DataParallel constructor noted in section 7.

**INFERRED — latency significance:** RAFT computes all-pairs correlations over n=HW/64 feature positions and performs 20 recurrent iterations on each accepted candidate. At 224/320/512 modes n is 784/1280/3072, respectively. Its basic correlation alone is `O(n²*256)` and stores an `O(n²)` pyramid. It is a credible major bottleneck, but “more expensive than memory attention” is **UNKNOWN / NEEDS RUNTIME VERIFICATION**. Sources: `dust3r/core/raft.py::RAFT.forward` (~98–139); `dust3r/core/corr.py::CorrBlock` (~18–27, 53–60).

# 9. Existing Heuristics

The table covers active reconstruction rules and relevant dead/inactive rules. Differentiability refers to the **decision**, not whether nearby tensor calculations support gradients. All locations and decisions below are **VERIFIED FROM CODE**; RL suitability and rationale are **INFERRED**.

| Heuristic | Location | Inputs | Decision | Future Effect | Differentiable | RL Candidate / rationale |
|---|---|---|---|---|---|---|
| Recent-key cosine gate, 0.92 | `dust3r/model.py::check_sim/add_mem_check` (378–416) | Current key, last wm*P spatial keys | If max mean patch cosine >0.92, skip spatial write | Changes future spatial keys/values, ages, uncertainty references, pruning cadence; avoids this candidate's RAFT | No | **Yes**, clean sequential write boundary |
| Four-frame temporal retention | `dust3r/model.py::add_mem_check` (409–413) | Candidate history length | Keep latest four temporal K/V frames | Bounds dynamic recent context; older rejected candidates disappear | No | **Yes**, retention/resource action is sequential; maintain shape/index correspondence |
| Spatial acceptance independent of uncertainty | `dust3r/model.py::add_mem_check` (417–428) | Similarity result, check_uncertainty flag | Every survivor written; uncertainty only attached | Accepted dynamics can remain in future bank | No | **Yes** for write decision; there is no existing uncertainty write cutoff to replace |
| Fixed long trigger 20k / top-k 20k | `dust3r/model.py::__init__/add_mem_check/memory_prune` (244–254, 439–441, 519) | lm token count, score tensor | Prune if lm>20k; keep 20k total spatial tokens | Controls context/memory cost; destroys frame layout | No | **Yes**, resource allocation, but current indexing defects raise risk |
| Nine-write recency protection | `dust3r/model.py::memory_prune` (513–514) | Accepted-write ages, zero importance | Set ages<9 to score 1e8 | Favors recent writes, ties elsewhere; not guaranteed retention | No | **Yes**, future memory survival decision; baseline scores currently broken |
| Probability threshold 0.001 | `dust3r/model.py::memory_read` (494–496) | Dense post-softmax attention | Drop small weights; renormalize | Alters fused reference and future keys/values | Hard threshold no | **Conditional**; presently saves no dense attention computation, with zero-row risk |
| Uncertainty flow-mask rule | `dust3r/utils/geometry.py::detect_dynamic_regions_3d_distance` (44–45, 64) | Flow norm vs global mean | Uncertainty only where norm>mean | Determines future logit weighting | No for mask | **Not a clean first action**; pixel rule, geometry semantics, not a compact resource gate |
| Fixed uncertainty decay/clamp | `dust3r/model.py::memory_read` (478–482) | Stored pooled uncertainty | exp(-20U), clamp[0.1,1] | Reweights all future reads | Mostly yes, except gate/clamp boundaries | **Low priority**; numeric weighting rather than an existing sequential keep/write action |
| Compute uncertainty flag | `dust3r/model.py::Endo3R.forward/add_mem_check` (735, 789–801, 417–428) | Flag, index, similarity survivor | Compute RAFT after first accepted candidate | Cost and future uncertainty; flag also changes temporal retrieval topology | No | **Yes**, expense gate if metadata/retrieval side effects are accounted for |
| Frame sampling interval | `dust3r/datasets/base_many_view_dataset.py::sample_frame_idx` (49–55); `demo.py::get_args_parser` (46) | Sorted input list, kf_every (default10) | Input `[::kf_every]` | Changes every later observation, motion, and reconstruction frame coverage | No | **Conditional**, external sampling action rather than memory-only action |
| Evaluation-only interval variable | `dust3r/model.py::Endo3R.forward/add_mem_check` (747, 414) | N and index gap | No active decision; interval condition commented | None in current path | N/A | **No existing rule to replace** |
| Parsed confidence threshold | `demo.py::get_args_parser/main` (45, 165–199) | conf_thresh option | Parsed but never applied | None in normal demo | N/A | **No existing active gate**; mem_c likewise unused |
| First-frame focal, shared intrinsics | `demo.py::main` (152–159); `dust3r/post_process.py::estimate_focal_knowing_depth` (36–58) | First points, image center | Ten robust iterations; share one focal | All later PnP/depth use same intrinsics | Estimator tensor math yes; fixed sharing not an action | **No for initial memory controller scope**, but affects metrics |
| PnP RANSAC settings | `demo.py::main` (176–196) | All global point/pixel correspondences | 100 iterations, reprojection error2, unfiltered points | Pose and local depth; no feedback into online memory | No | **Low priority**; downstream, not sequential memory selection |
| Offline next-view confidence | `dust3r/model.py::find_initial_pair/find_next_best_view` (674–733) | Mean (conf−1)/conf | Pick maximum pair/view | Would determine reconstruction order | Argmax no | **Inactive for baseline**; offline caller method is missing |

# 10. RL Insertion Candidates

These are **INFERRED** action boundaries supported by actual code. Ranking prioritizes implementation cleanliness, sequential effects, reconstruction influence, and compute opportunity. It does not choose an RL algorithm, policy, reward, or final architecture. No controller exists in the supplied repository.

| Rank / candidate | Exact insertion location | Already available state signals | Possible action choices | Future state and sequential nature | Overhead / implementation risk |
|---|---|---|---|---|---|
| **1. Spatial write gate** | `dust3r/model.py::SpatialTempralMemory.add_mem_check` (414–428), after temporal update and before uncertainty/append | Current K/V, similarity against recent spatial keys, pooled/current prediction confidence, index, temporal/spatial occupancy, wm/lm, ages, historical U. Current U is **not yet available** without RAFT | `WRITE_SPATIAL` / `TEMP_ONLY` | Determines future K/V, spatial reference index, age increments, prune timing, and retrieval; truly sequential across pair steps | Small action space; policy latency unknown. Reusing computed similarity avoids duplicate work. Saves conditional RAFT on skipped writes and future retrieval/storage; image/decoder/key/value work has already occurred. Risk: preserve initial memory availability and do not accidentally alter temporal write |
| **2. Retrieval extent** | `dust3r/model.py::Endo3R.forward` (758–767), around `sp_mem.memory_read`; implementation boundary `memory_read` (464–501) | Carried query key, bank K/V/U, occupancy, ages, previous confidence/decoding signals; current heads not yet computed | Full bank / reduced spatial subset / temporal-only where available | Changes current fused reference, subsequent pointmaps, future keys/values, and memory trajectory; sequential despite bank not being modified by the read | Potential direct O(PMC) reduction. Subset construction/state feature cost must be counted. Medium/high risk: joint softmax, duplicate tokens, empty bank/fallback, U alignment |
| **3. Expensive uncertainty trigger** | `dust3r/model.py::SpatialTempralMemory.add_mem_check` (417–424), immediately before `uncertainty_map` | Accepted candidate, current confidence/points, reference index/points, key novelty, historical U, cost budget | Run RAFT uncertainty / skip computation using an explicitly defined metadata alternative in a future design | Affects future spatial logit weights and future computation; sequential via retained metadata | Large conditional expense opportunity; controller overhead unmeasured. Medium/high risk: simply toggling existing global uncertainty_check drops temporal retrieval, and new U must retain token alignment. No trustworthy current uncertainty signal before the expensive call |
| **4. Retention at prune** | `dust3r/model.py::SpatialTempralMemory.add_mem_check/memory_prune` (439–441, 513–528) | Ages, K/V, U, occupancy; mem_attn present but unusable zero signal; mem_c lacks token alignment after prune | Keep/drop groups or select retention extent; precise granularity deferred | Defines surviving information for all later queries; strongest explicit resource-history decision | Controller can run only at prune boundaries; high risk: token/frame identity is already lost, topk is patch-level, buffers/confidence unsynchronized; lightweight action granularity needs evidence |

**INFERRED:** current-write skipping can reduce future work and avoid RAFT, but cannot remove that same pair's encoder/decoder/value encoder cost because the gate comes later. Moving a decision earlier is a separate design change, not a fact about the present insertion point. Source: `dust3r/model.py::Endo3R.forward` (~754–801).

State signals that require new computation, especially optical flow and fresh attention matrices, must not be described as free RL observations. Controller overhead, host/device synchronizations, quality effect, and compute savings are **UNKNOWN / NEEDS RUNTIME VERIFICATION**.

# 11. Efficient Attention Insertion Candidates

Ranked by code scope and potential saving versus risk, **STATIC ESTIMATE ONLY**. Exact kernel substitutions and approximate attention replacements have different risks: preserving the same operation/projections need not change pretrained weight shapes, while approximation changes learned behavior. No implementation or architecture is selected.

| Rank / target | Exact boundary and required interface | Weights affected | Dimensions | Expected saving / risk | Distillation feasibility |
|---|---|---|---|---|---|
| **1. Existing memory read operator** | `dust3r/model.py::SpatialTempralMemory.memory_read` (470–501); return fused `[B,P,1024]` with query residual | CroCo QKV/decoder weights untouched; preserve learned norm_q/k/v. Approximation changes downstream inputs | Q `[B,P,1024]`, K/V `[B,M,1024]` | Best isolated sequence-dependent target; saving grows with M. Medium/high geometry risk because future memory is generated from changed predictions | Teacher read output directly observable; no supplied trainer/loss |
| **2. Long/spatial portion only** | Same method, K/V concat (464–466), uncertainty prefix (474–482), joint softmax/threshold (484–496) | Can preserve CroCo and temporal tensors and memory norms | Spatial S up to L+4P, temporal <=4P | Largest variable bank portion. Fewer pretrained components touched, but **no separate long-attention module exists**; joint normalization/fusion must stay compatible. Savings/quality unknown | Paired teacher memory output feasible; teacher trajectories must be recorded |
| **3. Value encoder attention** | `dust3r/model.py::Endo3R.value_encoder/encode_value` (573–579, 646–651); preserve final `[B,P,1024]` | Endo3R learned value block weights; backbone untouched | Six layers, 16 heads; default C1024 | Fixed cost even when spatial write rejected; clean module boundary. Changes every future stored V; medium/high quality risk | Teacher layer/value outputs directly available |
| **4. Short-term portion only** | `dust3r/model.py::SpatialTempralMemory.memory_read` (464–496); preserve overall fusion `[B,P,1024]` | Backbone untouched; shared memory norms preserved | <=4P temporal tokens | Smaller bounded savings; dynamics-sensitive and shares spatial softmax; not an independent drop-in | Feasible joint teacher read targets; no separate temporal teacher output currently returned |
| **5. Decoder cross-attention (and separately self-attention)** | `croco/models/blocks.py::CrossAttention.forward/Attention.forward` (149–168, 94–111); preserve each output `[B,P,768]` and all decoder-layer outputs | Exact attention kernel may reuse Q/K/V/output weights; approximate mechanism changes pretrained cross-view computation | Q/K=P, C768, heads/layers checkpoint-defined; two streams | Potential large aggregate savings; direct geometric correspondence risk for approximation. DPT hook outputs must remain intact | Layer and final outputs available; absent Endo3R training release is a blocker |
| **6. Encoder self-attention** | `croco/models/blocks.py::Attention.forward` (94–111) called by `dust3r/model.py::_encode_image` (153–167); output `[B,P,1024]` | Exact kernel can preserve QKV/output/RoPE weights; approximation affects all incoming pretrained features | P fixed by resolution, C1024, heads/layers checkpoint-defined | Many layers; scope broad. Approximation propagates through keys, values, decoder, and outputs | Encoder layer features available; teacher/retraining setup absent |

**INFERRED — long-only versus DUSt3R/CroCo:** code establishes a narrower pretrained-weight scope for retrieval changes. It does **not** prove that an approximate long-only mechanism is safer for reconstruction: temporal and spatial retrieval are coupled, and an exact backbone attention kernel change may preserve semantics more closely than an approximate memory change. A long-only experiment requires joint normalization and uncertainty/threshold semantics to be specified and measured. Sources: `dust3r/model.py::memory_read` (~464–501); `croco/models/blocks.py::Attention/CrossAttention` (~94–168).

Before considering Kimi/linear/Flash/other variants, distinguish reducing memory-token count, reducing asymptotic attention work, changing kernels without changing math, and avoiding RAFT/value computation. The code does not justify choosing among them yet.

# 12. Training and Loss Audit

**VERIFIED FROM CODE:** Endo3R training is not released in this clone. `README.md` (~24–26) explicitly leaves release of training code unchecked. The supplied trainer `croco/pretrain.py::main/train_one_epoch` (~117–140, 190–247) constructs `CroCoNet` and `MaskedMSE`, not Endo3R.

| Requested supervision | Supplied Endo3R training evidence | Exact source / status |
|---|---|---|
| Depth | No released Endo3R criterion/trainer | `README.md` (24–26); **UNKNOWN / NEEDS TRAINING RELEASE** |
| Pointmaps / reconstruction | No released Endo3R criterion/trainer | Same; pointmap prediction itself is `dust3r/heads/postprocess.py::postprocess` (63–72), not a loss |
| Pose | No released Endo3R criterion/trainer | Optional head in `dust3r/heads/dpt_head.py::PixelwiseTaskWithDPT` (239–252) does not establish supervision |
| Confidence | No released Endo3R criterion/trainer | `dust3r/heads/postprocess.py::reg_dense_conf` (102–111) is an activation, not confidence loss |
| Temporal consistency / flow | No released Endo3R training implementation | Auxiliary scene-optimization losses exist below, separate from network training |
| Memory output / attention | No direct supervision implemented in supplied code | `dust3r/model.py::memory_read/Endo3R.forward` (456–509, 735–823); original training supervision unknown |
| Memory selection | Discrete hand-coded gate/prune; no selection objective or RL code | `dust3r/model.py::add_mem_check/memory_prune` (402–440, 511–537); original training recipe unknown |
| Uncertainty | Active inference computes analytic point disagreement through frozen flow | `dust3r/model.py::__init__/uncertainty_map` (259–262, 444–453); no uncertainty loss in this supplied path |
| Other geometry terms | No established Endo3R network training terms | Auxiliary optimization is not evidence of the authors' network criterion |
| RGB patch reconstruction | **Verified generic CroCo pretraining**, not Endo3R geometry training | `croco/models/croco.py::CroCoNet.forward` (231–249); `croco/models/criterion.py::MaskedMSE.forward` (24–37) |

`dust3r/inference.py::loss_of_one_batch` (~59–80) accepts an external criterion but defines none; `inference` (~83–98) passes `None`. The apparent training branch in `Endo3R.forward` calls undefined `sp_mem.add_mem` at ~785, so it is not a usable substitute trainer.

**VERIFIED FROM CODE — auxiliary optimization only:** `dust3r/cloud_opt/base_opt.py::BasePCOptimizer.forward` (~372–399) uses confidence-weighted point alignment. `dust3r/cloud_opt/optimizer.py::PointCloudOptimizer.forward` (~489–546) adds optional camera temporal smoothness, RAFT/ego-flow agreement outside dynamic masks, and scale-invariant depth regularization. `relative_pose_loss` (~548–561) penalizes relative rotation/translation. `dust3r/utils/goem_opt.py::depth_regularization_si_weighted` (~15–36) supplies weighted log-depth regularization. These optimize scene parameters through `global_alignment_loop/global_alignment_iter` in `base_opt.py` (~455–456, 507–523), not Endo3R weights, and are absent from normal online `demo.main`.

**PAPER INTENT / COMMENT ONLY:** the paper describes dynamics-aware flow, monocular depth, and confidence-aware point regression objectives; their Endo3R trainer and weights are not present here. [Endo3R paper, §§2.3–2.4](https://arxiv.org/html/2504.03198v1#S2.SS3). Repository release status: `README.md` (~24–26).

**INFERRED — distillation:** `feat_fuse` is observable at `dust3r/model.py::Endo3R.forward` (~758–763); `encode_value` returns `[B,P,1024]` (~646–651); decoder/encoder intermediate features can be captured; `return_memory=True` returns final memory (~820–821). These are possible teacher targets, but no distillation loss, data recipe, or working Endo3R trainer is supplied. Comparing intermediate outputs requires aligned queries and teacher memory trajectories, not just the same final frame.

**VERIFIED FROM CODE — reusable evaluation signals:**

- `eval/depth_evaluation.py::compute_errors` (~13–31): AbsRel, SqRel, RMSE, log-RMSE, δ<1.25/1.25²/1.25³. `depth_evaluation` (~34–79) uses depth range `(0.0001,100)`, one median scale alignment across the sequence, clipping, and per-frame averages.
- `eval/depth_evaluation.py::eval_single_scene` (~98–115) reads sorted GT/predictions by position, divides GT by 1000, and resizes to 320×256 using nearest-neighbor interpolation.
- `dust3r/utils/vo_eval.py::eval_metrics` (~158–241) computes associated-trajectory ATE and one-frame RPE with alignment and scale correction. Its wrapper/import paths are not all working in this clone.

**INFERRED — reward feasibility:** these GT-supported metrics could be reused in future episodes, but sequence-wide scale fitting depends on later frames and can hide absolute-scale error. Sample RGB frames lack GT reward data. Compute/resource measurements require instrumentation and are not existing reward terms. No PPO reward or training design is chosen.

## Minimum baseline reproduction requirements

| Requirement | Path / assumptions | Evidence |
|---|---|---|
| DUSt3R checkpoint | `./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`; `args.model` and `model` entries; loaded before Endo3R weights | `dust3r/model.py::Endo3R.__init__/load_model` (552–562, 53–69) |
| Endo3R checkpoint | `./checkpoints/endo3r.pth`; `['model']`, `strict=False`; missing/unexpected keys currently not logged | `demo.py::get_args_parser/main` (41, 92–95) |
| RAFT checkpoint | `./checkpoints/raft-things.pth`; loaded into DataParallel, then unwrapped; state-key compatibility needs verification | `dust3r/model.py::SpatialTempralMemory.__init__` (259–262) |
| Documented environment | Python3.11, CUDA12.1 PyTorch install; optional cuRoPE build | `README.md` (41–52) |
| Python dependencies | Mostly unpinned `requirements.txt`; explicit huggingface_hub>=0.22.0 and imageio imports omitted from list | `requirements.txt` (1–15); `dust3r/model.py` (11–12, 30–31); `dust3r/utils/image.py` (16) |
| CUDA/device | Demo defaults cuda:0, B=1; uncertainty images hardcode `.cuda()`; model/device portability needs verification | `demo.py` (44, 106); `dust3r/model.py::uncertainty_map` (445–446) |
| Inference data | Sorted RGB directory; optional matching `.npz` with camera_pose/camera_intrinsics and optional `*_depth.png` with maximum_depth metadata | `dust3r/datasets/demo.py::Demo._get_views` (30–98) |
| No-GT fallback | Identity pose, pseudo fx=fy=1 intrinsics, unit depth; not actual ground truth or required calibrated inference | `dust3r/datasets/demo.py::Demo._get_views` (39–40, 71–79) |
| Sampling / image dimensions | Demo defaults kf_every10, resolution224; evaluated320 gives320×256; normalize RGB to[-1,1] | `demo.py::get_args_parser/main` (46–47, 98–105); `dust3r/utils/image.py::ImgNorm` (26) |
| Save outputs | `--save_result` enables depth PNG/NPY and reconstruction `result.npy`; demo otherwise still creates directories | `demo.py::main` (89, 143–164, 200–227) |
| Full evaluation data | SCARED datasets8–9 and Hamlyn GT layouts; exact loader assumptions in evaluation scripts | `README.md` (54–56); `eval/eval.py` (36–85); `eval/depth_evaluation.py` (119–155) |

**VERIFIED FROM CODE:** README documents downloads for Endo3R and DUSt3R but omits mandatory RAFT. Its documented eval command calls nonexistent root `eval.py`; the actual file is `eval/eval.py`, with additional wrapper bugs in section 13. Sources: `README.md` (~57–67, 81–88); `dust3r/model.py::__init__` (~260).

**PAPER INTENT / COMMENT ONLY — dataset recipe:** README names SCARED1–7/StereoMIS/C3VD/EndoMapper and Cholec80/AutoLaparo/EndoVis17/18 for training, but the released loaders/trainer do not establish that mixture or preprocessing recipe. Source: `README.md` (~54–56); `dust3r/datasets/__init__.py`; `croco/pretrain.py::main` (~117–140).

**UNKNOWN / NEEDS RUNTIME VERIFICATION:** modern `torch.load` behavior for the DUSt3R checkpoint's serialized `args` object, checkpoint/state-key compatibility, actual layer/head counts, FP32 fit on 4GiB, pretrained head mode, and baseline quality. No checkpoint was downloaded to resolve these unknowns.

# 13. Bugs / Suspicious Logic

Severity means impact within the identified path: **BLOCKER** stops that path; **HIGH** can seriously affect correctness/reproducibility; **MEDIUM** is conditional or materially misleading; **LOW** limited impact; **INFO** clarification/dead code. Findings were not fixed.

| Severity | Finding / evidence label | Trigger and consequence | File / function / approximate lines |
|---|---|---|---|
| **BLOCKER** | Local inference cannot import; assets absent — verified environment | evo import fails; all three required checkpoints missing; no baseline profile possible | `dust3r/utils/vo_eval.py` (6); `dust3r/model.py::Endo3R.__init__/SpatialTempralMemory.__init__` (562, 260); section7 commands |
| **BLOCKER** | Undefined training method — VERIFIED FROM CODE | `self.training=True` calls nonexistent `sp_mem.add_mem` after initialization | `dust3r/model.py::Endo3R.forward` (784–785); full `SpatialTempralMemory` method inventory (242–549) |
| **BLOCKER** | Undefined offline method — VERIFIED FROM CODE | `--offline` calls nonexistent `Endo3R.offline_reconstruction` | `demo.py::main` (133); `dust3r/model.py::Endo3R` (550–856) |
| **HIGH** | Importance accumulation commented — VERIFIED FROM CODE | All mem_attn values remain zero; usage-based pruning ineffective; read reduction discarded | `dust3r/model.py::add_spatial_mem/memory_read/memory_prune` (360–364, 504–507, 513) |
| **HIGH** | Post-prune recent-frame similarity assumption — POTENTIAL BUG | topk reorders arbitrary patches; last wm*P no longer recent coherent frame grids; redundancy decisions become semantically invalid | `dust3r/model.py::memory_prune/check_sim` (519–528, 383–394) |
| **HIGH** | Zero attention row after threshold — POTENTIAL BUG, isolated reproduction | M>1000 allows all probabilities below0.001; divide by zero -> NaN fused features | `dust3r/model.py::memory_read` (494–498) |
| **HIGH** | Frozen RAFT remains train mode — VERIFIED FROM CODE | BatchNorm context statistics remain training behavior; no propagation from Endo3R.eval() to fresh external model | `dust3r/model.py::__init__/Endo3R.forward` (259–263, 735–739); `dust3r/core/raft.py::__init__` (55) |
| **HIGH** | Device/parallel portability — POTENTIAL BUG | `.cuda()` binds images to default device; DataParallel unwrapped before execution; nondefault/multi-GPU/CPU routes need verification. Single-GPU CPU mismatch is not established | `dust3r/model.py::__init__/uncertainty_map/pred_intrinsic/pred_poses` (259–262, 445–446, 833, 856) |
| **HIGH** | Invalid-correspondence uncertainty NaNs — POTENTIAL BUG, isolated reproduction | No valid correspondences -> min inf -> NaNs; zero masks do not sanitize; NaN maximum can skip weighting | `dust3r/utils/geometry.py::detect_dynamic_regions_3d_distance` (58–64); `dust3r/model.py::memory_read` (478) |
| **HIGH** | Whole-sequence GPU retention — VERIFIED FROM CODE / inferred scaling | All frames transferred upfront; both predictions for each pair retained; O(NHW) GPU residency despite K/V cap | `demo.py::main` (108–112); `dust3r/model.py::Endo3R.forward` (744–745, 805–816) |
| **HIGH** | Evaluation entrypoint/arguments broken — VERIFIED FROM CODE | README root eval.py absent; SCARED evaluator lacks data_type and takes Hamlyn branch; Hamlyn passes unsupported demo option, concatenates literals twice, omits save_result, uses wrong output root | `README.md` (85–88); `eval/eval.py` (64–67, 72–83); `eval/depth_evaluation.py` (119–121, 143–155); `demo.py::get_args_parser` (37–52) |
| **HIGH**, conditional | Optional learned pose processed twice — POTENTIAL BUG | has_pose head transforms translation norm d→expm1(d) twice; may amplify translation. Actual checkpoint head unknown; normal demo uses PnP | `dust3r/heads/dpt_head.py::postprocess_pose/PoseDecoder.forward/PixelwiseTaskWithDPT.forward` (36–69, 117, 252) |
| **MEDIUM** | Age division/recency ties — POTENTIAL BUG / verified algebra | New 0/0 overwritten by age mask, older scores zero; arbitrary retention, not FIFO or actual use scores; topk can drop protected entries when over budget | `dust3r/model.py::memory_prune` (513–526) |
| **MEDIUM** | “Disable uncertainty” changes read topology — VERIFIED FROM CODE | temporal K/V maintained but ignored without mem_uncert; uncertainty ablation is also temporal-memory ablation | `dust3r/model.py::memory_read/add_mem_check` (464–469, 426–428) |
| **MEDIUM** | Unbounded confidence/index metadata — VERIFIED FROM CODE | mem_c grows per accepted frame, spatial_buffer never pruned or mapped to surviving tokens | `dust3r/model.py::add_mem_c/add_mem_check/memory_prune` (308–315, 418, 427, 523–533) |
| **MEDIUM** | Hidden B=1 assumptions — VERIFIED FROM CODE | metadata pooling reshapes to[1,1,-1]; batch-wide similarity max governs all samples; demo focal/PnP selects sample0 | `dust3r/model.py::add_mem_c/add_mem_uncertainty/check_sim` (309, 334, 394); `demo.py::main` (170–183) |
| **MEDIUM** | Fixed P/shape assumptions — VERIFIED FROM CODE | num_patches captured once; same-shape pair concatenate; 16-pixel pooling/pose reshape hardcoded; varying sizes or incompatible checkpoint patches can fail | `dust3r/model.py::encode_image_pairs/add_mem_check/add_mem_c` (623–626, 404–405, 309); `dust3r/heads/dpt_head.py::PoseDecoder.forward` (98–104) |
| **MEDIUM** | PnP success unchecked — POTENTIAL BUG | failed solvePnPRansac proceeds to Rodrigues; ignores confidence/valid mask and all outliers except RANSAC | `demo.py::main` (179–196); `dust3r/model.py::Endo3R.pred_poses` (845–856) |
| **MEDIUM** | Fewer than two frames — POTENTIAL BUG | loop produces no res2/preds, then appends res2; no short-input guard | `dust3r/model.py::Endo3R.forward` (748–750, 816) |
| **MEDIUM** | Depth file without matching metadata — POTENTIAL BUG | input_metadata undefined when *_depth.png exists but .npz does not | `dust3r/datasets/demo.py::Demo._get_views` (67–82) |
| **MEDIUM**, optional | Payload pruning shape/index mismatch — POTENTIAL BUG | Token indices are expanded to[B,K,P,3] and used on unspecified point/image payload layouts. Rank or token-versus-frame index mismatch can fail gathering; mem_imgs may be None despite the mem_pts guard | `dust3r/model.py::memory_prune/add_mem_pts/add_mem_img` (530–533, 317–331); default forward passes neither payload |
| **MEDIUM**, optional | Temporal payload retention not trimmed — VERIFIED FROM CODE | passing pts_cur/img_cur appends payloads indefinitely although temporal K/V are shortened | `dust3r/model.py::add_temp_mem/add_mem_check` (347–348, 409–413) |
| **MEDIUM** | Repeated RAFT construction/load and unneeded intermediate output storage — VERIFIED FROM CODE | per-forward model load; each of20 iterations upsampled/retained despite test_mode returning final result | `dust3r/model.py::__init__` (259–260); `dust3r/core/raft.py::forward` (121–142) |
| **MEDIUM** | Evaluation file alignment/numerics — POTENTIAL BUG | sorted positional pairing without correspondence/length checks; sequence median can be zero/NaN; empty masks fail concatenation | `eval/depth_evaluation.py::eval_single_scene/depth_evaluation` (98–115, 52–62) |
| **MEDIUM**, auxiliary | Stale optimization/pose imports — VERIFIED FROM CODE | absent third_party.raft, sam2, dust3r.demo, dust3r.eval_metadata prevent auxiliary paths; not normal online dependency | `dust3r/cloud_opt/optimizer.py` (13–14); `dust3r/pose_eval.py` (14–16) |
| **MEDIUM**, auxiliary | NumPy alias undefined — POTENTIAL BUG | tum_to_pose_matrix uses np without NumPy import | `dust3r/utils/goem_opt.py::tum_to_pose_matrix` (1–13) |
| **LOW**, inactive | Expensive deep copies/best view undefined — POTENTIAL BUG | candidate search deep-copies full decoder lists/results; best_id undefined if none beats zero; not normal online forward | `dust3r/model.py::Endo3R.find_next_best_view` (700–733) |
| **LOW** | Position cache omits device — POTENTIAL BUG | reuse after device move can return cached coordinates on old device | `croco/models/blocks.py::PositionGetter.__call__` (201–206) |
| **LOW**, optional | Unbounded attention visualization — VERIFIED FROM CODE | save_attn appends flattened full matrices every read; disabled by default | `dust3r/model.py::memory_read` (486–490) |
| **LOW** | GPU host synchronizations — VERIFIED FROM CODE / inferred cost | NaN checks, max-based Python conditions, CPU shape reductions synchronize; impact unmeasured | `dust3r/model.py::Endo3R.forward/check_sim/memory_read` (772–773, 394, 478); `dust3r/utils/misc.py::transpose_to_landscape` (88–101) |
| **INFO** | Dead aliases/options/comments — VERIFIED FROM CODE | spat_mem_*, temp_mem_c unused; interval gate commented; eval arg unused; demo conf_thresh/vis/vis_cam unused; key comment calls an MLP cross-attention | `dust3r/model.py::init_mem/add_mem_check/Endo3R.forward` (273–286, 414, 735, 770); `demo.py::get_args_parser` (45, 48–49) |
| **INFO** | Checkpoint/dependency compatibility unknown | strict=False load status discarded; serialized args object and unpinned packages need real load checks; RoPE compilation optional | `demo.py::main` (94); `dust3r/model.py::load_model` (56–66); `requirements.txt`; `README.md` (48–52) |

Optional payload branch precision: for `top_k_indices [B,K,1]`, `unsqueeze(-1)` is `[B,K,1,1]` and expansion to `[B,K,P,3]` is legal. The suspicious part is its compatibility with the supplied payload layout and `mem_imgs`, not the expand operation itself. No payload inference path was run.

# 14. Paper-vs-Code Differences

The comparison is restricted to verified implementation and explicit statements in the accessible paper v1. It does not claim that this checkout is the authors' complete training implementation. Paper-source paraphrases below are intentionally brief. [Endo3R paper, §2.2](https://arxiv.org/html/2504.03198v1#S2.SS2).

| PAPER INTENT / COMMENT ONLY | VERIFIED FROM CODE | Difference / limitation |
|---|---|---|
| Older temporal keys/values move into spatial memory | `dust3r/model.py::add_mem_check` (406–428) inserts survivors immediately into both, trims temporal separately | No temporal-to-spatial transfer; recent entries duplicated |
| Sampson distance filters unreliable tokens | `dust3r/model.py::uncertainty_map` (450–452) calls Euclidean 3D disagreement; `dust3r/utils/geometry.py::detect_dynamic_regions_3d_distance` (16–67) | Active online uncertainty uses no epipolar/Sampson calculation |
| Tokens above an uncertainty threshold are eliminated | `dust3r/model.py::add_mem_check/memory_read` (417–428, 478–496) writes every similarity survivor and scales logits | No uncertainty-based write rejection/token removal; probability threshold is different |
| Confidence selects top-K memory tokens | `dust3r/model.py::add_spatial_mem/memory_prune` (372, 513–526) stores unused mem_c and ranks zero attention/age with recency protection | Prediction confidence does not rank pruning; attention importance not accumulated |

The paper's PnP-to-local-depth description is consistent with `demo.py::main` (~179–196); no difference is asserted there. [Endo3R paper, §2.1](https://arxiv.org/html/2504.03198v1#S2.SS1).

**VERIFIED FROM CODE — comment discrepancies:** memory_read's docstring describes `[B,T,P,C]` and confidence usage, while actual K/V are flattened tokens and confidence is not consumed (`dust3r/model.py::memory_read`, ~456–498). The key-encoding comment describes cross-attention; actual `encode_feat_key` concatenates features and applies an MLP (~591–602, 640–644, 770). `interval` is assigned, but its intended time-gap check is commented (~414, 747). These are code/comment differences, not additional unverified paper claims.

# 15. Recommended Measurements Before Architecture Design

These are measurement requirements, not a proposed architecture or implementation plan for RL/attention.

1. **Establish a runnable, identified baseline.** Inspect all checkpoint metadata/state incompatibilities, actual encoder/decoder heads/layers, patch size, output mode, model/RAFT device and mode. Record exact checkpoint hashes, package versions, dtype, image resolution, stride, and GPU. Verify two-frame and longer sequence outputs are finite. Resolve/explicitly document the known broken branches in a separately authorized phase. Relevant boundaries: `dust3r/model.py::load_model/Endo3R.forward/SpatialTempralMemory.__init__` (~53–69, 735–823, 259–262).
2. **Measure every compute category from section7.** Report full-step and exclusive timings, RAFT cold load/conditional calls, key MLPs, pointmap patch embedding/value encoder, both prediction heads, and CPU PnP. Compare 224/320/512 modes only when they fit. This determines whether attention, RAFT, MLPs, or output retention actually dominates.
3. **Measure retrieval as a function of actual M.** Record accepted writes, temporal/spatial tokens, before/after prune states, attention sparsity and zero rows, duplicate recent entries, uncertainty distribution, and mem_count/mem_attn. Contrast pre-prune and post-prune similarity decisions. Confirm budgets in tokens rather than frames. Sources: `dust3r/model.py::add_mem_check/check_sim/memory_read/memory_prune` (~378–537).
4. **Separate quality and memory effects.** Report depth metrics with the exact existing sequence-scale alignment and also track absolute scale/scale drift; pose metrics with stated trajectory alignment; pointmap consistency, dynamic/occlusion regions, and failures. Keep sequences/stride/frame coverage identical across comparisons. Existing evaluators need their argument/path defects addressed before their results can be trusted. Sources: `eval/depth_evaluation.py::depth_evaluation` (~34–79); `dust3r/utils/vo_eval.py::eval_metrics` (~158–241).
5. **Audit uncertainty value versus cost.** Count write acceptance, RAFT frequency, image resolution, valid correspondence fraction, flow-mask fraction, U finite rate, and effect of logit scaling for positive/negative logits. Test uncertainty removal independently of temporal retrieval: the existing false flag conflates them. Sources: `dust3r/utils/geometry.py::detect_dynamic_regions_3d_distance` (~44–64); `dust3r/model.py::memory_read` (~464–482).
6. **Measure exact attention kernel opportunities before approximations.** For each attention family, record Q/K/V shape/stride/dtype, head dimensions, RoPE, dropout, threshold and uncertainty operations, backward needs, and materialized matrix peaks. Validate numerical/output agreement against original operators. Benchmark complete block latency as well as QK/AV, since projection/MLP cost may dominate. Sources: `croco/models/blocks.py::Attention/CrossAttention/DecoderBlock` (~81–191); `dust3r/model.py::memory_read` (~456–509).
7. **Test insertion interfaces with teacher traces.** Capture original fused memory outputs, value outputs, decoder DPT hooks, pointmaps and confidence across complete trajectories. Measure the sensitivity of later reconstructions to altered reads/writes; a local feature error alone is insufficient. Establish a working training/data release or a separately authorized training setup before committing to distillation. Sources: `dust3r/model.py::Endo3R.forward/encode_value` (~758–821, 646–651); `dust3r/heads/dpt_head.py::create_dpt_head` (~270–271).
8. **Assess sequential-controller viability before PPO-based design.** Log currently available decision signals at the four insertion points, their extraction latency, action frequency, and downstream memory/quality/computation changes. Establish GT/reward coverage, episode/reset semantics, first-write/empty-memory behavior, sequence-length distribution, and how scale-aligned metrics depend on future observations. Measure a small controller's GPU and host overhead separately; do not assume it is negligible. No specific reward/policy/action architecture is selected here.
9. **Measure long-sequence residency beyond attention.** Separate model weights, temporal/spatial K/V, metadata, concatenation/norm copies, attention matrices, input frames, all pairwise predictions, and optional visualization buffers. Check memory slopes against N and accepted-write count. The existing implementation does not provide a strict whole-process long-sequence memory bound. Sources: `demo.py::main` (~108–112); `dust3r/model.py::Endo3R.forward/init_mem/add_mem_c/memory_read` (~744–821, 265–290, 308–315, 464–498).

# 16. Final Readiness Verdict

**READY_TO_PLAN_WITH_BLOCKERS**

The real code boundaries, tensor interfaces, heuristics, sequential state, and retrieval math are sufficiently mapped to discuss candidate experiments. A final architecture, baseline speed claim, or training-dependent research commitment is premature: baseline imports/checkpoints are missing, Endo3R training is unreleased and its supplied training branch is broken, several memory/numerical issues compromise comparisons, and no end-to-end profiling or reconstruction-quality verification has been performed. This verdict approves no architecture and does not assume any bug has been fixed.

## Files inspected

Content inspection by the root and parallel read-only audits covered:

```text
README.md
requirements.txt
.gitignore
demo.py
eval/eval.py
eval/depth_evaluation.py
dust3r/model.py
dust3r/inference.py
dust3r/image_pairs.py (call/search inspection)
dust3r/patch_embed.py
dust3r/post_process.py
dust3r/pose_eval.py
dust3r/datasets/__init__.py
dust3r/datasets/demo.py
dust3r/datasets/base_many_view_dataset.py
dust3r/datasets/base/base_stereo_view_dataset.py
dust3r/datasets/utils/transforms.py
dust3r/heads/__init__.py
dust3r/heads/dpt_head.py
dust3r/heads/linear_head.py
dust3r/heads/postprocess.py
dust3r/core/raft.py
dust3r/core/corr.py
dust3r/core/extractor.py
dust3r/utils/image.py
dust3r/utils/misc.py
dust3r/utils/geometry.py
dust3r/utils/path_to_croco.py
dust3r/utils/pose_utils.py
dust3r/utils/vo_eval.py
dust3r/utils/viz_demo.py
dust3r/utils/goem_opt.py
dust3r/cloud_opt/__init__.py
dust3r/cloud_opt/base_opt.py
dust3r/cloud_opt/optimizer.py
dust3r/cloud_opt/modular_optimizer.py
dust3r/cloud_opt/init_im_poses.py
croco/README.MD
croco/pretrain.py
croco/models/croco.py
croco/models/blocks.py
croco/models/criterion.py
croco/models/pos_embed.py
croco/models/dpt_block.py
croco/models/curope/__init__.py
croco/models/curope/curope2d.py
```

Additionally, all 97 `.py` files received an AST syntax parse, and repository-wide attention/memory/checkpoint/import searches were performed. AST coverage is not a claim that every file was semantically reviewed. External/local references read: `C:\Users\ADMIN\.codex\RTK.md`; installed `torch/nn/parallel/data_parallel.py`; [paper v1](https://arxiv.org/html/2504.03198v1).

## Runtime commands executed

Representative commands/checks (all read-only; root runtime/import probes used `-B` to avoid bytecode writes; parallel static inspections also used `python -c` and `python -X utf8 -c` without importing repository modules):

```text
git status --short
git rev-parse HEAD
git submodule status
git ls-files
git ls-files --others --exclude-standard
rg --files [including hidden/no-ignore checkpoint and AGENTS filters]
rg -n [targeted function/memory/attention/loss/import searches]
python --version
nvidia-smi
python -B -c "import sys,importlib.util,json; ..."  # executable/module discovery
python -B -c "import torch; ..."                  # torch/CUDA versions and availability
python -B -c "import dust3r.model"                # failed: missing evo
python -B -c "from pathlib import Path; import ast; ..."  # 97-file AST/asset inventory
python -B -c "import torch,inspect; ..."          # DataParallel source and tiny scalar/threshold probes
python -B -c $auditProbe                          # in-memory AST-extracted toy memory/uncertainty probes
python -B -c [source-line printing / AST method inventories / numeric budget arithmetic]
python -c [parallel-agent static reads / AST inventories / module discovery]
python -X utf8 -c [parallel-agent report verification]
```

After reading RTK instructions, commands used an `rtk` prefix. Because RTK is unavailable, a process-local PowerShell function forwarded executable/arguments without installing files. One shell-quoting attempt and one stdin-forwarding attempt failed; the latter was interrupted and the successful checks used `-c`. These were audit harness failures, not repository failures. No inference/profiling/demo/evaluation/training command was executed; no CUDA synchronization timing results exist.

**Audit artifact:** `docs/END3R_ATTENTION_RL_AUDIT.md`.

**Source files changed:** none. **Verified git status:** `?? docs/` (`--untracked-files=all` expands this to `?? docs/END3R_ATTENTION_RL_AUDIT.md`); tracked source diff empty. The document is the only untracked file.
