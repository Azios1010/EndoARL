# SCARED server guide: môi trường → dữ liệu → fit → validation → PPO → benchmark

Guide này là điểm vào cho toàn bộ công việc trên server. Các lệnh dưới đây được
viết cho Linux/Bash, chạy từ repository root **sau khi từng gate trước đó đạt**.
Chưa chạy dataset, CUDA, fitting hay PPO trong công việc local. Không chạy toàn
bộ tài liệu như script tự động; đặc biệt không chạy study `all --server-run` để
thay thế checkpoint selection và actor validation.

Trạng thái do operator báo: `dataset_2/keyframe_4/1729` có bốn pixel GT gốc nhưng
zero sau320×256; hybrid checkpoint mới hai bước smoke; chưa có attention checkpoint
được chọn/certify hoặc actor đã validate; budgets chưa chọn. Local không có
dataset/checkpoint để chứng minh các vấn đề này đã giải quyết trên server.

## 1. Đọc theo nhánh và xác định phần đã có

| Tài liệu | Nội dung | Trạng thái |
| --- | --- | --- |
| [Server environment](SCARED_SERVER_ENVIRONMENT.md) | GPU/driver, Python/Torch/Triton/FLA, checkpoints, kernel proof, environment lock | Có lệnh provisioning chỉ dành cho server; runtime chưa verified |
| Guide này | Real-GT coverage, training budgets, attention candidates, full selection, PPO/actor validation, ba seed, benchmark table | Công cụ đã implement; cần real-data/kernel proof |
| [StereoAnything labels](SCARED_STEREOANYTHING_LABEL_GUIDE.md) | Upstream compatibility, numeric exporter, pseudo-target loader/contracts và tests | Có đặc tả đầy đủ; exporter/target-kind integration **chưa implement** |
| [Partial-GT evaluation](SCARED_PARTIAL_GT_IMPLEMENTATION.md) | Full RGB inference/FPS, sparse depth/pose scoring, certification versioning và tests | Có đặc tả đầy đủ; scorer/certification mở rộng **chưa implement** |
| [Protocol](SCARED_EXPERIMENT_PROTOCOL.md), [paper comparison](PAPER_MATCHED_COMPARISON.md) | Identity, causal ordering, geometry, training/evaluation limitations | Các giới hạn phải giữ trong report |

**Track A — hiện được hỗ trợ:** real-GT-backed fine-tuning; TRAIN có thể chọn các
valid contiguous runs; validation/test vẫn strict real GT. Không cần StereoAnything.

**Track B — nhãn gần paper hơn:** tạo StereoAnything pseudo-depth trong run riêng,
bổ sung exporter và target-kind support trước fitting; held-out scoring vẫn real GT.

**Partial-GT held-out:** nếu validation7/test8–9 có empty/missing GT, full benchmark
và certification bị chặn. Training-only filtering không sửa được tình huống này.
Thực hiện guide partial-GT trước; không có CLI toggle an toàn sẵn để bỏ qua.

## 2. Paper so với thí nghiệm đang chạy

Nguồn chính: [Endo3R paper §2–3](https://arxiv.org/html/2504.03198v1),
[README tác giả](https://github.com/wrld/Endo3R#prepare-datasets). Release chưa
cung cấp training code gốc; không suy ra optimizer, epoch, loss weights hay
checkpoint selection của paper từ inference source.

| Hạng mục | Paper / release | Project hiện tại | Cần công bố |
| --- | --- | --- | --- |
| Reconstruction output | Pointmap/confidence; depth/pose suy từ pointmap | Giữ encoder/decoder/DPT outputs; chưa Gaussian | Ablation memory, không thay output representation |
| Memory | Dense temporal/spatial retrieval trong release | Exact Flash ST≤4 frames, fixed-state KDA LT, learned fusion | Architecture mới, baseline official tách riêng |
| Uncertainty | Paper mô tả Sampson; released code dùng geometry_3d | Sampson/RAFT cho WRITE của cả hai hybrid | Official baseline giữ geometry_3d, không sửa âm thầm |
| Training labels | StereoAnything depth từ rectified stereo | A00 disparity hiện từ original `scene_points` GT | Ghi `real_gt`, không gọi là StereoAnything |
| Training recipe | Tám dataset; SCARED1–7; flow/depth/confidence losses | SCARED1–6 fit; validation7; masked common-frame point L1 | Fine-tuning recipe mới, chưa original recipe |
| Evaluation | SCARED8–9,320×256; pose five-frame | Cùng split/resolution, project crop/scale/aggregation được khai báo | Chưa đủ chứng minh `paper_exact` |
| FPS | Có số paper; hardware/timing scope chưa đủ để đồng nhất | Rerun cùng GPU/precision/scope cho ba branch | Paper row là reference; speedup chỉ giữa matched reruns |

Backbone released đã thấy SCARED7 khi pretrained. Validation7 là holdout cho
**hybrid/controller fine-tuning**, không phải dữ liệu chưa từng được backbone thấy.
Refit1–7 cần validation độc lập; không vừa train7 vừa certify7, không dùng8–9 select.

Ba experiment chính: `legacy_official`, `hybrid_sampson`, `hybrid_ppo`.
Hai hybrid dùng **cùng certified attention checkpoint trong từng seed** và cùng
Sampson/RAFT; config chỉ khác WRITE policy. Sampson không phải kernel attention.
Giữ paper reference riêng trong catalog [`paper_reference.json`](../configs/experiments/paper_reference.json).

## 3. Cập nhật code, freeze checkout và khai báo paths

Những file local mới phải được commit/push trước để server có công cụ coverage và
selection. Server không thể pull chúng nếu chỉ tồn tại trong working tree local.
Guide này không tự commit/push. Trên checkout server sạch, kiểm tra remote đúng
`Azios1010/EndoARL`, fetch rồi fast-forward branch main từ remote tương ứng:

```bash
git remote -v
git status --short
git branch --show-current
# Chỉ khi main và working tree sạch. Dùng đúng remote đã kiểm tra.
export REPO_REMOTE=origin
git fetch "$REPO_REMOTE" main
git merge --ff-only "$REPO_REMOTE/main"
git rev-parse HEAD
```

Nếu remote đúng tên `endoarl`, đặt `REPO_REMOTE=endoarl`. Không reset/stash/ghi đè
thay đổi server một cách tự động. Nếu diverged, review/commit thay đổi riêng trước.
Lưu source commit **và** source-tree fingerprint; matching reports còn kiểm tra
source/config/tests và top-level Markdown như README. Giữ checkout bất biến từ fit đến quality/timing; không
sửa code hoặc guide giữa các run cần join. Artifacts đặt ngoài repo.

Hoàn tất [environment guide](SCARED_SERVER_ENVIRONMENT.md), activate đúng env rồi
khai báo paths thực. `OUT` phải là root mới để tránh trộn artifacts cũ:

```bash
set -euo pipefail
export FULL_MANIFEST=/work/scared_prepared/verified_manifest.json
export OUT=/work/endoarl_scared_run_new
export TRAIN_MANIFEST="$OUT/train_supervision.json"
export DUST3R_CHECKPOINT=/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
export ENDO3R_CHECKPOINT=/models/endo3r.pth
export RAFT_CHECKPOINT="$PWD/checkpoints/raft-things.pth"
export PROFILE_RGB=/data/SCARED/VERIFIED_TRAIN_OR_VALIDATION_LEFT_RGB
test ! -e "$OUT"
mkdir -p "$OUT"
for path in "$DUST3R_CHECKPOINT" "$ENDO3R_CHECKPOINT" "$RAFT_CHECKPOINT"; do
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 1; }
done
test -d "$PROFILE_RGB"
sha256sum "$DUST3R_CHECKPOINT" "$ENDO3R_CHECKPOINT" "$RAFT_CHECKPOINT" > "$OUT/checkpoint_hashes.txt"
git rev-parse HEAD > "$OUT/source_commit.txt"
git status --short > "$OUT/source_status.txt"
```

Legacy official hardcodes `checkpoints/raft-things.pth` và visible CUDA0.
Đặt `CUDA_VISIBLE_DEVICES` phù hợp trước Python, luôn dùng `--device cuda:0` cho
comparison có official. Không đổi `.cuda()`/RAFT/train-mode logic của legacy để
cứu baseline rồi vẫn gắn nhãn official. Baseline lỗi phải được báo; `LEGACY_CLEAN`
nếu cần là experiment riêng với danh sách fixes.

## 4. Data preparation và provenance

Nếu manifest verified đã có, giữ nguyên và chuyển sang inventory. Nếu chưa có:
review [`native mapping example`](../configs/experiments/scared_native_mapping.example.json)
và source index của server, không lấy example chưa verified làm calibration thật.

```bash
export SCARED_ROOT=/data/SCARED
export VERIFIED_MAPPING=/work/data_config/scared_verified_mapping.json
export NEW_MANIFEST=/work/scared_prepared/verified_manifest.json
test -f "$VERIFIED_MAPPING"
test ! -e "$NEW_MANIFEST"
python tools/prepare_scared.py --data-root "$SCARED_ROOT" \
  --mapping "$VERIFIED_MAPPING" --output "$NEW_MANIFEST" --dry-run
# Chỉ sau review dry summary và metadata.
python tools/prepare_scared.py --data-root "$SCARED_ROOT" \
  --mapping "$VERIFIED_MAPPING" --output "$NEW_MANIFEST"
```

`--source-index VERIFIED_INDEX.json` thay `--mapping` nếu có explicit frame index.
`--extract-video` chỉ khi mapping đã khai báo layout/index chính xác. Có thể thêm
`--verify-arrays` để strict verify; sparse GT có thể khiến bước đó dừng, lúc này
dùng inventory riêng để lấy report từng frame, không fake validity. Tool không
acquire/extract archive tự động và không ghi vào source dataset.

Phải xác minh RGB/GT/stereo frame IDs, timestamps, rectification grid, intrinsics,
pose direction W2C/C2W, axes, millimetres/metres, baseline và valid-mask semantics.
Không tự thêm identity pose hoặc đoán offset video. Split1–6/7/8–9 và patient
identities phải disjoint cho fitting. GT có kinematic-warp/sparse hạn chế phải nêu.

Sau preparation cập nhật `FULL_MANIFEST` nếu path khác. Manifest đầy đủ luôn là
nguồn identity evaluation/certification. Không đổi nó thành TRAIN-filtered source.

```bash
test -f "$FULL_MANIFEST"
sha256sum "$FULL_MANIFEST" "$DUST3R_CHECKPOINT" "$ENDO3R_CHECKPOINT" "$RAFT_CHECKPOINT" \
  > "$OUT/input_hashes.txt"
```

## 5. Inventory coverage và tạo TRAIN valid runs

Implementation: [`audit_supervision`, `build_training_manifest`, `validate_supervision_manifest`](../dust3r/data/supervision.py)
đi qua cùng decode/crop/nearest-exact/final geometry checks của
[`SequenceDataset._load_frame`](../dust3r/training/sequence_dataset.py).
Mỗi frame được ghi raw/cropped/resized/final count/fraction; per-frame errors
không làm mất các frame còn lại trong inventory. Provenance/schema/split sai vẫn fatal.

```bash
# Inventory lần đầu chỉ để review; một pixel là crash threshold, chưa đủ fit quality.
python tools/audit_scared_supervision.py \
  --manifest "$FULL_MANIFEST" --config configs/scared/hybrid_sampson.yaml \
  --report "$OUT/gt_coverage_initial.json" \
  --min-valid-pixels 1 --min-valid-fraction 0
```

Đọc count của frame1729, histogram per dataset, train/validation/test failures và
length các contiguous runs. Chọn ngưỡng supervision dựa trên tín hiệu thực, không
bảo đảm chất lượng chỉ vì `>=1` pixel. Ghi hai threshold và lý do trước export:

```bash
: "${MIN_VALID_PIXELS:?choose and record a positive supervision threshold}"
: "${MIN_VALID_FRACTION:?choose and record a coverage fraction from 0 to 1}"
python tools/audit_scared_supervision.py \
  --manifest "$FULL_MANIFEST" --config configs/scared/hybrid_sampson.yaml \
  --report "$OUT/gt_coverage_selected.json" --training-manifest "$TRAIN_MANIFEST" \
  --min-valid-pixels "$MIN_VALID_PIXELS" --min-valid-fraction "$MIN_VALID_FRACTION"
```

Output paths phải mới. Tool ghi report rồi exit2 nếu không còn valid clip; không
được xem report tồn tại là export thành công. Chỉ TRAIN được chọn thành maximal
valid contiguous runs≥9 raw frames. Không nối qua frame thiếu GT/gap. Derived
manifest giữ absolute target paths, source IDs và first-camera origin của sequence
gốc, công bố excluded frames/run quá ngắn. Source stride đã áp dụng; derived loader
stride1. V1 study stride1; thay stride/resolution/cliplen phải audit lại.

`--manifest` / `training.data_manifest` = full original source.
`--supervision-manifest` / `training.supervision_manifest` = selected TRAIN.
Attention và PPO fit dùng selected TRAIN; certification/evaluation/FPS dùng full
original manifest. Checkpoint bind cả hai hashes; đổi target selection sau fit
cần run mới hoặc exact-resume sẽ từ chối.

**Dừng nếu validation7/test8–9 có empty GT:** hiện evaluator/certifier strict.
Full held-out không được sửa bằng filtering. Thực hiện
[partial-GT implementation](SCARED_PARTIAL_GT_IMPLEMENTATION.md) với protocol mới.
Đối với sparse TRAIN-only, workflow hiện tại có thể tiếp tục khi held-out đạt gate.

## 6. Chọn và công bố budgets

Configs giữ budgets **UNSELECTED**; cả direct trainer lẫn study execution không
tự chọn1000 steps/1 epoch/1 PPO update. Operator cần đặt biến explicit trước run:

| Biến | Quyết định | Cách chọn / yêu cầu |
| --- | --- | --- |
| `ATTENTION_STEPS` | Positive optimizer-step ceiling | Pilot step cost/VRAM và validation learning curve |
| `ATTENTION_EPOCHS` | Positive epoch ceiling | Number of valid clips và train/val gap; dừng khi một ceiling đạt |
| `CANDIDATE_EVERY` | Positive save interval | Full validation cost; candidates được lưu ở fresh-clip boundary |
| `MIN_FIT_STEPS` | Minimum candidate steps accepted | Loại smoke theo protocol công bố; không chứng minh convergence |
| `PPO_UPDATES` | Positive **total** PPO update target | Validation quality/compute frontier; có thể nhiều increasing budgets |
| `MIN_VALID_PIXELS/FRACTION` | TRAIN supervision thresholds | Coverage inventory; không tune trên test |

```bash
: "${ATTENTION_STEPS:?select attention optimizer-step budget}"
: "${ATTENTION_EPOCHS:?select attention epoch ceiling}"
: "${CANDIDATE_EVERY:?select candidate save interval}"
: "${MIN_FIT_STEPS:?select minimum fitted candidate steps}"
: "${PPO_UPDATES:?select PPO total update budget}"
export ATTENTION_STEPS ATTENTION_EPOCHS CANDIDATE_EVERY MIN_FIT_STEPS PPO_UPDATES
export MIN_VALID_PIXELS MIN_VALID_FRACTION
python - <<'PY'
import json, os
from pathlib import Path
names = ['ATTENTION_STEPS', 'ATTENTION_EPOCHS', 'CANDIDATE_EVERY', 'MIN_FIT_STEPS', 'PPO_UPDATES', 'MIN_VALID_PIXELS']
values = {name: int(os.environ[name]) for name in names}
assert all(value > 0 for value in values.values()), values
values['MIN_VALID_FRACTION'] = float(os.environ['MIN_VALID_FRACTION'])
assert 0 <= values['MIN_VALID_FRACTION'] <= 1
assert values['MIN_FIT_STEPS'] <= values['ATTENTION_STEPS']
with (Path(os.environ['OUT']) / 'budget_decisions.json').open('x') as handle:
    json.dump({'values': values, 'origin': 'operator-selected; not recovered paper hyperparameters'}, handle, indent=2)
PY
```

Chưa có con số production nào được chọn trong guide. Pilot budget nhỏ chỉ đo
correctness/step cost, phải ở OUT riêng và không được gọi là fitted/certified.
Khi thay budget based on validation, ghi decision record mới; không sửa artifacts cũ.

## 7. Materialize ba seed và preflight

```bash
STUDY=(python tools/run_scared_study.py
  --manifest "$FULL_MANIFEST" --supervision-manifest "$TRAIN_MANIFEST"
  --dust3r-checkpoint "$DUST3R_CHECKPOINT" --endo3r-checkpoint "$ENDO3R_CHECKPOINT"
  --raft-checkpoint "$RAFT_CHECKPOINT" --profile-input "$PROFILE_RGB"
  --output-root "$OUT" --device cuda:0 --seeds 0 1 2 --trusted-checkpoints
  --attention-steps "$ATTENTION_STEPS" --attention-epochs "$ATTENTION_EPOCHS"
  --ppo-updates "$PPO_UPDATES")

# Materialization/dry plan; không launch fitting.
"${STUDY[@]}" --stage all --write-configs > "$OUT/study_plan.json"
# Kernel proof + data/module dry validation, chưa train reconstruction.
"${STUDY[@]}" --stage preflight --write-configs --server-run > "$OUT/preflight_execution.json"
```

**Không chạy `--stage all --server-run` cho production:** study `baseline/certify`
hiện dùng prefix64; `fit` chưa tạo periodic candidates; `ppo` chưa tự select actor.
Các bước manual full validation/candidate selection dưới đây thay thế chúng.
Study `preflight/profile/evaluate/benchmark/summarize` dùng được trong workflow này.

Review `seed_N/kernel_conformance.json`: actual forced SDPA Flash + FLA pinned,
finite forward/backward, ST output `[1,4,1024]` của probe, KDA state `[1,4,64,64]`
FP32 bounded. `P=4` chỉ là probe nhỏ; real320×256 với patch16 có P320. Không dùng
`--reference`, generic `sdpa` hoặc CPU proof làm bằng chứng efficient backend.

Dry trainer có thể exit0 khi `requirements_ready=false`. Không chỉ nhìn exit code;
review `blocking_requirements`, `missing_training_budgets`, config/data contracts.
Study preflight dry thiếu budgets chưa chứng minh fit-ready; chạy complete dry sau:

```bash
for SEED in 0 1 2; do
  python train_hybrid_memory.py --config "$OUT/seed_$SEED/config_hybrid_sampson.json" \
    --manifest "$FULL_MANIFEST" --supervision-manifest "$TRAIN_MANIFEST" \
    --seed "$SEED" --total-steps "$ATTENTION_STEPS" --epochs "$ATTENTION_EPOCHS" \
    --dry-run --validate-data --validate-modules --device cuda:0 \
    > "$OUT/seed_$SEED/attention_dry.json"
done
python - <<'PY'
import json, os
from pathlib import Path
for seed in (0, 1, 2):
    report = json.loads((Path(os.environ['OUT']) / f'seed_{seed}/attention_dry.json').read_text())
    assert report['requirements_ready'], report['blocking_requirements']
    assert not report['missing_training_budgets'], report
    assert report['config_valid'], report
print('All attention dry readiness reports pass; kernel/reconstruction proof remains separate.')
PY
```

Dry module construction chỉ standalone memory/optimizer grouping, chưa load full
released reconstruction. Không dùng dry-pass thay runtime proof. Environment
guide chỉ chứng minh imports/tiny kernels; whole-model checks cần bước riêng sau.

### 7.1. Bounded whole-model smoke, tách khỏi production fitting

Chỉ thực hiện trên server đã qua kernel/data gate. Smoke checkpoint lưu ở thư
mục riêng, không được đưa vào production candidate selection hoặc dùng để resume
T1 production. Hai steps kiểm tra forward/backward, không chứng minh memory đã fit.

```bash
test ! -e "$OUT/smoke_seed_0"
mkdir -p "$OUT/smoke_seed_0"
python demo_hybrid.py --config "$OUT/seed_0/config_legacy_official.json" \
  --input "$PROFILE_RGB" --max-frames 8 --device cuda:0 --trusted-checkpoints \
  --run-inference --output "$OUT/smoke_seed_0/legacy_predictions.pt"
python train_hybrid_memory.py --config "$OUT/seed_0/config_hybrid_sampson.json" \
  --manifest "$FULL_MANIFEST" --supervision-manifest "$TRAIN_MANIFEST" \
  --stage t1 --seed 0 --device cuda:0 --trusted-checkpoints --run-training \
  --total-steps 2 --epochs 1 --log-jsonl "$OUT/smoke_seed_0/attention_smoke.jsonl" \
  --output "$OUT/smoke_seed_0/hybrid_smoke.pt"
python demo_hybrid.py --config "$OUT/seed_0/config_hybrid_sampson.json" \
  --input "$PROFILE_RGB" --max-frames 8 --device cuda:0 --trusted-checkpoints \
  --hybrid-checkpoint "$OUT/smoke_seed_0/hybrid_smoke.pt" \
  --run-inference --output "$OUT/smoke_seed_0/hybrid_predictions.pt"
python - <<'PY'
import os, torch
from pathlib import Path
root = Path(os.environ['OUT']) / 'smoke_seed_0'
for name in ('legacy_predictions.pt', 'hybrid_predictions.pt'):
    payload = torch.load(root / name, map_location='cpu', weights_only=True)
    predictions = payload['predictions']
    assert len(predictions) >= 2
    for result in predictions:
        points = result.get('pts3d', result.get('pts3d_in_other_view'))
        confidence = result['conf']
        assert tuple(points.shape[-3:]) == (256, 320, 3), points.shape
        assert torch.isfinite(points).all() and torch.isfinite(confidence).all()
print('Bounded full-model predictions finite; inspect gradient/loss/freeze logs separately.')
PY
```

`PROFILE_RGB` phải là một TRAIN/validation sequence có filenames lexical order
khớp source time, tối thiểu tám frames để chạy đúng bounds trên; không trộn nhiều
sequence hay gọi loader RGB này là GT/data alignment validation. Review trainer
parameter-status/logs: new memory nhận gradient hữu hạn, frozen modules không
nhận update, loss hữu hạn, TBPTT state detach đúng. Smoke chỉ gate correctness.

### 7.2. Legacy original-forward vs streaming parity

Đây là bounded **server validation snippet**, chưa thực thi local. Nó so output
của original `Endo3R.forward` với adapter `LegacyOfficialStream` trên cùng weights/
frames/RNG, không thay legacy code. Tolerance `rtol=1e-4, atol=1e-5` là provisional;
fail phải lưu sai khác/max-error và review, không tăng tolerance để che regression.

```bash
python - <<'PY' > "$OUT/smoke_seed_0/legacy_parity.json"
import json, os, sys, torch
from contextlib import redirect_stdout
from pathlib import Path
from demo_hybrid import load_frames, load_inference_model, prepare_config
from dust3r.training.hybrid_trainer import move_tensors
from dust3r.online.streaming import LegacyOfficialStream
config = prepare_config(str(Path(os.environ['OUT']) / 'seed_0/config_legacy_official.json'))
def seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
with redirect_stdout(sys.stderr), torch.no_grad():
    model, hashes = load_inference_model(config, device='cuda:0', trusted_checkpoints=True)
    frames = [move_tensors(frame, 'cuda:0') for frame in load_frames(os.environ['PROFILE_RGB'], (256, 320), 8)]
    seed()
    predictions, pairs = model(frames, uncertainty_check=config.uncertainty.enabled)
    reference = move_tensors(predictions, 'cpu')
    del predictions, pairs
    seed()
    runner = LegacyOfficialStream(model, len(frames), uncertainty_check=config.uncertainty.enabled)
    streamed = []
    for index, frame in enumerate(frames):
        pair = runner.step(frame, terminal=index == len(frames) - 1)
        if pair is not None:
            streamed.append(move_tensors(pair.res1, 'cpu'))
            if pair.terminal:
                streamed.append(move_tensors(pair.res2, 'cpu'))
    assert len(streamed) == len(reference) == len(frames)
    max_point_error = max_conf_error = 0.0
    for original, adapter in zip(reference, streamed):
        a = original.get('pts3d', original.get('pts3d_in_other_view'))
        b = adapter['pts3d']
        for tensor in (a, b, original['conf'], adapter['conf']):
            assert torch.isfinite(tensor).all()
        max_point_error = max(max_point_error, (a - b).abs().max().item())
        max_conf_error = max(max_conf_error, (original['conf'] - adapter['conf']).abs().max().item())
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(original['conf'], adapter['conf'], rtol=1e-4, atol=1e-5)
print(json.dumps({'passed': True, 'frames': len(frames), 'checkpoint_hashes': hashes,
                  'max_point_abs_error': max_point_error, 'max_conf_abs_error': max_conf_error,
                  'scope': 'bounded legacy output parity; not throughput proof'}, indent=2))
PY
```

Nếu snippet lỗi, stdout JSON có thể chưa đầy đủ: status FAILED, giữ stderr và
không dùng empty output làm proof. So camera scorer với released evaluator là
gate khác: cần matched input/preprocessing/pose-scale protocol và ghi numerical
differences; hiện không có CLI tự chứng nhận `paper_exact`.

## 8. T1: fit attention trước PPO

```bash
for SEED in 0 1 2; do
  test ! -e "$OUT/seed_$SEED/hybrid_t1.pt"
  python train_hybrid_memory.py --config "$OUT/seed_$SEED/config_hybrid_sampson.json" \
    --manifest "$FULL_MANIFEST" --supervision-manifest "$TRAIN_MANIFEST" \
    --stage t1 --seed "$SEED" --device cuda:0 --trusted-checkpoints --run-training \
    --total-steps "$ATTENTION_STEPS" --epochs "$ATTENTION_EPOCHS" \
    --checkpoint-every "$CANDIDATE_EVERY" --checkpoint-dir "$OUT/seed_$SEED/candidates" \
    --log-jsonl "$OUT/seed_$SEED/attention_training.jsonl" \
    --output "$OUT/seed_$SEED/hybrid_t1.pt"
done
```

T1: AdamW LR1e-4 train ST QKV/output, LT projections/gates/output, fusion. Released
image encoder, decoder, DPT heads, `attn_head_1/2`, `value_encoder`, shared
`norm_q/k/v` và RAFT frozen. Heuristic WRITE deterministic threshold0.92; RL off.
Supervision: calibrated real-GT common-first-camera `res1/res2`, true-valid masks,
fixed detached first-frame scale và radius-normalized point L1. Chín raw frames/
tám pairs, TBPTT bốn pairs. Confidence/depth/pose/temporal loss weights mặc định
zero, không quảng cáo đây là recovered paper loss. Optional causal prefix-aligned
teacher feature loss cần cache/teacher/preprocess/action hashes; không bắt buộc v1.

Source: [`HybridTrainer`](../dust3r/training/hybrid_trainer.py),
[`ReconstructionLoss`](../dust3r/training/losses.py),
[`train_hybrid_memory.py`](../train_hybrid_memory.py).
Không retain toàn bộ video autograd graph. Candidates là **unvalidated**; save ở
clip boundary nên global_step có thể vượt save interval. Log actual steps, epoch
ceiling và `step_ceiling_reached`; checkpoint hai bước smoke không thành production
chỉ vì load được. Review finite loss/gradients và reconstruction validation curve.

T2 chỉ khi T1 curve cho thấy cần adaptation: run riêng từ T1 weights với stage`t2`,
key/value/shared norms LR1e-5, optional last decoder blocks/heads LR5e-6, new memory
LR1e-4. Image encoder vẫn frozen. Certify lại checkpoint T2; không mở toàn network
hoặc PPO cùng lúc để cứu loss. Lưu config/flags/unfrozen list của run này.

## 9. Full validation, checkpoint selection và certification

Không `--max-frames` cho full selection; prefix64 chỉ readiness probe. Giữ cùng
checkout/initial weights/full manifest/IDs/preprocessing. Mỗi output report mới;
evaluator thường ghi đè nếu path tồn tại nên shell phải chặn overwrite.

```bash
for SEED in 0 1 2; do
  ROOT="$OUT/seed_$SEED"
  test ! -e "$ROOT/legacy_validation.json"
  python eval/eval_hybrid.py --config "$ROOT/config_legacy_official.json" \
    --manifest "$FULL_MANIFEST" --split validation --device cuda:0 \
    --trusted-checkpoints --run-evaluation --output "$ROOT/legacy_validation.json"

  CANDIDATES=()
  for checkpoint in "$ROOT/candidates/"*.pt "$ROOT/hybrid_t1.pt"; do
    [[ -f "$checkpoint" ]] || continue
    name=$(basename "$checkpoint" .pt)
    report="$ROOT/validation_${name}.json"
    test ! -e "$report"
    python eval/eval_hybrid.py --config "$ROOT/config_hybrid_sampson.json" \
      --manifest "$FULL_MANIFEST" --split validation --device cuda:0 \
      --trusted-checkpoints --run-evaluation --hybrid-checkpoint "$checkpoint" \
      --output "$report" || { echo 'Validation failed; stop selection.' >&2; exit 1; }
    CANDIDATES+=(--candidate "$checkpoint" "$report")
  done
  test "${#CANDIDATES[@]}" -gt 0
  python tools/select_hybrid_checkpoint.py --baseline-report "$ROOT/legacy_validation.json" \
    "${CANDIDATES[@]}" --ppo-manifest "$FULL_MANIFEST" --min-training-steps "$MIN_FIT_STEPS" \
    > "$ROOT/checkpoint_selection_preview.json"

  # Chỉ sau review preview, completeness và learning curves; không phải auto-accept.
  test ! -e "$ROOT/hybrid_certified.pt"
  test ! -e "$ROOT/hybrid_validation.json"
  python tools/select_hybrid_checkpoint.py --baseline-report "$ROOT/legacy_validation.json" \
    "${CANDIDATES[@]}" --ppo-manifest "$FULL_MANIFEST" --min-training-steps "$MIN_FIT_STEPS" \
    --write-selection --output-checkpoint "$ROOT/hybrid_certified.pt" \
    --output-report "$ROOT/hybrid_validation.json"
done
```

Chạy block theo seed và pause review trước phần write-selection. Tool rank mean
frame-uniform normalized point L1, tie depth AbsRel; mỗi sequence phải giữ cả hai
error trong5% legacy, actual Flash/FLA, source/data/frame contracts khớp. Invalid
candidates ghi errors, không tự repair. Nếu một evaluation run lỗi, giải quyết
completeness trước selection, không chọn một subset thuận lợi của curve.

`MIN_FIT_STEPS` chỉ loại smoke, không chứng minh hội tụ. Review train/val gap,
sequence-level variance và chất lượng hình học. Certification dùng GT camera/
fixed first-frame scale; đây là safety gate cho PPO, **khác deployment/paper metrics**.
Certifier ghi checkpoint mới và report `ppo_certification.validated_for_ppo=true`;
candidate cũ giữ nguyên. Test8–9 không tham gia chọn attention checkpoint.

## 10. Profile action costs và freeze calibration

Sau cả ba seed có certified checkpoints, dùng bounded TRAIN/validation RGB để
profile, không lấy test8–9 tune reward. Study profile dùng seed đầu làm hardware/
config cost calibration dùng chung cho cùng protocol; ghi rõ đây là measured proxy,
không per-frame oracle runtime. Review representativeness và component scope.

```bash
test ! -e "$OUT/component_profile.json"
test ! -e "$OUT/cost_calibration.json"
"${STUDY[@]}" --stage profile --write-configs --server-run > "$OUT/profile_execution.json"
sha256sum "$OUT/cost_calibration.json" > "$OUT/cost_calibration.sha256"
```

Đo synchronized CUDA component costs và xác minh units/hardware/precision. KDA
state cố định: reward không phạt growing LT occupancy; chi phí action là RAFT,
KDA update và geometry. Freeze calibration file trước PPO. Regenerate calibration,
đổi hardware/config/reconstruction checkpoint tạo environment/run mới; exact
PPO resume đòi metadata/hash giống nhau.

Nếu cần profile per seed riêng, tạo OUT/calibration configs riêng và giữ heuristic/
PPO config identical ngoài policy; không ghi đè cost file giữa hai PPO budgets.

## 11. PPO fitting và tạo budget candidates

R0: toàn bộ reconstruction/memory/fusion/key/value/RAFT frozen; chỉ actor/critic
30→64→64→2/1 train. Actor không đọc GT/current RAFT/future frames hoặc full attention
maps. TEMP_ONLY không gọi RAFT/đổi KDA state; forced first viable WRITE; terminal
pair không learned action/long write. Quality reward thuộc **next** decoded left
output, gắn previous action; DenseGT targets từ selected real-GT TRAIN clips.

Dry validate từng seed; dry PPO chỉ construct tiny actor/critic và check paths,
chưa chứng minh certified checkpoint load hợp lệ. Startup real training còn check
certificate, data/environment contracts và calibration. Review JSON như T1.

```bash
for SEED in 0 1 2; do
  ROOT="$OUT/seed_$SEED"
  python train_rl_controller.py --config "$ROOT/config_hybrid_ppo.json" \
    --manifest "$FULL_MANIFEST" --supervision-manifest "$TRAIN_MANIFEST" \
    --seed "$SEED" --hybrid-checkpoint "$ROOT/hybrid_certified.pt" \
    --updates "$PPO_UPDATES" --dry-run --validate-data --validate-modules \
    --device cuda:0 > "$ROOT/ppo_dry_u${PPO_UPDATES}.json"
done
python - <<'PY'
import json, os
from pathlib import Path
for seed in (0, 1, 2):
    path = Path(os.environ['OUT']) / f"seed_{seed}/ppo_dry_u{os.environ['PPO_UPDATES']}.json"
    report = json.loads(path.read_text())
    assert report['requirements_ready'], report['blocking_requirements']
    assert not report['missing_training_budgets'], report
print('PPO dry requirements pass; certified loading/runtime still validated at startup.')
PY

for SEED in 0 1 2; do
  ROOT="$OUT/seed_$SEED"
  test ! -e "$ROOT/ppo_updates_${PPO_UPDATES}.pt"
  test ! -e "$ROOT/actor_updates_${PPO_UPDATES}.pt"
  python train_rl_controller.py --config "$ROOT/config_hybrid_ppo.json" \
    --manifest "$FULL_MANIFEST" --supervision-manifest "$TRAIN_MANIFEST" \
    --seed "$SEED" --hybrid-checkpoint "$ROOT/hybrid_certified.pt" \
    --updates "$PPO_UPDATES" --device cuda:0 --trusted-checkpoints --run-training \
    --log-jsonl "$ROOT/ppo_training_u${PPO_UPDATES}.jsonl" \
    --output "$ROOT/ppo_updates_${PPO_UPDATES}.pt" \
    --actor-output "$ROOT/actor_updates_${PPO_UPDATES}.pt"
done
```

Trainer hiện chỉ lưu cuối run, **không có periodic PPO-save flag**. Để so budget,
chạy increasing TOTAL updates với output/actor filenames mới; dùng
`--resume-checkpoint previous_ppo_training.pt` để tiếp tục actor/critic/optimizers/
RNG trên cùng config/calibration/data/reconstruction. `--updates` là tổng ceiling,
không số update cộng thêm; episode reset ở resume, không replay cursor giữa video.
Ví dụ operator chọn U2>U1 thì command như trên thêm resume U1 và output U2.
Đổi policy/reward hyperparameters là run mới, không exact-resume.

PPO run hoàn tất chưa chứng minh actor tốt; actor artifact ban đầu có
`validated_for_deployment=false`. Không sửa cờ này bằng tay để bypass validation.

## 12. Validation actor: quality + compute, không test selection

Đây là deployment scorer, khác report GT-camera `eval_hybrid` để certify attention.
Đánh giá validation7 cho official, hybrid heuristic và từng PPO budget bằng cùng
full source IDs/protocol. Dùng candidate ID riêng `ppo_uU`; renderer không join hai
actor khác nhau dưới cùng `(hybrid_ppo,seed)`.

```bash
for SEED in 0 1 2; do
  ROOT="$OUT/seed_$SEED"
  for EXP in legacy_official hybrid_sampson; do
    WEIGHTS=()
    if [[ "$EXP" == hybrid_sampson ]]; then
      WEIGHTS=(--hybrid-checkpoint "$ROOT/hybrid_certified.pt")
    fi
    test ! -e "$ROOT/quality_val_$EXP.json"
    test ! -e "$ROOT/performance_val_$EXP.json"
    python eval/eval_benchmark.py --config "$ROOT/config_$EXP.json" --manifest "$FULL_MANIFEST" \
      --split validation --seed "$SEED" --experiment-id "$EXP" "${WEIGHTS[@]}" \
      --device cuda:0 --trusted-checkpoints --run-evaluation --output "$ROOT/quality_val_$EXP.json"
    python tools/benchmark_experiment.py --config "$ROOT/config_$EXP.json" --manifest "$FULL_MANIFEST" \
      --split validation --seed "$SEED" --experiment-id "$EXP" "${WEIGHTS[@]}" \
      --repeats 3 --warmup-frames 4 --device cuda:0 --trusted-checkpoints --server-run \
      --output "$ROOT/performance_val_$EXP.json"
  done

  ID="ppo_u${PPO_UPDATES}"
  ACTOR="$ROOT/actor_updates_${PPO_UPDATES}.pt"
  test ! -e "$ROOT/quality_val_$ID.json"
  test ! -e "$ROOT/performance_val_$ID.json"
  python eval/eval_benchmark.py --config "$ROOT/config_hybrid_ppo.json" --manifest "$FULL_MANIFEST" \
    --split validation --seed "$SEED" --experiment-id "$ID" \
    --hybrid-checkpoint "$ROOT/hybrid_certified.pt" --actor-checkpoint "$ACTOR" \
    --device cuda:0 --trusted-checkpoints --run-evaluation --output "$ROOT/quality_val_$ID.json"
  python tools/benchmark_experiment.py --config "$ROOT/config_hybrid_ppo.json" --manifest "$FULL_MANIFEST" \
    --split validation --seed "$SEED" --experiment-id "$ID" \
    --hybrid-checkpoint "$ROOT/hybrid_certified.pt" --actor-checkpoint "$ACTOR" \
    --repeats 3 --warmup-frames 4 --device cuda:0 --trusted-checkpoints --server-run \
    --output "$ROOT/performance_val_$ID.json"
done
```

Khi thêm budget, chỉ chạy PPO candidate portion với ID mới; không ghi đè baseline/
heuristic reports. Render validation preview với danh sách explicit quality+timing
paths của mọi candidate, output riêng. Actor component overhead có thể N/A nếu
không cung cấp synchronized component profile; total latency đo được vẫn cần
review. Final study benchmark sẽ thêm actor profile bằng TRAIN/validation RGB.

**Acceptance rule phải khóa trước selection**, không tune trên test: ví dụ quality
non-inferiority tolerance đã công bố so heuristic, full prediction/GT coverage,
finite camera metrics, và total latency/RAFT calls giảm đủ vượt repeat variability.
Tolerance/desired gain là operator decision, không có threshold actor certified
được recovered từ paper. Khi chưa chọn thì status `ACTOR_ACCEPTANCE_UNSELECTED`.
Không buộc actor thắng nếu experiment evidence cho thấy trade-off không tốt.

Hiện **chưa có actor selector/certifier CLI**. Người vận hành review curve/frontier
và viết external `actor_acceptance.json` trong seed root: chosen actor path/SHA,
attention SHA, validation quality/timing report paths/SHAs, locked tolerance/
selection rule, metrics/rates, accepted/rejected và reviewer/reason. Đây là manual
experiment acceptance record, không giả certificate trong checkpoint.

Sau accepted record, copy **byte-for-byte** actor đã chọn tới expected `actor.pt`:

```bash
# Lặp từng seed với actor đã được chấp nhận; không mặc định chọn budget cuối.
export SEED=0
export SELECTED_ACTOR="$OUT/seed_$SEED/actor_updates_${PPO_UPDATES}.pt"
python - <<'PY'
import hashlib, os
from pathlib import Path
root = Path(os.environ['OUT']) / f"seed_{os.environ['SEED']}"
source = Path(os.environ['SELECTED_ACTOR'])
assert (root / 'actor_acceptance.json').is_file(), 'Write/review manual acceptance record first.'
import json
record = json.loads((root / 'actor_acceptance.json').read_text())
raw = source.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert record.get('accepted') is True
assert record.get('chosen_actor_sha256') == digest
with (root / 'actor.pt').open('xb') as handle:
    handle.write(raw)
print('Selected actor copied without modifying its metadata:', digest)
PY
```

Lặp cho seeds1/2 với actual selected actor; không sửa `validated_for_deployment`
flag. Study executor hiện không có actor-acceptance automation gate; manual record
và frozen test procedure là trách nhiệm experiment. Nếu actor không đạt quality/
compute acceptance, giữ result thất bại, tăng budget/đổi hyperparameters chỉ dựa
validation trong run mới. Không chọn bằng test performance.

## 13. Held-out evaluation, timing và bảng benchmark ba seed

Sau attention và actor được chọn/frozen, kiểm tra expected artifacts và fresh
quality/timing paths. Không fit/select tiếp bằng test8–9.

```bash
for SEED in 0 1 2; do
  for artifact in hybrid_certified.pt hybrid_validation.json actor.pt actor_acceptance.json kernel_conformance.json; do
    test -f "$OUT/seed_$SEED/$artifact"
  done
  for EXP in legacy_official hybrid_sampson hybrid_ppo; do
    test ! -e "$OUT/seed_$SEED/quality_$EXP.json"
    test ! -e "$OUT/seed_$SEED/performance_$EXP.json"
  done
done
"${STUDY[@]}" --stage evaluate benchmark summarize --write-configs --server-run \
  > "$OUT/test_execution.json"

# Kiểm tra tất cả18 quality/performance artifacts trước merge bảng ba seed.
REPORTS=(
  "$OUT"/seed_{0,1,2}/quality_{legacy_official,hybrid_sampson,hybrid_ppo}.json
  "$OUT"/seed_{0,1,2}/performance_{legacy_official,hybrid_sampson,hybrid_ppo}.json
)
for report in "${REPORTS[@]}"; do test -f "$report"; done
test ! -e "$OUT/benchmark_all_seeds"
python tools/render_benchmark.py --reports "${REPORTS[@]}" \
  --catalog configs/experiments/scared_study.json \
  --paper-reference configs/experiments/paper_reference.json \
  --output "$OUT/benchmark_all_seeds"
```

Artifacts: `benchmark.csv`, `.json`, `.md`, `.html`, quality-vs-FPS/VRAM SVG.
Missing measurements phải N/A; không tự điền paper numbers vào measured row.
Pilot seed0 không thành average ba seed. Review per-sequence aggregation, failures,
coverage và actual seed count trước dùng aggregate/variance cho report.

Metric bảng: depth AbsRel/SqRel/RMSE/logRMSE/delta1; pose ATE/RPE translation và
rotation trên five-frame windows khi GT/PnP hỗ trợ; FPS, latency p50/p95, peak VRAM,
WRITE/RAFT/KDA rates, actor component overhead nếu có. Ghi units/scale/alignment.
Timing streaming input/output residency khác original demo full-GPU residency;
streaming matched reruns và official numerical parity phải được công bố.
Whole-sequence exact depth median dùng CPU samples, không constant-memory scorer.

Paper reference là published row riêng. Chỉ so speedup với matched official rerun
cùng GPU/driver/precision/warmup/repeats/sequence lengths/scope; không chia paper
FPS với local/server FPS rồi tuyên bố architectural speedup. Giữ
`paper_training_recipe_reproduced=false`; label/split/loss/timing/crop chưa đủ
đồng nhất để gọi exact paper reproduction.

## 14. Resume, lỗi phổ biến và điều kiện dừng

| Dấu hiệu | Kiểm tra | Hành động đúng |
| --- | --- | --- |
| `Missing real GT coverage` | Inventory raw/crop/resize/final, source IDs/masks | TRAIN valid runs; held-out thì partial-GT implementation, không fake mask |
| No valid9-frame clip | Histogram/runs, supervision threshold, target validity | Dừng; review label quality/clip protocol, không nối discontinuities |
| Dry exit0 nhưng chưa ready | JSON `requirements_ready`, blockers/budgets | Xử lý từng blocker trước real run |
| Flash/KDA import/kernel/JIT error | Environment pin, GPU/driver, dtype, Triton, proof | Dừng efficient experiment; CPU reference chỉ correctness |
| Missing RAFT weights | Verified checkpoint + official path/hash | Operator cung cấp local checkpoint; không auto-download |
| Legacy NaN/crash | Official config/code/provenance và scope | Report baseline failure; clean legacy nếu cần là separate experiment |
| Candidate only smoke / val regress | Actual steps/curve/coverage/gradients | Chọn budget/refit từ validation; không bật certificate thủ công |
| Source/hash/config mismatch | Checkout drift, data/target selection, report source IDs | Run mới hoặc re-evaluate matched sources; không chỉnh hash để pass |
| PPO resume rejected | Config/data/calibration/attention SHA; total update budget | Giữ environment cũ hoặc run mới, không nới validation |
| Actor all-write/all-skip | Forced initialization vs learned rows, obs norms, reward components | Review validation quality/compute; collapsed policy không auto-success |
| Actor checkpoint stale | Environment contract/attention/Sampson config | Retrain actor cùng frozen environment mới |
| Report files already exist | Unique OUT/candidate identity | Không overwrite; chọn output mới và giữ provenance |
| Renderer join rejected | experiment_id/seed/protocol/checkpoint/input/timing contracts | Sửa run identity/report nguồn bằng rerun; không edit metrics |

Attention exact resume: `--resume-checkpoint PATH --resume-state`, cùng stage/config/
training data contract/optimizer settings và budgets tổng đã review; trainer resume
ở fresh clip boundary. Weights-only adaptation dùng run/stage mới, không gọi exact
continuation. PPO resume cần training-state artifact, không chỉ actor inference file.
Không hứa checkpoint giữ chính xác cursor/autograd giữa video.

## 15. Checklist đóng ba blocker và artifact archive

| Blocker | Điều kiện thực sự đã giải quyết | Artifact cần lưu |
| --- | --- | --- |
| GT coverage | Inventory mọi split; TRAIN valid runs đủ tín hiệu; full held-out scoring hợp lệ hoặc partial-GT protocol đã implement/test | Inventory, excludes/counts, source+TRAIN manifests/SHAs, protocol coverage |
| Attention chỉ smoke / actor chưa train | Real kernel gradients; meaningful learning curve; full-val selected/certified checkpoint; PPO actor validated quality/compute | Proofs, fit JSONL/candidates, all validation reports, certified checkpoint, PPO states/actors, acceptance record |
| Budget chưa chọn | Explicit positive steps/epochs/updates và minimum fit/save intervals, stopping rationale + actual work | `budget_decisions.json`, actual counters/curve, change records |

Các bước code đã có chỉ tạo điều kiện xử lý; không coi lệnh được viết hoặc tests
synthetic pass là real-data blocker đã đóng. StereoAnything/pseudo-target và
partial-GT guides giải thích đủ cách triển khai các nhánh thiếu, nhưng chưa cung
cấp implementation hoặc measurement tương ứng.

Archive ngoài repo: source commit/tree/dirty status, input/checkpoint/manifest hashes,
mapping/units/calibration/rectification provenance, environment lock/hardware,
resolved configs/seeds/budgets, coverage/exclusions, model freeze/trainable groups,
all fit/PPO logs, candidates và selection reasons, calibration, certified attention,
accepted actor/report hashes, all quality/perf reports và rendered table. Đảm bảo
không lẫn TRAIN filtering với evaluation sample set.

**Readiness hiện tại: READY_FOR_SERVER_VALIDATION_WITH_BLOCKERS.** Cần server
environment/data/kernel proof, budgets, real fitting/selection/actor validation;
held-out partial-GT và StereoAnything branch cần code theo guides riêng nếu dùng.
Guide hoàn chỉnh không đồng nghĩa đã có certified checkpoint hoặc paper-exact result.

## 16. Kiểm chứng local đã thực hiện

Ở lượt triển khai coverage/selection trước: full CPU suite248 tests,247 passed,
một Windows symlink test skipped do quyền hệ thống; sau hai guardrail certification
mới,41 focused tests coverage/selection/integration/validation đều pass. Hai test
mới không nằm trong tổng full-suite đã chạy trước đó. AST19 files, CLI help,
reference construction/optimizer dry và study37-command dry plan đạt.
Fixture4 raw pixels→zero resized tái hiện sparse-GT case bằng synthetic data,
không phải measurement của frame1729 trên server.

Lượt mở rộng guide chỉ thay documentation. Static checks trên bốn guides đạt:
32 Bash blocks bằng `bash -n`,13 Python snippets bằng AST,28 local links và23
CLI references/flags. Chỉ kiểm tra syntax, không thực thi snippets provisioning/
inference/training. Có `git diff --check`; không install/download, model-hub call, real
training, PPO fitting hoặc CUDA benchmark. Source changes từ lượt trước được giữ
nguyên; chưa commit/push trong lượt documentation này.
