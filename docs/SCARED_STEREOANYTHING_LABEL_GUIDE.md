# StereoAnything → nhãn SCARED: môi trường, exporter và target contract

**Trạng thái: cần bổ sung exporter và pseudo-target support trước khi chạy.**
Đây là hướng dẫn kỹ thuật đầy đủ cho nhánh nhãn gần paper hơn, không phải một CLI
đã có. Không có `export_scared_stereoanything.py` trong repository hiện tại.
Không thay disparity A00 hiện có hoặc tự set pseudo-depth thành genuine GT.

Quay lại [guide server](SCARED_SERVER_FIT_GUIDE.md). Nhánh real-GT supervision
không bắt buộc StereoAnything; nhánh paper-style labels cần công việc bên dưới.

## 1. Nguồn và phạm vi tái lập

[Endo3R §3.1](https://arxiv.org/html/2504.03198v1) dùng StereoAnything trên stereo
rectified để tạo nhãn training. [Hướng dẫn StereoAnything chính thức](https://github.com/XiandaGuo/OpenStereo/blob/v2/docs/StereoAnything.md)
nêu NMRF Swin-T config và `StereoAnything-NMRF_swinT.bin`. Ví dụ evaluation KITTI
của upstream không export raw numeric disparity cho SCARED.

Exact StereoAnything revision, toàn bộ training mixture và loss recipe của
Endo3R không được release. Tạo được nhãn này vẫn chưa đủ tuyên bố `paper_exact`.

Các điểm compatibility sau được kiểm chứng trên upstream branch `v2`; branch
có thể thay đổi, cần lưu commit SHA khi triển khai:

| Điểm | Source upstream | Hệ quả |
| --- | --- | --- |
| NMRF bị comment khỏi registry | [stereo/modeling/__init__.py](https://github.com/XiandaGuo/OpenStereo/blob/v2/stereo/modeling/__init__.py) | Latest clone + config NMRF chưa chắc build được |
| Backbone preload dùng hardcoded private checkpoint path | [NMRF backbone.py](https://github.com/XiandaGuo/OpenStereo/blob/v2/stereo/modeling/models/nmrf/backbone.py) | Released stereo bin không tự giải quyết đường dẫn riêng |
| Inference xuất visualization PNG | [tools/infer.py](https://github.com/XiandaGuo/OpenStereo/blob/v2/tools/infer.py) | Không lấy color image làm numeric disparity |

Source path NMRF có thể đổi giữa revision. Kiểm tra bằng `git ls-tree`/`rg` trước
khi patch; bảng chỉ ghi cấu trúc đã inspect, không hứa path của mọi checkout.
Registry còn kéo dependencies model khác; requirements cũ và API `timm.layers`
có nguy cơ không khớp. Không ghi một version timm tùy đoán thành đã validated.

## 2. Chọn revision và môi trường trên server

Tạo môi trường **riêng** cho label generator. Không thay Torch/timm của môi trường
EndoARL đã certify. Không chạy các bước này tại local, không để model auto-download.

Operator cung cấp OpenStereo checkout và released checkpoint đã xác minh theo
quyền truy cập của mình. Trong checkout đó, dùng lịch sử Git để xác định revision
có đủ NMRF registry, config và inference; không có historical pin đã verified ở đây:

```bash
git rev-parse HEAD
git log --all --oneline -- stereo/modeling/__init__.py docs/StereoAnything.md
git ls-tree -r HEAD --name-only | grep -E 'nmrf|StereoAnything|infer.py'
git show HEAD:stereo/modeling/__init__.py
git status --short
```

Các lệnh này chỉ inspection. Chọn pin trên checkout sạch riêng; lưu hash và diff
nếu phải patch external code. Requirements để review: PyTorch/CUDA pair đúng
hardware, timm API đúng backbone, yacs, omegaconf, einops và các dependency upstream
thực import. `requirements.txt` của revision không mặc nhiên là dependency lock.

NMRF dùng custom multi-scale deformable attention/downsampling ops. Kiểm tra
`make.sh`/`setup.py` của revision, CUDA_HOME, nvcc, C++ compiler và matching Torch
ABI; build chỉ trên server trong environment này. Không copy binary từ máy khác
mà không kiểm tra GPU/ABI. Ghi compiler/CUDA/version/build log và op import proof.

Hai hướng xử lý có thể review:

- Dùng historical revision thực có NMRF hoạt động và pin commit sau smoke test.
- Nếu dùng v2, tạo external patch được đặt tên/lưu diff: đăng ký NMRF, tách import
  model không dùng, bỏ redundant backbone preload **chỉ khi** full released stereo
  checkpoint cung cấp mọi learned weight; hoặc thay hardcoded path bằng verified
  local backbone checkpoint. Không đổi checkpoint semantics âm thầm.

**Stop gate:** checkpoint không đủ weights, registry không build được, missing
custom ops, network preload còn tồn tại hoặc không biết raw disparity convention.
Không chuyển sang model stereo khác rồi gọi nó là StereoAnything.

## 3. Input SCARED cần đủ gì?

Mỗi frame: synchronized rectified left/right RGB và frame ID, source resolution,
K_left/K_right của **rectified source grid**, stereo extrinsics/baseline và unit.
Lưu source RGB hashes và rectification provenance. Raw calibration của camera
chưa rectified không thay thế rectified-grid calibration.

Không đưa image đã normalize `[-1,1]` của Endo3R vào NMRF. Dùng đúng evaluating
transform của selected NMRF config; lưu normalize/channel-order/pad/crop/resize.
Stereo pair và disparity phải cùng grid. Nếu resize ngang với hệ số s_x,
disparity theo pixel cũng đổi s_x; baseline metres không đổi.

## 4. Numeric exporter: API đã inspect và việc cần implement

Upstream API v2 đã inspect: config loader tạo EasyDict, build_trainer trả
`trainer.model`, evaluating transform từ config, model eval returns
`model(sample)['disp_pred']` với disparity `[B,H,W]`. Kiểm tra lại cùng pin trước code.
Không copy một constructor signature phỏng đoán cho revision khác.

Quy trình exporter cần viết:

1. Parse local source index, config, verified checkpoint, output root mới và GPU.
   Source/data/model path thiếu thì fail; output hiện hữu không ghi đè.
2. Construct đúng NMRF model/transform từ revision được pin. `run_mode='infer'`,
   `dist_mode=False` theo API upstream; không tạo trainer training/dataloader vô ích.
3. Checkpoint loading phải kiểm tra mọi learned weight và shape. Upstream loader
   `strict=False` có thể bỏ keys; exporter phải fail trên missing/shape mismatch,
   trừ allowlist đã review, được ghi trong provenance. Không giữ random parameters.
4. Frozen `.eval()` + `torch.no_grad()`, dtype đã sanity-check; không gọi hub.
5. Transform synchronized float32 RGB arrays cùng H×W×3 theo đúng config; add batch
   dimension và device; chạy `disp_pred`, giữ numeric float32 sau unpad.
6. Xác nhận NMRF forward đã gỡ internal pad-to-32, **không unpad lần hai**.
   Exporter chỉ gỡ external padding còn lại;
   config còn có `RightTopPad` tối thiểu544×960. Với transform này,
   `pad_top=max(544-H,0)`, `pad_right=max(960-W,0)`; lấy
   `disp[pad_top:pad_top+H, :W]` sau forward nếu còn external padding.
   Các constants này thuộc config đã inspect, không hardcode cho config khác.
7. Kiểm tra output grid chính xác source H×W. Lưu `.npy` numeric disparity/depth
   và boolean mask; optional color PNG chỉ để QA, không làm training target.
8. Tạo report frame-by-frame với invalid reasons, left/right ID, valid pixels,
   coverage, calibration mapping và checkpoint/config/source hashes.

Implementation đề xuất ngoài core legacy: `tools/export_scared_stereo_targets.py`
và `dust3r/data/stereo_targets.py`. Đây là file **chưa có**, không đưa command dùng
tên này vào automation trước khi thực hiện và kiểm thử.

## 5. Disparity → depth → common-frame pointmap

Với rectified pair, quy ước `d = x_left - x_right` và K xác minh:

Điều kiện: matched horizontal focal lengths, same-row rectified geometry và
standard left/right baseline ordering. K_left/K_right tùy ý không đủ để áp dụng
công thức này. Nếu calibration không đáp ứng, rectify lại có provenance hoặc
dùng triangulation đã đặc tả/verified; không tự bỏ principal-point offset.

```text
denom = disparity - (cx_left - cx_right)
depth_m = fx_left * baseline_m / denom
```

Chỉ giữ finite positive denominator/depth, in-bounds correspondence và model
validity nếu model cung cấp. Occlusion/confidence handling phải được công bố;
không tự gọi mọi finite value là chắc chắn đúng. Nếu model không có validity,
mask này là numeric/geometric validity, không phải ground-truth occlusion mask.

Pointmap camera:
`X=(x-cx)*Z/fx`, `Y=(y-cy)*Z/fy`, `Z=depth_m`.
Nếu dùng GT pose để đưa sang first-camera frame, áp dụng
`T_first_from_world @ T_world_from_current`; origin là first RGB frame của sequence
nguồn. Pose là real GT; **depth vẫn là stereo pseudo-label**. Target file giữ
origin được khai báo bằng `first_camera_origin_camera_to_world`; loader sau đó
đưa output targets sang first selected clip camera như prediction của clip.
Không đổi nhãn origin của file sang clip camera mà không transform tọa độ.
Không đổi unit khi crop một run training từ giữa sequence.

StereoAnything không cung cấp temporal poses. Common-frame point L1 trainer
hiện tại vẫn cần verified pose từng frame. Nếu không có pose, dừng nhánh trainer
này hoặc đặc tả depth-only/self-supervised trainer riêng; nhãn stereo một mình
không thay thế đủ supervision hiện có.

Sanity checks: rectification epipolar residuals, baseline sign/unit, source-grid
cx offset, reprojection to right view, median range và visualization overlay.
Đối chiếu real GT TRAIN/validation để QA label generator, không tune bằng test8–9.

## 6. Target-kind support: blocker cần giải quyết trong EndoARL

Code hiện tại [`SequenceDataset`](../dust3r/training/sequence_dataset.py) yêu cầu
genuine-GT provenance; [`validate_supervision_manifest`](../dust3r/data/supervision.py)
ràng buộc target paths của derived manifest giống source real GT. Thay path sang
pseudo `.npy` sẽ bị từ chối, đúng mục đích. Không bypass bằng `depth_or_pointmap_gt=true`.

Contract mới cần tách:

- `input_source_manifest_sha256`: canonical full RGB/calibration/pose timeline.
- `training_target_manifest_sha256`: stereo targets riêng, train-only identity.
- `target_kind`: `real_gt`, `stereo_pseudo`, `teacher_proxy`.
- `generator`: repo revision + patch SHA, checkpoint SHA, config/transform SHA,
  precision/kernel version, disparity convention, output units, grid/rectification.
- Target mask semantics/coverage, frame/source IDs và first-camera origin.

Thêm explicit training-target loader/converter; evaluator/certifier hard-reject
pseudo target kind. Reuse numeric point L1 được phép với mode ghi rõ
`stereo_pseudo_supervised`, không quảng cáo nó là original supervised GT recipe.
Checkpoint/data-contract/resume/teacher-cache checks phải bind cả input và target.
Real-GT validation/test manifest giữ nguyên; pseudo label không được làm scoring GT.

PPO v1 tiếp tục `DenseGTReward` trên derived **real-GT** TRAIN runs. Hybrid memory
đã fit pseudo-label có thể được certify bằng real-GT validation rồi freeze làm PPO
environment. Chuyển reward sang pseudo depth là proxy-reward experiment riêng,
phải implement một target-aware reward mode; không giả verified GT.

## 7. Test, artifacts và thứ tự triển khai

| Bước | Artifact | Success gate | Failure gate |
| --- | --- | --- | --- |
| Pin upstream + env | revision, patch diff, pip/build lock | Complete weights và custom ops load offline | Missing/random weights, incompatible timm/API |
| One rectified TRAIN pair | raw disparity, depth/mask, QA report | Exact source grid, sane units/correspondences, finite masked output | Colored PNG as data, wrong grid/calibration |
| Numeric exporter tests | Tests/fixture report | Pad/unpad, cx offset, resize-disparity, invalid masks, fresh output, IDs deterministic | Data overwrite or hidden invalids |
| Target-kind integration | New schema + loader/contracts/tests | Pseudo training allowed explicitly, real-GT eval guard intact | Pseudo passes certification as GT |
| Label coverage audit | Counts/masks/provenance for each frame | Enough contiguous TRAIN clips at declared threshold | Dense appearance claimed as genuine coverage |
| New hybrid fit run | New checkpoint/learning curves | Real-GT validation selects/certifies | Reuse old cert after changing labels/config |
| PPO + final benchmark | Frozen env actor and real-GT metrics | Same attention weights for heuristic/PPO, held-out untouched | Test selection or changed baseline |

Tests tối thiểu: raw numeric output roundtrip, checkpoint-key completeness,
no auto-download, no source overwrite, matching stereo IDs, pad/unpad arbitrary
H/W, known plane with unequal cx, baseline units, invalid/OOB/NaN masking,
first-camera origin preservation và pseudo-vs-real evaluator rejection.

**SERVER_VALIDATION_REQUIRED:** source pin/patch, dependencies/custom CUDA ops,
released weights và one-pair QA trước bulk label generation. Guide không cho phép
bỏ các gate bằng cách dùng random model hoặc target giả. Không có nhãn nào được
tạo/tải về trong công việc documentation local.
