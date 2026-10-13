# SCARED partial-GT: protocol và hướng dẫn bổ sung code

**Trạng thái: BLOCKED — đặc tả triển khai, chưa phải tính năng chạy được.**
Guide này xử lý validation/test có RGB hợp lệ nhưng GT depth/pointmap thiếu hoặc
còn zero pixel sau crop/resize. Công cụ coverage đã có; scorer partial-GT chưa có.
Không có flag `--allow-partial-gt` trong CLI hiện tại. Không thêm flag vào lệnh chạy
trước khi triển khai, kiểm thử và version hóa protocol bên dưới.

Quay lại [quy trình server](SCARED_SERVER_FIT_GUIDE.md).

## 1. Phân biệt ba tình huống

| Tình huống | TRAIN hiện tại | Validation / test hiện tại | Hướng xử lý |
| --- | --- | --- | --- |
| GT có coverage đủ ở mọi frame | Chạy được nếu các gate khác đạt | Chạy được | Giữ protocol hiện tại |
| TRAIN có frame zero GT, held-out đủ GT | Derived TRAIN-only valid contiguous runs | Giữ nguyên held-out/full RGB | Dùng inventory + supervision manifest đã implement |
| Validation/test có frame zero hoặc thiếu GT | Không sửa bằng TRAIN filtering | Strict loader/scorer dừng | Bổ sung partial-GT protocol riêng trước full benchmark |

Không chuyển test sang manifest TRAIN, nối qua lỗ hổng frame hoặc đổi zero thành
GT hợp lệ. Không lấy metric từ số ít frame rồi gắn nhãn full-sequence mà bỏ coverage.

## 2. Vì sao code hiện tại chưa đủ?

Các điểm được kiểm chứng từ source:

- [`SequenceDataset._validate_sequence` và `_load_frame`](../dust3r/training/sequence_dataset.py)
  yêu cầu target/calibration/pose đã verified và GT có valid pixels. Tham số
  `allow_empty_gt` chỉ phục vụ inspection trong audit, không phải evaluation mode.
- [`StreamingSequenceEvaluator.add_frame`](../dust3r/evaluation/streaming.py)
  kiểm tra GT/mask trước camera estimation. Zero-GT ở frame đầu có thể đặt
  `_first_prediction_failed`, khiến prediction camera ở các frame sau cũng thất bại.
  Đây là sự phụ thuộc cần loại khỏi scorer mới, không sửa âm thầm legacy network.
- [`eval/eval_hybrid.py`](../eval/eval_hybrid.py) dùng một first-frame episode scale
  cho certification. First frame thiếu GT khiến scale không xác định.
- [`benchmark`](../dust3r/experiments/benchmark.py) vẫn đi qua strict dataset;
  dù timing model không cần GT, loader hiện chặn một số video partial-GT.
- [`DenseGTReward`](../dust3r/rl/rewards.py) cần first-GT hợp lệ và target tương lai
  hợp lệ. Missing GT là vấn đề dữ liệu, không phải model failure để phạt bằng error cap.

## 3. Protocol mới phải khóa trước khi code

Tên đề xuất: `scared_partial_gt_v2` — **PROVISIONAL** cho đến khi được review.
Đây là protocol của project, chưa có bằng chứng khớp toàn bộ evaluator của paper.

1. **RGB timeline:** giữ mọi frame/source ID trong manifest gốc, cùng stride1,
   cùng crop/resize320×256 và sequence reset cho cả ba model.
2. **Inference và timing:** mọi RGB đều được xử lý theo timeline gốc; GT availability
   không quyết định memory update, camera inference hay mẫu số FPS.
3. **Depth scoring eligibility:** xác định chỉ từ target thực và mask sau cùng,
   trước khi xem prediction. Ngưỡng pixel/fraction được khóa từ protocol; metric
   dùng đúng tập GT-source IDs/mask cho mọi model. TRAIN threshold có thể khác.
4. **Prediction failure:** khi frame đủ GT nhưng prediction/camera thất bại, ghi
   model failure; không bỏ frame khỏi denominator eligibility rồi báo chất lượng đẹp.
   Nếu không có quy tắc phạt đã thống nhất, headline quality mang status failed/N/A.
   Failure trên frame không có depth GT cũng phải được ghi trong full-RGB
   prediction coverage; GT absence không xóa lỗi model/camera hoặc timeline failure.
5. **Missing GT:** frame không đủ target ghi `gt_unavailable`/`insufficient_coverage`,
   vẫn giữ prediction và latency; không tính depth error giả bằng zero.
6. **Corrupt GT:** file hỏng, calibration sai hoặc provenance giả là lỗi dữ liệu.
   Không tự chuyển mọi exception thành `gt_unavailable`.
7. **Pose:** tính khi pose GT và prediction camera tồn tại, không yêu cầu depth GT
   của cùng frame. Window gồm năm frame liên tiếp trên timeline RGB gốc;
   không nén các frame có pose/GT thành timeline mới.
8. **Scale:** deployment depth giữ quy tắc scale hiện tại và công bố tập pixel/frame
   dùng để fit scale: median GT depth / median predicted depth trên cùng fixed
   genuine-GT support toàn sequence. Nếu không có GT nào đủ điều kiện, depth metrics là N/A.
   Không suy luận scale metric từ reference pose để cứu depth nếu protocol không cho phép.
9. **Certification:** giữ strict first-RGB-frame scale cho phiên bản đầu. Nếu first
   frame không đủ GT, sequence không đủ điều kiện certification hiện hành và phải dừng.
   Muốn dùng first-valid-GT anchor cần protocol/certificate schema mới, không đổi
   lặng lẽ index của camera origin hoặc cấp chứng nhận bằng evaluator hiện tại.

Ngay cả khi first frame đủ GT, validation có missing GT ở frame khác vẫn không
được certify bằng v1 hiện tại: `certify_for_ppo` đòi finite metrics ở mọi report
frame. Mở rộng certification cho sparse support cần protocol/schema mới và
fixed GT-support hash, không chỉ thêm khả năng chọn scale anchor.

GT không được vào actor, encoder hoặc memory. Chỉ scorer/reward training có quyền
đọc target. Memory state reset ở ranh giới sequence/patient như hiện tại.

## 4. Interface dữ liệu đề xuất

**Các tên dưới đây là API cần bổ sung, chưa tồn tại trong repository.**

`SequenceInput`: `episode_id`, `source_frame_id`, `source_index`, RGB, true_shape,
source-grid/prepared-grid mapping, sequence reset flag. Không chứa GT cho actor.

`ScoringTarget`: cùng identity, `target_kind=real_gt`, optional depth/pointmap/mask,
optional K/pose, `depth_status`, `pose_status`, `valid_pixel_count`,
`coverage_fraction`, provenance/hash và lý do không có target.

Status hợp lệ: `available`, `missing`, `insufficient_coverage`, `corrupt`.
Schema validation không coi pseudo-depth là real-GT. Target-file thiếu chỉ được
coi là missing nếu manifest nguồn khai báo rõ; file được khai báo tồn tại mà biến
mất sau audit là integrity error.

Reader mới ghép input và optional scoring target theo ID, không zip hai list đã
filter. Camera-to-world/first-camera origin vẫn là frame RGB đầu của sequence.
Không thay origin thành frame có GT đầu tiên khi tạo pointmap targets.

## 5. Bản đồ file cần sửa / thêm

| File | Công việc cần làm | Điều phải giữ |
| --- | --- | --- |
| `dust3r/data/evaluation_sequence.py` — mới, đề xuất | Full-RGB input reader + optional real-GT scoring targets | Source IDs/order/origin; strict corruption handling |
| `dust3r/training/sequence_dataset.py` | Tách helper geometry/crop có thể dùng chung | TRAIN và DenseGT strict mặc định; không thay legacy forward |
| `dust3r/evaluation/streaming.py` | Camera/depth prediction chạy trước GT eligibility; scorer xử lý availability riêng | Camera được suy ra từ prediction; GT chỉ scoring |
| `dust3r/evaluation/reports.py` | Schema/protocolID/coverage fields và fingerprint mới | Không đọc nhầm partial-v2 như protocol cũ |
| `eval/eval_benchmark.py` | Explicit protocol dispatch tới reader/scorer mới | Không có silent fallback hay filtered test |
| `dust3r/experiments/benchmark.py`, `tools/benchmark_experiment.py` | Full-RGB timing loader độc lập GT | Warmup/repeats/precision/timing scope unchanged |
| `eval/eval_hybrid.py`, `dust3r/training/validation.py`, `dust3r/training/selection.py` | Chỉ khi mở rộng certification: GT eligibility và camera-origin/scale contract versioned | Strict v1 không bị nới lỏng |
| `dust3r/experiments/study.py` | Kiểm tra đồng nhất coverage/protocol trước aggregate | Không select checkpoint/actor bằng test |
| `tests/test_partial_gt_evaluation.py` — mới, đề xuất | Fixtures và contract tests bên dưới | Real-GT provenance bắt buộc |

Các path ở hàng đầu/mới là đề xuất. Kiểm tra module protocol thực tế khi coding;
không tạo dependency vào module chưa tồn tại chỉ vì nó được nêu trong guide.

## 6. Thứ tự scorer mới

```text
for every original RGB frame:
    predict pointmap/confidence from model
    estimate camera/intrinsics from prediction only
    derive predicted depth; record prediction status
    fetch real scoring target by source_frame_id
    record target availability and final-mask coverage
    if target depth eligible:
        accumulate paired depth vectors or record prediction failure
    if target pose eligible:
        retain pair of GT/predicted poses with original ID
    record latency independently of target eligibility
finalize one sequence median depth scale, metrics and original five-frame pose windows
```

GT failure không được set `_first_prediction_failed`. Biến này chỉ phản ánh
prediction/camera failure nếu scorer mới còn cần nó.

## 7. Coverage/report bắt buộc

Mỗi sequence phải có:

- `rgb_frame_count`, `prediction_frame_count`, `prediction_failures` và original IDs.
- `depth_gt_available_count`, `depth_gt_eligible_count`, `depth_scored_count`,
  `depth_gt_missing_count`, `depth_gt_insufficient_count`, `depth_gt_corrupt_count`.
- Valid-pixel count/fraction mỗi frame; coverage summary và GT eligibility SHA256.
- `pose_gt_available_count`, `pose_prediction_count`, total/eligible/scored/failed
  five-frame windows và window source IDs.
- Metric protocolID, resize/crop, calibration/target source hashes, scale rule,
  depth units, pose units và các giá trị N/A với lý do.
- Timing denominator, warmup IDs, successful/failed prediction counts; không lấy
  `depth_scored_count` làm số frame để tính FPS.

Aggregator từ chối các model có eligibility/crop/scale/GT source không khớp.
Một model ít scored frame hơn do prediction failure không trở thành model tốt hơn.
GT-support fingerprint phải hash original IDs, actual final masks, availability,
crop/resize và depth-range semantics; equal pixel counts không chứng minh equal pixels.

## 8. Certification và PPO

V1 nên giữ PPO trên derived TRAIN valid runs và `DenseGTReward` strict đã có.
Không cần mở PPO partial-GT để giải quyết benchmark held-out partial-GT.

Nếu sau này muốn RL trên full partial-GT episodes, phải đặc tả reward sparse
và delayed association/GAE khi không có quality target; đây là nghiên cứu riêng.
Không tự gán quality zero, không chia episode ở lỗ hổng rồi gọi đó là full-video
policy environment. Cost component có thể vẫn đo được, nhưng không đủ thay thế
supervised quality reward mà không công bố thay đổi objective.

Đối với certification mới, candidate/baseline phải có cùng original RGB coverage,
cùng GT eligible IDs/masks, đủ coverage tối thiểu đã khóa, actual Flash/FLA và
finite outputs. Chỉ sau tests mới cấp schema/certificate version mới; utility
selection hiện tại không tự cấp certificate cho report giả lập mới.

## 9. Test và success gate trước server run

1. Frame đầu zero-depth-GT nhưng RGB/camera prediction hợp lệ: later cameras vẫn chạy.
2. Một frame missing depth ở giữa: mọi RGB được infer; pose vẫn có thể score;
   depth counts chính xác; RGB/source ID timeline không rút ngắn.
3. Tất cả depth-GT missing: depth N/A rõ ràng; latency vẫn có; không NaN aggregate.
4. GT đủ nhưng model NaN/PnP fail: báo model failure, không giảm GT eligibility.
5. Broken/misidentified target file: integrity error, không silently missing.
6. Same predictions full-GT: scorer mới bằng scorer hiện tại trong tolerance.
7. Pose windows dùng original consecutive IDs, không nối qua missing poses/gaps.
8. Hai model eligibility khác: aggregator/certifier từ chối join.
9. First-frame scale missing: strict certification vẫn từ chối; không chọn anchor ngầm.
10. Pseudo targets bị evaluator/certifier từ chối; actor không thể đọc target.
11. FPS numerator là timed RGB scope; GT absence không làm thay đổi reset/write actions.
12. Test patient reset, real partial-GT sequence sanity và full coverage report.

**Success gate:** CPU fixtures đạt; full-GT parity đạt; một sequence thật chạy full
RGB, counts/source IDs khớp inventory; ba branch chia sẻ target eligibility; review
protocol xong. **Failure gate:** source/GT corruption, giả provenance, prediction
failure bị giấu hoặc compressed timeline. Dừng benchmark/certification bị ảnh hưởng.

`Scale undefined` chặn depth-quality/certification khi đáng lẽ có đủ GT để scale.
Expected all-depth-GT-missing có depth N/A vẫn được chạy full-RGB inference/FPS
và pose nếu đủ pose GT; không dừng timing chỉ vì target được khai báo absent.

Guide này hoàn tất hướng dẫn triển khai; không chứng nhận tính năng partial-GT đã
được implement. Khi chưa đạt gate, kết quả held-out bị chặn phải ghi BLOCKED/N/A.
