# Môi trường Linux server cho EndoARL Flash/KDA

Tài liệu này chỉ mô tả các lệnh **người vận hành chạy trên server**. Không lệnh
cài đặt, build CUDA, tải package, tải dataset/checkpoint hay training nào được
chạy trong quá trình viết tài liệu tại local. Sau phần môi trường, tiếp tục
[hướng dẫn fit và benchmark](SCARED_SERVER_FIT_GUIDE.md).

Các lệnh provisioning có thể truy cập PyPI/PyTorch/GitHub để lấy **package**;
chỉ chạy khi đã cho phép chuẩn bị môi trường server. Dataset SCARED và ba
checkpoint phải đã có sẵn. Không dùng model hub hoặc lệnh tải weights trong
quy trình này. Không chạy toàn bộ các block như một script tự động.

## 1. Điều kiện đầu vào và lựa chọn còn provisional

| Thành phần | Chọn ban đầu | Trạng thái / kiểm chứng |
| --- | --- | --- |
| OS / Python | Linux x86_64, Python 3.11 trong venv riêng | PROVISIONAL; kiểm tra wheel và import trên máy thật |
| GPU | NVIDIA CUDA, native BF16 cho config v1 | NEEDS_RUNTIME_VALIDATION; tên GPU chưa đủ chứng minh kernel chạy |
| PyTorch / torchvision | `2.7.1` / `0.22.1` | PROVISIONAL; đây là cặp wheel chính thức, chưa certify EndoARL trên GPU cụ thể |
| CUDA wheel | Chọn **một** index `cu126` hoặc `cu128` | PROVISIONAL; dựa GPU, driver và kernel probe, không tự đổi khi gặp lỗi |
| Triton | Version do wheel Torch đã chọn resolve | Khóa đúng version thực tế; không nâng Triton độc lập |
| FLA source | `9f38d24980c46d46bd38614e743cdacd21906578` | LOCKED theo adapter; xác minh metadata VCS trước probe |
| Short attention | Forced PyTorch SDPA Flash | LOCKED cho config SCARED v1; không cần cài `flash-attn` nếu probe thành công |
| Long attention | `fla.ops.kda` qua adapter nội bộ | LOCKED; cần forward và backward thực tế |

Contract trong code: [ShortTermFlashMemory._forced_sdpa_flash](../dust3r/memory/short_flash.py),
[KDAAdapter.update](../dust3r/memory/kda_adapter.py),
[config validation](../dust3r/memory/config.py) và
[kernel probe](../tools/check_memory_backends.py). CUDA/package requirements
ngoài repo cần được xác minh riêng; có import thành công chưa đủ cho kernel gate.

PyTorch công bố cặp `torch==2.7.1`, `torchvision==0.22.1` cho cả hai index
CUDA 12.6/12.8. Việc có wheel không chứng minh GPU, driver và FLA của thí nghiệm
tương thích. [Nguồn PyTorch](https://pytorch.org/get-started/previous-versions/).

Ở đúng commit FLA đã pin, `pyproject.toml` yêu cầu Python >=3.10; CUDA extra
yêu cầu Torch >=2.7 và Triton >=3.3. Base dependencies có `transformers>=4.45`
và `einops`; `causal-conv1d` nằm ở extra khác. Không cài full Kimi model,
`causal-conv1d`, TileLang hay benchmark extras cho adapter KDA hiện tại.
[Metadata FLA tại commit đã pin](https://raw.githubusercontent.com/fla-org/flash-linear-attention/9f38d24980c46d46bd38614e743cdacd21906578/pyproject.toml).

Không cam kết một phiên bản transformers mới nhất sẽ dùng được với mọi stack
Torch cũ. Review dependency resolution, lưu version đã resolve và coi import /
kernel probe là điều kiện bắt buộc. Nếu cần thay một version, tạo môi trường
thử nghiệm riêng và ghi lại thay đổi; không nâng package ngay giữa một run.

## 2. Kiểm kê hardware trước khi cài

Các đường dẫn dưới đây là ví dụ; đổi sang đường dẫn có quyền ghi trên server.
Không dùng một venv chung với dự án khác. Python 3.11, driver NVIDIA, compiler
và công cụ hệ thống phải do người vận hành quản lý; tài liệu không tự cài driver
hay dùng `sudo` để thay môi trường host.

```bash
export ENDOARL_REPO=/work/EndoARL
export ENDOARL_ENV=/work/venvs/endoarl_flashkda_py311
export ENV_ARTIFACTS=/work/endoarl_environment_v1

test -d "$ENDOARL_REPO"
test ! -e "$ENDOARL_ENV"
test ! -e "$ENV_ARTIFACTS"
mkdir -p "$ENV_ARTIFACTS"
cd "$ENDOARL_REPO"

python3.11 --version
uname -a > "$ENV_ARTIFACTS/host.txt"
nvidia-smi > "$ENV_ARTIFACTS/nvidia_smi.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv > "$ENV_ARTIFACTS/gpu_inventory.csv"
git rev-parse HEAD > "$ENV_ARTIFACTS/repository_commit.txt"
git status --short > "$ENV_ARTIFACTS/repository_status.txt"
git submodule status --recursive > "$ENV_ARTIFACTS/submodules.txt"
```

Dừng nếu `nvidia-smi` lỗi, GPU đang có workload khác hoặc thiếu CroCo sources.
Chọn GPU được cấp riêng; nếu cần, đặt `CUDA_VISIBLE_DEVICES` **trước khi** chạy
Python. Ví dụ chọn GPU vật lý số 1 nhưng mọi lệnh dự án vẫn dùng `cuda:0`:

```bash
export CUDA_VISIBLE_DEVICES=1
```

Legacy có `.cuda()` và giả định visible device 0; không dùng `--device cuda:1`
để sửa giả định này. Dùng cùng GPU visible và resource allocation cho cả ba
nhánh benchmark.

Trường CUDA trong `nvidia-smi` không phải version toolkit `nvcc` đã cài.
Wheel Torch chứa CUDA runtime; custom CUDA builds có thể cần toolkit riêng.
Triton/PTX có thể đòi driver mới hơn mức minor compatibility tối thiểu; không
dựa riêng vào một số driver để tuyên bố tương thích.
[Hạn chế CUDA minor compatibility của NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## 3. Venv mới và cặp Torch/vision

Các block cài đặt dưới đây chỉ dành cho server được phép provisioning. Nếu
`python3.11 -m venv` không dùng được, dừng để chuẩn bị Python/venv đúng cách;
không cài vào `(base)`, Python hệ thống hoặc env đang phục vụ workload khác.

```bash
python3.11 -m venv "$ENDOARL_ENV"
source "$ENDOARL_ENV/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

python - <<'PY'
import sys
from pathlib import Path
import os
assert sys.version_info[:2] == (3, 11), sys.version
assert sys.prefix != sys.base_prefix, 'Fresh venv required'
assert Path(sys.prefix).resolve() == Path(os.environ['ENDOARL_ENV']).resolve()
print(sys.executable)
PY

python -m pip install --upgrade pip setuptools wheel

# Chọn một index sau khi review driver/GPU; giá trị này là PROVISIONAL.
export TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cu126
# Hoặc cu128, trong một venv mới riêng; không chạy cả hai lần trong cùng env.
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url "$TORCH_WHEEL_INDEX"
```

Không cần torchaudio cho đường inference/training này. Sau khi cài Torch,
khóa phiên bản **thực tế**, kể cả suffix CUDA và Triton đã được resolve:

```bash
export CONSTRAINTS="$ENV_ARTIFACTS/core_constraints.txt"
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import os
names = ('torch', 'torchvision', 'triton')
versions = {name: version(name) for name in names}
assert versions['torch'].split('+')[0] == '2.7.1', versions
assert versions['torchvision'].split('+')[0] == '0.22.1', versions
parts = versions['triton'].split('.')
assert tuple(map(int, parts[:2])) >= (3, 3), versions
path = Path(os.environ['CONSTRAINTS'])
with path.open('x', encoding='utf-8') as handle:
    handle.write('\n'.join(f'{name}=={versions[name]}' for name in names) + '\n')
print(versions)
PY
export PIP_CONSTRAINT="$CONSTRAINTS"
```

Không chạy `pip install -U triton`, `triton-nightly` hoặc thay Torch khi gặp lỗi
FLA. Constraint conflict là điều kiện dừng để review stack; không bỏ constraint
để pip tự quyết định một kiến trúc môi trường khác.

## 4. Dependencies dự án và FLA source đúng commit

`requirements.txt` hiện để Torch/vision không pin, nên luôn dùng constraint.
`PyYAML` cần cho YAML config; `imageio`, Pillow, packaging và
`huggingface-hub` có import trong code nhưng requirements tối thiểu chưa liệt kê
đầy đủ. Cài thư viện hub không đồng nghĩa dùng API tải model. Các gói build
dưới đây chuẩn bị việc package/build có kiểm soát trên server.

```bash
python -m pip install -c "$CONSTRAINTS" \
  -r "$ENDOARL_REPO/requirements.txt" \
  PyYAML imageio Pillow packaging huggingface-hub ninja psutil
python -m pip check

export FLA_REVISION=9f38d24980c46d46bd38614e743cdacd21906578
python -m pip install -c "$CONSTRAINTS" --no-build-isolation \
  "flash-linear-attention[cuda] @ git+https://github.com/fla-org/flash-linear-attention.git@${FLA_REVISION}"
python -m pip check
```

Lệnh FLA dùng source tại SHA cố định và CUDA extra, không `main`, không
`--upgrade`, không nâng Triton độc lập. `--no-build-isolation` dùng Torch và
setuptools/wheel đã cài trong venv; nếu build/import lỗi, giữ log và dừng.
Đối chiếu với [hướng dẫn FLA source installation](https://raw.githubusercontent.com/fla-org/flash-linear-attention/9f38d24980c46d46bd38614e743cdacd21906578/INSTALL.md).

Kiểm tra provenance trước khi chạy kernel; package version đơn thuần không
chứng minh đúng commit:

```bash
python - <<'PY' > "$ENV_ARTIFACTS/fla_identity.json"
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
expected = os.environ['FLA_REVISION']
dist = distribution('flash-linear-attention')
direct = json.loads(dist.read_text('direct_url.json') or '{}')
actual = direct.get('vcs_info', {}).get('commit_id')
assert actual == expected, f'FLA VCS mismatch: {actual!r} != {expected}'
for line in Path(os.environ['CONSTRAINTS']).read_text().splitlines():
    name, pinned = line.split('==', 1)
    assert version(name) == pinned, f'{name} changed after dependency resolution'
print(json.dumps({'package_version': version('flash-linear-attention'),
                  'required_commit': expected, 'direct_url': direct}, indent=2))
PY
```

FLA có các gói/phân tách mới ở những release khác; không tự thay source đã pin
bằng `fla-core` latest hoặc một package KDA khác khi import lỗi. Adapter hiện
gọi `fla.ops.kda.chunk_kda` và `fused_recurrent_kda`; đây là interface cần tồn tại.

## 5. Kiểm tra imports, CUDA và BF16

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MPLBACKEND=Agg
cd "$ENDOARL_REPO"

python - <<'PY' > "$ENV_ARTIFACTS/runtime_imports.json"
import json
import sys
from importlib.metadata import version
from contextlib import redirect_stdout
import torch
import torchvision
import cv2
import yaml
import imageio
with redirect_stdout(sys.stderr):
    import dust3r
    from dust3r.model import Endo3R
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() >= 1, 'No visible GPU'
assert torch.cuda.is_bf16_supported(including_emulation=False), 'v1 requires native BF16'
device = torch.cuda.current_device()
probe = torch.ones(8, 8, device='cuda:0', dtype=torch.bfloat16)
assert torch.isfinite(probe @ probe).all()
torch.cuda.synchronize()
print(json.dumps({'python': sys.version, 'executable': sys.executable,
                  'torch': str(torch.__version__), 'vision': str(torchvision.__version__),
                  'cuda_runtime': torch.version.cuda, 'triton': version('triton'),
                  'gpu': torch.cuda.get_device_name(device),
                  'compute_capability': list(torch.cuda.get_device_capability(device)),
                  'native_bf16': True, 'cv2': cv2.__version__,
                  'fla_api_present': callable(chunk_kda) and callable(fused_recurrent_kda)}, indent=2))
PY
```

`import Endo3R` không khởi tạo model/RAFT hoặc load weights. Nếu OpenCV lỗi
`libGL`/thư viện hệ thống, chuẩn bị đúng runtime OS với quản trị viên. Không
cài đồng thời `opencv-python` và `opencv-python-headless` để thử che lỗi import.
Nếu cần một lựa chọn headless, lập env riêng và ghi rõ dependency thay đổi.

BF16 tensor matmul thành công vẫn chưa xác minh Flash/KDA. Nếu BF16 không hỗ
trợ, dừng config v1; một thử nghiệm FP16 phải có config/probe riêng, không đổi
dtype âm thầm rồi báo kết quả BF16.
Tham số loại emulation trong snippet có ở
[source Torch v2.7.1](https://raw.githubusercontent.com/pytorch/pytorch/v2.7.1/torch/cuda/__init__.py).

## 6. Flash backend và custom RoPE

Config SCARED dùng `memory.short.backend=sdpa_flash` và `require_flash=true`.
Code ép `SDPBackend.FLASH_ATTENTION`, không cho math fallback. PyTorch ghi rõ
fused SDPA có hạn chế input; buộc backend mới cho biết kernel thực sự chạy được.
[Tài liệu SDPA PyTorch 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html).

Không cài package `flash-attn` cho đường `sdpa_flash` nếu probe đạt. Nếu chủ động
chọn backend `flash_attn`, cần một version/revision đã review, đúng toolkit /
Torch ABI / GPU; ghi lựa chọn này vào config mới rồi probe lại. Không dùng API
FlashAttention-3/4 thay cho `flash_attn.flash_attn_func` hiện tại.
[Hướng dẫn package FlashAttention chính thức](https://github.com/Dao-AILab/flash-attention).

CroCo RoPE là một yếu tố khác với memory Flash: code có PyTorch fallback
khi không thấy extension `curope`. Điều này có thể làm FPS thấp dù memory
kernel tốt. Kiểm tra backend đang dùng:

```bash
python - <<'PY' > "$ENV_ARTIFACTS/rope_backend.json"
import json
import sys
from contextlib import redirect_stdout
with redirect_stdout(sys.stderr):
    import dust3r.utils.path_to_croco
    from models.pos_embed import RoPE2D
print(json.dumps({'rope_module': RoPE2D.__module__,
                  'rope_class': RoPE2D.__name__}, indent=2))
PY
```

Nếu chọn build extension, toolkit `nvcc`, C++ compiler và CUDA ABI phải tương
thích với Torch đã khóa. Đây là bước **tùy chọn riêng trên server**, không yêu
cầu cho correctness tests. Build từ bản copy ở ngoài repo để không thay source
tree đang bind với checkpoint; kiểm tra script `croco/models/curope/setup.py`
trước vì nó tạo flags cho nhiều CUDA architectures.

```bash
nvcc --version
c++ --version
test ! -e "$ENV_ARTIFACTS/curope_src"
cp -a "$ENDOARL_REPO/croco/models/curope" "$ENV_ARTIFACTS/curope_src"
MAX_JOBS=4 python -m pip install -c "$CONSTRAINTS" --no-build-isolation \
  "$ENV_ARTIFACTS/curope_src"
python -m pip check
```

Chạy lại kiểm tra RoPE sau build. Tất cả legacy/hybrid/PPO phải dùng cùng
backend RoPE và version extension; ghi nhận lựa chọn trước benchmark. Không
đổi backend giữa quality run và timing run.

## 7. Weights có sẵn, SHA256 và đường RAFT legacy

Chỉ trỏ đến file đã có; tên file không chứng minh đúng weights. Lưu SHA256,
nguồn checkpoint và commit/release được sử dụng. Không thêm lệnh `gdown`,
`wget`, `.from_pretrained(...)` hay model-hub login để lấy weights tại đây.

```bash
export DUST3R_CHECKPOINT=/models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
export ENDO3R_CHECKPOINT=/models/endo3r.pth
export RAFT_CHECKPOINT=/models/raft-things.pth
test -f "$DUST3R_CHECKPOINT"
test -f "$ENDO3R_CHECKPOINT"
test -f "$RAFT_CHECKPOINT"
sha256sum "$DUST3R_CHECKPOINT" "$ENDO3R_CHECKPOINT" "$RAFT_CHECKPOINT" \
  > "$ENV_ARTIFACTS/pretrained_checkpoints.sha256"

# Nếu path legacy đã tồn tại, phải cùng SHA với RAFT đã chọn; không ghi đè.
mkdir -p "$ENDOARL_REPO/checkpoints"
if [[ ! -e "$ENDOARL_REPO/checkpoints/raft-things.pth" ]]; then
  ln -s "$RAFT_CHECKPOINT" "$ENDOARL_REPO/checkpoints/raft-things.pth"
fi
test -f "$ENDOARL_REPO/checkpoints/raft-things.pth"
cmp "$RAFT_CHECKPOINT" "$ENDOARL_REPO/checkpoints/raft-things.pth"
```

Legacy đọc `checkpoints/raft-things.pth` tương đối với repository cwd; config
RAFT khác không thay đường này. Hybrid có device-aware wrapper riêng. Symlink
chỉ đặt file đã có vào path legacy, không sửa source, không sửa trạng thái RAFT
của baseline. `--trusted-checkpoints` chỉ dùng sau khi review nguồn/hashes;
DUSt3R released metadata có pickle và không được coi file bất kỳ là đáng tin.

## 8. Kernel probe và dry readiness có giới hạn

Probe template không cần dataset/checkpoint và dùng tensors rất nhỏ; đây là
kiểm tra backend, không phải chứng minh training hoặc quality đã đạt:

```bash
python tools/check_memory_backends.py \
  --config configs/scared/hybrid_sampson.yaml \
  --server-kernels --device cuda:0 \
  > "$ENV_ARTIFACTS/kernel_template_probe.json"
python - <<'PY'
import json
import os
from pathlib import Path
report = json.loads((Path(os.environ['ENV_ARTIFACTS']) / 'kernel_template_probe.json').read_text())
assert report.get('passed') is True
assert report.get('reference_mode') is False
assert report['actual_short_backend'] == 'sdpa_flash'
assert report['actual_long_backend'] == 'fla_kda_chunk'
assert report['output_shape'] == [1, 4, 1024]
assert report['state_shape'] == [1, 4, 64, 64]
assert report['finite_gradients'] and all(report['finite_gradients'].values())
print('Template kernel probe passed; real-resolution/sequence checks remain required')
PY
```

Success condition: finite forward/backward, actual forced Flash, actual FLA
chunk update/final-state gradients, canonical FP32 KDA state. Short branch
512-D/8×64, long branch 4 heads với K=V=64; memory input/output 1024-D. Probe
không chứng minh kernel behavior ở P=320, full model autocast, RAFT geometry,
clip TBPTT, chất lượng reconstruct hay speedup. Các kiểm tra đó thực hiện tiếp
theo hướng dẫn chính với output mới, có bounds rõ ràng.

Khi đã resolve config/data paths trong hướng dẫn chính, chạy study preflight
để lưu `seed_N/kernel_conformance.json` đúng config thay vì tự đổi tên proof
template. CLI dry-run có thể exit 0 khi JSON `requirements_ready=false`; xem
`blocking_requirements` và `missing_training_budgets` trước khi chấp nhận:

```bash
# Các biến này được tạo ở hướng dẫn chính, không phải paths placeholder chưa có.
: "${RESOLVED_CONFIG:?resolved hybrid config required}"
: "${FULL_MANIFEST:?verified full manifest required}"
: "${TRAIN_MANIFEST:?audited TRAIN-only supervision manifest required}"
: "${ATTENTION_STEPS:?explicit attention step budget required}"
: "${ATTENTION_EPOCHS:?explicit epoch limit required}"
python train_hybrid_memory.py --config "$RESOLVED_CONFIG" \
  --manifest "$FULL_MANIFEST" --supervision-manifest "$TRAIN_MANIFEST" \
  --total-steps "$ATTENTION_STEPS" --epochs "$ATTENTION_EPOCHS" \
  --dry-run --validate-data --validate-modules --device cuda:0 \
  > "$ENV_ARTIFACTS/hybrid_training_readiness.json"
```

Dry module construction chỉ tạo memory và parameter groups. Nó không load
released model weights hoặc chứng minh whole-model training gradients. PPO dry
validation cũng không thay thế attention certification và actor validation.

## 9. Khóa environment và audit đường không dùng hub

Sau khi imports/probe đạt, lưu lock và dừng thay packages giữa một thí nghiệm:

```bash
python -m pip check > "$ENV_ARTIFACTS/pip_check.txt"
python -m pip freeze --all > "$ENV_ARTIFACTS/environment_lock.txt"
python -m pip inspect > "$ENV_ARTIFACTS/pip_inspect.json"
python -m torch.utils.collect_env > "$ENV_ARTIFACTS/torch_collect_env.txt"
sha256sum "$ENV_ARTIFACTS/core_constraints.txt" \
  "$ENV_ARTIFACTS/environment_lock.txt" \
  > "$ENV_ARTIFACTS/environment_hashes.sha256"

python - <<'PY' > "$ENV_ARTIFACTS/source_identity.json"
import json
from dust3r.training.provenance import source_provenance
print(json.dumps(source_provenance(), indent=2))
PY

# Read-only tìm loader/network entrypoints để review call path thực sự sử dụng.
rg -n 'from_pretrained|hf_hub_download|snapshot_download|torch\.hub|gdown' \
  demo_hybrid.py train_hybrid_memory.py train_rl_controller.py \
  dust3r/training/hybrid_trainer.py dust3r/memory/uncertainty.py
git status --short
```

`rg` không có match sẽ trả exit 1, không phải môi trường lỗi; có match thì
review nơi gọi, không xóa code hoặc kết luận ngay là có download. Đường
`load_initialized_model`/uncertainty của pipeline dùng file local và từ chối
weights thiếu. Offline flags giới hạn các thư viện Hugging Face; chúng không
là firewall cho mọi Python library. Nếu server cần chứng minh không network
trong inference/training, chạy bằng chính sách egress của server/container và
giữ local files/cache cần thiết. Không gọi helper hub chỉ để "thử login".

Một `pip freeze` không pin driver, compiler, GPU và nội dung source extension.
Giữ các inventory/provenance/probe ở trên, cùng revision/source hashes và logs.
Artifacts ở ngoài repo giúp checkpoint/evaluation source fingerprint không
thay đổi do tạo config/log. Không `git pull`, sửa source hoặc nâng packages
giữa fit → full validation → checkpoint selection → quality/timing.

## 10. StereoAnything là môi trường và nhãn riêng

Nếu chọn nhãn gần training paper hơn, dùng environment riêng cho OpenStereo /
StereoAnything với checkpoint đã có, revision đã review và custom CUDA builds
của chính dự án đó. Không cài các extension/stack Torch của OpenStereo vào venv
EndoARL để giải quyết một lỗi exporter. Giữ nhãn pseudo-depth/mask/provenance ở
output riêng, không ghi đè genuine GT hoặc A00 disparity.

Repo hiện chưa có SCARED StereoAnything exporter hoàn chỉnh và genuine-GT
loader/reward không chấp nhận pseudo-label giả danh GT. Việc chuẩn bị venv này
chưa xóa blocker target-kind/exporter. Tham khảo
[hướng dẫn StereoAnything của tác giả](https://github.com/XiandaGuo/OpenStereo/blob/master/docs/StereoAnything.md)
và phần nhãn ở hướng dẫn chính trước khi lập run riêng.

## 11. Gate kết thúc phần môi trường

**READY_FOR_PREFLIGHT** chỉ khi venv đúng, Torch/vision/Triton không trôi version,
FLA đúng SHA, imports thành công, native BF16 phù hợp config, Flash/FLA template
probe hữu hạn và local checkpoints có hash/path xác minh. Đây chưa phải
`READY_FOR_FIT`: vẫn cần coverage manifest, supervision threshold, full-model
bounded checks, explicit budgets, attention fitting và validation selection.

Nếu gate thất bại, lưu log, version/hardware và lỗi đầy đủ. Không chuyển sang
CPU reference, generic SDPA, KDA khác hoặc bỏ uncertainty rồi báo như cùng
thí nghiệm. Điều chỉnh có chủ ý phải thành config/run mới có provenance.
