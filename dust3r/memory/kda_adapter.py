"""Optional FLA boundary and differentiable FP32 delta-rule reference.

SERVER_VALIDATION_REQUIRED: pinned FLA 9f38d24980c46d46bd38614e743cdacd21906578,
Torch>=2.7/Triton>=3.3 on a validated CUDA environment. No package is downloaded.
"""

import importlib
import math
from typing import Optional

import torch
from torch import Tensor

from .config import FLA_REVISION


class KDABackendUnavailable(RuntimeError):
    pass


def kda_reference(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
                  initial_state: Optional[Tensor] = None, scale: Optional[float] = None):
    """Sequential key-first channel-gated delta rule, all arithmetic FP32.

    Inputs q/k [B,P,H,K], v [B,P,H,V], g [B,P,H,K], beta [B,P,H].
    Return post-update token outputs and final [B,H,K,V] state. One frame is
    one P-token invocation; patch order is preserved exactly.
    """
    if q.ndim != 4 or k.shape != q.shape or g.shape != k.shape:
        raise ValueError("KDA q/k/log-decay require equal [B,P,H,K] shapes")
    if v.shape[:3] != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("Invalid KDA value/beta layout")
    if q.shape[1] == 0:
        raise ValueError("A KDA frame must contain at least one patch")
    if any(x.device != q.device for x in (k, v, g, beta)):
        raise ValueError("KDA tensors must share one device")
    if not all(torch.isfinite(x).all() for x in (q, k, v, g, beta)):
        raise ValueError("Nonfinite KDA inputs")
    if torch.any(g > 0) or torch.any(beta < 0) or torch.any(beta > 1):
        raise ValueError("KDA requires nonpositive log decay and beta in [0,1]")
    b, _, h, key_dim = q.shape
    state_shape = (b, h, key_dim, v.shape[-1])
    if initial_state is None:
        state = torch.zeros(state_shape, device=q.device, dtype=torch.float32)
    else:
        if initial_state.shape != state_shape or initial_state.device != q.device:
            raise ValueError("KDA initial state shape/device mismatch")
        state = initial_state.float()
    scale = scale if scale is not None else key_dim ** -0.5
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
        outputs = []
        for patch in range(q.shape[1]):
            decayed = state * g[:, patch].exp().unsqueeze(-1)
            prediction = torch.einsum("bhk,bhkv->bhv", k[:, patch], decayed)
            correction = beta[:, patch].unsqueeze(-1) * (v[:, patch] - prediction)
            state = decayed + k[:, patch].unsqueeze(-1) * correction.unsqueeze(-2)
            outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, patch] * scale, state))
        return torch.stack(outputs, dim=1), state


class KDAAdapter:
    """External state never escapes the canonical FP32 key-first layout."""

    def __init__(self, backend: str = "fla_kda", chunk_size: int = 64,
                 implementation_revision: str = FLA_REVISION, inference_mode: str = "chunk"):
        if backend not in {"fla_kda", "torch_reference"}:
            raise ValueError("KDA backend must be fla_kda or torch_reference")
        if backend == "fla_kda" and implementation_revision != FLA_REVISION:
            raise ValueError("Unvalidated FLA revision; use the revision pinned in the plan")
        if chunk_size not in {32, 64} or inference_mode not in {"chunk", "fused_recurrent"}:
            raise ValueError("Invalid KDA chunk size or inference mode")
        self.backend, self.chunk_size = backend, chunk_size
        self.implementation_revision = implementation_revision
        self.inference_mode = inference_mode
        self.last_backend = "not_run"

    def validate_backend(self) -> None:
        if self.backend == "torch_reference":
            return
        self._load("chunk_kda")

    @staticmethod
    def _load(name: str):
        try:
            module = importlib.import_module("fla.ops.kda")
            return getattr(module, name)
        except (ImportError, AttributeError, OSError) as exc:
            raise KDABackendUnavailable(
                "FLA KDA is unavailable or incompatible. Required fla revision " + FLA_REVISION +
                "; SERVER_VALIDATION_REQUIRED on the provisioned training server. "
                "Select torch_reference explicitly for CPU correctness development; no fallback was used.") from exc

    def update(self, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, state: Tensor,
               training: bool = False) -> Tensor:
        q = torch.zeros_like(k)
        if self.backend == "torch_reference":
            self.last_backend = "torch_reference"
            _, final_state = kda_reference(q, k, v, g, beta, state)
            return final_state
        self.validate_backend()
        if not k.is_cuda or k.dtype not in {torch.float16, torch.bfloat16}:
            raise KDABackendUnavailable("fla_kda requires CUDA FP16/BF16 q/k/v; reference is never selected automatically")
        if k.shape[-1] > 256 or state.dtype != torch.float32:
            raise ValueError("FLA KDA requires key_dim<=256 and FP32 initial state")
        if training or torch.is_grad_enabled() or self.inference_mode == "chunk":
            op = self._load("chunk_kda")
            _, final_state = op(q=q, k=k, v=v, g=g.float(), beta=beta.to(k.dtype),
                                initial_state=state, output_final_state=True,
                                use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                                use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                                safe_gate=False, state_v_first=False,
                                return_intermediate_states=False, chunk_size=self.chunk_size,
                                disable_recompute=False)
            self.last_backend = "fla_kda_chunk"
        else:
            # Inference-only API is never used as an autograd substitute.
            op = self._load("fused_recurrent_kda")
            _, final_state = op(q=q, k=k, v=v, g=g.float(), beta=beta.to(k.dtype),
                                initial_state=state, output_final_state=True,
                                use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                                use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                                state_v_first=False)
            self.last_backend = "fla_kda_fused_recurrent"
        if final_state is None or final_state.shape != state.shape:
            raise RuntimeError("FLA did not return the expected canonical final KDA state")
        return final_state.float()
