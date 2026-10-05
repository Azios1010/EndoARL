"""Exact recent-frame attention with explicit, honestly reported backends."""

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ShortMemoryConfig
from .types import HybridMemoryState, SharedNorms


class FlashBackendUnavailable(RuntimeError):
    """An explicitly requested Flash kernel could not execute."""


def exact_attention_reference(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """FP32 dense exact softmax, layouts [B,H,Q/K,D]; no approximation."""
    if k.shape[-2] == 0:
        raise ValueError("An empty bank must bypass attention, not enter softmax")
    with torch.autocast(device_type=q.device.type, enabled=False):
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
        return torch.matmul(torch.softmax(logits, dim=-1), v.float())


class ShortTermFlashMemory(nn.Module):
    def __init__(self, config: Optional[ShortMemoryConfig] = None, input_dim: int = 1024,
                 shared_norms: Optional[SharedNorms] = None):
        super().__init__()
        self.config = config or ShortMemoryConfig()
        c = self.config
        if c.model_dim != c.num_heads * c.head_dim or not 1 <= c.max_frames <= 4:
            raise ValueError("Invalid short memory heads/dimensions/frame bound")
        if c.dropout != 0:
            raise ValueError("v1 uses zero attention dropout")
        if c.backend not in {"flash", "sdpa", "sdpa_flash", "flash_attn", "math_reference"}:
            raise ValueError(f"Unknown short backend: {c.backend}")
        if c.require_flash and c.backend in {"sdpa", "math_reference"}:
            raise ValueError("require_flash cannot label generic SDPA/math as Flash")
        self.input_dim = input_dim
        self._shared_norms = shared_norms
        if shared_norms is None:
            self.norms = nn.ModuleDict({name: nn.LayerNorm(input_dim) for name in ("q", "k", "v")})
        self.q_proj = nn.Linear(input_dim, c.model_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, c.model_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, c.model_dim, bias=False)
        self.out_proj = nn.Linear(c.model_dim, input_dim, bias=False)
        for layer in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        self.last_backend = "not_run"

    def _norm(self, name: str, x: Tensor) -> Tensor:
        return self._shared_norms.apply(name, x) if self._shared_norms else self.norms[name](x)

    def _validate(self, x: Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [B,P,{self.input_dim}], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite short-memory features")

    def project_kv(self, k: Tensor, v: Tensor, normalized: bool = False):
        self._validate(k)
        self._validate(v)
        if k.shape != v.shape:
            raise ValueError("Short K/V must have the same shape")
        if not normalized:
            k, v = self._norm("k", k), self._norm("v", v)
        shape = (*k.shape[:2], self.config.num_heads, self.config.head_dim)
        return (self.k_proj(k.to(self.k_proj.weight.dtype)).reshape(shape),
                self.v_proj(v.to(self.v_proj.weight.dtype)).reshape(shape))

    def _forced_sdpa_flash(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if not q.is_cuda or q.dtype not in {torch.float16, torch.bfloat16}:
            raise FlashBackendUnavailable("Flash attention requires CUDA FP16/BF16 tensors")
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    def _official_flash(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if not q.is_cuda or q.dtype not in {torch.float16, torch.bfloat16}:
            raise FlashBackendUnavailable("Official flash-attn requires CUDA FP16/BF16 tensors")
        try:
            from flash_attn import flash_attn_func
        except (ImportError, OSError) as exc:
            raise FlashBackendUnavailable("flash-attn is absent/unloadable; provision it on the server") from exc
        result = flash_attn_func(q.transpose(1, 2).contiguous(),
                                 k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(),
                                 dropout_p=0.0, causal=False)
        return result.transpose(1, 2)

    def attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        backend = self.config.backend
        if backend == "math_reference":
            self.last_backend = "math_reference"
            return exact_attention_reference(q, k, v)
        if backend == "sdpa":
            self.last_backend = "sdpa_dispatch_unverified"
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        failures = []
        if backend in {"flash", "sdpa_flash"}:
            try:
                result = self._forced_sdpa_flash(q, k, v)
                self.last_backend = "sdpa_flash"
                return result
            except (RuntimeError, ImportError, OSError) as exc:
                failures.append(f"forced SDPA Flash: {exc}")
                if backend == "sdpa_flash":
                    raise FlashBackendUnavailable("; ".join(failures)) from exc
        try:
            result = self._official_flash(q, k, v)
            self.last_backend = "flash_attn"
            return result
        except (RuntimeError, ImportError, OSError) as exc:
            failures.append(f"official flash-attn: {exc}")
            raise FlashBackendUnavailable("No requested exact Flash backend is available. "
                                          "SERVER_VALIDATION_REQUIRED: " + "; ".join(failures) +
                                          ". Select math_reference explicitly for CPU correctness tests.") from exc

    def read(self, query: Tensor, state: HybridMemoryState, normalized: bool = False) -> Tensor:
        self._validate(query)
        if state.short_occupancy == 0:
            return torch.zeros_like(query)
        if query.shape[:2] != (state.batch_size, state.num_patches):
            raise ValueError("Query batch/patch layout differs from episode state")
        source_dtype = query.dtype
        q = query if normalized else self._norm("q", query)
        q = self.q_proj(q.to(self.q_proj.weight.dtype)).reshape(
            *query.shape[:2], self.config.num_heads, self.config.head_dim).transpose(1, 2)
        k = state.short_k.flatten(1, 2).transpose(1, 2)
        v = state.short_v.flatten(1, 2).transpose(1, 2)
        output = self.attend(q.contiguous(), k.contiguous(), v.contiguous())
        output = output.transpose(1, 2).reshape(*query.shape[:2], self.config.model_dim)
        return self.out_proj(output.to(self.out_proj.weight.dtype)).to(source_dtype)

    forward = read
