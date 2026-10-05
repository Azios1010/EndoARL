"""Bounded visual KDA state, separate pre-update read and frame-chunk write."""

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import LongMemoryConfig
from .kda_adapter import KDAAdapter
from .types import HybridMemoryState, SharedNorms, WriteDiagnostics


class LongTermKDAMemory(nn.Module):
    def __init__(self, config: Optional[LongMemoryConfig] = None, input_dim: int = 1024,
                 shared_norms: Optional[SharedNorms] = None):
        super().__init__()
        self.config = config or LongMemoryConfig()
        c = self.config
        if c.model_dim != c.num_heads * c.key_dim or min(c.key_dim, c.value_dim, c.num_heads) <= 0:
            raise ValueError("Invalid KDA model/head dimensions")
        self.input_dim = input_dim
        self._shared_norms = shared_norms
        if shared_norms is None:
            self.norms = nn.ModuleDict({name: nn.LayerNorm(input_dim) for name in ("q", "k", "v")})
        self.q_proj = nn.Linear(input_dim, c.num_heads * c.key_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, c.num_heads * c.key_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, c.num_heads * c.value_dim, bias=False)
        self.out_proj = nn.Linear(c.num_heads * c.value_dim, input_dim, bias=False)
        self.beta_proj = nn.Linear(input_dim, c.num_heads, bias=True)
        self.decay_proj = nn.Linear(input_dim, c.num_heads * c.key_dim, bias=False)
        self.A_log = nn.Parameter(torch.zeros(c.num_heads))
        self.dt_bias = nn.Parameter(torch.full((c.num_heads, c.key_dim),
                                              math.log(math.expm1(c.initial_frame_decay_rate))))
        for layer in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.normal_(self.beta_proj.weight, std=1e-3)
        nn.init.constant_(self.beta_proj.bias, math.log(c.initial_beta / (1 - c.initial_beta)))
        nn.init.zeros_(self.decay_proj.weight)
        self.adapter = KDAAdapter(c.backend, c.kernel_chunk_size, c.implementation_revision, c.inference_mode)

    def _norm(self, name: str, x: Tensor) -> Tensor:
        return self._shared_norms.apply(name, x) if self._shared_norms else self.norms[name](x)

    def _validate(self, x: Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != self.input_dim or x.shape[1] < 1:
            raise ValueError(f"KDA visual input must be [B,P,{self.input_dim}] with P>0")
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite KDA visual features")

    def reset(self, batch_size: int, device=None) -> Tensor:
        c = self.config
        return torch.zeros(batch_size, c.num_heads, c.key_dim, c.value_dim,
                           dtype=torch.float32, device=device or self.q_proj.weight.device)

    def read(self, query: Tensor, state, normalized: bool = False) -> Tensor:
        self._validate(query)
        if isinstance(state, HybridMemoryState):
            if not state.has_long_memory:
                return torch.zeros_like(query)
            memory = state.kda_state
        else:
            memory = state
        expected = (query.shape[0], self.config.num_heads, self.config.key_dim, self.config.value_dim)
        if memory.shape != expected or memory.dtype != torch.float32 or memory.device != query.device:
            raise ValueError("KDA state must be canonical FP32 [B,H,K,V] on the query device")
        q = query if normalized else self._norm("q", query)
        q = self.q_proj(q.to(self.q_proj.weight.dtype)).reshape(
            *query.shape[:2], self.config.num_heads, self.config.key_dim)
        q = F.normalize(q.float(), dim=-1, eps=1e-6)
        with torch.autocast(device_type=query.device.type, enabled=False):
            output = torch.einsum("bphk,bhkv->bphv", q / math.sqrt(self.config.key_dim), memory.float())
        output = output.flatten(-2)
        return self.out_proj(output.to(self.out_proj.weight.dtype)).to(query.dtype)

    def update(self, k: Tensor, v: Tensor, update_weight: Tensor, state: Tensor,
               normalized: bool = False):
        self._validate(k)
        self._validate(v)
        if k.shape != v.shape:
            raise ValueError("Long K/V must have the same raw shape")
        if update_weight.shape != (*k.shape[:2], 1):
            raise ValueError("KDA write confidence must have shape [B,P,1]")
        if update_weight.device != k.device or not torch.isfinite(update_weight).all():
            raise ValueError("Write confidence must be finite on the feature device")
        if torch.any(update_weight < 0) or torch.any(update_weight > 1):
            raise ValueError("Write confidence must be in [0,1]")
        if not torch.any(update_weight > 0):
            return state, WriteDiagnostics(False, False, "all patches invalid")
        if not normalized:
            k, v = self._norm("k", k), self._norm("v", v)
        c = self.config
        b, p = k.shape[:2]
        kn = k.to(self.k_proj.weight.dtype)
        vn = v.to(self.v_proj.weight.dtype)
        key = self.k_proj(kn).reshape(b, p, c.num_heads, c.key_dim)
        key = F.normalize(key.float(), dim=-1, eps=1e-6).to(key.dtype)
        value = self.v_proj(vn).reshape(b, p, c.num_heads, c.value_dim)
        beta_base = torch.sigmoid(self.beta_proj(kn).float())
        gate_input = self.decay_proj(kn).float().reshape(b, p, c.num_heads, c.key_dim)
        log_scale = self.A_log.float().clamp(-5, 5).exp().view(1, 1, c.num_heads, 1)
        unclamped_decay = -log_scale * F.softplus(gate_input + self.dt_bias.float())
        decay_base = unclamped_decay.clamp(-5, 0)
        weight = update_weight.detach().float()
        if not c.uncertainty_write_gate:
            # Ablate disagreement attenuation while keeping completely invalid
            # patches unable to decay or overwrite state. RAFT statistics/rates
            # are still measured; uncertainty.enabled=false skips RAFT entirely.
            weight = (weight > 0).float()
        beta = beta_base * weight
        decay = decay_base * (weight.unsqueeze(-1) / p)
        new_state = self.adapter.update(key, value, decay, beta, state, training=self.training)
        if not torch.isfinite(new_state).all():
            raise FloatingPointError("KDA update produced nonfinite state")
        diagnostics = WriteDiagnostics(
            True, True, beta_mean=float(beta.detach().mean()), decay_mean=float(decay.detach().mean()),
            beta_saturation_fraction=float(((beta_base.detach() < 1e-4) | (beta_base.detach() > 1-1e-4)).float().mean()),
            decay_saturation_fraction=float((unclamped_decay.detach() < -5).float().mean()))
        return new_state, diagnostics

    forward = read
