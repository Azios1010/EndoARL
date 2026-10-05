"""Small per-frame channel gate, no attention distributions required."""

import math
from typing import Optional

import torch
from torch import Tensor, nn

from .config import FusionConfig
from .types import ReadDiagnostics


class HybridMemoryFusion(nn.Module):
    def __init__(self, config: Optional[FusionConfig] = None, input_dim: int = 1024):
        super().__init__()
        self.config = config or FusionConfig(output_dim=input_dim)
        c = self.config
        if c.output_dim != input_dim or not 0 < c.initial_short_preference < 1:
            raise ValueError("Fusion must preserve input dimension and have a valid initial gate")
        self.input_dim = input_dim
        self.gate = nn.Sequential(nn.Linear(3 * input_dim, c.hidden_dim), nn.GELU(),
                                  nn.Linear(c.hidden_dim, input_dim))
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
        nn.init.normal_(self.gate[2].weight, std=1e-3)
        nn.init.constant_(self.gate[2].bias, math.log(c.initial_short_preference / (1-c.initial_short_preference)))

    def forward(self, query: Tensor, short_output: Optional[Tensor] = None,
                long_output: Optional[Tensor] = None):
        if query.ndim != 3 or query.shape[-1] != self.input_dim:
            raise ValueError("Fusion query shape differs from model dimension")
        for value in (query, short_output, long_output):
            if value is not None and (value.shape != query.shape or not torch.isfinite(value).all()):
                raise FloatingPointError("Fusion requires finite, matching [B,P,D] inputs")
        hs, hl = short_output is not None, long_output is not None
        st = short_output if hs else torch.zeros_like(query)
        lt = long_output if hl else torch.zeros_like(query)
        if hs and hl:
            pooled = torch.cat([value.float().mean(dim=1) for value in (query, st, lt)], dim=-1)
            gate = torch.sigmoid(self.gate(pooled.to(self.gate[0].weight.dtype))).unsqueeze(1)
        elif hs:
            gate = torch.ones(query.shape[0], 1, self.input_dim, device=query.device, dtype=query.dtype)
        else:
            gate = torch.zeros(query.shape[0], 1, self.input_dim, device=query.device, dtype=query.dtype)
        st_contribution, lt_contribution = gate * st, (1 - gate) * lt
        fused = (query + st_contribution + lt_contribution).to(query.dtype)
        if not torch.isfinite(fused).all():
            raise FloatingPointError("Hybrid fusion produced a nonfinite feature")
        with torch.no_grad():
            rms = lambda value: value.float().square().mean(dim=(1, 2)).sqrt()
            st_norm = st_contribution.float().flatten(1).norm(dim=-1)
            lt_norm = lt_contribution.float().flatten(1).norm(dim=-1)
            diagnostics = ReadDiagnostics(
                gate.float().mean(dim=(1, 2)), gate.float().std(dim=(1, 2), unbiased=False),
                rms(st), rms(lt), st_norm / (st_norm + lt_norm + 1e-8), hs, hl)
        return fused, diagnostics
