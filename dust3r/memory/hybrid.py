"""Explicit episodic memory API; policy, uncertainty and references stay outside."""

from dataclasses import replace
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MemoryConfig, validate_memory_config
from .fusion import HybridMemoryFusion
from .long_kda import LongTermKDAMemory
from .short_flash import ShortTermFlashMemory
from .types import HybridMemoryState, MemoryCandidate, SharedNorms, WriteConfidence


class HybridMemory(nn.Module):
    def __init__(self, config: Optional[MemoryConfig] = None, shared_norms: Optional[SharedNorms] = None):
        super().__init__()
        self.config = validate_memory_config(config or MemoryConfig())
        if self.config.backend != "hybrid_flash_kda":
            raise ValueError("HybridMemory cannot represent the legacy backend")
        self._shared_norms = shared_norms
        if shared_norms is None:
            self.norms = nn.ModuleDict({name: nn.LayerNorm(self.config.input_dim) for name in ("q", "k", "v")})
            self._shared_norms = SharedNorms(self.norms["q"], self.norms["k"], self.norms["v"])
        self.short = ShortTermFlashMemory(self.config.short, self.config.input_dim, self._shared_norms)
        self.long = LongTermKDAMemory(self.config.long, self.config.input_dim, self._shared_norms)
        self.fusion = HybridMemoryFusion(self.config.fusion, self.config.input_dim)

    def reset(self, batch_size: int = 1, image_hw=(256, 320), patch_hw=(16, 20),
              device=None, training: bool = False, sequence_id: Optional[str] = None) -> HybridMemoryState:
        """patch_hw is the encoder PATCH GRID (rows, cols), not pixel patch size."""
        if batch_size != 1:
            raise ValueError("v1 owns one independent video state and requires B=1")
        if len(image_hw) != 2 or len(patch_hw) != 2 or min(*image_hw, *patch_hw) <= 0:
            raise ValueError("Image and patch-grid dimensions must be positive pairs")
        device = device or self.short.q_proj.weight.device
        c, p = self.config, patch_hw[0] * patch_hw[1]
        cache_shape = (batch_size, 0, p, c.short.num_heads, c.short.head_dim)
        cache_dtype = self.short.k_proj.weight.dtype
        return HybridMemoryState(
            short_k=torch.empty(cache_shape, dtype=cache_dtype, device=device),
            short_v=torch.empty(cache_shape, dtype=cache_dtype, device=device),
            kda_state=self.long.reset(batch_size, device),
            recent_key_descriptors=torch.empty(batch_size, 0, c.input_dim, dtype=torch.float32, device=device),
            recent_long_descriptors=torch.empty(batch_size, 0, c.input_dim, dtype=torch.float32, device=device),
            previous_uncertainty_stats=torch.zeros(batch_size, 3, dtype=torch.float32, device=device),
            short_frame_indices=torch.empty(0, dtype=torch.long, device=device),
            image_hw=tuple(image_hw), patch_hw=tuple(patch_hw), sequence_id=sequence_id,
            retain_short_raw=training and c.short.raw_shadow_training_only)

    def _validate(self, x: Tensor, state: HybridMemoryState) -> None:
        if x.shape != (state.batch_size, state.num_patches, self.config.input_dim):
            raise ValueError("Feature shape differs from episode batch/patch layout/model dimension")
        if x.device != state.kda_state.device:
            raise ValueError("Features/state must be on the same explicit device")
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite hybrid memory features")

    def read(self, query: Tensor, state: HybridMemoryState):
        self._validate(query, state)
        if state.short_occupancy == 0 and not state.has_long_memory:
            return self.fusion(query)
        normalized = self._shared_norms.apply("q", query)
        st = self.short.read(normalized, state, normalized=True) if state.short_occupancy else None
        lt = self.long.read(normalized, state, normalized=True) if state.has_long_memory else None
        return self.fusion(query, st, lt)

    def append_short(self, k: Tensor, v: Tensor, descriptor: Optional[Tensor], state: HybridMemoryState):
        self._validate(k, state)
        self._validate(v, state)
        kp, vp = self.short.project_kv(self._shared_norms.apply("k", k),
                                        self._shared_norms.apply("v", v), normalized=True)
        bound = self.config.short.max_frames
        if descriptor is None:
            descriptor = F.normalize(k.detach().float().mean(dim=1), dim=-1, eps=1e-6)
        if descriptor.shape != (state.batch_size, self.config.input_dim) or not torch.isfinite(descriptor).all():
            raise ValueError("Descriptor must be finite [B,input_dim]")
        descriptor = F.normalize(descriptor.detach().float(), dim=-1, eps=1e-6)
        if state.short_occupancy:
            if state.short_k.dtype != kp.dtype or state.short_v.dtype != vp.dtype:
                raise ValueError("Projected short-cache dtype changed within the episode; reset or reproject explicitly")
            short_k = torch.cat((state.short_k, kp.unsqueeze(1)), dim=1)[:, -bound:]
            short_v = torch.cat((state.short_v, vp.unsqueeze(1)), dim=1)[:, -bound:]
        else:
            # Empty reset buffers may use FP32 parameter dtype while autocast
            # projects BF16. Concatenating would silently promote the cache.
            short_k, short_v = kp.unsqueeze(1), vp.unsqueeze(1)
        raw_k, raw_v = state.short_raw_k, state.short_raw_v
        if state.retain_short_raw:
            raw_k = k.unsqueeze(1) if raw_k is None else torch.cat((raw_k, k.unsqueeze(1)), dim=1)
            raw_v = v.unsqueeze(1) if raw_v is None else torch.cat((raw_v, v.unsqueeze(1)), dim=1)
            raw_k, raw_v = raw_k[:, -bound:], raw_v[:, -bound:]
        indices = torch.cat((state.short_frame_indices,
                             torch.tensor([state.candidate_index], device=k.device, dtype=torch.long)))[-bound:]
        return replace(state, short_k=short_k, short_v=short_v, short_raw_k=raw_k, short_raw_v=raw_v,
                       short_frame_indices=indices, short_occupancy=short_k.shape[1],
                       short_cursor=(state.short_cursor + 1) % bound,
                       recent_key_descriptors=torch.cat((state.recent_key_descriptors,
                                                         descriptor.unsqueeze(1)), dim=1)[:, -bound:],
                       processed_candidate_count=state.processed_candidate_count + 1)

    def write_long(self, candidate: MemoryCandidate, write_confidence, state: HybridMemoryState):
        """Commit KDA tensor/presence only; runner owns references and counters."""
        self._validate(candidate.key, state)
        self._validate(candidate.value, state)
        weight = write_confidence.weights if isinstance(write_confidence, WriteConfidence) else write_confidence
        new_tensor, diagnostics = self.long.update(candidate.key, candidate.value, weight, state.kda_state)
        if not diagnostics.success:
            return state, diagnostics
        return replace(state, kda_state=new_tensor, has_long_memory=True), diagnostics

    def detach(self, state: HybridMemoryState, reproject: bool = False) -> HybridMemoryState:
        """TBPTT boundary after optimizer step; retain values, never the old graph."""
        result = state.detached()
        if reproject and result.short_occupancy:
            if result.short_raw_k is None or result.short_raw_v is None:
                raise ValueError("Reprojection requires training-only raw short shadows")
            b, t, p, d = result.short_raw_k.shape
            kp, vp = self.short.project_kv(result.short_raw_k.reshape(b, t*p, d),
                                           result.short_raw_v.reshape(b, t*p, d))
            shape = (b, t, p, self.config.short.num_heads, self.config.short.head_dim)
            result = replace(result, short_k=kp.reshape(shape), short_v=vp.reshape(shape))
        return result

    def reproject(self, state: HybridMemoryState) -> HybridMemoryState:
        return self.detach(state, reproject=True)

    forward = read
