"""Episode-owned state and small diagnostics, never model checkpoint buffers."""

from dataclasses import dataclass, field, fields, replace
from typing import Optional, Tuple
import weakref

import torch
from torch import Tensor, nn


class SharedNorms:
    """Weak, non-Module references avoid registering Endo3R norms twice."""

    def __init__(self, norm_q: nn.Module, norm_k: nn.Module, norm_v: nn.Module):
        self._refs = {name: weakref.ref(module) for name, module in
                      (("q", norm_q), ("k", norm_k), ("v", norm_v))}

    def apply(self, name: str, value: Tensor) -> Tensor:
        module = self._refs[name]()
        if module is None:
            raise RuntimeError("The Endo3R owner of shared memory norms was destroyed")
        return module(value)


@dataclass
class ReadDiagnostics:
    gate_mean: Tensor
    gate_std: Tensor
    short_rms: Tensor
    long_rms: Tensor
    contribution_ratio: Tensor
    has_short: bool = False
    has_long: bool = False


@dataclass
class WriteDiagnostics:
    success: bool
    update_called: bool
    reason: str = ""
    beta_mean: float = 0.0
    decay_mean: float = 0.0
    beta_saturation_fraction: float = 0.0
    decay_saturation_fraction: float = 0.0


@dataclass
class MemoryCandidate:
    key: Tensor
    value: Tensor
    descriptor: Optional[Tensor] = None
    image: Optional[Tensor] = None
    pointmap: Optional[Tensor] = None
    valid_mask: Optional[Tensor] = None
    confidence: Optional[Tensor] = None
    candidate_index: int = -1
    arrival_index: int = -1


@dataclass
class WriteConfidence:
    weights: Tensor
    uncertainty_stats: Tensor
    raft_called: bool = False
    diagnostics: dict = field(default_factory=dict)

    @property
    def has_any_positive_patch(self) -> bool:
        return bool(torch.any(self.weights > 0).item())


@dataclass
class ObservationPacket:
    observation: Tensor
    action_mask: Tensor
    descriptor: Optional[Tensor] = None
    heuristic_similarity: Optional[Tensor] = None
    current_step: int = 0

    @property
    def raw_observation(self) -> Tensor:
        return self.observation


@dataclass
class PolicyDecision:
    action: Tensor
    log_prob: Optional[Tensor] = None
    value: Optional[Tensor] = None
    forced: Optional[Tensor] = None
    action_mask: Optional[Tensor] = None


@dataclass
class HybridMemoryState:
    short_k: Tensor
    short_v: Tensor
    kda_state: Tensor
    recent_key_descriptors: Tensor
    recent_long_descriptors: Tensor
    previous_uncertainty_stats: Tensor
    short_frame_indices: Tensor
    image_hw: Tuple[int, int]
    patch_hw: Tuple[int, int]
    short_raw_k: Optional[Tensor] = None
    short_raw_v: Optional[Tensor] = None
    short_cursor: int = 0
    short_occupancy: int = 0
    has_long_memory: bool = False
    last_long_descriptor: Optional[Tensor] = None
    has_last_long_descriptor: bool = False
    last_long_reference_index: int = -1
    last_long_reference_arrival_index: int = -1
    last_long_reference_image: Optional[Tensor] = None
    last_long_reference_pointmap: Optional[Tensor] = None
    last_long_reference_valid_mask: Optional[Tensor] = None
    has_previous_raft_quality: bool = False
    last_write_step: int = -1
    write_count: int = 0
    write_attempt_count: int = 0
    raft_call_count: int = 0
    kda_update_count: int = 0
    current_step: int = 0
    processed_candidate_count: int = 0
    candidate_index: int = -1
    arrival_index: int = -1
    predicted_radius_scale: Optional[Tensor] = None
    has_predicted_radius_scale: bool = False
    sequence_id: Optional[str] = None
    patch_layout_id: str = "encoder_row_major_v1"
    retain_short_raw: bool = False

    @property
    def batch_size(self) -> int:
        return self.kda_state.shape[0]

    @property
    def num_patches(self) -> int:
        return self.patch_hw[0] * self.patch_hw[1]

    def detached(self) -> "HybridMemoryState":
        return replace(self, **{field.name: getattr(self, field.name).detach()
                               for field in fields(self)
                               if isinstance(getattr(self, field.name), Tensor)})
