"""Explicit external-write-policy ablation; original memory class stays intact."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
import weakref

import torch

from .types import ReadDiagnostics, WriteDiagnostics


@dataclass
class LegacyAdapterState:
    short_k: torch.Tensor
    recent_key_descriptors: torch.Tensor
    recent_long_descriptors: torch.Tensor
    previous_uncertainty_stats: torch.Tensor
    image_hw: tuple
    patch_hw: tuple
    sequence_id: str | None = None
    short_occupancy: int = 0
    has_long_memory: bool = False
    last_long_descriptor: torch.Tensor | None = None
    has_last_long_descriptor: bool = False
    last_long_reference_image: torch.Tensor | None = None
    last_long_reference_pointmap: torch.Tensor | None = None
    last_long_reference_valid_mask: torch.Tensor | None = None
    last_long_reference_index: int = -1
    last_long_reference_arrival_index: int = -1
    has_previous_raft_quality: bool = False
    predicted_radius_scale: torch.Tensor | None = None
    has_predicted_radius_scale: bool = False
    last_write_step: int = -1
    write_count: int = 0
    write_attempt_count: int = 0
    raft_call_count: int = 0
    kda_update_count: int = 0  # Legacy writes are not KDA updates.
    current_step: int = 0
    processed_candidate_count: int = 0
    candidate_index: int = -1
    arrival_index: int = -1


class LegacyMemoryAdapter:
    """Lazy factory for an action-gated subclass; NEVER used by LEGACY_OFFICIAL.

    Dense retrieval, token pruning and old uncertainty remain original. This
    adapter records failures rather than repairing those baseline mechanisms.
    """

    def __init__(self, model, config):
        self._model = weakref.ref(model)
        self.config = config
        self.memory = None
        self.last_commit_info = {}

    def reset(self, batch_size, image_hw, patch_hw, device, training=False, sequence_id=None):
        if batch_size != 1 or training:
            raise ValueError('LEGACY_POLICY is a B=1 frozen inference/RL environment, not the broken legacy trainer.')
        required = Path('checkpoints/raft-things.pth')
        if not required.is_file():
            raise FileNotFoundError('LEGACY_POLICY retains the original mandatory RAFT path: '
                                    f'{required}. No checkpoint is downloaded.')
        from dust3r.model import SpatialTempralMemory

        class ActionGatedMemory(SpatialTempralMemory):
            pending_action = 1

            def check_sim(self, *args, **kwargs):
                return self.pending_action == 0

        model = self._model()
        self.memory = ActionGatedMemory(model.norm_q, model.norm_k, model.norm_v)
        d = self.config.memory.input_dim
        return LegacyAdapterState(torch.empty(1, 0, d, device=device),
                    torch.empty(1, 0, d, device=device), torch.empty(1, 0, d, device=device),
                    torch.zeros(1, 3, device=device), tuple(image_hw), tuple(patch_hw), sequence_id)

    def read(self, query, state):
        output = self.memory.memory_read(query, res=True)
        zeros = query.new_zeros(1)
        # Separate ST/long outputs/gates do not exist in this joint-softmax bank.
        magnitude = (output - query).float().square().mean((1, 2)).sqrt().detach()
        diag = ReadDiagnostics(zeros, zeros, zeros, magnitude, zeros,
                               has_short=state.short_occupancy > 0, has_long=state.has_long_memory)
        return output, diag

    def commit(self, candidate, action, state, terminal=False):
        memory = self.memory
        # Original add_mem_check already inserts ST. Never call a second append.
        memory.pending_action = 0 if terminal else int(action)
        uncertainty = self.config.uncertainty.enabled
        had_long = state.has_long_memory
        requested = action == 1 and not terminal
        previous_frames = len(memory.spatial_buffer)
        kwargs = dict(index=candidate.candidate_index, check_uncertainty=uncertainty,
                      cur_conf=candidate.confidence)
        if uncertainty and candidate.candidate_index > 0:
            if state.last_long_reference_pointmap is None:
                raise RuntimeError('Legacy policy has no accepted reference after initial write.')
            kwargs.update(ref_view={'origin_img': state.last_long_reference_image.permute(0, 2, 3, 1)},
                          cur_view={'origin_img': candidate.image.permute(0, 2, 3, 1)},
                          ref_pointmap=state.last_long_reference_pointmap,
                          cur_pts3d=candidate.pointmap)
        memory.add_mem_check(candidate.key, candidate.value, **kwargs)
        success = len(memory.spatial_buffer) > previous_frames
        p = state.patch_hw[0] * state.patch_hw[1]
        state.short_k = memory.temp_mem_k
        state.short_occupancy = memory.temp_mem_k.shape[1] // p
        state.recent_key_descriptors = torch.cat((state.recent_key_descriptors,
                                                 candidate.descriptor[:, None]), 1)[:, -4:]
        state.processed_candidate_count += 1
        state.write_attempt_count += int(requested)
        raft_called = success and uncertainty and candidate.candidate_index > 0
        state.raft_call_count += int(raft_called)
        if success:
            state.has_long_memory = True
            state.write_count += 1
            state.last_write_step = state.current_step
            state.last_long_descriptor = candidate.descriptor.detach().clone()
            state.has_last_long_descriptor = True
            state.recent_long_descriptors = torch.cat((state.recent_long_descriptors,
                                                       candidate.descriptor[:, None]), 1)[:, -4:]
            state.last_long_reference_image = candidate.image.detach().clone()
            state.last_long_reference_pointmap = candidate.pointmap.detach().clone()
            state.last_long_reference_valid_mask = candidate.valid_mask.detach().clone()
            state.last_long_reference_index = candidate.candidate_index
            state.last_long_reference_arrival_index = candidate.arrival_index
            if raft_called and memory.mem_uncert is not None:
                u = memory.mem_uncert[:, -p:].float()
                if torch.isfinite(u).all():
                    state.previous_uncertainty_stats = torch.stack(
                        (u.mean(1), u.std(1, unbiased=False), torch.zeros_like(u.mean(1))), -1).detach()
                    # The old implementation exposes no trustworthy valid-flow ratio.
                    state.has_previous_raft_quality = False
        self.last_commit_info = dict(requested_write=requested, successful_write=success,
                    raft_called=raft_called, kda_updated=False, geometry_called=raft_called,
                    forced=not had_long and not terminal,
                    candidate_index=candidate.candidate_index, arrival_index=candidate.arrival_index)
        return WriteDiagnostics(success, False, 'legacy_spatial_write' if success else 'TEMP_ONLY')

    def detach(self, state, reproject=False):
        return replace(state, **{field.name: getattr(state, field.name).detach()
                                 for field in fields(state)
                                 if isinstance(getattr(state, field.name), torch.Tensor)})
