"""Cheap, causal actor observations. This module has no GT or RAFT input API."""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .types import ObservationPacket


ACTOR_STATE_DIM = 30
CONTINUOUS_INDICES = (0, 1, 2, 3, 4, 10, 24, 25, 28, 29)
FEATURE_NAMES = (
    "confidence_log_mean", "confidence_log_std", "confidence_log_p10",
    "confidence_log_p50", "confidence_log_p90", "prediction_valid_ratio",
    "cosine_last_long", "cosine_short_max", "cosine_short_mean",
    "descriptor_distance_last_long", "key_log_rms", "has_long_memory",
    "long_write_log_count", "long_write_ratio", "steps_since_long_write",
    "current_step_log", "short_occupancy", "has_long_descriptor",
    "previous_uncertainty_mean", "previous_uncertainty_std",
    "previous_valid_flow_ratio", "has_previous_raft_quality",
    "fusion_gate_mean", "fusion_gate_std", "short_output_log_rms",
    "long_output_log_rms", "short_contribution_ratio", "has_short_memory",
    "point_radius_log_median", "point_radius_log_spread",
)


def pooled_descriptor(key: Tensor) -> Tensor:
    if key.ndim != 3:
        raise ValueError("Memory key must have shape [B,P,D].")
    if not torch.isfinite(key).all():
        raise ValueError("Nonfinite memory keys cannot enter actor observations.")
    return F.normalize(key.detach().float().mean(dim=1), dim=-1, eps=1e-6)


def _pointmap(prediction: Mapping[str, Tensor]) -> Tensor:
    for name in ("pts3d", "pointmap", "pts3d_in_other_view"):
        if name in prediction:
            return prediction[name]
    raise ValueError("Prediction must expose a canonical pts3d/pointmap tensor.")


def _scalar(value, batch: int, device, *, default=0.0) -> Tensor:
    if value is None:
        value = default
    result = torch.as_tensor(value, device=device, dtype=torch.float32).detach()
    if result.numel() == 1:
        return result.reshape(1).expand(batch)
    if result.numel() != batch:
        raise ValueError("Memory diagnostics must have one scalar per sequence.")
    return result.reshape(batch)


class ObservationBuilder:
    """Produce the plan's detached 30 scalars from PRE-UPDATE memory.

    Quantiles use a deterministic <=16x16 sample. Full-map mean/std/validity
    reductions never materialize attention matrices. Only the first predicted
    radius scale is initialized here; counts/references are not updated.
    """

    state_dim = ACTOR_STATE_DIM
    feature_names = FEATURE_NAMES

    @torch.no_grad()
    def build(self, prediction, key: Tensor, state, diagnostics=None) -> ObservationPacket:
        pointmap = _pointmap(prediction).detach().float()
        confidence = prediction.get("conf", prediction.get("confidence"))
        if confidence is None:
            raise ValueError("Current reconstruction confidence is required.")
        confidence = confidence.detach().float()
        if confidence.ndim == 4 and confidence.shape[-1] == 1:
            confidence = confidence.squeeze(-1)
        if pointmap.ndim != 4 or pointmap.shape[-1] != 3:
            raise ValueError("Pointmap must have shape [B,H,W,3].")
        batch, height, width, _ = pointmap.shape
        if confidence.shape != (batch, height, width) or key.shape[0] != batch:
            raise ValueError("Confidence, pointmap and key batch/spatial shapes differ.")
        device = key.device
        if pointmap.device != device or confidence.device != device:
            raise ValueError("Actor input tensors must share their device.")
        descriptor = pooled_descriptor(key)
        result = torch.zeros(batch, ACTOR_STATE_DIM, device=device)
        confidence_valid = torch.isfinite(confidence) & (confidence > 0)
        point_valid = torch.isfinite(pointmap).all(-1)
        joint_valid = confidence_valid & point_valid
        if (~joint_valid.flatten(1).any(1)).any():
            raise ValueError("Entirely invalid reconstruction: terminate the episode explicitly.")
        rows = torch.linspace(0, height - 1, min(16, height), device=device).long().unique()
        cols = torch.linspace(0, width - 1, min(16, width), device=device).long().unique()
        radii = torch.linalg.vector_norm(torch.nan_to_num(pointmap), dim=-1)
        sampled_radius = radii[:, rows[:, None], cols[None, :]]
        sampled_conf = confidence[:, rows[:, None], cols[None, :]]
        sampled_valid = joint_valid[:, rows[:, None], cols[None, :]]
        scale = getattr(state, "predicted_radius_scale", None)
        if scale is None:
            scale = torch.stack([radii[b][point_valid[b]].median().clamp_min(1e-6)
                                 for b in range(batch)]).detach()
            state.predicted_radius_scale = scale
            if hasattr(state, "predicted_radius_scale_valid"):
                state.predicted_radius_scale_valid = True
            if hasattr(state, "has_predicted_radius_scale"):
                state.has_predicted_radius_scale = True
        scale = _scalar(scale, batch, device).clamp_min(1e-6)
        probabilities = torch.tensor([0.1, 0.5, 0.9], device=device)
        for b in range(batch):
            z = confidence[b][confidence_valid[b]].clamp(1e-4, 1e4).log()
            result[b, 0] = z.mean() if z.numel() else 0
            result[b, 1] = z.std(unbiased=False) if z.numel() else 0
            sampled_z = sampled_conf[b][torch.isfinite(sampled_conf[b]) & (sampled_conf[b] > 0)]
            if sampled_z.numel():
                result[b, 2:5] = torch.quantile(sampled_z.clamp(1e-4, 1e4).log(), probabilities)
            radius_sample = sampled_radius[b][sampled_valid[b]]
            if radius_sample.numel():
                rq = torch.quantile(radius_sample, probabilities)
                result[b, 28] = torch.log1p(rq[1] / scale[b])
                result[b, 29] = torch.log1p((rq[2] - rq[0]).clamp_min(0) / scale[b])
        result[:, 5] = joint_valid.float().flatten(1).mean(1)
        last_descriptor = getattr(state, "last_long_descriptor", None)
        if last_descriptor is not None:
            last = F.normalize(last_descriptor.detach().float(), dim=-1, eps=1e-6)
            result[:, 6] = (descriptor * last).sum(-1).clamp(-1, 1)
            result[:, 9] = ((descriptor - last).norm(dim=-1) / 2).clamp(0, 1)
            result[:, 17] = 1
        recent = getattr(state, "recent_key_descriptors", None)
        if recent is not None and recent.shape[1]:
            scores = torch.einsum("bd,btd->bt", descriptor, recent.detach().float())
            result[:, 7] = scores.max(1).values.clamp(-1, 1)
            result[:, 8] = scores.mean(1).clamp(-1, 1)
        result[:, 10] = torch.log1p(key.detach().float().mean(1).square().mean(-1).sqrt())
        has_long = bool(getattr(state, "has_long_memory", False))
        step = int(getattr(state, "current_step", 0))
        writes = int(getattr(state, "write_count", 0))
        processed = int(getattr(state, "processed_candidate_count", 0))
        result[:, 11] = float(has_long)
        result[:, 12] = min(math.log1p(writes) / math.log(1025), 1)
        result[:, 13] = min(writes / max(processed, 1), 1)
        if has_long:
            elapsed = max(step - int(getattr(state, "last_write_step", step)), 0)
            result[:, 14] = min(math.log1p(elapsed) / math.log(257), 1)
        result[:, 15] = min(math.log1p(step) / math.log(1025), 1)
        occupancy = int(getattr(state, "short_occupancy", 0))
        result[:, 16] = min(occupancy / 4, 1)
        has_quality = bool(getattr(state, "has_previous_raft_quality", False))
        quality = getattr(state, "previous_uncertainty_stats", None)
        if has_quality and quality is not None:
            quality = quality.detach().float()
            result[:, 18] = quality[:, 0].clamp(0, 1)
            result[:, 19] = (2 * quality[:, 1]).clamp(0, 1)
            result[:, 20] = quality[:, 2].clamp(0, 1)
            result[:, 21] = 1
        for index, name, multiplier, logarithmic in (
            (22, "gate_mean", 1, False), (23, "gate_std", 2, False),
            (24, "short_rms", 1, True), (25, "long_rms", 1, True),
            (26, "contribution_ratio", 1, False),
        ):
            value = _scalar(getattr(diagnostics, name, None), batch, device)
            result[:, index] = torch.log1p(value.clamp_min(0)) if logarithmic else (value * multiplier).clamp(0, 1)
        result[:, 27] = float(occupancy > 0)
        recent_long = getattr(state, "recent_long_descriptors", None)
        heuristic_similarity = torch.full((batch,), -1.0, device=device)
        if recent_long is not None and recent_long.shape[1]:
            heuristic_similarity = torch.einsum("bd,btd->bt", descriptor, recent_long.detach().float()).max(1).values
        mask = torch.ones(batch, 2, dtype=torch.bool, device=device)
        if not has_long:
            mask[:, 0] = False
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite actor observation; no silent replacement is allowed.")
        return ObservationPacket(observation=result.detach(), action_mask=mask,
                                 descriptor=descriptor, heuristic_similarity=heuristic_similarity.detach(),
                                 current_step=step)
