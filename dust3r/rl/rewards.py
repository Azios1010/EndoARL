"""Training-only causal rewards with explicit supervision provenance and costs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _points(prediction):
    if isinstance(prediction, torch.Tensor):
        return prediction
    for key in ("pts3d", "pointmap", "pts3d_in_other_view"):
        value = _get(prediction, key)
        if value is not None:
            return value
    raise ValueError("Reward prediction/target must expose pointmap coordinates.")


@dataclass(frozen=True)
class CostCalibration:
    baseline: float
    raft: float = 0.0
    kda: float = 0.0
    geometry: float = 0.0
    hardware_id: str = ""
    image_hw: tuple[int, int] | None = None

    def __post_init__(self):
        if self.baseline <= 0 or any(value < 0 for value in (self.raft, self.kda, self.geometry)):
            raise ValueError("Calibrated baseline must be positive and component costs nonnegative.")
        if not all(torch.isfinite(torch.tensor(value)) for value in (self.baseline, self.raft, self.kda, self.geometry)):
            raise ValueError("Cost calibration must be finite.")


@dataclass
class RewardResult:
    reward: float
    quality_error: float
    components: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    failure: bool = False

    def __float__(self):
        return self.reward


class RewardComputer:
    """No actor may hold this object; input is the NEXT decoded left output."""
    mode = "abstract"

    def __init__(self, quality_weight=1.0, write_weight=0.0, raft_weight=0.01,
                 kda_geometry_weight=0.01, error_cap=5.0, calibration=None):
        self.quality_weight = float(quality_weight)
        self.write_weight = float(write_weight)
        self.raft_weight = float(raft_weight)
        self.kda_geometry_weight = float(kda_geometry_weight)
        self.error_cap = float(error_cap)
        self.calibration = calibration
        if min(quality_weight, write_weight, raft_weight, kda_geometry_weight) < 0 or error_cap <= 0:
            raise ValueError("Reward weights must be nonnegative and error cap positive.")
        if (raft_weight or kda_geometry_weight) and calibration is None:
            raise ValueError("Compute-sensitive reward requires measured CostCalibration; no guessed runtime costs.")

    def reset(self, initial_prediction=None, initial_target=None):
        pass

    def _result(self, error, write_diagnostics=None, failure=False):
        if not torch.isfinite(torch.tensor(error)):
            error, failure = self.error_cap, True
        error = min(max(float(error), 0), self.error_cap)
        requested = bool(_get(write_diagnostics, "write_attempt", _get(write_diagnostics, "requested_write", False)))
        raft = bool(_get(write_diagnostics, "raft_called", False))
        kda = bool(_get(write_diagnostics, "kda_updated", _get(write_diagnostics, "update_success", False)))
        geometry = bool(_get(write_diagnostics, "geometry_called", _get(write_diagnostics, "geometry_computed", raft)))
        calibration = self.calibration
        components = {"quality": -self.quality_weight * error,
                      "write": -self.write_weight * requested,
                      "raft": 0.0, "kda": 0.0, "geometry": 0.0}
        if calibration is not None:
            components["raft"] = -self.raft_weight * raft * calibration.raft / calibration.baseline
            components["kda"] = -self.kda_geometry_weight * kda * calibration.kda / calibration.baseline
            components["geometry"] = -self.kda_geometry_weight * geometry * calibration.geometry / calibration.baseline
        return RewardResult(sum(components.values()), error, components, failure=failure)

    def failure_reward(self, write_diagnostics=None):
        return self._result(self.error_cap, write_diagnostics, failure=True)

    def compute(self, next_prediction, target=None, write_diagnostics=None, *, terminal=False):
        raise NotImplementedError


class DenseGTReward(RewardComputer):
    """Fixed episode alignment and masked common-frame point L1.

    The target must explicitly assert provenance_verified/verified and contain
    pointmap and valid_mask. A fabricated demo target is deliberately rejected.
    """
    mode = "dense_gt"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = None
        self.radius_scale = None

    def _validated_target(self, target, prediction):
        verified = _get(target, "provenance_verified", _get(target, "verified", False))
        if not verified:
            raise ValueError("DenseGTReward requires provenance-verified calibrated GT, not demo placeholders.")
        points = _points(target).detach().to(prediction.device, torch.float32)
        mask = _get(target, "valid_mask")
        if mask is None:
            raise ValueError("Dense GT reward requires an explicit GT validity mask.")
        mask = torch.as_tensor(mask, device=prediction.device, dtype=torch.bool)
        if points.shape != prediction.shape or mask.shape != prediction.shape[:-1]:
            raise ValueError("Reward GT must match common-coordinate prediction shapes [B,H,W,3].")
        mask = mask & torch.isfinite(points).all(-1)
        if not mask.any():
            raise ValueError("No valid GT: discard this episode, do not fabricate a model-failure reward.")
        return points, mask

    @torch.no_grad()
    def reset(self, initial_prediction=None, initial_target=None):
        self.scale, self.radius_scale = None, None
        if initial_prediction is None or initial_target is None:
            raise ValueError("Dense reward reset requires first prediction and verified first GT.")
        predicted = _points(initial_prediction).detach().float()
        target, mask = self._validated_target(initial_target, predicted)
        if not torch.isfinite(predicted[mask]).all():
            raise ValueError("Initial prediction is nonfinite on GT-valid pixels.")
        target_radius = target[mask].norm(dim=-1).median()
        predicted_radius = predicted[mask].norm(dim=-1).median()
        if target_radius <= 1e-6 or predicted_radius <= 1e-6:
            raise ValueError("Initial episode scale is degenerate.")
        self.scale = (target_radius / predicted_radius).detach()
        self.radius_scale = target_radius.clamp_min(1e-6).detach()

    @torch.no_grad()
    def compute(self, next_prediction, target=None, write_diagnostics=None, *, terminal=False):
        if self.scale is None:
            raise RuntimeError("Reward episode must be reset before scoring future reconstruction.")
        predicted = _points(next_prediction).detach().float()
        target_points, mask = self._validated_target(target, predicted)
        if not torch.isfinite(predicted[mask]).all():
            return self.failure_reward(write_diagnostics)
        error = ((self.scale * predicted - target_points).abs().mean(-1)[mask].mean() /
                 self.radius_scale).item()
        result = self._result(error, write_diagnostics)
        result.metrics = {"point_l1_normalized": float(error), "gt_valid_ratio": mask.float().mean().item()}
        return result


class TeacherProxyReward(DenseGTReward):
    """Teacher agreement, explicitly not GT reconstruction quality."""
    mode = "teacher_proxy"

    def __init__(self, *args, teacher_id=None, **kwargs):
        if not teacher_id:
            raise ValueError("Teacher proxy reward needs an identified frozen teacher checkpoint.")
        self.teacher_id = str(teacher_id)
        super().__init__(*args, **kwargs)

    def _validated_target(self, target, prediction):
        if not _get(target, "teacher_verified", False) or _get(target, "teacher_id") != self.teacher_id:
            raise ValueError("Proxy target must identify the same verified causal frozen teacher.")
        points = _points(target).detach().to(prediction.device, torch.float32)
        mask = _get(target, "valid_mask")
        if mask is None:
            mask = torch.isfinite(points).all(-1)
        return super()._validated_target({"pointmap": points, "valid_mask": mask,
                                          "provenance_verified": True}, prediction)


class SequenceGTReward(RewardComputer):
    """Sparse terminal reward; requires prevalidated sequence metrics and scales."""
    mode = "sequence_gt"

    def __init__(self, *args, metric_weights=None, metric_scales=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_weights = metric_weights or {"reconstruction": 1.0}
        self.metric_scales = metric_scales or {"reconstruction": 1.0}
        if (set(self.metric_weights) != set(self.metric_scales) or
                any(s <= 0 for s in self.metric_scales.values()) or
                any(weight < 0 for weight in self.metric_weights.values())):
            raise ValueError("Terminal metrics require explicit positive scales for each weighted metric.")

    def compute(self, next_prediction=None, target=None, write_diagnostics=None, *, terminal=False):
        if not terminal:
            return self._result(0.0, write_diagnostics)
        if not _get(target, "provenance_verified", _get(target, "verified", False)):
            raise ValueError("Sequence terminal metrics require verified GT/protocol.")
        metrics = _get(target, "metrics", {})
        missing = set(self.metric_weights) - set(metrics)
        if missing:
            raise ValueError(f"Missing verified terminal metrics: {sorted(missing)}")
        if any(not torch.isfinite(torch.tensor(float(metrics[name]))) or float(metrics[name]) < 0
               for name in self.metric_weights):
            raise ValueError("GT terminal error metrics must be finite and nonnegative.")
        error = sum(self.metric_weights[name] * float(metrics[name]) / self.metric_scales[name]
                    for name in self.metric_weights)
        result = self._result(error, write_diagnostics)
        result.metrics = {name: float(metrics[name]) for name in self.metric_weights}
        return result
