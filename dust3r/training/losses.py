"""New defensible pointmap objective, NOT the unreleased Endo3R authors' loss.

The first-frame scale is fit once on the true GT mask and is detached. Future
predictions are never individually realigned and invalid predictions never
remove difficult pixels from supervision.
"""
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def pointmap_from_output(output: Mapping[str, Tensor]) -> Tensor:
    for key in ("pts3d", "pts3d_in_other_view"):
        if key in output:
            return output[key]
    raise KeyError("Prediction must expose pts3d or pts3d_in_other_view")


def _validate(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError(f"Expected matching [...,3] pointmaps, got {prediction.shape}, {target.shape}")
    if mask.shape != target.shape[:-1]:
        raise ValueError("GT valid_mask shape must match the pointmap spatial dimensions")
    mask = mask.to(device=prediction.device, dtype=torch.bool)
    if not mask.any():
        raise ValueError("Missing supervision: real GT valid_mask has no valid pixels")
    if not torch.isfinite(target[mask]).all():
        raise ValueError("Nonfinite GT on the true GT mask")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite prediction; prediction validity cannot shrink the GT mask")
    return mask


@dataclass(frozen=True)
class EpisodeScale:
    scale: Tensor
    radius: Tensor

    @classmethod
    def fit(cls, prediction: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-6):
        target = target.to(device=prediction.device, dtype=torch.float32)
        mask = _validate(prediction, target, mask)
        with torch.no_grad():
            radius = target.float()[mask].norm(dim=-1).median()
            predicted_radius = prediction.float()[mask].norm(dim=-1).median()
            if not torch.isfinite(radius + predicted_radius) or radius <= eps or predicted_radius <= eps:
                raise ValueError("Degenerate first-frame median radius; cannot fit episode scale")
            return cls((radius / predicted_radius).detach(), radius.detach())


class ReconstructionLoss(nn.Module):
    """Masked common-first-camera point L1 with optional confidence calibration."""

    def __init__(self, pointmap_weight: float = 1.0, confidence_weight: float = 0.0,
                 confidence_mode_verified: bool = False):
        super().__init__()
        if pointmap_weight < 0 or confidence_weight < 0:
            raise ValueError("Loss weights must be nonnegative")
        if confidence_weight and not confidence_mode_verified:
            raise ValueError("Confidence calibration requires verified checkpoint confidence semantics")
        self.pointmap_weight = pointmap_weight
        self.confidence_weight = confidence_weight

    def frame(self, output: Mapping[str, Tensor], target: Tensor, mask: Tensor,
              episode_scale: EpisodeScale):
        pred = pointmap_from_output(output)
        target = target.to(device=pred.device, dtype=torch.float32)
        mask = _validate(pred, target, mask)
        scale = episode_scale.scale.to(pred.device)
        radius = episode_scale.radius.to(pred.device)
        # GT may be undefined outside its supplied mask. Sanitizing those values
        # avoids zero upstream gradients multiplying a NaN local derivative.
        safe_target = torch.where(mask[..., None], target, torch.zeros_like(target))
        error = (scale * pred.float() - safe_target).abs().mean(dim=-1) / radius
        point_loss = error[mask].mean()
        confidence_loss = point_loss.new_zeros(())
        if self.confidence_weight:
            if "conf" not in output:
                raise KeyError("Optional confidence loss needs prediction['conf']")
            conf = output["conf"].float()
            if conf.ndim == error.ndim + 1 and conf.shape[-1] == 1:
                conf = conf.squeeze(-1)
            if conf.shape != error.shape or not torch.isfinite(conf).all():
                raise ValueError("Confidence shape/nonfinite output is invalid")
            conf = conf.clamp(1, 100)
            confidence_loss = (conf * error - 0.2 * conf.log())[mask].mean()
        loss = self.pointmap_weight * point_loss + self.confidence_weight * confidence_loss
        return loss, {"pointmap": point_loss.detach(), "confidence": confidence_loss.detach(),
                      "valid_pixels": int(mask.sum())}

    def forward(self, res1, res2, gt1, gt2, episode_scale):
        left, left_metrics = self.frame(res1, gt1["pointmap"], gt1["valid_mask"], episode_scale)
        right, right_metrics = self.frame(res2, gt2["pointmap"], gt2["valid_mask"], episode_scale)
        return 0.5 * (left + right), {"left": left_metrics, "right": right_metrics}


def feature_distillation_loss(student: Tensor, teacher: Tensor, cosine_weight: float = 0.1) -> Tensor:
    """Prefix-aligned optional teacher feature loss; teacher is always detached."""
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("Distillation expects matching [B,P,C] features")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise FloatingPointError("Nonfinite teacher/student features cannot supervise distillation")
    teacher = teacher.detach().to(device=student.device, dtype=torch.float32)
    student = student.float()
    return F.l1_loss(student, teacher) + cosine_weight * (1 - F.cosine_similarity(student, teacher, dim=-1)).mean()
