"""Predicted camera geometry for Sampson writes and paper-style evaluation.

All geometry comes from model pointmaps. Calibration/poses from a dataset are
deliberately absent from this API. OpenCV is imported only by PnP execution;
pure torch functions remain usable on CPU without it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class CameraEstimate:
    intrinsics: Tensor
    world_to_camera: Tensor
    valid: Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def estimate_intrinsics(pointmap: Tensor, valid_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Masked square-pixel Weiszfeld focal estimate in FIRST camera coordinates.

    Uses the released Endo3R principal-point convention ``(W/2,H/2)`` and
    ten robust iterations, as ``post_process.estimate_focal_knowing_depth``.
    The common-frame XYZ of later frames MUST NOT independently initialize K.
    Invalid/degenerate estimates have ``valid=False``; no guessed focal is used.
    """
    if pointmap.ndim != 4 or pointmap.shape[-1] != 3:
        raise ValueError('Predicted pointmap must be [B,H,W,3].')
    b, h, w, _ = pointmap.shape
    pts = pointmap.detach().float()
    mask = torch.isfinite(pts).all(-1) & (pts[..., 2] > 1e-6)
    if valid_mask is not None:
        if valid_mask.shape != (b, h, w):
            raise ValueError('Camera validity mask must be [B,H,W].')
        mask &= valid_mask.bool()
    yy, xx = torch.meshgrid(torch.arange(h, device=pts.device),
                            torch.arange(w, device=pts.device), indexing='ij')
    pixels = torch.stack((xx - w / 2., yy - h / 2.), -1).float()
    matrices = torch.eye(3, device=pts.device).expand(b, 3, 3).clone()
    matrices[:, 0, 2] = w / 2.
    matrices[:, 1, 2] = h / 2.
    validity = torch.zeros(b, device=pts.device, dtype=torch.bool)
    for batch in range(b):
        xyz = pts[batch][mask[batch]]
        px = pixels[mask[batch]]
        if xyz.shape[0] < 6:
            continue
        rays = xyz[:, :2] / xyz[:, 2:]
        xy_px = (rays * px).sum(-1)
        xy_xy = rays.square().sum(-1)
        denom = xy_xy.sum()
        if not torch.isfinite(denom) or denom <= 1e-12:
            continue
        focal = xy_px.sum() / denom
        for _ in range(10):
            weights = torch.linalg.vector_norm(px - focal * rays, dim=-1).clamp_min(1e-8).reciprocal()
            denom = (weights * xy_xy).sum()
            focal = (weights * xy_px).sum() / denom.clamp_min(1e-12)
        if torch.isfinite(focal) and focal > 0:
            matrices[batch, 0, 0] = focal
            matrices[batch, 1, 1] = focal
            validity[batch] = True
    return matrices, validity


class PredictedCameraEstimator:
    """CPU PnP from dense predicted common-frame XYZ and pixel correspondences.

    ``intrinsics`` should be anchored on the first predicted frame. The optional
    initialization without K is provided for that first frame only. Pose output
    maps the common coordinate frame into each camera (world-to-camera).

    SERVER_VALIDATION_REQUIRED: OpenCV server version, PnP failure/inlier rates,
    focal stability and image/pointmap conventions on real SCARED sequences.
    """

    def __init__(self, max_points: int = 4096, reprojection_error_px: float = 8.0,
                 iterations: int = 100):
        if max_points < 6 or reprojection_error_px <= 0 or iterations <= 0:
            raise ValueError('PnP requires max_points>=6 and positive error/iterations.')
        self.max_points = int(max_points)
        self.reprojection_error_px = float(reprojection_error_px)
        self.iterations = int(iterations)

    @torch.no_grad()
    def __call__(self, pointmap: Tensor, valid_mask: Tensor | None = None,
                 confidence: Tensor | None = None, intrinsics: Tensor | None = None) -> CameraEstimate:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ImportError('Sampson uncertainty/predicted-camera evaluation requires local '
                              'OpenCV (cv2) and NumPy on the server. No dependency is downloaded.') from exc
        if pointmap.ndim != 4 or pointmap.shape[-1] != 3:
            raise ValueError('PnP pointmap must be [B,H,W,3].')
        b, h, w, _ = pointmap.shape
        pts = pointmap.detach().float()
        mask = torch.isfinite(pts).all(-1)
        if valid_mask is not None:
            if valid_mask.shape != (b, h, w):
                raise ValueError('PnP validity mask must match the pointmap.')
            mask &= valid_mask.bool()
        if confidence is not None:
            if confidence.shape != (b, h, w):
                raise ValueError('PnP confidence must be [B,H,W].')
            mask &= torch.isfinite(confidence) & (confidence > 0)
        if intrinsics is None:
            intrinsics, valid_k = estimate_intrinsics(pts, mask)
        else:
            if intrinsics.shape != (b, 3, 3) or intrinsics.device != pts.device:
                raise ValueError('Predicted intrinsics must share device and shape [B,3,3].')
            intrinsics = intrinsics.detach().float()
            valid_k = (torch.isfinite(intrinsics).all((1, 2)) &
                       (intrinsics[:, 0, 0] > 0) & (intrinsics[:, 1, 1] > 0) &
                       (torch.linalg.det(intrinsics).abs() > 1e-12))
        poses = torch.eye(4, device=pts.device).expand(b, 4, 4).clone()
        valid = torch.zeros(b, device=pts.device, dtype=torch.bool)
        inlier_ratios, errors, reasons = [], [], []
        for batch in range(b):
            indices = torch.nonzero(mask[batch].flatten(), as_tuple=False).flatten()
            if not valid_k[batch] or indices.numel() < 6:
                inlier_ratios.append(0.0)
                errors.append(None)
                reasons.append('invalid_intrinsics_or_insufficient_correspondences')
                continue
            if indices.numel() > self.max_points:
                select = torch.linspace(0, indices.numel() - 1, self.max_points,
                                        device=pts.device).round().long()
                indices = indices[select]
            objects = pts[batch].reshape(-1, 3)[indices].cpu().numpy().astype(np.float64)
            pixels = torch.stack((indices % w, indices // w), -1).cpu().numpy().astype(np.float64)
            camera = intrinsics[batch].cpu().numpy().astype(np.float64)
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objects, pixels, camera, np.zeros(4), iterationsCount=self.iterations,
                    reprojectionError=self.reprojection_error_px, confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if not success or inliers is None or len(inliers) < 6:
                    raise ValueError('pnp_failed_or_insufficient_inliers')
                rotation, _ = cv2.Rodrigues(rvec)
                pose = np.eye(4)
                pose[:3, :3], pose[:3, 3] = rotation, tvec.reshape(3)
                if not np.isfinite(pose).all():
                    raise ValueError('nonfinite_pnp_pose')
                camera_xyz = objects @ rotation.T + tvec.reshape(1, 3)
                if np.mean(camera_xyz[:, 2] > 0) < 0.5:
                    raise ValueError('pnp_points_behind_camera')
                projected, _ = cv2.projectPoints(objects, rvec, tvec, camera, np.zeros(4))
                error = np.linalg.norm(projected[:, 0] - pixels, axis=-1)
                inlier_error = float(np.sqrt(np.mean(error[inliers[:, 0]] ** 2)))
                poses[batch] = torch.from_numpy(pose).to(pts.device, torch.float32)
                valid[batch] = True
                inlier_ratios.append(float(len(inliers) / len(objects)))
                errors.append(inlier_error)
                reasons.append('ok')
            except (cv2.error, ValueError) as exc:
                inlier_ratios.append(0.0)
                errors.append(None)
                reasons.append(str(exc))
        return CameraEstimate(intrinsics, poses, valid, dict(
            camera_source='predicted_pointmap_pnp', pnp_max_points=self.max_points,
            pnp_inlier_ratio=inlier_ratios, pnp_inlier_rmse_px=errors, pnp_reason=reasons))


def skew_symmetric(vector: Tensor) -> Tensor:
    if vector.ndim != 2 or vector.shape[-1] != 3:
        raise ValueError('Translation must be [B,3].')
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)


@torch.no_grad()
def fundamental_from_cameras(current: CameraEstimate, reference: CameraEstimate,
                             min_translation: float = 1e-6) -> tuple[Tensor, Tensor, Tensor]:
    """F for x_ref^T F x_cur=0; reject pure rotation/invalid predicted cameras."""
    if min_translation <= 0:
        raise ValueError('Minimum camera translation must be positive.')
    b = current.intrinsics.shape[0]
    for camera in (current, reference):
        if (camera.intrinsics.shape != (b, 3, 3) or camera.world_to_camera.shape != (b, 4, 4)
                or camera.valid.shape != (b,)):
            raise ValueError('Camera layout must be K[B,3,3], w2c[B,4,4], valid[B].')
    matrices = current.intrinsics.new_zeros((b, 3, 3), dtype=torch.float32)
    valid = current.valid.bool() & reference.valid.bool()
    baselines = matrices.new_zeros(b)
    for batch in range(b):
        if not valid[batch]:
            continue
        kc, kr = current.intrinsics[batch].float(), reference.intrinsics[batch].float()
        tc, tr = current.world_to_camera[batch].float(), reference.world_to_camera[batch].float()
        if not all(torch.isfinite(t).all() for t in (kc, kr, tc, tr)):
            valid[batch] = False
            continue
        try:
            relative = tr @ torch.linalg.inv(tc)
            translation = relative[:3, 3]
            baseline = torch.linalg.vector_norm(translation)
            baselines[batch] = baseline
            if baseline < min_translation:
                valid[batch] = False
                continue
            f = (torch.linalg.inv(kr).T @ skew_symmetric(translation.unsqueeze(0))[0]
                 @ relative[:3, :3] @ torch.linalg.inv(kc))
            norm = torch.linalg.vector_norm(f)
            if not torch.isfinite(f).all() or norm <= 1e-12:
                valid[batch] = False
                continue
            matrices[batch] = f / norm
        except RuntimeError:
            valid[batch] = False
    return matrices, valid, baselines


@torch.no_grad()
def sampson_residual_squared(fundamental: Tensor, current_xy: Tensor,
                             reference_xy: Tensor, denominator_epsilon: float = 1e-12
                             ) -> tuple[Tensor, Tensor]:
    """Squared Sampson residual in pixels squared, with a separate valid mask.

    Inputs are F[B,3,3] and corresponding pixel coordinates [B,...,2].
    Invalid coordinates/zero denominators produce a finite placeholder zero,
    NEVER confidence; the caller must combine the returned validity mask.
    """
    if (fundamental.ndim != 3 or fundamental.shape[1:] != (3, 3) or
            current_xy.shape != reference_xy.shape or current_xy.ndim < 3 or
            current_xy.shape[0] != fundamental.shape[0] or current_xy.shape[-1] != 2):
        raise ValueError('Sampson expects F[B,3,3] and matching [B,...,2] coordinates.')
    if denominator_epsilon <= 0:
        raise ValueError('Sampson denominator epsilon must be positive.')
    shape = current_xy.shape[:-1]
    b = shape[0]
    cur, ref = current_xy.float().reshape(b, -1, 2), reference_xy.float().reshape(b, -1, 2)
    finite = torch.isfinite(cur).all(-1) & torch.isfinite(ref).all(-1)
    cur = torch.where(torch.isfinite(cur), cur, torch.zeros_like(cur))
    ref = torch.where(torch.isfinite(ref), ref, torch.zeros_like(ref))
    cur_h = torch.cat((cur, torch.ones_like(cur[..., :1])), -1)
    ref_h = torch.cat((ref, torch.ones_like(ref[..., :1])), -1)
    f = fundamental.float()
    fx = torch.bmm(cur_h, f.transpose(1, 2))
    ftx = torch.bmm(ref_h, f)
    numerator = (ref_h * fx).sum(-1).square()
    denominator = fx[..., :2].square().sum(-1) + ftx[..., :2].square().sum(-1)
    valid = (finite & torch.isfinite(f).all((1, 2))[:, None] &
             torch.isfinite(numerator) & torch.isfinite(denominator) &
             (denominator > denominator_epsilon))
    residual = numerator / denominator.clamp_min(denominator_epsilon)
    valid &= torch.isfinite(residual)
    residual = torch.where(valid, residual, torch.zeros_like(residual))
    return residual.reshape(shape), valid.reshape(shape)
