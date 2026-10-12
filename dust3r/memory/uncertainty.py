"""Write-only uncertainty estimation; no network import or download at import time."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .types import WriteConfidence


class _RAFTArgs(SimpleNamespace):
    def __contains__(self, key):
        return hasattr(self, key)


class UncertaintyEstimator(nn.Module):
    """Own exactly one frozen RAFT, or run neutral confidence when disabled.

    Injecting ``flow_model`` is an explicit testing/integration hook, not an
    automatic replacement for missing weights. Real RAFT is imported only
    after validating the configured local checkpoint.
    """

    def __init__(self, config, checkpoint_path=None, flow_model=None, patch_hw=(16, 16),
                 camera_estimator=None):
        super().__init__()
        self.config = config
        self.enabled = bool(config.enabled)
        self.metric = getattr(config, 'metric', 'geometry_3d')
        if self.metric not in {'geometry_3d', 'sampson'}:
            raise ValueError('Uncertainty metric must be geometry_3d or sampson.')
        self.patch_hw = tuple(patch_hw)
        self.camera_estimator = camera_estimator
        self.reset()
        self.flow_model = None
        if self.enabled:
            if flow_model is None:
                checkpoint = Path(checkpoint_path or 'checkpoints/raft-things.pth')
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f'Uncertainty is enabled but RAFT checkpoint is missing: {checkpoint}. '
                        'Supply this local checkpoint or explicitly set uncertainty.enabled=false. '
                        'No automatic download is performed.')
                try:
                    from dust3r.core.raft import RAFT
                except ImportError as exc:
                    raise ImportError('Unable to import the repository RAFT dependencies; '
                                      'prepare the server environment explicitly.') from exc
                flow_model = RAFT(_RAFTArgs(small=False, mixed_precision=False,
                                           alternate_corr=False, dropout=0.0))
                weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if isinstance(weights, dict) and 'state_dict' in weights:
                    weights = weights['state_dict']
                if not isinstance(weights, dict):
                    raise ValueError('RAFT checkpoint must contain a parameter dictionary.')
                weights = {key.removeprefix('module.'): value for key, value in weights.items()}
                flow_model.load_state_dict(weights, strict=True)
            self.flow_model = flow_model
            self.flow_model.requires_grad_(False)
            self.flow_model.eval()

    def reset(self):
        """Clear bounded predicted-camera caches at every episode boundary.

        These detached inference caches are not buffers, checkpoint weights or
        actor features. A shared estimator must not serve concurrent episodes.
        """
        self._predicted_intrinsics = None
        self._predicted_intrinsics_valid = None
        self._reference_pointmap = None
        self._reference_camera = None
        self.last_diagnostics = {}
        return self

    @torch.no_grad()
    def _anchor_intrinsics(self, candidate):
        if self._predicted_intrinsics is None:
            from .camera_geometry import estimate_intrinsics
            k, valid = estimate_intrinsics(candidate.pointmap, candidate.valid_mask)
            self._predicted_intrinsics = k.detach()
            self._predicted_intrinsics_valid = valid.detach()

    @torch.no_grad()
    def _camera(self, candidate):
        from .camera_geometry import CameraEstimate, PredictedCameraEstimator
        if self.camera_estimator is None:
            self.camera_estimator = PredictedCameraEstimator()
        camera = self.camera_estimator(candidate.pointmap, valid_mask=candidate.valid_mask,
                                       confidence=getattr(candidate, 'confidence', None),
                                       intrinsics=self._predicted_intrinsics)
        if not isinstance(camera, CameraEstimate):
            raise TypeError('camera_estimator must return CameraEstimate, with predicted K and w2c.')
        camera.valid = camera.valid.bool() & self._predicted_intrinsics_valid
        return camera

    @torch.no_grad()
    def _sampson_uncertainty(self, candidate, reference, current_xy, reference_xy, valid):
        from .camera_geometry import fundamental_from_cameras, sampson_residual_squared
        # OnlineRunner first initializes K on its forced first accepted write.
        # The standalone estimator also accepts an explicit accepted reference.
        self._anchor_intrinsics(reference)
        if self._reference_pointmap is not reference.pointmap:
            self._reference_camera = self._camera(reference)
            self._reference_pointmap = reference.pointmap
        current_camera = self._camera(candidate)
        f, camera_valid, baseline = fundamental_from_cameras(
            current_camera, self._reference_camera,
            min_translation=getattr(self.config, 'sampson_min_translation', 1e-6))
        residual, residual_valid = sampson_residual_squared(f, current_xy, reference_xy)
        valid = valid & residual_valid & camera_valid[:, None, None]
        sigma = getattr(self.config, 'sampson_sigma_px', 1.0)
        if sigma <= 0:
            raise ValueError('sampson_sigma_px must be positive.')
        uncertainty = (residual / (sigma * sigma)).clamp(0, 1)
        uncertainty = torch.where(valid, uncertainty, torch.ones_like(uncertainty))
        camera_reason = []
        for batch in range(valid.shape[0]):
            if not current_camera.valid[batch] or not self._reference_camera.valid[batch]:
                camera_reason.append('invalid_predicted_camera')
            elif baseline[batch] < getattr(self.config, 'sampson_min_translation', 1e-6):
                camera_reason.append('degenerate_translation_or_pure_rotation')
            elif not camera_valid[batch]:
                camera_reason.append('invalid_fundamental_matrix')
            else:
                camera_reason.append('ok')
        means = []
        for batch in range(valid.shape[0]):
            samples = residual[batch][valid[batch]]
            means.append(float(samples.mean()) if samples.numel() else None)
        self.last_diagnostics = dict(
            uncertainty_metric='sampson', sampson_residual_units='pixels_squared',
            sampson_sigma_px=float(sigma), camera_source='predicted_pointmap_only',
            baseline_common_frame_units=baseline.detach().cpu().tolist(),
            camera_valid=camera_valid.detach().cpu().tolist(), camera_reason=camera_reason,
            sampson_mean_px2=means, current_camera=current_camera.diagnostics,
            reference_camera=self._reference_camera.diagnostics)
        return uncertainty, valid

    def train(self, mode=True):
        super().train(mode)
        if self.flow_model is not None:
            self.flow_model.eval()
        return self

    def _patch_reduce(self, image, patch_hw):
        ph, pw = patch_hw
        if image.shape[-2] % ph or image.shape[-1] % pw:
            raise ValueError('Image size must be divisible by the verified patch size.')
        return F.avg_pool2d(image, kernel_size=(ph, pw), stride=(ph, pw))

    @torch.no_grad()
    def neutral(self, candidate, patch_hw=None):
        patch_hw = tuple(patch_hw or self.patch_hw)
        if self.enabled and self.metric == 'sampson':
            self._anchor_intrinsics(candidate)
        valid = torch.isfinite(candidate.pointmap).all(dim=-1)
        if candidate.valid_mask is not None:
            valid = valid & candidate.valid_mask.bool()
        weights = self._patch_reduce(valid.float().unsqueeze(1), patch_hw)
        weights = weights.flatten(2).transpose(1, 2).contiguous()
        if weights.shape[1] != candidate.key.shape[1]:
            raise ValueError('Uncertainty patch grid does not match memory key ordering/size.')
        self.last_diagnostics = dict(uncertainty_metric=self.metric, neutral=True,
                                     neutral_reason='first_write_or_uncertainty_disabled')
        return WriteConfidence(weights=weights, uncertainty_stats=weights.new_zeros((weights.shape[0], 3)),
                               raft_called=False, diagnostics=dict(self.last_diagnostics))

    @torch.no_grad()
    def estimate(self, candidate, accepted_reference=None, patch_hw=None):
        patch_hw = tuple(patch_hw or self.patch_hw)
        if accepted_reference is None or not self.enabled:
            return self.neutral(candidate, patch_hw)
        cur = candidate.pointmap.float()
        ref = accepted_reference.pointmap.float()
        image = candidate.image
        ref_image = accepted_reference.image
        if cur.shape != ref.shape or cur.ndim != 4 or cur.shape[-1] != 3:
            raise ValueError('Current/reference pointmaps must share [B,H,W,3].')
        b, h, w, _ = cur.shape
        if image.shape != (b, 3, h, w) or ref_image.shape != image.shape:
            raise ValueError('RAFT crops must match the pointmap shape [B,3,H,W].')
        if image.device != cur.device or ref.device != cur.device or ref_image.device != cur.device:
            raise ValueError('RAFT images and pointmaps must be on the same device.')
        if self.flow_model is None:
            raise RuntimeError('Enabled uncertainty has no loaded RAFT model.')
        first_param = next(self.flow_model.parameters(), None)
        if first_param is not None and first_param.device != cur.device:
            raise ValueError('Move the registered UncertaintyEstimator to the model device before use.')
        self.flow_model.eval()
        # Only FP32, independent of outer mixed precision. RAFT pads to /8.
        with torch.autocast(device_type=cur.device.type, enabled=False):
            rgb, ref_rgb = image.float(), ref_image.float()
            if not torch.isfinite(rgb).all() or not torch.isfinite(ref_rgb).all():
                raise ValueError('RAFT images must be finite RGB values in [0,255].')
            if rgb.min() < 0 or rgb.max() > 255 or ref_rgb.min() < 0 or ref_rgb.max() > 255:
                raise ValueError('RAFT requires RGB in [0,255], not encoder-normalized images.')
            pad_h, pad_w = (-h) % 8, (-w) % 8
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode='replicate')
            ref_rgb = F.pad(ref_rgb, (0, pad_w, 0, pad_h), mode='replicate')
            result = self.flow_model(rgb, ref_rgb, iters=self.config.raft_iters, test_mode=True)
            flow = result[-1] if isinstance(result, (tuple, list)) else result
            flow = flow[..., :h, :w].float()
            if flow.shape != (b, 2, h, w):
                raise ValueError('RAFT must return current-to-reference flow [B,2,H,W].')

            flow_finite = torch.isfinite(flow).all(dim=1)
            safe_flow = torch.where(torch.isfinite(flow), flow, torch.zeros_like(flow))
            yy, xx = torch.meshgrid(torch.arange(h, device=cur.device),
                                    torch.arange(w, device=cur.device), indexing='ij')
            x = xx.float().unsqueeze(0) + safe_flow[:, 0]
            y = yy.float().unsqueeze(0) + safe_flow[:, 1]
            in_bounds = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
            grid = torch.stack((2 * x / max(w - 1, 1) - 1,
                                2 * y / max(h - 1, 1) - 1), dim=-1)
            ref_valid = torch.isfinite(ref).all(dim=-1)
            if accepted_reference.valid_mask is not None:
                ref_valid &= accepted_reference.valid_mask.bool()
            cur_valid = torch.isfinite(cur).all(dim=-1)
            if candidate.valid_mask is not None:
                cur_valid &= candidate.valid_mask.bool()
            sampled_valid = F.grid_sample(ref_valid.float().unsqueeze(1), grid,
                                          mode='bilinear', padding_mode='zeros', align_corners=True)[:, 0]
            valid = flow_finite & in_bounds & cur_valid & (sampled_valid >= 1 - 1e-6)
            if self.metric == 'sampson':
                current_xy = torch.stack((xx, yy), -1).float().unsqueeze(0).expand(b, h, w, 2)
                reference_xy = torch.stack((x, y), -1)
                uncertainty, valid = self._sampson_uncertainty(
                    candidate, accepted_reference, current_xy, reference_xy, valid)
            else:
                safe_ref = torch.where(ref_valid.unsqueeze(-1), ref, torch.zeros_like(ref))
                sampled = F.grid_sample(safe_ref.permute(0, 3, 1, 2), grid,
                                        mode='bilinear', padding_mode='zeros', align_corners=True)
                safe_cur = torch.where(cur_valid.unsqueeze(-1), cur, torch.zeros_like(cur))
                distance = torch.linalg.vector_norm(safe_cur - sampled.permute(0, 2, 3, 1), dim=-1)
                radius = torch.linalg.vector_norm(safe_ref, dim=-1)
                scales = []
                for batch in range(b):
                    values = radius[batch][ref_valid[batch]]
                    scales.append(values.median().clamp_min(1e-6) if values.numel()
                                  else radius.new_tensor(1e-6))
                scale = torch.stack(scales)[:, None, None]
                uncertainty = (distance / (self.config.distance_scale_fraction * scale + 1e-6)).clamp(0, 1)
                uncertainty = torch.where(valid, uncertainty, torch.ones_like(uncertainty))
                self.last_diagnostics = dict(uncertainty_metric='geometry_3d',
                                             distance_scale_fraction=self.config.distance_scale_fraction)
            valid_fraction = self._patch_reduce(valid.float().unsqueeze(1), patch_hw)
            summed_u = self._patch_reduce((uncertainty * valid).unsqueeze(1), patch_hw)
            patch_u = torch.where(valid_fraction > 0, summed_u / valid_fraction.clamp_min(1e-6),
                                  torch.ones_like(valid_fraction))
            weights = valid_fraction * torch.exp(-self.config.confidence_alpha * patch_u).clamp(
                self.config.confidence_floor_on_valid, 1)
            weights = weights.flatten(2).transpose(1, 2).contiguous()
            if weights.shape[1] != candidate.key.shape[1]:
                raise ValueError('Pooled uncertainty and encoder token layout differ.')
            stats = []
            for batch in range(b):
                values = uncertainty[batch][valid[batch]]
                mean = values.mean() if values.numel() else uncertainty.new_tensor(1.)
                std = values.std(unbiased=False) if values.numel() else uncertainty.new_tensor(0.)
                stats.append(torch.stack((mean, std, valid[batch].float().mean())))
            if not torch.isfinite(weights).all():
                raise RuntimeError('Sanitized uncertainty produced nonfinite write confidence.')
            return WriteConfidence(weights=weights, uncertainty_stats=torch.stack(stats), raft_called=True,
                                   diagnostics=dict(self.last_diagnostics))
