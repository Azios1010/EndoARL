"""CPU/synthetic tests; no SCARED, RAFT checkpoint or CUDA dependency."""
import sys
import unittest
import importlib.util
from unittest.mock import patch

import torch
from torch import nn

from dust3r.memory.camera_geometry import (
    CameraEstimate, PredictedCameraEstimator, estimate_intrinsics,
    fundamental_from_cameras, sampson_residual_squared,
)
from dust3r.memory.config import UncertaintyConfig
from dust3r.memory.types import MemoryCandidate
from dust3r.memory.uncertainty import UncertaintyEstimator


def predicted_candidate(h=8, w=8):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    # Positive varying depth avoids a degenerate ray grid and is in camera0.
    z = 2.0 + 0.01 * x + 0.02 * y
    points = torch.stack(((x - w / 2.) * z / 100., (y - h / 2.) * z / 100., z), -1)
    return MemoryCandidate(
        torch.ones(1, h * w // 4, 8), torch.ones(1, h * w // 4, 8),
        image=torch.zeros(1, 3, h, w, dtype=torch.uint8), pointmap=points.unsqueeze(0),
        valid_mask=torch.ones(1, h, w, dtype=torch.bool), confidence=torch.ones(1, h, w))


class SyntheticFlow(nn.Module):
    def __init__(self, dx=0., dy=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.dx, self.dy, self.calls = dx, dy, 0

    def forward(self, image, reference, iters, test_mode):
        self.calls += 1
        flow = image.new_zeros((image.shape[0], 2, *image.shape[-2:]))
        flow[:, 0] = self.dx
        flow[:, 1] = self.dy
        return flow


class SyntheticCameras:
    """Camera estimator injection returns known geometry, never GT metadata."""
    def __init__(self, reference_pointmap, translation=1., valid=True, rotation=False):
        self.reference_pointmap = reference_pointmap
        self.translation, self.valid, self.rotation = translation, valid, rotation
        self.calls = 0

    def __call__(self, pointmap, valid_mask=None, confidence=None, intrinsics=None):
        self.calls += 1
        b = pointmap.shape[0]
        transform = torch.eye(4).expand(b, 4, 4).clone()
        if pointmap is not self.reference_pointmap:
            transform[:, 0, 3] = self.translation
            if self.rotation:
                angle = torch.tensor(.3)
                c, s = torch.cos(angle), torch.sin(angle)
                transform[:, :3, :3] = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return CameraEstimate(intrinsics, transform.to(pointmap.device),
                              torch.full((b,), self.valid, dtype=torch.bool, device=pointmap.device),
                              dict(camera_source='synthetic_predicted_geometry'))


class SampsonTests(unittest.TestCase):
    def test_masked_first_camera_focal(self):
        candidate = predicted_candidate()
        candidate.pointmap[:, :2, :2] = float('nan')
        k, valid = estimate_intrinsics(candidate.pointmap, candidate.valid_mask)
        self.assertTrue(valid.all())
        self.assertAlmostEqual(float(k[0, 0, 0]), 100., places=4)
        self.assertEqual(float(k[0, 0, 2]), 4.)
        self.assertEqual(float(k[0, 1, 2]), 4.)

    def test_degenerate_focal_is_invalid(self):
        k, valid = estimate_intrinsics(torch.ones(1, 8, 8, 3))
        self.assertFalse(valid.any())
        self.assertTrue(torch.isfinite(k).all())

    def test_fundamental_convention_and_squared_residual(self):
        k = torch.tensor([[[100., 0, 4], [0, 100., 4], [0, 0, 1]]])
        cur_pose = torch.eye(4).unsqueeze(0)
        cur_pose[:, 0, 3] = 1.
        current = CameraEstimate(k, cur_pose, torch.tensor([True]))
        reference = CameraEstimate(k, torch.eye(4).unsqueeze(0), torch.tensor([True]))
        f, valid, baseline = fundamental_from_cameras(current, reference)
        self.assertTrue(valid.all())
        self.assertEqual(float(baseline[0]), 1.)
        xy = torch.tensor([[[2., 3.], [4., 5.]]])
        shifted = xy + torch.tensor([[[1., 0.], [1., 2.]]])
        squared, valid = sampson_residual_squared(f, xy, shifted)
        self.assertTrue(valid.all())
        self.assertTrue(torch.allclose(squared, torch.tensor([[0., 2.]]), atol=1e-6))
        scaled, scaled_valid = sampson_residual_squared(f * 100, xy, shifted)
        self.assertTrue(torch.equal(valid, scaled_valid))
        self.assertTrue(torch.allclose(squared, scaled, atol=1e-6))

    def test_invalid_coordinates_zero_denominator_finite_not_valid(self):
        xy = torch.tensor([[[float('nan'), 1.], [1., 2.]]])
        squared, valid = sampson_residual_squared(torch.zeros(1, 3, 3), xy, torch.zeros_like(xy))
        self.assertFalse(valid.any())
        self.assertTrue(torch.isfinite(squared).all())

    def make_estimator(self, reference, flow=None, translation=1., valid=True, rotation=False):
        camera = SyntheticCameras(reference.pointmap, translation, valid, rotation)
        flow = flow or SyntheticFlow()
        estimator = UncertaintyEstimator(UncertaintyConfig(metric='sampson'),
            flow_model=flow, camera_estimator=camera, patch_hw=(2, 2))
        return estimator, flow, camera

    def test_first_write_skips_raft_and_camera_pnp(self):
        reference = predicted_candidate()
        estimator, flow, camera = self.make_estimator(reference)
        confidence = estimator.estimate(reference, None)
        self.assertFalse(confidence.raft_called)
        self.assertEqual(flow.calls, 0)
        self.assertEqual(camera.calls, 0)
        self.assertTrue(torch.equal(confidence.weights, torch.ones(1, 16, 1)))
        self.assertTrue(estimator._predicted_intrinsics_valid.all())

    def test_epipolar_correspondence_high_confidence(self):
        reference, current = predicted_candidate(), predicted_candidate()
        estimator, flow, _ = self.make_estimator(reference)
        estimator.estimate(reference, None)
        confidence = estimator.estimate(current, reference)
        self.assertTrue(confidence.raft_called)
        self.assertEqual(flow.calls, 1)
        self.assertTrue(torch.allclose(confidence.weights, torch.ones(1, 16, 1)))
        self.assertEqual(confidence.diagnostics['sampson_residual_units'], 'pixels_squared')
        self.assertEqual(confidence.diagnostics['camera_source'], 'predicted_pointmap_only')
        self.assertEqual(confidence.diagnostics['camera_reason'], ['ok'])
        self.assertFalse(confidence.weights.requires_grad)
        self.assertFalse(flow.training)
        self.assertFalse(flow.weight.requires_grad)

    def test_vertical_flow_has_expected_sampson_write_strength(self):
        reference, current = predicted_candidate(), predicted_candidate()
        estimator, _, _ = self.make_estimator(reference, SyntheticFlow(dy=1.))
        confidence = estimator.estimate(current, reference)
        expected = torch.exp(torch.tensor(-5. * .5))
        self.assertTrue(torch.allclose(confidence.weights[0, :4, 0], expected.expand(4), atol=1e-6))
        self.assertAlmostEqual(confidence.diagnostics['sampson_mean_px2'][0], .5, places=5)
        # The last row flows outside the image: its patch valid fraction is 1/2.
        self.assertTrue(torch.allclose(confidence.weights[0, -4:, 0], expected.expand(4) * .5, atol=1e-6))

    def test_pure_rotation_or_invalid_camera_rejects_write(self):
        reference, current = predicted_candidate(), predicted_candidate()
        for translation, valid, rotation in ((0., True, False), (0., True, True), (1., False, False)):
            estimator, _, _ = self.make_estimator(reference, translation=translation,
                                                 valid=valid, rotation=rotation)
            confidence = estimator.estimate(current, reference)
            self.assertFalse(confidence.has_any_positive_patch)
            self.assertTrue(torch.isfinite(confidence.weights).all())
            self.assertEqual(float(confidence.uncertainty_stats[0, 2]), 0.)
            self.assertNotEqual(confidence.diagnostics['camera_reason'], ['ok'])

    def test_nan_and_out_of_bounds_flow_rejects_write(self):
        reference, current = predicted_candidate(), predicted_candidate()
        for dx, dy in ((float('nan'), 0.), (float('inf'), 0.), (1000., 0.)):
            estimator, _, _ = self.make_estimator(reference, SyntheticFlow(dx=dx, dy=dy))
            confidence = estimator.estimate(current, reference)
            self.assertFalse(confidence.has_any_positive_patch)
            self.assertTrue(torch.isfinite(confidence.weights).all())

    def test_reference_cache_bounded_and_reset(self):
        reference, current = predicted_candidate(), predicted_candidate()
        estimator, _, camera = self.make_estimator(reference)
        estimator.estimate(current, reference)
        estimator.estimate(current, reference)
        self.assertEqual(camera.calls, 3)  # reference once, current twice
        estimator.reset()
        self.assertIsNone(estimator._reference_pointmap)
        self.assertIsNone(estimator._predicted_intrinsics)
        estimator.estimate(current, reference)
        self.assertEqual(camera.calls, 5)

    def test_missing_optional_opencv_has_dependency_error(self):
        candidate = predicted_candidate()
        with patch.dict(sys.modules, {'cv2': None}):
            with self.assertRaisesRegex(ImportError, 'No dependency is downloaded'):
                PredictedCameraEstimator()(candidate.pointmap)

    @unittest.skipUnless(importlib.util.find_spec('cv2') is not None, 'OpenCV is not installed locally')
    def test_existing_cpu_opencv_recovers_synthetic_predicted_pose(self):
        reference = predicted_candidate(h=16, w=16)
        k, valid = estimate_intrinsics(reference.pointmap)
        self.assertTrue(valid.all())
        # Common coordinates are camera0. A camera translated by +1 in w2c
        # observes points with common XYZ = its own ray points minus [1,0,0].
        current = predicted_candidate(h=16, w=16)
        current.pointmap[..., 0] -= 1.
        camera = PredictedCameraEstimator()(current.pointmap, intrinsics=k)
        self.assertTrue(camera.valid.all(), camera.diagnostics)
        self.assertTrue(torch.allclose(camera.world_to_camera[0, :3, :3], torch.eye(3), atol=1e-3))
        self.assertTrue(torch.allclose(camera.world_to_camera[0, :3, 3], torch.tensor([1., 0., 0.]), atol=1e-3))
        self.assertLess(camera.diagnostics['pnp_inlier_rmse_px'][0], 1e-3)


if __name__ == '__main__':
    unittest.main()
