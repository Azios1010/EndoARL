"""Deployment camera prediction from Endo3R pointmaps, never GT cameras.

Source convention: demo.py::main (focal Weiszfeld, PnP100/error2) and
dust3r/model.py::Endo3R.pred_intrinsic/pred_poses. Unlike legacy helpers, this
module handles failed PnP explicitly and is CPU/device independent.
"""
from dataclasses import dataclass

import numpy as np


class CameraPredictionError(ValueError):
    """A failed prediction counts as an evaluation failure, not a hidden mask."""


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pointmap_numpy(output):
    if isinstance(output, dict):
        names = [name for name in ("pts3d", "pts3d_in_other_view") if name in output]
        if len(names) != 1:
            raise CameraPredictionError("Prediction must provide exactly one Endo3R pointmap key.")
        output = output[names[0]]
    points = as_numpy(output).astype(np.float64, copy=False)
    if points.ndim == 4 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 3 or points.shape[-1] != 3:
        raise CameraPredictionError("Evaluation requires a B=1 [H,W,3] pointmap.")
    if not np.isfinite(points).all():
        raise CameraPredictionError("Nonfinite predicted pointmap; invalid predictions are not masked away.")
    return points


def estimate_intrinsics(first_pointmap):
    """CPU float64 Weiszfeld10 counterpart of released focal estimator.

    Principal point=(W/2,H/2), fx=fy, no known camera calibration. Float64
    improves reference robustness; released code computes torch float32.
    """
    points = pointmap_numpy(first_pointmap)
    height, width, _ = points.shape
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack((xx - width / 2, yy - height / 2), -1).reshape(-1, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rays = points[..., :2] / points[..., 2:3]
    rays = np.nan_to_num(rays, nan=0., posinf=0., neginf=0.).reshape(-1, 2)
    dot_xy_px = np.sum(rays * pixels, axis=-1)
    dot_xy_xy = np.sum(rays * rays, axis=-1)
    denominator = float(dot_xy_xy.mean())
    if denominator <= 0:
        raise CameraPredictionError("Degenerate first pointmap cannot determine positive focal length.")
    focal = float(dot_xy_px.mean() / denominator)
    for _ in range(10):
        distances = np.linalg.norm(pixels - focal * rays, axis=-1)
        weights = 1. / np.maximum(distances, 1e-8)
        denominator = float((weights * dot_xy_xy).mean())
        if denominator <= 0:
            raise CameraPredictionError("Degenerate focal fit.")
        focal = float((weights * dot_xy_px).mean() / denominator)
    if not np.isfinite(focal) or focal <= 0:
        raise CameraPredictionError("Predicted focal must be finite and positive; no GT focal fallback.")
    intrinsic = np.array([[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]])
    return intrinsic


@dataclass(frozen=True)
class PredictedCamera:
    intrinsic: np.ndarray
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    inlier_count: int


def predict_camera(pointmap, intrinsic=None, *, pnp_seed=0, cv2_module=None):
    """Estimate camera from predicted common-frame points and their pixel grid.

    No target/calibration/pose argument exists, preventing deployment leakage.
    Dense correspondences match demo.py; deterministic OpenCV RNG is seeded
    per frame. Solver flags are ITERATIVE, iterations=100, reprojectionError=2.
    SERVER_VALIDATION_REQUIRED: installed OpenCV version and real PnP success.
    """
    points = pointmap_numpy(pointmap)
    intrinsic = estimate_intrinsics(points) if intrinsic is None else as_numpy(intrinsic).astype(np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0:
        raise CameraPredictionError("Only finite positive predicted intrinsics are accepted.")
    height, width, _ = points.shape
    if height * width < 6:
        raise CameraPredictionError("Dense PnP needs at least six points.")
    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise ImportError("Predicted-camera benchmark requires local OpenCV (cv2). No package is installed automatically.") from exc
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack((xx, yy), -1).reshape(-1, 2).astype(np.float32)
    if hasattr(cv2_module, "setRNGSeed"):
        cv2_module.setRNGSeed(int(pnp_seed) % (2 ** 31 - 1))
    try:
        success, rotation, translation, inliers = cv2_module.solvePnPRansac(
            points.reshape(-1, 3).astype(np.float32), pixels,
            intrinsic.astype(np.float32), np.zeros(4, dtype=np.float32),
            iterationsCount=100, reprojectionError=2., flags=cv2_module.SOLVEPNP_ITERATIVE)
        if not success or rotation is None or translation is None:
            raise CameraPredictionError("Predicted-camera PnP failed; no GT or identity fallback.")
        rotation_matrix = cv2_module.Rodrigues(rotation)[0]
    except CameraPredictionError:
        raise
    except Exception as exc:
        raise CameraPredictionError(f"PnP raised {type(exc).__name__}: {exc}") from exc
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = np.asarray(translation).reshape(3)
    if not np.isfinite(transform).all() or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4):
        raise CameraPredictionError("PnP returned a nonfinite/improper camera transform.")
    return PredictedCamera(intrinsic.copy(), transform, np.linalg.inv(transform),
                           0 if inliers is None else int(np.asarray(inliers).size))


def predicted_depth(pointmap, camera, *, first_frame=False):
    """Legacy demo retains first pointmap Z and transforms subsequent maps."""
    points = pointmap_numpy(pointmap)
    if first_frame:
        return points[..., 2].copy()
    transform = camera.world_to_camera
    return (points @ transform[:3, :3].T + transform[:3, 3])[..., 2]


@dataclass(frozen=True)
class CameraDepthPrediction:
    valid: bool
    depth: object = None
    camera: object = None
    error: object = None


class PredictedCameraEstimator:
    """Prediction-only sequential camera/depth API shared by GPU benchmarking.

    Holds only a 3x3 predicted intrinsic and a counter, never a pointmap/depth
    history. A missing OpenCV dependency propagates clearly. Failed geometry
    returns valid=False; first-camera failure cannot use a future/GT fallback.
    """
    def __init__(self, *, camera_predictor=predict_camera, pnp_seed=0):
        self.camera_predictor = camera_predictor
        self.pnp_seed = int(pnp_seed)
        self.reset()

    def reset(self):
        self.intrinsic = None
        self.index = 0
        self.first_failed = False

    def predict(self, output):
        index = self.index
        self.index += 1
        try:
            points = pointmap_numpy(output)
            if index == 0:
                self.intrinsic = estimate_intrinsics(points)
            elif self.first_failed or self.intrinsic is None:
                raise CameraPredictionError("First predicted camera failed; no GT/future fallback.")
            camera = self.camera_predictor(points, self.intrinsic, pnp_seed=self.pnp_seed + index)
            depth = predicted_depth(points, camera, first_frame=index == 0)
            if not np.isfinite(depth).all():
                raise CameraPredictionError("Nonfinite predicted local depth.")
            return CameraDepthPrediction(True, depth, camera)
        except (CameraPredictionError, np.linalg.LinAlgError) as exc:
            if index == 0:
                self.first_failed = True
            return CameraDepthPrediction(False, error=str(exc))
