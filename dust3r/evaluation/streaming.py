"""One-frame evaluator: CPU scalar depth vectors, never a sequence of GPU maps.

Whole-sequence median alignment necessarily sees all GT-valid depth samples.
Exact sample vectors are retained on CPU (O(number of valid pixels)), while
only 4x4 poses and per-frame scalar reports persist otherwise. This is an
offline scoring transform, never an actor/memory input.
"""
import hashlib
import json

import numpy as np

from .camera import (CameraPredictionError, as_numpy, estimate_intrinsics,
                     pointmap_numpy, predict_camera, predicted_depth)
from .metrics import DEPTH_KEYS, POSE_KEYS, depth_errors, pose_window_metrics

DEPTH_PROTOCOL = "DEPTH_RELEASED_CODE_PREDICTED_CAMERA_V1"
POSE_PROTOCOL = "POSE_C2W_SIM3_5FRAME_STRIDE1_ADJACENT_RPE_V1"
SCALE_PROTOCOL = "sequence_median_gt_over_median_prediction_gt_valid_pixels"
CAMERA_SOURCE = "first_predicted_pointmap_weiszfeld10_focal_dense_predicted_pointmap_pnp100_error2"


def canonical_sha256(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _single_array(value, ndim):
    value = as_numpy(value)
    if value.ndim == ndim + 1 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != ndim:
        raise ValueError(f"Expected B=1 target with {ndim} unbatched dimensions.")
    return value


class StreamingSequenceEvaluator:
    def __init__(self, episode_id, *, expected_frames=None, provenance=None,
                 camera_predictor=predict_camera, pnp_seed=0, min_depth=1e-4,
                 max_depth=100., pose=True):
        if not 0 < min_depth < max_depth or expected_frames is not None and expected_frames < 2:
            raise ValueError("Positive depth bounds and at least two expected frames are required.")
        self.episode_id = str(episode_id)
        self.expected_frames = expected_frames
        self.provenance = dict(provenance or {})
        self.camera_predictor = camera_predictor
        self.pnp_seed = int(pnp_seed)
        self.min_depth, self.max_depth = float(min_depth), float(max_depth)
        self.pose_enabled = bool(pose)
        self.intrinsic = None
        self.first_gt_pose = None
        self.frames = []
        self.depth_vectors = []
        self.predicted_poses, self.gt_poses = [], []
        self.frame_identities = []
        self._first_prediction_failed = False
        self._finalized = False

    def add_frame(self, index, output, target, *, frame_identity=None):
        """Score one final frame prediction; target never reaches camera fit.

        Call once for each pair's res1, then once for final pair's res2. An
        invalid prediction is reported as failed instead of masking its
        pixels, changing frame counts, or substituting GT camera geometry.
        """
        if self._finalized or index != len(self.frames):
            raise ValueError("Frames must be submitted once in increasing contiguous index order.")
        self.frame_identities.append(str(index) if frame_identity is None else str(frame_identity))
        record = {"frame": int(index), "status": "failed", "failure": None,
                  "gt_valid_pixel_count": 0, "depth_metrics": None}
        self.frames.append(record)
        self.depth_vectors.append(None)
        self.predicted_poses.append(None)
        self.gt_poses.append(None)
        try:
            if target.get("provenance_verified") is not True:
                raise ValueError("Benchmark requires verified real GT provenance.")
            gt_pose = _single_array(target["camera_to_world"], 2).astype(np.float64)
            if gt_pose.shape != (4, 4) or not np.isfinite(gt_pose).all():
                raise ValueError("GT pose must be finite 4x4 C2W for scoring.")
            if index == 0:
                self.first_gt_pose = _single_array(target.get("first_camera_to_world", gt_pose), 2).astype(np.float64).copy()
            self.gt_poses[index] = gt_pose.copy()
            mask = _single_array(target["valid_mask"], 2).astype(bool)
            if target.get("depth") is not None:
                gt_depth = _single_array(target["depth"], 2).astype(np.float64)
            else:
                gt_points = _single_array(target["pointmap"], 3).astype(np.float64)
                if self.first_gt_pose is None:
                    raise ValueError("First GT camera unavailable for common-frame GT scoring.")
                gt_to_camera = np.linalg.solve(gt_pose, self.first_gt_pose)
                gt_depth = (gt_points @ gt_to_camera[:3, :3].T + gt_to_camera[:3, 3])[..., 2]
            if mask.shape != gt_depth.shape:
                raise ValueError("GT mask/depth pixel alignment mismatch.")
            mask &= np.isfinite(gt_depth) & (gt_depth > self.min_depth) & (gt_depth < self.max_depth)
            record["gt_valid_pixel_count"] = int(mask.sum())
            if not mask.any():
                raise ValueError("Frame has zero real GT-valid positive in-range depth pixels.")
            points = pointmap_numpy(output)
            if points.shape[:2] != gt_depth.shape:
                raise CameraPredictionError("Prediction/GT cropped resolution mismatch.")
            if index == 0:
                self.intrinsic = estimate_intrinsics(points)
            elif self._first_prediction_failed or self.intrinsic is None:
                raise CameraPredictionError("First predicted camera was invalid; later frames have no GT-camera fallback.")
            # The only inputs to this deployment step are predictions/seed.
            camera = self.camera_predictor(points, self.intrinsic, pnp_seed=self.pnp_seed + index)
            depth = predicted_depth(points, camera, first_frame=index == 0)
            if not np.isfinite(depth).all():
                raise CameraPredictionError("Predicted local depth is nonfinite.")
            valid_pred = depth[mask].astype(np.float64, copy=True)
            valid_gt = gt_depth[mask].astype(np.float64, copy=True)
            self.depth_vectors[index] = (valid_gt, valid_pred)
            self.predicted_poses[index] = np.array(camera.camera_to_world, dtype=np.float64, copy=True)
            record.update(status="ok", pnp_inlier_count=int(camera.inlier_count),
                          nonpositive_predicted_depth_ratio=float(np.mean(valid_pred <= 0)))
        except (ValueError, FloatingPointError, KeyError, np.linalg.LinAlgError) as exc:
            record["failure"] = {"type": type(exc).__name__, "reason": str(exc)}
            if index == 0:
                self._first_prediction_failed = True
        # Missing OpenCV is a missing environment dependency, so propagates
        # ImportError rather than being disguised as failed model geometry.
        return record

    def finalize(self):
        if self._finalized:
            raise ValueError("Evaluator has already been finalized.")
        if not self.frames or self.expected_frames is not None and len(self.frames) != self.expected_frames:
            raise ValueError("Evaluation must cover exactly all expected frames.")
        self._finalized = True
        arrays = [item for item in self.depth_vectors if item is not None]
        scale = None
        scale_failure = None
        if arrays:
            gt_median = float(np.median(np.concatenate([item[0] for item in arrays])))
            pred_median = float(np.median(np.concatenate([item[1] for item in arrays])))
            if np.isfinite(pred_median) and pred_median > 0:
                scale = gt_median / pred_median
            else:
                scale_failure = "Sequence predicted median must be positive and finite."
        for record, vectors in zip(self.frames, self.depth_vectors):
            if vectors is not None and scale is not None:
                gt, prediction = vectors
                with np.errstate(over="ignore", invalid="ignore"):
                    scaled = prediction * scale
                if not np.isfinite(scaled).all():
                    record.update(status="failed", failure={"type": "ScaleFailure", "reason": "Nonfinite scaled prediction."})
                else:
                    record["depth_metrics"] = depth_errors(gt, np.clip(scaled, self.min_depth, self.max_depth))
            elif vectors is not None:
                record.update(status="failed", failure={"type": "ScaleFailure", "reason": scale_failure})
        scored = [frame for frame in self.frames if frame["depth_metrics"] is not None]
        depth_summary = {key: float(np.mean([frame["depth_metrics"][key] for frame in scored])) if scored else None
                         for key in DEPTH_KEYS}
        windows, window_failures = [], []
        if self.pose_enabled:
            for start in range(max(0, len(self.frames) - 4)):
                pred, gt = self.predicted_poses[start:start + 5], self.gt_poses[start:start + 5]
                if any(value is None for value in pred + gt):
                    window_failures.append({"start": start, "reason": "missing_or_failed_pose"})
                    continue
                try:
                    windows.append({"start": start, **pose_window_metrics(pred, gt)})
                except ValueError as exc:
                    window_failures.append({"start": start, "reason": str(exc)})
        pose_summary = {key: float(np.mean([item[key] for item in windows])) if windows else None for key in POSE_KEYS}
        failures = [{"frame": frame["frame"], **frame["failure"]}
                    for frame in self.frames if frame["failure"] is not None]
        result = {"episode_id": self.episode_id, "protocol": DEPTH_PROTOCOL,
                  "prediction_camera_source": CAMERA_SOURCE, "scale_protocol": SCALE_PROTOCOL,
                  "scale_ratio": scale, "scale_failure": scale_failure,
                  "depth_clip_m": [self.min_depth, self.max_depth],
                  "depth_aggregation": "per_frame_mean_then_frame_weighted_across_sequences",
                  "pose_protocol": POSE_PROTOCOL if self.pose_enabled else None,
                  "pose_alignment": "independent_positive_Sim3_per_5_frame_window_first_rotation_when_degenerate",
                  "pose_window_stride": 1, "pose_units": {"translation": "mm", "rotation": "degrees"},
                  "frame_count": len(self.frames), "scored_frame_count": len(scored),
                  "failed_frame_count": len(self.frames) - len(scored), "failures": failures,
                  "status": "complete" if len(scored) == len(self.frames) else "incomplete",
                  "frame_indices_sha256": canonical_sha256(self.frame_identities),
                  "provenance": self.provenance, "provenance_sha256": canonical_sha256(self.provenance),
                  "depth_metrics": depth_summary, "pose_metrics": pose_summary,
                  "pose_window_count": len(windows), "pose_failed_window_count": len(window_failures),
                  "pose_windows": windows, "pose_window_failures": window_failures,
                  "frame_metrics": self.frames}
        self.depth_vectors.clear()
        return result


def aggregate_sequences(sequences):
    """Frame-weight depth; window-weight pose; publish denominators/failures."""
    if not sequences:
        raise ValueError("A benchmark needs at least one sequence.")
    protocols = {(item["protocol"], item["scale_protocol"], item["prediction_camera_source"],
                  item["pose_protocol"], tuple(item["depth_clip_m"])) for item in sequences}
    if len(protocols) != 1:
        raise ValueError("Cannot aggregate results from different evaluation protocols.")
    valid_frames = sum(item["scored_frame_count"] for item in sequences)
    pose_windows = sum(item["pose_window_count"] for item in sequences)
    def weighted(keys, category, weight):
        values = {}
        for key in keys:
            eligible = [(item[category][key], item[weight]) for item in sequences if item[category][key] is not None]
            denominator = sum(count for _, count in eligible)
            values[key] = sum(value * count for value, count in eligible) / denominator if denominator else None
        return values
    return {"sequence_count": len(sequences), "frame_count": sum(item["frame_count"] for item in sequences),
            "scored_frame_count": valid_frames, "failed_frame_count": sum(item["failed_frame_count"] for item in sequences),
            "pose_window_count": pose_windows,
            "pose_failed_window_count": sum(item["pose_failed_window_count"] for item in sequences),
            "status": "complete" if all(item["status"] == "complete" for item in sequences) else "incomplete",
            "depth_metrics": weighted(DEPTH_KEYS, "depth_metrics", "scored_frame_count"),
            "pose_metrics": weighted(POSE_KEYS, "pose_metrics", "pose_window_count")}
