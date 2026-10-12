"""Pure NumPy depth and pose metrics, with no model or CUDA dependency."""
import numpy as np

DEPTH_KEYS = ("abs_rel", "sq_rel", "rmse_m", "rmse_log", "delta1", "delta2", "delta3")
POSE_KEYS = ("ate_mm", "rpe_translation_mm", "rpe_rotation_deg")


def depth_errors(gt, prediction):
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if gt.shape != prediction.shape or not gt.size:
        raise ValueError("Depth metrics require matching nonempty valid pixel arrays.")
    if not np.isfinite(gt).all() or not np.isfinite(prediction).all() or (gt <= 0).any() or (prediction <= 0).any():
        raise ValueError("Depth metrics require finite positive depths; mask/clip explicitly before calling.")
    difference = prediction - gt
    ratio = np.maximum(gt / prediction, prediction / gt)
    values = (np.mean(np.abs(difference) / gt), np.mean(difference ** 2 / gt),
              np.sqrt(np.mean(difference ** 2)), np.sqrt(np.mean((np.log(prediction) - np.log(gt)) ** 2)),
              np.mean(ratio < 1.25), np.mean(ratio < 1.25 ** 2), np.mean(ratio < 1.25 ** 3))
    return dict(zip(DEPTH_KEYS, map(float, values)))


def _pose_array(poses):
    value = np.asarray(poses, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Pose metrics require finite [N,4,4] C2W transforms.")
    if not np.allclose(value[:, 3], [0., 0., 0., 1.], atol=1e-6):
        raise ValueError("Camera poses must be homogeneous C2W transforms.")
    rotations = value[:, :3, :3]
    if not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=1e-4) or (np.linalg.det(rotations) <= 0).any():
        raise ValueError("Camera rotations must be proper orthonormal matrices.")
    return value


def align_pose_window(prediction, gt):
    """Fit positive-scale Sim3 on camera centers, rotate pose orientations.

    Collinear/stationary windows have ambiguous SVD rotation. Use the first
    camera orientation to fix rotation in that case; stationary scale=1.
    This explicit convention is NOT asserted to match underspecified paper
    trajectory alignment; it is logged as a separately named protocol.
    """
    prediction, gt = _pose_array(prediction), _pose_array(gt)
    if prediction.shape != gt.shape or len(gt) < 2:
        raise ValueError("Alignment requires matching pose windows of at least two frames.")
    source, target = prediction[:, :3, 3], gt[:, :3, 3]
    sx, tx = source - source.mean(0), target - target.mean(0)
    variance = float(np.mean(np.sum(sx ** 2, axis=1)))
    covariance = tx.T @ sx / len(source)
    degenerate = np.linalg.matrix_rank(covariance, tol=1e-12) < 2
    if degenerate:
        rotation = gt[0, :3, :3] @ prediction[0, :3, :3].T
        scale = float(np.sum(tx * (sx @ rotation.T)) / np.sum(sx ** 2)) if variance > 1e-15 else 1.
    else:
        left, singular, right = np.linalg.svd(covariance)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(left @ right)
        rotation = left @ correction @ right
        scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive similarity scale cannot be fitted to predicted trajectory.")
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    aligned = prediction.copy()
    aligned[:, :3, :3] = rotation @ prediction[:, :3, :3]
    aligned[:, :3, 3] = scale * (source @ rotation.T) + translation
    return aligned, {"scale": scale, "degenerate_alignment": bool(degenerate)}


def pose_window_metrics(prediction, gt):
    """ATE center RMSE; adjacent-frame mean RPE translation/rotation."""
    aligned, diagnostics = align_pose_window(prediction, gt)
    gt = _pose_array(gt)
    ate = np.sqrt(np.mean(np.sum((aligned[:, :3, 3] - gt[:, :3, 3]) ** 2, axis=-1)))
    gt_relative = np.linalg.inv(gt[:-1]) @ gt[1:]
    pred_relative = np.linalg.inv(aligned[:-1]) @ aligned[1:]
    error = np.linalg.inv(gt_relative) @ pred_relative
    cosine = np.clip((np.trace(error[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)
    return {"ate_mm": float(ate * 1000),
            "rpe_translation_mm": float(np.linalg.norm(error[:, :3, 3], axis=1).mean() * 1000),
            "rpe_rotation_deg": float(np.rad2deg(np.arccos(cosine)).mean()), **diagnostics}
