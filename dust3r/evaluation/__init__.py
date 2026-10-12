"""CPU benchmark metrics with explicitly versioned evaluation protocols.

Model and optional OpenCV imports are lazy. Importing this package never loads
a checkpoint, downloads data or initializes a CUDA kernel.
"""
from .camera import CameraPredictionError, PredictedCamera, PredictedCameraEstimator, predict_camera
from .metrics import DEPTH_KEYS, depth_errors, pose_window_metrics
from .streaming import StreamingSequenceEvaluator, aggregate_sequences

__all__ = ["CameraPredictionError", "PredictedCamera", "PredictedCameraEstimator", "predict_camera",
           "DEPTH_KEYS", "depth_errors", "pose_window_metrics",
           "StreamingSequenceEvaluator", "aggregate_sequences"]
