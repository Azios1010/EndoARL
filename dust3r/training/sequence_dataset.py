"""Strict, local-only manifest dataset with real calibrated GT provenance.

Manifest v1 has ``sequences``. Every sequence declares id, patient_id, split,
provenance (verified=true, source, depth_or_pointmap_gt=true,
calibration_gt=true, pose_gt=true, units), and ordered frames. Each frame has
rgb, valid_mask (or an explicitly verified target-validity rule), intrinsics
(3x3), camera_to_world (4x4), and either depth plus
depth_scale, or pointmap plus pointmap_space (camera/world/first_camera).
Paths are relative to the manifest directory unless absolute. There are no
identity-pose, unit-depth, or calibration fallbacks.
"""
from dataclasses import dataclass
import copy
import json
from pathlib import Path
from typing import Mapping, Optional
from collections.abc import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class SequenceSample:
    episode_id: str
    patient_id: str
    split: str
    frames: tuple
    targets: tuple
    provenance: dict


class _StreamingView(Sequence):
    """Indexable frame/target view; slicing explicitly materializes that slice."""
    def __init__(self, sample, item):
        self.sample, self.item = sample, item

    def __len__(self):
        return self.sample.frame_count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        return self.sample.get_frame(index)[self.item]


class StreamingSequenceSample:
    """CPU-resident metadata plus one decoded frame cache, independent of video length.

    Call ``get_frame(i)`` for an aligned ``(frame, target)`` pair. The coordinate
    origin is the first selected frame (including clips and stride selection).
    Never move the complete views to CUDA; move each returned pair instead.
    """
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.sequence, self.entries, self.start = dataset.samples[index]
        self.episode_id = f"{self.sequence['id']}:{self.start}"
        self.patient_id = self.sequence["patient_id"]
        self.split = self.sequence["split"]
        self.provenance = dict(self.sequence["provenance"])
        self.frame_count = len(self.entries)
        self.first_pose = _matrix(self.entries[0]["camera_to_world"], (4, 4), "camera_to_world")
        self.original_first_pose = _matrix(self.sequence["frames"][0]["camera_to_world"], (4, 4), "camera_to_world")
        self._cached_index, self._cached_pair = None, None
        self.frames, self.targets = _StreamingView(self, 0), _StreamingView(self, 1)

    def get_frame(self, index):
        if index < 0:
            index += self.frame_count
        if not 0 <= index < self.frame_count:
            raise IndexError(index)
        if self._cached_index != index:
            self._cached_pair = self.dataset._load_frame(self.sequence, self.entries[index],
                                                       self.start, index, self.first_pose,
                                                       self.original_first_pose)
            self._cached_index = index
        return self._cached_pair

    def iter_frames(self):
        for index in range(self.frame_count):
            yield self.get_frame(index)

    def clear_cache(self):
        self._cached_index, self._cached_pair = None, None

    def source_frame_id(self, index):
        if not 0 <= index < self.frame_count:
            raise IndexError(index)
        return self.entries[index].get("frame_id", self.start + index)

    def prefix(self, frame_count):
        """Independent lazy prefix with unchanged first-camera coordinate origin."""
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or not 2 <= frame_count <= self.frame_count:
            raise ValueError("A benchmark prefix must contain 2..frame_count frames")
        sample = copy.copy(self)
        sample.entries = self.entries[:frame_count]
        sample.frame_count = frame_count
        sample.clear_cache()
        sample.frames, sample.targets = _StreamingView(sample, 0), _StreamingView(sample, 1)
        return sample


def _matrix(value, shape, name):
    try:
        matrix = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be an explicitly specified finite {shape} matrix") from exc
    if tuple(matrix.shape) != shape or not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {shape} matrix")
    if name == "intrinsics":
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or matrix.det().abs() < 1e-8:
            raise ValueError("Intrinsics must have positive focal lengths and be invertible")
    else:
        if not torch.allclose(matrix[3], torch.tensor([0., 0., 0., 1.]), atol=1e-5):
            raise ValueError("camera_to_world must be homogeneous C2W, not an unspecified pose")
        rotation = matrix[:3, :3]
        if not torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-3) or rotation.det() <= 0:
            raise ValueError("camera_to_world rotation is not a proper rotation")
    return matrix


def backproject_common_frame(depth: Tensor, intrinsics: Tensor, camera_to_world: Tensor,
                             first_camera_to_world: Tensor) -> Tensor:
    """Backproject metric depth and apply T_first^-1 T_frame, no DUSt3R dependency."""
    if depth.ndim != 2:
        raise ValueError("Depth must have [H,W] shape")
    height, width = depth.shape
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                            torch.arange(width, dtype=torch.float32), indexing="ij")
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    points = (pixels @ torch.linalg.inv(intrinsics).T) * depth[..., None]
    transform = torch.linalg.solve(first_camera_to_world, camera_to_world)
    return points @ transform[:3, :3].T + transform[:3, 3]


class SequenceDataset(Dataset):
    def __init__(self, manifest, split: Optional[str] = None, image_hw=None,
                 sequence_raw_frames: Optional[int] = None, stride: int = 1):
        self.manifest_path = Path(manifest).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Verified GT sequence manifest is required: {self.manifest_path}")
        with self.manifest_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != 1:
            raise ValueError("Sequence manifest must declare schema_version=1")
        self.root = self.manifest_path.parent
        self.image_hw = tuple(image_hw) if image_hw is not None else None
        if self.image_hw is not None and (len(self.image_hw) != 2 or min(self.image_hw) <= 0):
            raise ValueError("image_hw must be positive (H,W)")
        if stride < 1 or sequence_raw_frames is not None and sequence_raw_frames < 2:
            raise ValueError("stride>=1 and sequence_raw_frames>=2 are required")
        sequences = payload.get("sequences", [])
        if not sequences:
            raise ValueError("Manifest contains no sequences")
        self.samples = []
        patient_splits = {}
        sequence_ids = set()
        for sequence in sequences:
            self._validate_sequence(sequence)
            if sequence["id"] in sequence_ids:
                raise ValueError("Sequence ids must be unique")
            sequence_ids.add(sequence["id"])
            patient = sequence["patient_id"]
            if patient in patient_splits and patient_splits[patient] != sequence["split"]:
                raise ValueError(f"Patient {patient} leaks across train/validation/test splits")
            patient_splits[patient] = sequence["split"]
            if split is not None and sequence["split"] != split:
                continue
            frames = [{**entry, "frame_id": entry.get("frame_id", source_index)}
                      for source_index, entry in enumerate(sequence["frames"])
                      if source_index % stride == 0]
            if sequence_raw_frames is None:
                self.samples.append((sequence, frames, 0))
            else:
                for start in range(0, len(frames) - sequence_raw_frames + 1):
                    self.samples.append((sequence, frames[start:start + sequence_raw_frames], start))
        if not self.samples:
            raise ValueError(f"No eligible sequences for split={split!r} and requested clip length")

    def _path(self, path):
        resolved = Path(path)
        return resolved if resolved.is_absolute() else self.root / resolved

    def _validate_sequence(self, sequence):
        for key in ("id", "patient_id", "split"):
            if not isinstance(sequence.get(key), str) or not sequence[key]:
                raise ValueError(f"Every sequence requires nonempty {key}")
        provenance = sequence.get("provenance", {})
        for key in ("verified", "depth_or_pointmap_gt", "calibration_gt", "pose_gt"):
            if provenance.get(key) is not True:
                raise ValueError(f"Real GT provenance requires {key}=true; placeholders are forbidden")
        if not provenance.get("source") or provenance.get("units") not in {"m", "metres", "meters"}:
            raise ValueError("GT provenance requires a documented source and output units in metres (m/metres/meters)")
        frames = sequence.get("frames", [])
        if len(frames) < 2:
            raise ValueError("A sequence must contain at least two ordered frames")
        for frame in frames:
            for key in ("rgb",):
                if not frame.get(key) or not self._path(frame[key]).is_file():
                    raise FileNotFoundError(f"Required local {key} file missing: {frame.get(key)}")
            if frame.get("valid_mask"):
                if not self._path(frame["valid_mask"]).is_file():
                    raise FileNotFoundError(f"Required local valid_mask file missing: {frame['valid_mask']}")
            elif frame.get("valid_mask_rule") not in {"depth_positive_finite", "camera_xyz_positive_z_finite"}:
                raise ValueError("An explicit valid_mask file or verified target validity rule is required")
            _matrix(frame.get("intrinsics"), (3, 3), "intrinsics")
            _matrix(frame.get("camera_to_world"), (4, 4), "camera_to_world")
            if bool(frame.get("depth")) == bool(frame.get("pointmap")):
                raise ValueError("Every frame must provide exactly one real depth or pointmap target")
            target_key = "depth" if frame.get("depth") else "pointmap"
            if not self._path(frame[target_key]).is_file():
                raise FileNotFoundError(f"Required GT {target_key} file missing: {frame[target_key]}")
            if target_key == "depth" and ("depth_scale" not in frame or not 0 < float(frame["depth_scale"]) < float("inf")):
                raise ValueError("depth_scale must explicitly convert stored GT depth to metric units")
            if target_key == "pointmap" and frame.get("pointmap_space") not in {"camera", "world", "first_camera"}:
                raise ValueError("GT pointmap requires explicit coordinate space")
            if frame.get("valid_mask_rule") == "camera_xyz_positive_z_finite" and (target_key != "pointmap" or frame.get("pointmap_space") != "camera"):
                raise ValueError("Camera XYZ validity requires camera-space pointmap GT")
            if frame.get("valid_mask_rule") == "depth_positive_finite" and target_key != "depth":
                raise ValueError("Depth validity requires depth GT")
            if not 0 < float(frame.get("pointmap_scale", 1.)) < float("inf"):
                raise ValueError("pointmap_scale must be finite and positive")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _array(path):
        import numpy as np
        if path.suffix.lower() == ".npy":
            return torch.from_numpy(np.load(path, allow_pickle=False).copy())
        if path.suffix.lower() == ".npz":
            raise ValueError("Use explicit .npy arrays instead of ambiguous npz target keys")
        if path.suffix.lower() in {".tif", ".tiff"}:
            try:
                import tifffile
            except ImportError as exc:
                raise ImportError("SCARED TIFF GT requires tifffile on the server; no package is installed automatically") from exc
            return torch.from_numpy(np.asarray(tifffile.imread(path)).copy())
        from PIL import Image
        with Image.open(path) as image:
            return torch.from_numpy(np.asarray(image).copy())

    def _preprocess(self, rgb, depth_or_points, mask, intrinsics):
        """Joint center crop then resize; GT nearest, RGB bilinear, pixel-center K."""
        if self.image_hw is None or tuple(rgb.shape[:2]) == self.image_hw:
            return rgb, depth_or_points, mask, intrinsics
        out_h, out_w = self.image_hw
        in_h, in_w = rgb.shape[:2]
        ratio = out_w / out_h
        crop_w, crop_h = (round(in_h * ratio), in_h) if in_w / in_h > ratio else (in_w, round(in_w / ratio))
        left, top = (in_w - crop_w) // 2, (in_h - crop_h) // 2
        rgb = rgb[top:top + crop_h, left:left + crop_w]
        depth_or_points = depth_or_points[top:top + crop_h, left:left + crop_w]
        mask = mask[top:top + crop_h, left:left + crop_w]
        rgb = F.interpolate(rgb.permute(2, 0, 1)[None].float(), self.image_hw,
                            mode="bilinear", align_corners=False)[0].permute(1, 2, 0).round().clamp(0, 255).byte()
        tensor = depth_or_points.float()
        tensor = tensor[None, None] if tensor.ndim == 2 else tensor.permute(2, 0, 1)[None]
        tensor = F.interpolate(tensor, self.image_hw, mode="nearest")[0]
        depth_or_points = tensor[0] if depth_or_points.ndim == 2 else tensor.permute(1, 2, 0)
        mask = F.interpolate(mask[None, None].float(), self.image_hw, mode="nearest")[0, 0].bool()
        intrinsics = intrinsics.clone()
        intrinsics[0, 2] -= left
        intrinsics[1, 2] -= top
        sx, sy = out_w / crop_w, out_h / crop_h
        intrinsics[0] *= sx
        intrinsics[1] *= sy
        intrinsics[0, 2] += 0.5 * sx - 0.5
        intrinsics[1, 2] += 0.5 * sy - 0.5
        return rgb, depth_or_points, mask, intrinsics

    def _load_frame(self, sequence, entry, start, frame_index, first_pose, original_first_pose):
        rgb = self._array(self._path(entry["rgb"]))
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            raise ValueError("RGB files must decode to HxWx3 images")
        if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
            raise ValueError("RGB arrays must contain finite values in [0,255]")
        rgb = rgb[..., :3].byte()
        key = "depth" if entry.get("depth") else "pointmap"
        values = self._array(self._path(entry[key])).float()
        if entry.get("valid_mask"):
            raw_mask = self._array(self._path(entry["valid_mask"]))
            mask = torch.isfinite(raw_mask) & (raw_mask > 0)
            if mask.ndim == 3:
                if mask.shape[-1] != 1:
                    raise ValueError("GT validity mask must have one channel")
                mask = mask.squeeze(-1)
        elif entry["valid_mask_rule"] == "depth_positive_finite":
            mask = torch.isfinite(values) & (values > 0)
        else:
            if values.ndim != 3 or values.shape[-1] != 3:
                raise ValueError("Camera XYZ validity requires HxWx3 GT")
            mask = torch.isfinite(values).all(dim=-1) & (values[..., 2] > 0)
        intrinsics = _matrix(entry["intrinsics"], (3, 3), "intrinsics")
        pose = _matrix(entry["camera_to_world"], (4, 4), "camera_to_world")
        if values.shape[:2] != rgb.shape[:2] or mask.shape != rgb.shape[:2]:
            raise ValueError("RGB, calibrated GT, and validity mask must be pixel aligned")
        if key == "depth":
            if values.ndim != 2:
                raise ValueError("Depth GT must have HxW shape")
            values = values * float(entry["depth_scale"])
            mask &= torch.isfinite(values) & (values > 0)
        elif values.ndim != 3 or values.shape[-1] != 3:
            raise ValueError("Dense pointmap GT must have HxWx3 shape")
        else:
            values = values * float(entry.get("pointmap_scale", 1.))
        rgb, values, mask, intrinsics = self._preprocess(rgb, values, mask, intrinsics)
        if key == "depth":
            safe_depth = torch.where(mask, values, torch.zeros_like(values))
            points = backproject_common_frame(safe_depth, intrinsics, pose, first_pose)
            depth = values
        else:
            if entry["pointmap_space"] == "camera":
                transform = torch.linalg.solve(first_pose, pose)
            elif entry["pointmap_space"] == "world":
                transform = torch.linalg.inv(first_pose)
            else:
                transform = torch.linalg.solve(first_pose, original_first_pose)
            points = values @ transform[:3, :3].T + transform[:3, 3]
            # XYZ GT in a world/first-camera frame still defines metric
            # depth in the current optical camera, using calibrated pose.
            camera_transform = torch.linalg.solve(pose, first_pose)
            camera_points = points @ camera_transform[:3, :3].T + camera_transform[:3, 3]
            depth = camera_points[..., 2]
        mask &= torch.isfinite(points).all(dim=-1)
        mask &= torch.isfinite(depth) & (depth > 0)
        if not mask.any():
            raise ValueError(f"Missing real GT coverage: {sequence['id']} frame {frame_index}")
        height, width = rgb.shape[:2]
        image_origin = rgb.permute(2, 0, 1)[None]
        frame = {"img": image_origin.float() / 127.5 - 1., "origin_img": rgb[None],
                       "true_shape": torch.tensor([[height, width]]), "idx": frame_index,
                       "instance": f"{sequence['id']}:{entry.get('frame_id', start + frame_index)}"}
        target = {"pointmap": points[None], "valid_mask": mask[None],
                        "depth": None if depth is None else depth[None],
                        "intrinsics": intrinsics[None], "camera_to_world": pose[None],
                        "first_camera_to_world": first_pose[None],
                        "source_frame_id": entry.get("frame_id", start + frame_index),
                        "provenance_verified": True}
        return frame, target

    def stream(self, index):
        """Return lazy sequence metadata/views; allocates no RGB/GT tensors yet."""
        return StreamingSequenceSample(self, index)

    def __getitem__(self, index):
        stream = self.stream(index)
        pairs = tuple(stream.iter_frames())
        return SequenceSample(stream.episode_id, stream.patient_id, stream.split,
                              tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs),
                              stream.provenance)

    def validate_all(self):
        """Read targets locally; no inference and no invented targets."""
        frame_count = 0
        for index in range(len(self)):
            for _ in self.stream(index).iter_frames():
                frame_count += 1
        return {"samples": len(self), "loaded_frames": frame_count, "manifest": str(self.manifest_path)}
