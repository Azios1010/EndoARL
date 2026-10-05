"""Strict, local-only manifest dataset with real calibrated GT provenance.

Manifest v1 has ``sequences``. Every sequence declares id, patient_id, split,
provenance (verified=true, source, depth_or_pointmap_gt=true,
calibration_gt=true, pose_gt=true, units), and ordered frames. Each frame has
rgb, valid_mask, intrinsics (3x3), camera_to_world (4x4), and either depth plus
depth_scale, or pointmap plus pointmap_space (camera/world/first_camera).
Paths are relative to the manifest directory unless absolute. There are no
identity-pose, unit-depth, or calibration fallbacks.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Optional

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


def _matrix(value, shape, name):
    matrix = torch.as_tensor(value, dtype=torch.float32)
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
            frames = sequence["frames"][::stride]
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
            for key in ("rgb", "valid_mask"):
                if not frame.get(key) or not self._path(frame[key]).is_file():
                    raise FileNotFoundError(f"Required local {key} file missing: {frame.get(key)}")
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

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _array(path):
        import numpy as np
        if path.suffix.lower() == ".npy":
            return torch.from_numpy(np.load(path, allow_pickle=False).copy())
        if path.suffix.lower() == ".npz":
            raise ValueError("Use explicit .npy arrays instead of ambiguous npz target keys")
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

    def __getitem__(self, index):
        sequence, entries, start = self.samples[index]
        first_pose = _matrix(entries[0]["camera_to_world"], (4, 4), "camera_to_world")
        original_first_pose = _matrix(sequence["frames"][0]["camera_to_world"], (4, 4), "camera_to_world")
        frames, targets = [], []
        for frame_index, entry in enumerate(entries):
            rgb = self._array(self._path(entry["rgb"]))
            if rgb.ndim != 3 or rgb.shape[-1] < 3:
                raise ValueError("RGB files must decode to HxWx3 images")
            if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
                raise ValueError("RGB arrays must contain finite values in [0,255]")
            rgb = rgb[..., :3].byte()
            raw_mask = self._array(self._path(entry["valid_mask"]))
            mask = torch.isfinite(raw_mask) & (raw_mask > 0)
            if mask.ndim == 3:
                if mask.shape[-1] != 1:
                    raise ValueError("GT validity mask must have one channel")
                mask = mask.squeeze(-1)
            key = "depth" if entry.get("depth") else "pointmap"
            values = self._array(self._path(entry[key])).float()
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
                depth = None
            mask &= torch.isfinite(points).all(dim=-1)
            if not mask.any():
                raise ValueError(f"Missing real GT coverage: {sequence['id']} frame {frame_index}")
            height, width = rgb.shape[:2]
            image_origin = rgb.permute(2, 0, 1)[None]
            frames.append({"img": image_origin.float() / 127.5 - 1., "origin_img": rgb[None],
                           "true_shape": torch.tensor([[height, width]]), "idx": frame_index,
                           "instance": f"{sequence['id']}:{start + frame_index}"})
            targets.append({"pointmap": points[None], "valid_mask": mask[None],
                            "depth": None if depth is None else depth[None],
                            "intrinsics": intrinsics[None], "camera_to_world": pose[None],
                            "provenance_verified": True})
        return SequenceSample(f"{sequence['id']}:{start}", sequence["patient_id"], sequence["split"],
                              tuple(frames), tuple(targets), dict(sequence["provenance"]))

    def validate_all(self):
        """Read targets locally; no inference and no invented targets."""
        frame_count = 0
        for index in range(len(self)):
            frame_count += len(self[index].frames)
        return {"samples": len(self), "loaded_frames": frame_count, "manifest": str(self.manifest_path)}
