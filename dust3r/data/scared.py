"""Strict SCARED source-index/mapping conversion, with no acquisition code.

Native SCARED distributions and derived layouts differ. Discovery is deliberately
driven by explicit globs and JSON field mappings; a pose labelled ``camera_pose``
alone is NOT enough to infer C2W/W2C, optical axes, units or robot extrinsics.
The user must verify these declarations against the server's release. Neither
this module nor its CLI downloads data, weights or Python packages.
"""
import json
from pathlib import Path
import re

import numpy as np

from dust3r.training.sequence_dataset import SequenceDataset, _matrix


DEFAULT_SPLITS = {**{i: "train" for i in range(1, 7)}, 7: "validation", 8: "test", 9: "test"}
_UNIT_SCALE = {"m": 1., "metres": 1., "meters": 1., "mm": .001, "millimetres": .001}


def _field(value, path):
    """Traverse declared JSON keys, e.g. calibration.left.K; never guess keys."""
    for key in path.split(".") if path else ():
        if isinstance(value, list):
            value = value[int(key)]
        elif isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise ValueError(f"Declared JSON field is missing: {path}")
    return value


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve(root, value, field=None):
    """Resolve a literal matrix or an explicit local file/JSON-field reference."""
    if isinstance(value, dict) and "file" in value:
        path = Path(value["file"])
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise FileNotFoundError(f"Required calibration/pose file missing: {path}")
        if path.suffix.lower() == ".npy":
            loaded = np.load(path, allow_pickle=False).tolist()
        elif path.suffix.lower() == ".json":
            loaded = _json(path)
        elif path.suffix.lower() in {".txt", ".csv"}:
            loaded = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None).tolist()
        else:
            raise ValueError(f"Unsupported calibration/pose file: {path}")
        return _field(loaded, value.get("field", field))
    return value


def _required_file(root, value, description, parent_cache=None):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Explicit {description} path is required")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if parent_cache is None:
        path = path.resolve()
    else:
        # Source directories are immutable during one preparation call. Resolve
        # their shared ancestors once, while still checking every file and
        # resolving any final-file symlink independently (important on WSL).
        parent = path.parent
        if parent not in parent_cache:
            parent_cache[parent] = parent.resolve()
        path = parent_cache[parent] / path.name
        if path.is_symlink():
            path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required local {description} file missing: {path}")
    return path


def _checked_pose(root, value, specification):
    if specification.get("verified") is not True:
        raise ValueError("pose.verified=true is required after checking camera/robot extrinsics and transform direction")
    if specification.get("camera_axes") != "opencv":
        raise ValueError("pose.camera_axes=opencv (+x right,+y down,+z forward) must be explicitly verified")
    convention = specification.get("convention")
    if convention not in {"camera_to_world", "world_to_camera"}:
        raise ValueError("Explicit pose.convention must be camera_to_world or world_to_camera; robot poses require prior calibrated conversion")
    units = specification.get("translation_units")
    if units not in _UNIT_SCALE:
        raise ValueError("pose.translation_units must explicitly be m or mm")
    pose = np.asarray(_resolve(root, value), dtype=np.float64)
    # Validate rigid transforms before inversion; both C2W and W2C have the same
    # rigidity requirements. Scaling translation before inversion preserves units.
    _matrix(pose, (4, 4), "camera_to_world")
    pose = pose.copy()
    pose[:3, 3] *= _UNIT_SCALE[units]
    if convention == "world_to_camera":
        pose = np.linalg.inv(pose)
    _matrix(pose, (4, 4), "camera_to_world")
    return pose.tolist()


def _check_calibration(sequence):
    calibration = sequence.get("calibration", {})
    for key in ("verified", "rgb_rectified", "gt_rectified", "pixel_alignment_verified"):
        if calibration.get(key) is not True:
            raise ValueError(f"calibration.{key}=true is required; unrectified/unaligned data is not silently corrected")
    if not calibration.get("source") or not calibration.get("alignment_source"):
        raise ValueError("Calibration and frame/GT alignment need documented source descriptions")
    distortion = calibration.get("distortion")
    if distortion is None:
        raise ValueError("Declare distortion=[] for already rectified images, or the original known coefficients")
    coefficients = np.asarray(distortion, dtype=float)
    if coefficients.ndim != 1 or not np.isfinite(coefficients).all():
        raise ValueError("Distortion coefficients must be a finite vector")
    # Nonzero original distortion is allowed only because rectified RGB AND GT,
    # plus the post-rectification intrinsics, are explicitly required above.
    return calibration


def _video_frame(video, index, layout):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Optional SCARED video extraction requires opencv-python on the server; nothing is installed automatically") from exc
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open local RGB video: {video}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = capture.read()
        if not ok or bgr is None:
            raise ValueError(f"Missing explicitly indexed video frame {index}: {video}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if layout == "top_bottom_left_top":
            if h % 2:
                raise ValueError("Stacked stereo height must be even")
            return rgb[:h // 2]
        if layout == "side_by_side_left_first":
            if w % 2:
                raise ValueError("Stacked stereo width must be even")
            return rgb[:, :w // 2]
        if layout == "left_only":
            return rgb
        raise ValueError("rgb_video_layout must explicitly identify the left view")
    finally:
        capture.release()


def build_scared_manifest(source_root, source_index, output_path, *, dry_run=False,
                          extract_video=False, verify_arrays=False, splits=None):
    """Build manifest v1 from a verified explicit source-index dictionary/file.

    ``dry_run`` checks existing paths and declared geometry but writes/extracts
    nothing. ``verify_arrays`` additionally loads one frame at a time after
    conversion; it is incompatible with a video-extraction dry run because the
    planned RGB files do not exist yet. Existing source files are never changed.
    """
    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Server-local SCARED root is missing: {source_root}")
    output_path = Path(output_path).resolve()
    payload = _json(source_index) if isinstance(source_index, (str, Path)) else source_index
    if payload.get("schema_version") != 1 or not payload.get("sequences"):
        raise ValueError("SCARED source index needs schema_version=1 and nonempty sequences")
    split_map = DEFAULT_SPLITS if splits is None else {int(k): v for k, v in splits.items()}
    if set(split_map) != set(range(1, 10)) or set(split_map.values()) - {"train", "validation", "test"}:
        raise ValueError("Split map must assign datasets 1..9 to train/validation/test")
    sequences, ids, patient_splits, extraction = [], set(), {}, []
    parent_cache = {}
    for source in payload["sequences"]:
        dataset_id = source.get("dataset_id")
        if isinstance(dataset_id, bool) or not isinstance(dataset_id, int) or dataset_id not in split_map:
            raise ValueError("dataset_id must be an integer in 1..9")
        sequence_id, patient = source.get("id"), source.get("patient_id")
        if not isinstance(sequence_id, str) or not sequence_id or not isinstance(patient, str) or not patient:
            raise ValueError("Source sequences need explicit id and verified patient_id")
        if sequence_id in ids:
            raise ValueError(f"Duplicate sequence id: {sequence_id}")
        ids.add(sequence_id)
        split = split_map[dataset_id]
        if patient in patient_splits and patient_splits[patient] != split:
            raise ValueError(f"Patient {patient} leaks across splits")
        patient_splits[patient] = split
        root = source_root / source.get("root", ".")
        calibration = _check_calibration(source)
        gt = source.get("geometry", {})
        if gt.get("verified") is not True or not gt.get("source") or gt.get("units") not in _UNIT_SCALE:
            raise ValueError("geometry needs verified=true, a GT source and explicit m/mm units")
        if source.get("frame_alignment_verified") is not True:
            raise ValueError("frame_alignment_verified=true requires checking numeric indices/timestamps, not guessing nearest RGB")
        frames = source.get("frames", [])
        if len(frames) < 2:
            raise ValueError("Every sequence needs at least two explicitly aligned frames")
        frame_ids = [frame.get("frame_id") for frame in frames]
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in frame_ids) or len(set(frame_ids)) != len(frame_ids):
            raise ValueError("Frame ids must be unique nonnegative integer indices")
        frames = sorted(frames, key=lambda frame: frame["frame_id"])
        entries = []
        for frame in frames:
            frame_id = frame["frame_id"]
            if frame.get("rgb") and frame.get("rgb_video"):
                raise ValueError("Declare either left RGB image or video reference, never both")
            if frame.get("rgb_video"):
                if not extract_video:
                    raise ValueError("Video references require --extract-video on the server; --dry-run can validate without extraction")
                video = _required_file(root, frame["rgb_video"], "RGB video", parent_cache)
                video_index = frame.get("rgb_frame_index")
                if isinstance(video_index, bool) or not isinstance(video_index, int) or video_index < 0:
                    raise ValueError("Video frame index must explicitly match the GT frame; no FPS/timestamp guessing")
                layout = frame.get("rgb_video_layout", source.get("rgb_video_layout"))
                if layout not in {"top_bottom_left_top", "side_by_side_left_first", "left_only"}:
                    raise ValueError("Explicit rgb_video_layout is required for left-view extraction")
                safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", sequence_id)
                rgb = output_path.parent / "scared_left_rgb" / safe_id / f"{frame_id:08d}.npy"
                extraction.append((video, video_index, layout, rgb))
            else:
                rgb = _required_file(root, frame.get("rgb"), "left RGB", parent_cache)
            has_depth, has_points = bool(frame.get("depth")), bool(frame.get("pointmap"))
            if has_depth == has_points:
                raise ValueError("Exactly one depth or XYZ pointmap GT path is required")
            target_key = "depth" if has_depth else "pointmap"
            target_path = _required_file(root, frame[target_key], target_key, parent_cache)
            intrinsics = _resolve(root, frame.get("intrinsics", source.get("intrinsics")))
            intrinsics = _matrix(intrinsics, (3, 3), "intrinsics").tolist()
            pose = _checked_pose(root, frame.get("pose"), source.get("pose", {}))
            entry = {"frame_id": frame_id, "rgb": str(rgb), target_key: str(target_path),
                     "intrinsics": intrinsics, "camera_to_world": pose}
            if frame.get("valid_mask"):
                entry["valid_mask"] = str(_required_file(root, frame["valid_mask"], "valid mask", parent_cache))
            else:
                rule = frame.get("valid_mask_rule", gt.get("valid_mask_rule"))
                if rule not in {"depth_positive_finite", "camera_xyz_positive_z_finite"}:
                    raise ValueError("Missing validity mask: declare a verified depth/XYZ invalid-value rule, or provide mask files")
                entry["valid_mask_rule"] = rule
            if has_depth:
                entry["depth_scale"] = _UNIT_SCALE[gt["units"]]
                encoding = frame.get("depth_encoding", "metric_depth")
                if encoding != "metric_depth":
                    if encoding != "rectified_disparity" or gt.get("units") != "m" or gt.get("stored_units") != "pixels":
                        raise ValueError("Rectified disparity requires geometry.units=m (decoded output) and stored_units=pixels")
                    entry.update(depth_encoding=encoding,
                                 stereo_baseline_m=frame.get("stereo_baseline_m"),
                                 right_principal_x=frame.get("right_principal_x"))
                if entry.get("valid_mask_rule") == "camera_xyz_positive_z_finite":
                    raise ValueError("Camera XYZ mask rule cannot be used for depth")
            else:
                space = frame.get("pointmap_space", gt.get("space"))
                if space not in {"camera", "world", "first_camera"}:
                    raise ValueError("Pointmap coordinate space must be explicitly declared")
                if entry.get("valid_mask_rule") == "depth_positive_finite" or entry.get("valid_mask_rule") == "camera_xyz_positive_z_finite" and space != "camera":
                    raise ValueError("Validity rule is incompatible with the GT coordinate space")
                entry.update(pointmap_space=space, pointmap_scale=_UNIT_SCALE[gt["units"]])
            entries.append(entry)
        provenance = {"verified": True, "source": gt["source"], "depth_or_pointmap_gt": True,
                      "calibration_gt": True, "pose_gt": True, "units": "metres",
                      "dataset": "SCARED", "dataset_id": dataset_id,
                      "source_index": str(source_index) if isinstance(source_index, (str, Path)) else "explicit mapping/index",
                      "source_root": str(source_root), "source_units": gt["units"],
                      "pose_convention_input": source["pose"]["convention"],
                      "pose_convention_output": "camera_to_world", "camera_axes": "opencv",
                      "calibration": calibration, "frame_alignment_verified": True,
                      "verification_kind": "operator-declared; numerical checks do not prove sensor synchronization"}
        if gt.get("stored_units"):
            provenance["stored_geometry_units"] = gt["stored_units"]
        sequences.append({"id": sequence_id, "patient_id": patient, "split": split,
                          "provenance": provenance, "frames": entries})
    result = {"schema_version": 1, "dataset": "SCARED", "split_protocol": {
                  "name": "scared_subject_holdout_v1", "assignments": {str(k): v for k, v in split_map.items()},
                  "status": "proposed validation holdout; datasets 8-9 reserved for paper-aligned test"},
              "sequences": sequences}
    if dry_run:
        if verify_arrays:
            raise ValueError("--verify-arrays is a separate read-only validation after preparation; dry-run does not decode/preprocess data")
        return result
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output_path}; choose a fresh output path")
    for _, _, _, destination in extraction:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite extracted RGB: {destination}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for video, index, layout, destination in extraction:
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, _video_frame(video, index, layout), allow_pickle=False)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Metadata validation covers shape, paths, provenance and all patient splits.
    dataset = SequenceDataset(output_path)
    if verify_arrays:
        dataset.validate_all()
    return result


def discover_source_index(source_root, mapping):
    """Discover declared SCARED patterns and join RGB/GT/metadata by exact ids.

    Mapping ``defaults`` carries explicit calibration, pose/geometry semantics
    and alignment assertions; ``patterns`` carries sequence/rgb/target/metadata
    globs and JSON fields. Discovery is opt-in because release layouts differ.
    Missing or ambiguous files abort rather than silently dropping frames.
    """
    root = Path(source_root).resolve()
    mapping = _json(mapping) if isinstance(mapping, (str, Path)) else mapping
    if mapping.get("schema_version") != 1:
        raise ValueError("Discovery mapping needs schema_version=1")
    defaults, patterns = mapping.get("defaults", {}), mapping.get("patterns", {})
    for key in ("sequence_glob", "dataset_regex", "patient_ids", "index_regex", "metadata_glob", "pose_field"):
        if not patterns.get(key):
            raise ValueError(f"Discovery mapping requires patterns.{key}")
    index_regex = re.compile(patterns["index_regex"])
    dataset_regex = re.compile(patterns["dataset_regex"])

    def indexed(directory, pattern):
        found = {}
        for path in sorted(directory.glob(pattern)):
            match = index_regex.search(path.name)
            if not match or not path.is_file():
                raise ValueError(f"Pattern file has no unambiguous numeric frame id: {path}")
            number = int(match.group(1))
            if number in found:
                raise ValueError(f"Ambiguous duplicate frame id {number} for pattern {pattern}")
            found[number] = path
        if not found:
            raise FileNotFoundError(f"No files match {directory}/{pattern}")
        return found

    sequences = []
    for directory in sorted(root.glob(patterns["sequence_glob"])):
        if not directory.is_dir():
            continue
        relative = directory.relative_to(root)
        match = dataset_regex.search(relative.as_posix())
        if not match:
            raise ValueError(f"Cannot identify SCARED dataset id for {relative}")
        dataset_id = int(match.group(1))
        patient = patterns["patient_ids"].get(str(dataset_id))
        if not patient:
            raise ValueError(f"Verified patient identity missing for dataset {dataset_id}")
        metadata = indexed(directory, patterns["metadata_glob"])
        target_key = patterns.get("target_type")
        if target_key not in {"depth", "pointmap"} or not patterns.get("target_glob"):
            raise ValueError("Discovery requires target_type=depth/pointmap and target_glob")
        targets = indexed(directory, patterns["target_glob"])
        if set(targets) != set(metadata):
            raise ValueError("GT and pose metadata ids differ; do not guess synchronization or silently drop frames")
        rgb = indexed(directory, patterns["rgb_glob"]) if patterns.get("rgb_glob") else None
        if rgb is not None and set(rgb) != set(metadata):
            raise ValueError("Left RGB and GT ids differ; provide an explicit verified source index for partial GT")
        masks = indexed(directory, patterns["mask_glob"]) if patterns.get("mask_glob") else None
        if masks is not None and set(masks) != set(metadata):
            raise ValueError("Validity-mask and GT indices differ")
        if rgb is None:
            if not patterns.get("video") or not patterns.get("video_frame_index_field"):
                raise ValueError("Discovery needs rgb_glob or video + video_frame_index_field; video/GT index offset is not assumed")
            _required_file(directory, patterns["video"], "RGB video")
        sequence = dict(defaults)
        sequence.update(id=relative.as_posix().replace("/", "_"), dataset_id=dataset_id,
                        patient_id=patient, root=relative.as_posix(), frames=[])
        for number in sorted(metadata):
            document = _json(metadata[number])
            frame = {"frame_id": number, target_key: str(targets[number].relative_to(directory)),
                     "pose": _field(document, patterns["pose_field"])}
            if patterns.get("intrinsics_field"):
                frame["intrinsics"] = _field(document, patterns["intrinsics_field"])
            if rgb is not None:
                frame["rgb"] = str(rgb[number].relative_to(directory))
            else:
                frame.update(rgb_video=patterns["video"],
                             rgb_frame_index=_field(document, patterns["video_frame_index_field"]))
            if masks is not None:
                frame["valid_mask"] = str(masks[number].relative_to(directory))
            sequence["frames"].append(frame)
        sequences.append(sequence)
    if not sequences:
        raise FileNotFoundError("No sequence directories match the explicitly configured native pattern")
    return {"schema_version": 1, "sequences": sequences}


def manifest_summary(payload):
    counts = {split: {"sequences": 0, "frames": 0, "patients": set()} for split in ("train", "validation", "test")}
    for sequence in payload["sequences"]:
        item = counts[sequence["split"]]
        item["sequences"] += 1
        item["frames"] += len(sequence["frames"])
        item["patients"].add(sequence["patient_id"])
    for item in counts.values():
        item["patients"] = sorted(item["patients"])
    return {"schema_version": 1, "splits": counts, "total_frames": sum(c["frames"] for c in counts.values())}
