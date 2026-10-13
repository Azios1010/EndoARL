"""CPU GT coverage audit and explicitly separate, train-only supervision manifests.

Coverage uses SequenceDataset's actual decoding, calibrated geometry, center
crop and nearest-exact resize. It never fills missing GT or filters evaluation.
The one-pixel default prevents empty-target crashes; it is not a quality gate.
"""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path

from dust3r.training.sequence_dataset import SequenceDataset, _matrix


INVENTORY_SCHEMA = "endo3r_supervision_inventory_v1"
SUPERVISION_SCHEMA = "endo3r_training_supervision_v1"
_FRAME_PATHS = ("rgb", "depth", "pointmap", "valid_mask")


def manifest_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("sequences"), list):
        raise ValueError("A schema_version=1 source sequence manifest is required")
    if not payload["sequences"]:
        raise ValueError("Source manifest contains no sequences")
    _validate_splits(payload["sequences"])
    return path, payload


def _validate_splits(sequences):
    """Inspect all source splits even when the eventual output is train-only."""
    patients, identifiers = {}, set()
    for sequence in sequences:
        for key in ("id", "patient_id", "split"):
            if not isinstance(sequence.get(key), str) or not sequence[key]:
                raise ValueError(f"Every source sequence requires nonempty {key}")
        if sequence["split"] not in {"train", "validation", "test"}:
            raise ValueError("Source split must be train, validation or test")
        if sequence["id"] in identifiers:
            raise ValueError("Source sequence ids must be unique")
        identifiers.add(sequence["id"])
        patient = sequence["patient_id"]
        if patient in patients and patients[patient] != sequence["split"]:
            raise ValueError(f"Patient {patient} leaks across source train/validation/test splits")
        patients[patient] = sequence["split"]
    return patients


def _options(image_hw, source_stride, clip_raw_frames, min_valid_pixels,
             min_valid_fraction, expected_frame_id_step):
    if (len(image_hw) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
                                 for x in image_hw)):
        raise ValueError("image_hw must contain positive integer (height,width)")
    for name, number, lower in (("source_stride", source_stride, 1),
                                ("clip_raw_frames", clip_raw_frames, 2),
                                ("min_valid_pixels", min_valid_pixels, 1),
                                ("expected_frame_id_step", expected_frame_id_step, 1)):
        if isinstance(number, bool) or not isinstance(number, int) or number < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if (not isinstance(min_valid_fraction, (int, float)) or isinstance(min_valid_fraction, bool)
            or not math.isfinite(min_valid_fraction) or not 0 <= min_valid_fraction <= 1):
        raise ValueError("min_valid_fraction must be finite and in [0,1]")
    return {"image_hw": list(image_hw), "source_stride": source_stride,
            "clip_raw_frames": clip_raw_frames, "min_valid_pixels": min_valid_pixels,
            "min_valid_fraction": float(min_valid_fraction),
            "expected_frame_id_step": expected_frame_id_step}


def _numeric_frame_id(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _valid_runs(rows, options):
    """Do not join across an invalid selected frame or a source-index/ID gap."""
    runs, current = [], []
    stride = options["source_stride"]
    for row in rows:
        if not row["selected_by_source_stride"]:
            continue
        if row["status"] != "eligible":
            if current:
                runs.append(current)
                current = []
            continue
        if current:
            previous = current[-1]
            current_id, previous_id = map(_numeric_frame_id,
                                          (row["source_frame_id"], previous["source_frame_id"]))
            index_gap = row["source_index"] - previous["source_index"] != stride
            id_gap = (current_id is not None and previous_id is not None and
                      current_id - previous_id != stride * options["expected_frame_id_step"])
            if index_gap or id_gap:
                runs.append(current)
                current = []
        current.append(row)
    if current:
        runs.append(current)
    return runs


def audit_supervision(manifest, *, image_hw=(256, 320), source_stride=1,
                      clip_raw_frames=9, min_valid_pixels=1,
                      min_valid_fraction=0., expected_frame_id_step=1):
    """Audit every source frame, retaining per-frame failures instead of stopping.

    Malformed top-level schemas and patient leakage remain fatal. Individual
    missing files, malformed arrays and empty resized GT are recorded. No model,
    CUDA, downloads, or writes are involved.
    """
    path, payload = _read_manifest(manifest)
    options = _options(image_hw, source_stride, clip_raw_frames, min_valid_pixels,
                       min_valid_fraction, expected_frame_id_step)
    # Construction normally eagerly validates all files. This inspection-only
    # reader instead validates each entry with the same _validate_sequence and
    # _load_frame implementations, so one broken frame does not hide later rows.
    reader = SequenceDataset.__new__(SequenceDataset)
    reader.root, reader.image_hw = path.parent, tuple(image_hw)
    rows, sequence_reports = [], []
    patients = _validate_splits(payload["sequences"])
    for sequence in payload["sequences"]:
        sequence_rows = []
        frames = sequence.get("frames", [])
        if not isinstance(frames, list):
            raise ValueError(f"Sequence {sequence['id']} frames must be a list")
        id_counts = Counter(entry.get("frame_id", index) for index, entry in enumerate(frames)
                            if isinstance(entry, dict) and isinstance(entry.get("frame_id", index), (int, str)))
        for index, entry in enumerate(frames):
            row = {"sequence_id": sequence["id"], "patient_id": sequence["patient_id"],
                   "split": sequence["split"], "source_index": index,
                   "source_frame_id": entry.get("frame_id", index) if isinstance(entry, dict) else index,
                   "selected_by_source_stride": index % source_stride == 0,
                   "status": "error", "reasons": [], "coverage": {}}
            try:
                if not isinstance(entry, dict):
                    raise ValueError("Frame entry must be an object")
                identifier = row["source_frame_id"]
                if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or identifier == "":
                    raise ValueError("Source frame_id must be an integer or nonempty string")
                # Duplicating this entry only satisfies the normal sequence
                # minimum; no fake frame is decoded or included in a manifest.
                checked_sequence = {**sequence, "frames": [entry, entry]}
                reader._validate_sequence(checked_sequence)
                first_pose = _matrix(frames[0]["camera_to_world"], (4, 4), "camera_to_world")
                origin_pose = _matrix(sequence.get("first_camera_origin_camera_to_world",
                                                    frames[0]["camera_to_world"]),
                                      (4, 4), "camera_to_world")
                reader._load_frame(sequence, entry, 0, index, first_pose, origin_pose,
                                   coverage_trace=row["coverage"], allow_empty_gt=True)
                final = row["coverage"]["final"]
                if final["valid_pixels"] < min_valid_pixels:
                    row["reasons"].append("below_min_valid_pixels")
                if final["valid_fraction"] < min_valid_fraction:
                    row["reasons"].append("below_min_valid_fraction")
                if final["valid_pixels"] == 0:
                    row["reasons"].append("zero_final_valid_gt")
                if row["coverage"]["raw"]["valid_pixels"] > 0 and final["valid_pixels"] == 0:
                    row["reasons"].append("native_gt_disappears_after_crop_resize")
                if id_counts[identifier] != 1:
                    row["reasons"].append("duplicate_source_frame_id")
                row["status"] = "ineligible" if row["reasons"] else "eligible"
            except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError, OverflowError) as exc:
                row["reasons"].append(f"{type(exc).__name__}: {exc}")
            sequence_rows.append(row)
            rows.append(row)
        runs = _valid_runs(sequence_rows, options)
        eligible_runs = [run for run in runs if len(run) >= clip_raw_frames]
        sequence_reports.append({"sequence_id": sequence["id"], "patient_id": sequence["patient_id"],
                                 "split": sequence["split"], "source_frames": len(frames),
                                 "status_counts": dict(Counter(row["status"] for row in sequence_rows)),
                                 "empty_or_too_short_sequence": len(frames) < 2,
                                 "valid_runs": [{"source_indices": [row["source_index"] for row in run],
                                                 "source_frame_ids": [row["source_frame_id"] for row in run],
                                                 "length": len(run), "supports_requested_clip": len(run) >= clip_raw_frames}
                                                for run in runs],
                                 "train_clip_count": sum(len(run) - clip_raw_frames + 1 for run in eligible_runs)
                                 if sequence["split"] == "train" else 0})
    split_counts = {}
    for split in ("train", "validation", "test"):
        chosen = [row for row in rows if row["split"] == split]
        split_counts[split] = {"source_frames": len(chosen),
                               "status_counts": dict(Counter(row["status"] for row in chosen)),
                               "patients": sorted(p for p, s in patients.items() if s == split)}
    return {"schema": INVENTORY_SCHEMA, "source_manifest_path": str(path),
            "source_manifest_sha256": manifest_sha256(path), "options": options,
            "scope": "all_original_frames_all_splits; no evaluation/FPS filtering",
            "default_threshold_is_crash_prevention_only": min_valid_pixels == 1 and min_valid_fraction == 0,
            "coverage_semantics": "finite calibrated pointmap and positive finite current-camera depth; loader nearest-exact",
            "split_counts": split_counts, "sequence_reports": sequence_reports, "frames": rows,
            "failures": [copy.deepcopy(row) for row in rows if row["status"] != "eligible"],
            "sequence_failures": [{"sequence_id": item["sequence_id"], "split": item["split"],
                                   "reason": "sequence_has_fewer_than_two_frames"}
                                  for item in sequence_reports if item["empty_or_too_short_sequence"]],
            "train_clip_count": sum(item["train_clip_count"] for item in sequence_reports)}


def _relocate_frame(entry, source_root, index):
    entry = copy.deepcopy(entry)
    for key in _FRAME_PATHS:
        if entry.get(key):
            path = Path(entry[key])
            entry[key] = str((path if path.is_absolute() else source_root / path).resolve())
    entry["frame_id"] = entry.get("frame_id", index)
    entry["supervision_source_index"] = index
    return entry


def build_training_manifest(manifest, inventory):
    """Construct train-only maximal valid runs; do not write or alter the source."""
    path, payload = _read_manifest(manifest)
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("An audit_supervision inventory is required")
    if inventory.get("source_manifest_sha256") != manifest_sha256(path):
        raise ValueError("Source manifest changed after coverage inventory")
    options = inventory["options"]
    options = _options(**options)
    by_sequence = {}
    for row in inventory["frames"]:
        by_sequence.setdefault(row["sequence_id"], []).append(row)
    result = copy.deepcopy(payload)
    result["sequences"], exclusions = [], []
    included = set()
    for source_sequence in payload["sequences"]:
        if source_sequence["split"] != "train":
            continue
        for run in _valid_runs(by_sequence.get(source_sequence["id"], []), options):
            if len(run) < options["clip_raw_frames"]:
                continue
            indices = [row["source_index"] for row in run]
            sequence = copy.deepcopy(source_sequence)
            sequence.update(id=f"{source_sequence['id']}__supervision_{indices[0]}_{indices[-1]}",
                            source_sequence_id=source_sequence["id"], supervision_source_indices=indices,
                            first_camera_origin_camera_to_world=copy.deepcopy(
                                source_sequence.get("first_camera_origin_camera_to_world",
                                                    source_sequence["frames"][0]["camera_to_world"])),
                            frames=[_relocate_frame(source_sequence["frames"][i], path.parent, i) for i in indices])
            result["sequences"].append(sequence)
            included.update((source_sequence["id"], i) for i in indices)
    if not result["sequences"]:
        raise ValueError("No contiguous train-only run supports the requested clip length; inspect inventory failures")
    for row in inventory["frames"]:
        if (row["sequence_id"], row["source_index"]) in included:
            continue
        reason = ("held_out_split_preserved" if row["split"] != "train" else
                  "not_selected_by_source_stride" if not row["selected_by_source_stride"] else
                  "ineligible_coverage_or_decode" if row["status"] != "eligible" else
                  "valid_run_shorter_than_clip")
        exclusions.append({"sequence_id": row["sequence_id"], "patient_id": row["patient_id"],
                           "split": row["split"], "source_index": row["source_index"],
                           "source_frame_id": row["source_frame_id"], "reason": reason,
                           "coverage_reasons": copy.deepcopy(row["reasons"])})
    result["training_supervision"] = {"schema": SUPERVISION_SCHEMA,
                                      "source_manifest_path": str(path),
                                      "source_manifest_sha256": inventory["source_manifest_sha256"],
                                      **options, "loader_stride": 1, "frame_count": len(included),
                                      "sequence_count": len(result["sequences"]),
                                      "clip_count": inventory["train_clip_count"],
                                      "default_threshold_is_crash_prevention_only": inventory["default_threshold_is_crash_prevention_only"],
                                      "evaluation_filtering_permitted": False,
                                      "excluded_frames": exclusions,
                                      "selected_frame_coverage": [copy.deepcopy(row) for row in inventory["frames"]
                                                                  if (row["sequence_id"], row["source_index"]) in included]}
    return result


def validate_supervision_manifest(supervision_manifest, source_manifest, image_hw,
                                  stride=1, clip_frames=None):
    """Verify derived training provenance without decoding or altering any target.

    ``stride`` is the effective loader stride and must equal one: source stride
    was already applied when constructing the derived manifest. This validation
    is not a certification of reconstruction quality or completeness of GT.
    """
    source_path, source = _read_manifest(source_manifest)
    path, derived = _read_manifest(supervision_manifest)
    meta = derived.get("training_supervision", {})
    if meta.get("schema") != SUPERVISION_SCHEMA:
        raise ValueError("Derived training manifest lacks verified training_supervision metadata")
    if meta.get("source_manifest_sha256") != manifest_sha256(source_path):
        raise ValueError("Training supervision source manifest SHA256 mismatch")
    if list(image_hw) != meta.get("image_hw"):
        raise ValueError("Training supervision resolution differs from configured image_hw; regenerate coverage")
    if stride != 1 or meta.get("loader_stride") != 1:
        raise ValueError("Derived training supervision must use loader stride=1; source stride was already applied")
    options = _options(**{name: meta.get(name) for name in
                         ("image_hw", "source_stride", "clip_raw_frames", "min_valid_pixels",
                          "min_valid_fraction", "expected_frame_id_step")})
    if clip_frames is not None and clip_frames != options["clip_raw_frames"]:
        raise ValueError("Configured training clip length differs from supervision inventory; regenerate coverage")
    if meta.get("evaluation_filtering_permitted") is not False:
        raise ValueError("Training supervision must explicitly prohibit evaluation filtering")
    sources = {sequence["id"]: sequence for sequence in source["sequences"]}
    coverage_rows = meta.get("selected_frame_coverage", [])
    coverage = {(row["sequence_id"], row["source_index"]): row for row in coverage_rows}
    if len(coverage) != len(coverage_rows):
        raise ValueError("Training supervision repeats coverage evidence for one source frame")
    selected = set()
    for sequence in derived["sequences"]:
        original = sources.get(sequence.get("source_sequence_id"))
        if original is None or sequence["split"] != "train" or original["split"] != "train":
            raise ValueError("Training supervision may contain only original TRAIN sequences")
        for key in (key for key in original if key not in {"id", "frames"}):
            if sequence.get(key) != original.get(key):
                raise ValueError(f"Training supervision changed source {key}")
        expected_origin = original.get("first_camera_origin_camera_to_world", original["frames"][0]["camera_to_world"])
        if sequence.get("first_camera_origin_camera_to_world") != expected_origin:
            raise ValueError("Training supervision changed original first-camera coordinate origin")
        frames, indices = sequence.get("frames", []), sequence.get("supervision_source_indices", [])
        if len(indices) != len(frames) or len(frames) < options["clip_raw_frames"]:
            raise ValueError("Training supervision interval is too short or indices do not match")
        previous_id = None
        for position, (index, entry) in enumerate(zip(indices, frames)):
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(original["frames"]):
                raise ValueError("Training supervision source index is invalid")
            if index % options["source_stride"] != 0 or position and index - indices[position - 1] != options["source_stride"]:
                raise ValueError("Training supervision crosses a source-index gap or applies a different stride")
            expected = _relocate_frame(original["frames"][index], source_path.parent, index)
            actual = _relocate_frame(entry, path.parent, index)
            if actual != expected:
                raise ValueError("Training supervision changed original frame IDs, target metadata, calibration, or file paths")
            if entry.get("supervision_source_index") != index:
                raise ValueError("Frame source index metadata differs from interval metadata")
            current_id = _numeric_frame_id(entry["frame_id"])
            if (position and current_id is not None and previous_id is not None and
                    current_id - previous_id != options["source_stride"] * options["expected_frame_id_step"]):
                raise ValueError("Training supervision crosses a numerical source-frame-ID gap")
            previous_id = current_id
            key = (original["id"], index)
            if key in selected:
                raise ValueError("Training supervision repeats a source frame across intervals")
            selected.add(key)
            row = coverage.get(key, {})
            final = row.get("coverage", {}).get("final", {})
            if (row.get("status") != "eligible" or row.get("source_frame_id") != expected["frame_id"] or
                    row.get("split") != "train" or row.get("patient_id") != original["patient_id"] or
                    final.get("valid_pixels", -1) < options["min_valid_pixels"] or
                    final.get("valid_fraction", -1) < options["min_valid_fraction"]):
                raise ValueError("Selected frame lacks matching eligible coverage provenance")
            pixels = final.get("valid_pixels")
            total = options["image_hw"][0] * options["image_hw"][1]
            if (isinstance(pixels, bool) or not isinstance(pixels, int) or not 0 <= pixels <= total or
                    final.get("image_hw") != options["image_hw"] or final.get("total_pixels") != total or
                    not math.isclose(final.get("valid_fraction", -1), pixels / total, abs_tol=1e-12)):
                raise ValueError("Training supervision final coverage pixels/fraction/resolution are inconsistent")
    if meta.get("frame_count") != len(selected) or len(coverage) != len(selected):
        raise ValueError("Training supervision coverage/frame-count accounting mismatch")
    all_source = {(item["id"], index) for item in source["sequences"]
                  for index in range(len(item.get("frames", [])))}
    exclusions = meta.get("excluded_frames", [])
    excluded = {(row["sequence_id"], row["source_index"]) for row in exclusions}
    if len(excluded) != len(exclusions) or selected & excluded or selected | excluded != all_source:
        raise ValueError("Training supervision must disclose every excluded original frame exactly once")
    for row in exclusions:
        original = sources[row["sequence_id"]]
        entry = original["frames"][row["source_index"]]
        if (row.get("patient_id") != original["patient_id"] or row.get("split") != original["split"] or
                row.get("source_frame_id") != (entry.get("frame_id", row["source_index"]) if isinstance(entry, dict)
                                               else row["source_index"]) or not row.get("reason")):
            raise ValueError("Training supervision exclusion metadata differs from source")
    clip_count = sum(len(item["frames"]) - options["clip_raw_frames"] + 1 for item in derived["sequences"])
    if meta.get("sequence_count") != len(derived["sequences"]) or meta.get("clip_count") != clip_count:
        raise ValueError("Training supervision sequence/clip-count accounting mismatch")
    return {"manifest": str(path), "source_manifest": str(source_path),
            "source_manifest_sha256": meta["source_manifest_sha256"],
            "train_sequences": len(derived["sequences"]), "train_frames": len(selected),
            "train_clips": clip_count,
            "image_hw": list(image_hw), "loader_stride": 1,
            "source_stride": options["source_stride"],
            "threshold_is_quality_certification": False}


def write_json_new(path, payload):
    """Explicit output only; exclusive creation prevents accidental overwrites."""
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return str(path)
