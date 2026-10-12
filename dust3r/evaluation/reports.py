"""Strict benchmark report serialization and reproducibility fingerprints."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .streaming import (CAMERA_SOURCE, DEPTH_PROTOCOL, POSE_PROTOCOL,
                        SCALE_PROTOCOL, aggregate_sequences, canonical_sha256)

REPORT_SCHEMA = "endo3r_benchmark_v1"


def file_sha256(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local file required for reproducibility fingerprint: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_report(sequences, *, config, manifest_sha256, checkpoint_hashes,
                split, image_hw, seed, experiment_id=None, timing=None,
                actual_backends=None, additional_metadata=None):
    """No numeric field is fabricated for absent timing/pose/failed inference."""
    aggregate = aggregate_sequences(sequences)
    report = {"schema": REPORT_SCHEMA, "schema_version": 1,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "experiment_id": experiment_id,
              "baseline_identity": config.get("baseline_identity"),
              "protocol": DEPTH_PROTOCOL, "scale_protocol": SCALE_PROTOCOL,
              "prediction_camera_source": CAMERA_SOURCE,
              "pose_protocol": sequences[0]["pose_protocol"],
              "paper_exact_protocol_claim": False,
              "protocol_disclosures": [
                  "Depth conversion follows released demo.py predicted focal/PnP; first frame retains raw Z.",
                  "Depth scaling follows released depth_evaluation.py whole-sequence GT-valid median ratio.",
                  "GT validity mask is explicit; nonfinite predictions/PnP failures are counted, not silently skipped.",
                  "GT calibration/pose are used only for target construction/scoring, never prediction camera.",
                  "Pose alignment is explicit 5-frame positive Sim3, stride1, adjacent-frame RPE; paper details remain underspecified.",
                  "Float64 CPU focal/metric arithmetic may differ slightly from released float32 outputs.",
                  "Preprocessing must match across experiments; GT center crop/resize can differ from legacy evaluation's direct depth resize.",
                  "Incomplete reports publish successful-frame metrics with failure denominators; they are not quality-equivalent to complete runs.",
              ],
              "config": config, "config_sha256": canonical_sha256(config),
              "manifest_sha256": manifest_sha256, "checkpoint_hashes": dict(checkpoint_hashes),
              "split": split, "image_hw": list(image_hw), "seed": int(seed),
              "frame_indices_sha256": canonical_sha256([
                  [item["episode_id"], item["frame_indices_sha256"]] for item in sequences]),
              "provenance_sha256": canonical_sha256([
                  [item["episode_id"], item["provenance_sha256"]] for item in sequences]),
              "sequences": sequences, "aggregate": aggregate,
              "timing": timing, "actual_backends": actual_backends or {},
              "metadata": additional_metadata or {}}
    # Strict serialization rejects NaN/Infinity before an artifact is written.
    json.dumps(report, allow_nan=False)
    return report


def save_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
