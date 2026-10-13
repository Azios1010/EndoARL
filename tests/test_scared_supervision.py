"""Synthetic CPU-only GT coverage/run provenance tests; no external resources."""
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dust3r.data.supervision import (audit_supervision, build_training_manifest,
                                    manifest_sha256, validate_supervision_manifest, write_json_new)
from dust3r.training.sequence_dataset import SequenceDataset
from tools.audit_scared_supervision import main


def fixture(root, frame_count=7, *, split="train", prefix="train", first_camera=False):
    root = Path(root)
    frames = []
    for index in range(frame_count):
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = index
        np.save(root / f"{prefix}_rgb{index}.npy", rgb)
        np.save(root / f"{prefix}_mask{index}.npy", np.ones((4, 4), dtype=np.uint8))
        entry = {"frame_id": index, "rgb": f"{prefix}_rgb{index}.npy",
                 "valid_mask": f"{prefix}_mask{index}.npy",
                 "intrinsics": [[4., 0., 1.5], [0., 4., 1.5], [0., 0., 1.]],
                 "camera_to_world": pose.tolist()}
        if first_camera:
            xyz = np.zeros((4, 4, 3), dtype=np.float32)
            xyz[..., 0], xyz[..., 2] = index, 1.
            np.save(root / f"{prefix}_xyz{index}.npy", xyz)
            entry.update(pointmap=f"{prefix}_xyz{index}.npy", pointmap_space="first_camera")
        else:
            np.save(root / f"{prefix}_depth{index}.npy", np.ones((4, 4), dtype=np.float32))
            entry.update(depth=f"{prefix}_depth{index}.npy", depth_scale=1.)
        frames.append(entry)
    return {"id": prefix, "patient_id": f"patient_{prefix}", "split": split,
            "provenance": {"verified": True, "source": "synthetic only", "units": "m",
                           "depth_or_pointmap_gt": True, "calibration_gt": True, "pose_gt": True},
            "frames": frames}


def source_manifest(root, *sequences):
    path = Path(root) / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "split_protocol": {"name": "synthetic patient split"},
                                "sequences": list(sequences)}), encoding="utf-8")
    return path


class CoverageTests(unittest.TestCase):
    def test_sparse_native_pixels_disappear_and_normal_loader_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = fixture(directory, 5)
            sparse = np.zeros((4, 4), dtype=np.uint8)
            sparse[::2, ::2] = 1  # four native valid pixels, none selected at 2x2
            np.save(Path(directory) / "train_mask2.npy", sparse)
            path = source_manifest(directory, sequence)
            before = path.read_bytes()
            inventory = audit_supervision(path, image_hw=(2, 2), clip_raw_frames=2)
            row = inventory["frames"][2]
            self.assertEqual(row["coverage"]["raw"]["valid_pixels"], 4)
            self.assertEqual(row["coverage"]["cropped"]["valid_pixels"], 4)
            self.assertEqual(row["coverage"]["resized"]["valid_pixels"], 0)
            self.assertEqual(row["coverage"]["final"]["valid_pixels"], 0)
            self.assertEqual(row["status"], "ineligible")
            self.assertIn("native_gt_disappears_after_crop_resize", row["reasons"])
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "Missing real GT coverage"):
                SequenceDataset(path, image_hw=(2, 2))[0]

    def test_crop_loss_count_and_explicit_fraction_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = fixture(directory, 2)
            np.save(Path(directory) / "train_rgb0.npy", np.zeros((4, 8, 3), dtype=np.uint8))
            np.save(Path(directory) / "train_depth0.npy", np.ones((4, 8), dtype=np.float32))
            mask = np.zeros((4, 8), dtype=np.uint8)
            mask[1, 0] = 1
            np.save(Path(directory) / "train_mask0.npy", mask)
            inventory = audit_supervision(source_manifest(directory, sequence), image_hw=(2, 2),
                                          clip_raw_frames=2, min_valid_fraction=.5)
            row = inventory["frames"][0]
            self.assertEqual(row["coverage"]["crop_tlhw"], [0, 2, 4, 4])
            self.assertEqual(row["coverage"]["raw"]["valid_pixels"], 1)
            self.assertEqual(row["coverage"]["cropped"]["valid_pixels"], 0)
            self.assertIn("below_min_valid_fraction", row["reasons"])
            self.assertFalse(inventory["default_threshold_is_crash_prevention_only"])

    def test_bad_arrays_missing_files_empty_gt_do_not_hide_later_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = fixture(directory, 6)
            np.save(Path(directory) / "train_depth0.npy", np.ones((4, 4, 2), dtype=np.float32))
            np.save(Path(directory) / "train_mask1.npy", np.zeros((4, 4), dtype=np.uint8))
            (Path(directory) / "train_depth2.npy").unlink()
            inventory = audit_supervision(source_manifest(directory, sequence), image_hw=(4, 4), clip_raw_frames=3)
            self.assertEqual([row["status"] for row in inventory["frames"]],
                             ["error", "ineligible", "error", "eligible", "eligible", "eligible"])
            self.assertEqual(len(inventory["failures"]), 3)
            self.assertEqual(inventory["train_clip_count"], 1)

    def test_final_validity_excludes_nan_points_and_nonpositive_optical_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = fixture(directory, 2, first_camera=True)
            xyz = np.load(Path(directory) / "train_xyz1.npy")
            xyz[0, 0] = np.nan
            xyz[0, 1, 2] = -1
            np.save(Path(directory) / "train_xyz1.npy", xyz)
            path = source_manifest(directory, sequence)
            inventory = audit_supervision(path, image_hw=(4, 4), clip_raw_frames=2)
            target = SequenceDataset(path)[0].targets[1]
            self.assertEqual(inventory["frames"][1]["coverage"]["final"]["valid_pixels"],
                             int(target["valid_mask"].sum()))
            self.assertEqual(int(target["valid_mask"].sum()), 14)

    def test_empty_malformed_entries_and_duplicate_source_ids_are_disclosed(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = fixture(directory, 5)
            sequence["frames"][0] = {}
            sequence["frames"][1] = None
            sequence["frames"][3]["frame_id"] = sequence["frames"][2]["frame_id"]
            empty = {**deepcopy(sequence), "id": "empty", "frames": []}
            path = source_manifest(directory, sequence, empty)
            inventory = audit_supervision(path, image_hw=(4, 4), clip_raw_frames=2)
            self.assertEqual(inventory["frames"][0]["status"], "error")
            self.assertEqual(inventory["frames"][1]["status"], "error")
            # Missing original first pose is recorded on later entries too;
            # there is no guessed identity camera to make this sequence train.
            self.assertTrue(inventory["sequence_reports"][1]["empty_or_too_short_sequence"])
            self.assertEqual(inventory["sequence_failures"][0]["sequence_id"], "empty")
            sequence["frames"][0] = fixture(directory, 1)["frames"][0]
            path.write_text(json.dumps({"schema_version": 1, "sequences": [sequence]}), encoding="utf-8")
            inventory = audit_supervision(path, image_hw=(4, 4), clip_raw_frames=2)
            self.assertIn("duplicate_source_frame_id", inventory["frames"][2]["reasons"])
            self.assertIn("duplicate_source_frame_id", inventory["frames"][3]["reasons"])


class TrainingManifestTests(unittest.TestCase):
    def test_train_only_runs_preserve_relocated_paths_and_first_camera_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir, output_dir = Path(directory) / "source", Path(directory) / "elsewhere"
            source_dir.mkdir()
            sequence = fixture(source_dir, 5, first_camera=True)
            np.save(source_dir / "train_mask0.npy", np.zeros((4, 4), dtype=np.uint8))
            val = fixture(source_dir, 3, split="validation", prefix="val")
            test = fixture(source_dir, 3, split="test", prefix="test")
            path = source_manifest(source_dir, sequence, val, test)
            inventory = audit_supervision(path, image_hw=(4, 4), clip_raw_frames=3)
            derived = build_training_manifest(path, inventory)
            destination = output_dir / "train.json"
            write_json_new(destination, derived)
            summary = validate_supervision_manifest(destination, path, (4, 4), clip_frames=3)
            self.assertEqual(summary["train_frames"], 4)
            self.assertEqual(summary["train_clips"], 2)
            self.assertEqual([s["split"] for s in derived["sequences"]], ["train"])
            self.assertEqual(derived["split_protocol"], {"name": "synthetic patient split"})
            excluded = derived["training_supervision"]["excluded_frames"]
            self.assertEqual(len([row for row in excluded if row["reason"] == "held_out_split_preserved"]), 6)
            stream = SequenceDataset(destination, sequence_raw_frames=3, image_hw=(4, 4)).stream(0)
            self.assertEqual(stream.targets[0]["source_frame_id"], 1)
            self.assertAlmostEqual(float(stream.targets[0]["pointmap"][0, 0, 0, 0]), 0.)
            self.assertAlmostEqual(float(stream.targets[1]["pointmap"][0, 0, 0, 0]), 1.)
            self.assertTrue(torch.equal(stream.original_first_pose, torch.eye(4)))

    def test_source_id_gap_breaks_runs_and_short_runs_are_reported_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = fixture(directory, 7)
            for index in range(3, 7):
                sequence["frames"][index]["frame_id"] += 2
            inventory = audit_supervision(source_manifest(directory, sequence), image_hw=(4, 4), clip_raw_frames=4)
            self.assertEqual([r["length"] for r in inventory["sequence_reports"][0]["valid_runs"]], [3, 4])
            derived = build_training_manifest(Path(directory) / "manifest.json", inventory)
            self.assertEqual(derived["sequences"][0]["supervision_source_indices"], [3, 4, 5, 6])
            excluded = derived["training_supervision"]["excluded_frames"]
            self.assertEqual([row["reason"] for row in excluded], ["valid_run_shorter_than_clip"] * 3)

    def test_stride_is_applied_once_and_validation_rejects_double_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = fixture(directory, 7)
            path = source_manifest(directory, sequence)
            inventory = audit_supervision(path, image_hw=(4, 4), source_stride=2, clip_raw_frames=3)
            derived = build_training_manifest(path, inventory)
            self.assertEqual(derived["sequences"][0]["supervision_source_indices"], [0, 2, 4, 6])
            destination = Path(directory) / "train.json"
            write_json_new(destination, derived)
            validate_supervision_manifest(destination, path, (4, 4), clip_frames=3)
            with self.assertRaisesRegex(ValueError, "stride=1"):
                validate_supervision_manifest(destination, path, (4, 4), stride=2)
            self.assertEqual(SequenceDataset(destination).stream(0).frame_count, 4)

    def test_no_eligible_run_cannot_generate_fake_or_too_short_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = source_manifest(directory, fixture(directory, 3))
            inventory = audit_supervision(path, image_hw=(4, 4), clip_raw_frames=9)
            self.assertEqual(inventory["train_clip_count"], 0)
            with self.assertRaisesRegex(ValueError, "No contiguous"):
                build_training_manifest(path, inventory)

    def test_source_hash_resolution_gt_metadata_and_split_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = source_manifest(directory, fixture(directory, 3))
            inventory = audit_supervision(path, image_hw=(4, 4), clip_raw_frames=3)
            derived = build_training_manifest(path, inventory)
            destination = Path(directory) / "train.json"
            write_json_new(destination, derived)
            with self.assertRaisesRegex(ValueError, "resolution"):
                validate_supervision_manifest(destination, path, (2, 2))
            for key, value in (("depth_scale", 1000.), ("frame_id", 999)):
                modified = deepcopy(derived)
                modified["sequences"][0]["frames"][0][key] = value
                destination.write_text(json.dumps(modified), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed original"):
                    validate_supervision_manifest(destination, path, (4, 4))
            modified = deepcopy(derived)
            modified["sequences"][0]["split"] = "test"
            destination.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only original TRAIN"):
                validate_supervision_manifest(destination, path, (4, 4))
            destination.write_text(json.dumps(derived), encoding="utf-8")
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_supervision_manifest(destination, path, (4, 4))

    def test_original_patient_leak_is_rejected_before_train_only_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            train, test = fixture(directory, 3), fixture(directory, 3, split="test", prefix="test")
            test["patient_id"] = train["patient_id"]
            with self.assertRaisesRegex(ValueError, "leaks"):
                audit_supervision(source_manifest(directory, train, test), image_hw=(4, 4), clip_raw_frames=3)

    def test_no_overwrite_source_unchanged_and_cli_writes_failure_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = source_manifest(directory, fixture(directory, 3))
            before = manifest_sha256(path)
            report, train = Path(directory) / "audit.json", Path(directory) / "train.json"
            with redirect_stdout(StringIO()):
                status = main(["--manifest", str(path), "--report", str(report),
                               "--training-manifest", str(train), "--image-height", "4", "--image-width", "4",
                               "--clip-raw-frames", "9"])
            self.assertEqual(status, 2)
            self.assertTrue(report.is_file())
            self.assertFalse(train.exists())
            self.assertEqual(manifest_sha256(path), before)
            self.assertIn("training_manifest_error", json.loads(report.read_text(encoding="utf-8")))
            with self.assertRaises(FileExistsError):
                write_json_new(report, {})


if __name__ == "__main__":
    unittest.main()
