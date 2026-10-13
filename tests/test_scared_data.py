"""Synthetic CPU fixtures only: no dataset/checkpoint acquisition or model run."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dust3r.data.scared import build_scared_manifest, discover_source_index, manifest_summary, _required_file
from dust3r.training.sequence_dataset import SequenceDataset, StreamingSequenceSample


def source_fixture(directory, dataset_id=1, frame_count=5, points=False):
    directory = Path(directory)
    frames = []
    for i in range(frame_count):
        np.save(directory / f"rgb_{i:03}.npy", np.full((4, 6, 3), i, dtype=np.uint8))
        pose = np.eye(4)
        pose[0, 3] = i * 1000
        frame = {"frame_id": i, "rgb": f"rgb_{i:03}.npy", "pose": pose.tolist()}
        if points:
            xyz = np.zeros((4, 6, 3), dtype=np.float32)
            xyz[..., 2] = 1000
            np.save(directory / f"xyz_{i:03}.npy", xyz)
            frame["pointmap"] = f"xyz_{i:03}.npy"
        else:
            np.save(directory / f"depth_{i:03}.npy", np.full((4, 6), 1000, dtype=np.float32))
            frame["depth"] = f"depth_{i:03}.npy"
        frames.append(frame)
    sequence = {"id": f"dataset{dataset_id}_keyframe1", "dataset_id": dataset_id,
                "patient_id": f"patient{dataset_id}", "root": ".", "frames": frames,
                "intrinsics": [[100, 0, 2], [0, 100, 1], [0, 0, 1]],
                "calibration": {"verified": True, "rgb_rectified": True, "gt_rectified": True,
                                "pixel_alignment_verified": True, "distortion": [],
                                "source": "synthetic K", "alignment_source": "synthetic same-pixel sensor"},
                "pose": {"verified": True, "convention": "camera_to_world",
                         "translation_units": "mm", "camera_axes": "opencv"},
                "geometry": {"verified": True, "source": "synthetic metre scene encoded mm",
                             "units": "mm", "space": "camera",
                             "valid_mask_rule": "camera_xyz_positive_z_finite" if points else "depth_positive_finite"},
                "frame_alignment_verified": True}
    return {"schema_version": 1, "sequences": [sequence]}


class ScaredManifestTests(unittest.TestCase):
    def test_cached_parent_resolution_preserves_directory_and_file_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").mkdir()
            (root / "real" / "target.npy").write_bytes(b"fixture")
            try:
                (root / "alias").symlink_to(root / "real", target_is_directory=True)
                (root / "real" / "linked.npy").symlink_to(root / "real" / "target.npy")
            except OSError as exc:
                if getattr(exc, 'winerror', None) == 1314:
                    self.skipTest('Windows session lacks the privilege to create symlinks; validate this test on Linux/server')
                raise
            cache = {}
            for name in ("alias/target.npy", "alias/linked.npy"):
                self.assertEqual(_required_file(root, name, "target", cache), root / "real" / "target.npy")
            with self.assertRaises(FileNotFoundError):
                _required_file(root, "alias/missing.npy", "target", cache)

    def test_depth_units_c2w_and_split_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_fixture(directory)
            output = Path(directory) / "out" / "manifest.json"
            manifest = build_scared_manifest(directory, source, output, verify_arrays=True)
            self.assertEqual(manifest["sequences"][0]["split"], "train")
            self.assertEqual(manifest["split_protocol"]["assignments"]["7"], "validation")
            self.assertEqual(manifest["split_protocol"]["assignments"]["9"], "test")
            self.assertEqual(manifest["sequences"][0]["frames"][1]["camera_to_world"][0][3], 1.)
            sample = SequenceDataset(output)[0]
            self.assertTrue(torch.equal(sample.targets[0]["depth"], torch.ones(1, 4, 6)))
            self.assertAlmostEqual(float(sample.targets[1]["pointmap"][0, 0, 0, 0]), .98, places=5)
            self.assertEqual(manifest_summary(manifest)["total_frames"], 5)

    def test_world_to_camera_direction_is_explicitly_inverted(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_fixture(directory)
            source["sequences"][0]["pose"]["convention"] = "world_to_camera"
            manifest = build_scared_manifest(directory, source, Path(directory) / "manifest.json", dry_run=True)
            self.assertEqual(manifest["sequences"][0]["frames"][1]["camera_to_world"][0][3], -1.)

    def test_disparity_encoding_and_metric_stereo_metadata_survive_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_fixture(directory)
            sequence = source["sequences"][0]
            sequence["geometry"].update(units="m", stored_units="pixels")
            for frame in sequence["frames"]:
                np.save(Path(directory) / frame["depth"], np.full((4, 6), 4., dtype=np.float32))
                frame.update(depth_encoding="rectified_disparity", stereo_baseline_m=.004,
                             right_principal_x=0.)
            output = Path(directory) / "manifest.json"
            manifest = build_scared_manifest(directory, source, output, verify_arrays=True)
            self.assertEqual(manifest["sequences"][0]["frames"][0]["depth_encoding"], "rectified_disparity")
            torch.testing.assert_close(SequenceDataset(output)[0].targets[0]["depth"], torch.full((1, 4, 6), .2))
            sequence["geometry"]["stored_units"] = "mm"
            with self.assertRaisesRegex(ValueError, "stored_units"):
                build_scared_manifest(directory, source, Path(directory) / "bad.json", dry_run=True)

    def test_missing_pose_convention_cannot_be_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_fixture(directory)
            del source["sequences"][0]["pose"]["convention"]
            with self.assertRaisesRegex(ValueError, "pose.convention"):
                build_scared_manifest(directory, source, Path(directory) / "manifest.json", dry_run=True)

    def test_unrectified_or_unsynchronized_declaration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_fixture(directory)
            source["sequences"][0]["calibration"]["rgb_rectified"] = False
            with self.assertRaisesRegex(ValueError, "rgb_rectified"):
                build_scared_manifest(directory, source, Path(directory) / "manifest.json", dry_run=True)
            source["sequences"][0]["calibration"]["rgb_rectified"] = True
            source["sequences"][0]["frame_alignment_verified"] = False
            with self.assertRaisesRegex(ValueError, "frame_alignment_verified"):
                build_scared_manifest(directory, source, Path(directory) / "manifest.json", dry_run=True)

    def test_patient_leak_is_rejected_even_when_frame_ids_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_fixture(directory)
            second = deepcopy(source["sequences"][0])
            second.update(id="dataset8_keyframe1", dataset_id=8)
            source["sequences"].append(second)
            with self.assertRaisesRegex(ValueError, "leaks"):
                build_scared_manifest(directory, source, Path(directory) / "manifest.json", dry_run=True)

    def test_dry_run_never_creates_output_or_decodes_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_fixture(directory)
            output = Path(directory) / "must_not_exist" / "manifest.json"
            before = sorted(str(p) for p in Path(directory).rglob("*"))
            with patch.object(SequenceDataset, "_array", side_effect=AssertionError("Decoded data in dry run")):
                build_scared_manifest(directory, source, output, dry_run=True)
            self.assertFalse(output.parent.exists())
            self.assertEqual(before, sorted(str(p) for p in Path(directory).rglob("*")))

    def test_pointmap_units_and_invalid_values_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_fixture(directory, points=True)
            xyz = np.load(Path(directory) / "xyz_000.npy")
            xyz[0, 0, 2] = np.nan
            np.save(Path(directory) / "xyz_000.npy", xyz)
            output = Path(directory) / "manifest.json"
            build_scared_manifest(directory, source, output)
            target = SequenceDataset(output)[0].targets[0]
            self.assertFalse(target["valid_mask"][0, 0, 0])
            self.assertAlmostEqual(float(target["pointmap"][0, 1, 1, 2]), 1.)
            self.assertAlmostEqual(float(target["depth"][0, 1, 1]), 1.)

    def test_output_overwrite_and_duplicate_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_fixture(directory)
            output = Path(directory) / "manifest.json"
            build_scared_manifest(directory, source, output)
            with self.assertRaises(FileExistsError):
                build_scared_manifest(directory, source, output)
            source["sequences"][0]["frames"][1]["frame_id"] = 0
            with self.assertRaisesRegex(ValueError, "unique"):
                build_scared_manifest(directory, source, output, dry_run=True)

    def test_video_dry_run_requires_explicit_left_layout_without_importing_cv2(self):
        with tempfile.TemporaryDirectory() as directory:
            source = source_fixture(directory)
            (Path(directory) / "rgb.mp4").write_bytes(b"mock, not a real video")
            for frame in source["sequences"][0]["frames"]:
                del frame["rgb"]
                frame.update(rgb_video="rgb.mp4", rgb_frame_index=frame["frame_id"])
            source["sequences"][0]["rgb_video_layout"] = "top_bottom_left_top"
            output = Path(directory) / "out" / "manifest.json"
            with patch("dust3r.data.scared._video_frame", side_effect=AssertionError("Video decoded in dry run")):
                manifest = build_scared_manifest(directory, source, output, dry_run=True, extract_video=True)
            self.assertFalse(output.parent.exists())
            self.assertIn("scared_left_rgb", manifest["sequences"][0]["frames"][0]["rgb"])


class StreamingTests(unittest.TestCase):
    def test_stream_keeps_only_one_pair_and_loads_on_demand(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            build_scared_manifest(directory, source_fixture(directory), output)
            dataset = SequenceDataset(output)
            with patch.object(dataset, "_array", wraps=dataset._array) as load:
                stream = dataset.stream(0)
                self.assertIsInstance(stream, StreamingSequenceSample)
                self.assertEqual(load.call_count, 0)
                frame, target = stream.get_frame(0)
                self.assertEqual(load.call_count, 2)
                self.assertIs(stream.frames[0], frame)
                self.assertIs(stream.targets[0], target)
                self.assertEqual(load.call_count, 2)
                stream.get_frame(1)
                self.assertEqual(load.call_count, 4)
                self.assertEqual(stream._cached_index, 1)
                stream.clear_cache()
                self.assertIsNone(stream._cached_pair)

    def test_stride_clip_origin_is_selected_first_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            build_scared_manifest(directory, source_fixture(directory), output)
            dataset = SequenceDataset(output, stride=2, sequence_raw_frames=2)
            self.assertEqual(len(dataset), 2)
            stream = dataset.stream(1)  # original source frames 2,4, not 1,2
            first, second = stream.targets[0], stream.targets[1]
            self.assertEqual(first["source_frame_id"], 2)
            self.assertEqual(second["source_frame_id"], 4)
            self.assertAlmostEqual(float(first["pointmap"][0, 0, 0, 0]), -.02, places=5)
            self.assertAlmostEqual(float(second["pointmap"][0, 0, 0, 0]), 1.98, places=5)
            self.assertEqual(float(first["first_camera_to_world"][0, 0, 3]), 2.)

    def test_stream_materialized_equivalence_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            build_scared_manifest(directory, source_fixture(directory), output)
            dataset = SequenceDataset(output, image_hw=(2, 3))
            materialized = dataset[0]
            stream = dataset.stream(0)
            for i, (frame, target) in enumerate(stream.iter_frames()):
                self.assertTrue(torch.equal(frame["img"], materialized.frames[i]["img"]))
                self.assertTrue(torch.equal(target["pointmap"], materialized.targets[i]["pointmap"]))
            fresh = dataset.stream(0)
            self.assertIsNone(fresh._cached_pair)
            with self.assertRaises(IndexError):
                fresh.get_frame(5)

    def test_prefix_preserves_indices_and_has_independent_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            build_scared_manifest(directory, source_fixture(directory), output)
            stream = SequenceDataset(output, stride=2).stream(0)
            stream.get_frame(0)
            prefix = stream.prefix(2)
            self.assertEqual(prefix.frame_count, 2)
            self.assertEqual(stream.frame_count, 3)
            self.assertIsNone(prefix._cached_pair)
            self.assertEqual(prefix.source_frame_id(1), 2)
            self.assertTrue(torch.equal(prefix.targets[0]["pointmap"], stream.targets[0]["pointmap"]))
            self.assertEqual(prefix.episode_id, stream.episode_id)
            with self.assertRaises(ValueError):
                stream.prefix(1)


class NativeMappingTests(unittest.TestCase):
    def test_explicit_glob_json_field_discovery_and_exact_join(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / "dataset1" / "keyframe1"
            native.mkdir(parents=True)
            source = source_fixture(native)
            defaults = {k: v for k, v in source["sequences"][0].items()
                        if k not in {"id", "dataset_id", "patient_id", "root", "frames"}}
            for frame in source["sequences"][0]["frames"]:
                (native / f"metadata_{frame['frame_id']:03}.json").write_text(
                    json.dumps({"camera": {"pose": frame["pose"]}}), encoding="utf-8")
            mapping = {"schema_version": 1, "defaults": defaults,
                       "patterns": {"sequence_glob": "dataset*/keyframe*", "dataset_regex": r"dataset(\d+)",
                                    "patient_ids": {"1": "verified_patient1"}, "index_regex": r"_(\d+)\.",
                                    "rgb_glob": "rgb_*.npy", "target_type": "depth", "target_glob": "depth_*.npy",
                                    "metadata_glob": "metadata_*.json", "pose_field": "camera.pose"}}
            index = discover_source_index(directory, mapping)
            self.assertEqual(len(index["sequences"][0]["frames"]), 5)
            self.assertEqual(index["sequences"][0]["patient_id"], "verified_patient1")
            output = Path(directory) / "manifest.json"
            build_scared_manifest(directory, index, output, verify_arrays=True)
            (native / "rgb_004.npy").unlink()
            with self.assertRaisesRegex(ValueError, "ids differ"):
                discover_source_index(directory, mapping)


if __name__ == "__main__":
    unittest.main()
