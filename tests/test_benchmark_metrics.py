"""Synthetic CPU metrics/protocol checks. No dataset or checkpoint download."""
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from dust3r.evaluation.camera import (CameraPredictionError, PredictedCamera,
                                     estimate_intrinsics, predict_camera)
from dust3r.evaluation.metrics import depth_errors, pose_window_metrics
from dust3r.evaluation.reports import make_report, save_report
from dust3r.evaluation.streaming import StreamingSequenceEvaluator, aggregate_sequences


def pointmap(depth=1., height=8, width=10, focal=12.):
    yy, xx = np.mgrid[:height, :width]
    depth = np.broadcast_to(np.asarray(depth), (height, width))
    return np.stack(((xx - width / 2) * depth / focal,
                     (yy - height / 2) * depth / focal, depth), -1)


def target(depth=1., height=8, width=10, pose=None):
    return {"depth": np.broadcast_to(np.asarray(depth), (height, width)).copy(),
            "valid_mask": np.ones((height, width), dtype=bool),
            "camera_to_world": np.eye(4) if pose is None else pose,
            "intrinsics": np.diag([999., 999., 1.]),
            "provenance_verified": True}


def identity_camera(points, intrinsic, **kwargs):
    return PredictedCamera(intrinsic.copy(), np.eye(4), np.eye(4), len(points.reshape(-1, 3)))


class CameraTests(unittest.TestCase):
    def test_weizfeld_focal_uses_pointmap_only(self):
        estimated = estimate_intrinsics(pointmap())
        np.testing.assert_allclose(estimated, [[12., 0., 5.], [0., 12., 4.], [0., 0., 1.]], atol=1e-10)
        self.assertNotIn("target", inspect.signature(predict_camera).parameters)
        self.assertNotIn("pose", inspect.signature(predict_camera).parameters)

    def test_degenerate_and_nonfinite_focal_fail(self):
        with self.assertRaises(CameraPredictionError):
            estimate_intrinsics(np.zeros((8, 10, 3)))
        invalid = pointmap()
        invalid[0, 0, 0] = np.nan
        with self.assertRaises(CameraPredictionError):
            estimate_intrinsics(invalid)

    def test_pnp_failure_is_explicit_and_not_identity(self):
        class FailedCV2:
            SOLVEPNP_ITERATIVE = 0
            @staticmethod
            def solvePnPRansac(*args, **kwargs):
                return False, None, None, None
        with self.assertRaisesRegex(CameraPredictionError, "no GT or identity fallback"):
            predict_camera(pointmap(), cv2_module=FailedCV2)

    def test_mock_pnp_uses_dense_points_and_released_parameters(self):
        captured = {}
        class FakeCV2:
            SOLVEPNP_ITERATIVE = 0
            @staticmethod
            def solvePnPRansac(*args, **kwargs):
                captured.update(kwargs)
                captured["points"] = args[0].shape
                return True, np.zeros(3), np.array([0., 0., 2.]), np.arange(80)
            @staticmethod
            def Rodrigues(rotation):
                return np.eye(3), None
        camera = predict_camera(pointmap(), cv2_module=FakeCV2)
        self.assertEqual(captured["points"], (80, 3))
        self.assertEqual(captured["iterationsCount"], 100)
        self.assertEqual(captured["reprojectionError"], 2.)
        np.testing.assert_allclose(camera.camera_to_world[:3, 3], [0., 0., -2.])


class DepthTests(unittest.TestCase):
    def test_perfect_and_hand_computed_depth(self):
        perfect = depth_errors([1., 2.], [1., 2.])
        self.assertEqual(perfect["abs_rel"], 0.)
        self.assertEqual(perfect["delta3"], 1.)
        measured = depth_errors([1., 2.], [2., 4.])
        self.assertEqual(measured["abs_rel"], 1.)
        self.assertEqual(measured["sq_rel"], 1.5)
        self.assertAlmostEqual(measured["rmse_m"], math.sqrt(2.5))
        self.assertAlmostEqual(measured["rmse_log"], math.log(2.))

    def test_sequence_median_not_independent_per_frame_scale(self):
        evaluator = StreamingSequenceEvaluator("median", expected_frames=2, camera_predictor=identity_camera, pose=False)
        evaluator.add_frame(0, pointmap(2.), target(1.))
        evaluator.add_frame(1, pointmap(2.), target(3.))
        report = evaluator.finalize()
        self.assertEqual(report["scale_ratio"], 1.)
        self.assertAlmostEqual(report["depth_metrics"]["abs_rel"], (1. + 1. / 3.) / 2.)
        self.assertEqual(evaluator.depth_vectors, [])

    def test_gt_valid_samples_define_median_and_prediction_clipping(self):
        evaluator = StreamingSequenceEvaluator("mask", expected_frames=2, camera_predictor=identity_camera, pose=False)
        first_target = target()
        first_target["valid_mask"][0, 0] = False
        first = pointmap(2.)
        first[0, 0, 2] = 5000.
        evaluator.add_frame(0, first, first_target)
        second = pointmap(2.)
        second[0, 0, 2] = -2.
        evaluator.add_frame(1, second, target())
        report = evaluator.finalize()
        self.assertAlmostEqual(report["scale_ratio"], .5)
        self.assertEqual(report["frame_metrics"][0]["gt_valid_pixel_count"], 79)
        self.assertGreater(report["frame_metrics"][1]["nonpositive_predicted_depth_ratio"], 0.)
        self.assertTrue(np.isfinite(report["depth_metrics"]["rmse_log"]))

    def test_prediction_camera_never_receives_gt_K_or_pose(self):
        intrinsics = []
        def spy(points, intrinsic, **kwargs):
            intrinsics.append(intrinsic.copy())
            return identity_camera(points, intrinsic)
        evaluator = StreamingSequenceEvaluator("no_leak", expected_frames=2, camera_predictor=spy, pose=False)
        gt1, gt2 = target(), target()
        gt2["intrinsics"] *= 100
        gt2["camera_to_world"][0, 3] = 100
        evaluator.add_frame(0, pointmap(), gt1)
        evaluator.add_frame(1, pointmap(), gt2)
        self.assertEqual(evaluator.finalize()["depth_metrics"]["abs_rel"], 0.)
        np.testing.assert_allclose(intrinsics[0], intrinsics[1])
        self.assertAlmostEqual(intrinsics[0][0, 0], 12.)

    def test_nonfinite_output_counts_failure_outside_gt_mask(self):
        evaluator = StreamingSequenceEvaluator("failed", expected_frames=2, camera_predictor=identity_camera, pose=False)
        evaluator.add_frame(0, pointmap(), target())
        output, gt = pointmap(), target()
        output[0, 0, 0] = np.nan
        gt["valid_mask"][0, 0] = False
        evaluator.add_frame(1, output, gt)
        report = evaluator.finalize()
        self.assertEqual(report["failed_frame_count"], 1)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["failures"][0]["frame"], 1)
        json.dumps(report, allow_nan=False)

    def test_first_camera_failure_cannot_reinitialize_on_future_frame(self):
        evaluator = StreamingSequenceEvaluator("first_failed", expected_frames=2, camera_predictor=identity_camera, pose=False)
        evaluator.add_frame(0, np.zeros((8, 10, 3)), target())
        evaluator.add_frame(1, pointmap(), target())
        report = evaluator.finalize()
        self.assertEqual(report["failed_frame_count"], 2)
        self.assertIsNone(report["depth_metrics"]["abs_rel"])

    def test_exact_frame_coverage_and_order_enforced(self):
        evaluator = StreamingSequenceEvaluator("order", expected_frames=2, camera_predictor=identity_camera)
        with self.assertRaises(ValueError):
            evaluator.add_frame(1, pointmap(), target())
        evaluator.add_frame(0, pointmap(), target())
        with self.assertRaises(ValueError):
            evaluator.finalize()


class PoseAndReportTests(unittest.TestCase):
    def test_identity_and_perfect_scaled_trajectory(self):
        stationary = np.repeat(np.eye(4)[None], 5, axis=0)
        result = pose_window_metrics(stationary, stationary)
        self.assertEqual(result["ate_mm"], 0.)
        self.assertEqual(result["rpe_rotation_deg"], 0.)
        gt = stationary.copy()
        gt[:, :3, 3] = [[0., 0., 0.], [.01, 0., 0.], [.02, .01, 0.], [.03, -.01, .01], [.04, 0., 0.]]
        predicted = gt.copy()
        predicted[:, :3, 3] = gt[:, :3, 3] * 3 + [1., 2., 3.]
        result = pose_window_metrics(predicted, gt)
        self.assertLess(result["ate_mm"], 1e-9)
        self.assertLess(result["rpe_translation_mm"], 1e-9)
        self.assertLess(result["rpe_rotation_deg"], 1e-5)
        self.assertAlmostEqual(result["scale"], 1. / 3.)

    def test_collinear_pose_orientation_fixes_rotation_ambiguity(self):
        gt = np.repeat(np.eye(4)[None], 5, axis=0)
        gt[:, 0, 3] = np.arange(5) * .01
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        predicted = gt.copy()
        predicted[:, :3, :3] = rotation
        predicted[:, :3, 3] = 2 * gt[:, :3, 3] @ rotation.T + [1., 2., 3.]
        result = pose_window_metrics(predicted, gt)
        self.assertTrue(result["degenerate_alignment"])
        self.assertLess(result["ate_mm"], 1e-9)
        self.assertLess(result["rpe_rotation_deg"], 1e-5)

    def sequence(self, count, delta=0):
        evaluator = StreamingSequenceEvaluator(str(count), expected_frames=count, camera_predictor=identity_camera)
        for index in range(count):
            evaluator.add_frame(index, pointmap(1. + delta * (index % 2)), target())
        return evaluator.finalize()

    def test_five_frame_windows_failure_counts_and_frame_weighting(self):
        first, second = self.sequence(5), self.sequence(6, .2)
        aggregate = aggregate_sequences([first, second])
        self.assertEqual(aggregate["frame_count"], 11)
        self.assertEqual(aggregate["pose_window_count"], 3)
        self.assertAlmostEqual(aggregate["depth_metrics"]["abs_rel"], second["depth_metrics"]["abs_rel"] * 6 / 11)
        different = deepcopy(second)
        different["scale_protocol"] = "different"
        with self.assertRaises(ValueError):
            aggregate_sequences([first, different])

    def test_report_json_fingerprints_and_units(self):
        report = make_report([self.sequence(5)], config={"baseline_identity": "SYNTHETIC_TEST"},
                 manifest_sha256="a" * 64, checkpoint_hashes={"test": "b" * 64}, split="test",
                 image_hw=[8, 10], seed=1, experiment_id="cpu_synthetic")
        self.assertFalse(report["paper_exact_protocol_claim"])
        self.assertEqual(report["sequences"][0]["pose_units"]["translation"], "mm")
        self.assertEqual(len(report["frame_indices_sha256"]), 64)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "report.json"
            save_report(report, file)
            self.assertEqual(json.loads(file.read_text())["aggregate"]["failed_frame_count"], 0)
        report["timing"] = {"fps": float("nan")}
        with self.assertRaises(ValueError):
            save_report(report, Path("unused.json"))


class StreamIntegrationTests(unittest.TestCase):
    def test_online_evaluation_uses_final_left_and_terminal_right_without_GT_inputs(self):
        import torch
        from unittest.mock import patch
        from eval.eval_benchmark import evaluate_stream
        class Sample:
            episode_id, frame_count, provenance = "synthetic", 3, {"source": "synthetic_test_only"}
            def get_frame(self, index):
                return {"img": torch.zeros(1, 3, 8, 10), "idx": index, "instance": f"synthetic:{index}"}, target()
        class Runner:
            def __init__(self):
                self.model = torch.nn.Linear(1, 1)
                self.seen = []
                self.state = SimpleNamespace(write_count=1, raft_call_count=0, kda_update_count=1)
            def step(self, frame, terminal=False):
                self.seen.append(set(frame))
                if frame["idx"] == 0:
                    return None
                return SimpleNamespace(res1={"pts3d": pointmap()}, res2={"pts3d": pointmap()}, terminal=terminal)
        runner = Runner()
        original = StreamingSequenceEvaluator
        def configured(*args, **kwargs):
            kwargs["camera_predictor"] = identity_camera
            return original(*args, **kwargs)
        with patch("dust3r.evaluation.StreamingSequenceEvaluator", side_effect=configured):
            result = evaluate_stream(Sample(), runner=runner, pose=False)
        self.assertEqual(result["scored_frame_count"], 3)
        self.assertEqual(result["controller_counts"]["write_count"], 1)
        self.assertTrue(all(keys == {"img", "idx", "instance"} for keys in runner.seen))
        self.assertEqual(result["depth_metrics"]["abs_rel"], 0.)


if __name__ == "__main__":
    unittest.main()
