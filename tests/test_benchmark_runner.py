"""Synthetic throughput accounting, never GPU or published performance."""
from types import SimpleNamespace
from copy import deepcopy
import unittest

import numpy as np
import torch

from dust3r.evaluation.camera import PredictedCameraEstimator, PredictedCamera
from dust3r.evaluation.streaming import canonical_sha256
from dust3r.experiments.benchmark import (aggregate_performance, benchmark_sequence,
                                         canonical_hash, validate_actor_component_profile)


class ManualClock:
    def __init__(self):
        self.time = 0.
    def __call__(self):
        return self.time
    def advance(self, milliseconds):
        self.time += milliseconds / 1000


class Sample:
    frame_count, episode_id = 4, "cpu_synthetic_only"
    def __init__(self, clock):
        self.clock = clock
    def get_frame(self, index):
        self.clock.advance(3)
        return {"img": torch.zeros(1, 3, 8, 10), "index": index}, {"GT_MUST_NOT_ENTER_CAMERA": True}


class Runner:
    def __init__(self, model, config, frame_count, episode_id, clock, *, fail_at=None, no_output=False):
        self.model, self.clock = model, clock
        self.state = SimpleNamespace(kda_state=torch.zeros(1, 4, 8, 8))
        self.fail_at, self.no_output = fail_at, no_output
        self.last_commit_info = {}
    def step(self, frame, terminal=False):
        self.clock.advance(10)
        if frame["index"] == 0 or self.no_output:
            return None
        if frame["index"] == self.fail_at:
            raise FloatingPointError("synthetic failure")
        self.last_commit_info = {"requested_write": not terminal, "successful_write": not terminal,
                                 "raft_called": frame["index"] == 2, "kda_updated": not terminal}
        return SimpleNamespace(res1={"point": frame["index"] - 1},
                               res2={"point": frame["index"]}, terminal=terminal,
                               candidate_index=frame["index"] - 1, arrival_index=frame["index"])


class Camera:
    def __init__(self, clock, *, fail_at=None):
        self.clock, self.fail_at, self.inputs = clock, fail_at, []
    def predict(self, output):
        # Camera prediction and depth conversion cost must enter pipeline time.
        self.clock.advance(2)
        self.inputs.append(output)
        if output["point"] == self.fail_at:
            return SimpleNamespace(valid=False, error="synthetic camera failure")
        return SimpleNamespace(valid=True, depth=np.ones((8, 10)))


class BenchmarkAccountingTests(unittest.TestCase):
    def result(self, *, fail_at=None, camera_fail_at=None, no_output=False):
        clock = ManualClock()
        sample = Sample(clock)
        camera = Camera(clock, fail_at=camera_fail_at)
        syncs = []
        model = torch.nn.Linear(1, 1)
        runner_instances = []
        def runner_factory(*args):
            runner = Runner(*args, clock, fail_at=fail_at, no_output=no_output)
            runner_instances.append(runner)
            return runner
        result = benchmark_sequence(model, sample, None, "cpu", runner_factory=runner_factory,
                     camera_factory=lambda: camera, synchronize=lambda: syncs.append(clock()), clock=clock)
        return result, camera, runner_instances, syncs

    def test_camera_depth_time_and_N_output_fps_denominator(self):
        result, camera, _, syncs = self.result()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["output_frames"], 4)
        self.assertEqual(result["pair_count"], 3)
        self.assertAlmostEqual(result["pipeline_ms"], 3 * 10 + 4 * 2)
        self.assertAlmostEqual(result["fps"], 4000 / 38)
        self.assertAlmostEqual(result["input_preparation_ms"], 4 * 3)
        self.assertEqual([value["point"] for value in camera.inputs], [0, 1, 2, 3])
        self.assertTrue(all(set(value) == {"point"} for value in camera.inputs))
        self.assertGreaterEqual(len(syncs), 8)
        self.assertAlmostEqual(result["pair_ms"]["p50_ms"], 12.)
        self.assertAlmostEqual(result["frame_timings"][-1]["amortized_output_latency_ms"], 7.)

    def test_empty_outputs_do_not_report_completed_or_FPS(self):
        result, _, _, _ = self.result(no_output=True)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Incomplete output coverage", result["failure"])
        self.assertIsNone(result["fps"])

    def test_model_and_camera_failure_are_explicit(self):
        for kwargs in ({"fail_at": 2}, {"camera_fail_at": 2}):
            result, _, _, _ = self.result(**kwargs)
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["fps"])
            self.assertTrue(result["failure"])

    def test_failed_sequences_do_not_improve_headline_FPS(self):
        good, _, _, _ = self.result()
        failed, _, _, _ = self.result(fail_at=2)
        aggregate = aggregate_performance([good, failed])
        self.assertFalse(aggregate["coverage_complete"])
        self.assertEqual(aggregate["failed_sequences"], 1)
        self.assertIsNone(aggregate["fps"])
        self.assertAlmostEqual(aggregate["successful_sequences_only_fps"], good["fps"])

    def test_each_sequence_has_new_runner_no_prediction_history_in_report(self):
        result, camera, runners, _ = self.result()
        self.assertEqual(len(runners), 1)
        self.assertEqual(len(result["frame_timings"]), 3)
        self.assertTrue(all(not isinstance(value, torch.Tensor) for value in result.values()))
        self.assertEqual(result["kda_state_bytes"], 1 * 4 * 8 * 8 * 4)
        self.assertEqual(result["eligible_decisions"], 2)
        self.assertEqual(result["write_rate"], 1.)
        self.assertEqual(result["raft_call_rate"], .5)

    def test_performance_config_hash_matches_quality_hash(self):
        values = {"b": [1, 2], "a": {"x": True}}
        self.assertEqual(canonical_hash(values), canonical_sha256(values))

    def test_default_camera_factory_receives_quality_matching_PnP_seed(self):
        from unittest.mock import patch
        clock = ManualClock()
        model = torch.nn.Linear(1, 1)
        camera = Camera(clock)
        with patch("dust3r.evaluation.camera.PredictedCameraEstimator", return_value=camera) as factory:
            result = benchmark_sequence(model, Sample(clock), None, "cpu", pnp_seed=17,
                runner_factory=lambda *args: Runner(*args, clock), clock=clock, synchronize=lambda: None)
        factory.assert_called_once_with(pnp_seed=17)
        self.assertEqual(result["status"], "completed")

    def test_stateful_predicted_camera_performs_depth_conversion_and_bounded_reset(self):
        yy, xx = np.mgrid[:8, :10]
        points = np.stack(((xx - 5) / 12., (yy - 4) / 12., np.ones_like(xx)), -1)
        transform = np.eye(4)
        transform[2, 3] = 2.
        def predictor(points, intrinsic, **kwargs):
            return PredictedCamera(intrinsic, transform, np.linalg.inv(transform), 80)
        camera = PredictedCameraEstimator(camera_predictor=predictor)
        first = camera.predict({"pts3d": points})
        second = camera.predict({"pts3d": points})
        np.testing.assert_allclose(first.depth, 1.)
        np.testing.assert_allclose(second.depth, 3.)
        self.assertEqual(set(vars(camera)), {"camera_predictor", "pnp_seed", "intrinsic", "index", "first_failed"})
        camera.reset()
        self.assertIsNone(camera.intrinsic)
        self.assertEqual(camera.index, 0)


class ActorComponentProfileTests(unittest.TestCase):
    def artifact(self):
        config = {"policy": {"long_write_policy": "ppo"}, "training": {"seed": 0, "new_memory_dtype": "bfloat16"}}
        report = {"schema": "endo3r_component_profile_v1", "config": deepcopy(config),
                  "config_sha256": canonical_hash(config), "hardware": {"hardware_id": "server_gpu"},
                  "timing_scope": "synchronized_exclusive_components_not_headline_throughput",
                  "component_operations": {"policy": "LongWritePolicy.act"},
                  "original_checkpoint_hashes": {"endo3r": "e" * 64, "dust3r": "d" * 64},
                  "hybrid_checkpoint_sha256": "b" * 64, "actor_checkpoint_sha256": "a" * 64,
                  "source_provenance": {"source_tree_sha256": "c" * 64},
                  "exclusive_component_timings": {"policy": {"calls": 6, "mean_ms": .08}}}
        expected = {"config": config, "hardware_id": "server_gpu",
                    "original_checkpoint_hashes": report["original_checkpoint_hashes"],
                    "hybrid_checkpoint_sha256": "b" * 64, "actor_checkpoint_sha256": "a" * 64,
                    "source_tree_sha256": "c" * 64}
        return report, expected

    def test_valid_proxy_is_explicit_and_does_not_require_GT_or_test_frame_hash(self):
        report, expected = self.artifact()
        report.update(split="validation", frame_indices_sha256="different_validation_frames")
        result = validate_actor_component_profile(report, **expected)
        self.assertEqual(result["actor_mean_latency_ms"], .08)
        self.assertEqual(result["actor_profile_calls"], 6)
        self.assertEqual(result["actor_latency_source"], "separate_validation_component_profile_proxy")
        self.assertFalse(result["actor_latency_is_headline_per_frame_measurement"])

    def test_config_hardware_and_weight_mismatches_are_rejected(self):
        original, expected = self.artifact()
        mutations = (
            lambda r: r["config"]["training"].update(new_memory_dtype="float32"),
            lambda r: r.update(config_sha256="wrong"),
            lambda r: r["hardware"].update(hardware_id="another_gpu"),
            lambda r: r.update(original_checkpoint_hashes={"endo3r": "different"}),
            lambda r: r.update(hybrid_checkpoint_sha256="different"),
            lambda r: r.update(actor_checkpoint_sha256="different"),
            lambda r: r["source_provenance"].update(source_tree_sha256="different"),
        )
        for mutate in mutations:
            report = deepcopy(original)
            mutate(report)
            with self.assertRaises(ValueError):
                validate_actor_component_profile(report, **expected)

    def test_only_explicit_separately_synchronized_policy_act_measurements_are_allowed(self):
        original, expected = self.artifact()
        for mutate in (
            lambda r: r.update(schema="unidentified"),
            lambda r: r.update(timing_scope="headline_unsynchronized"),
            lambda r: r.update(component_operations={"policy": "observation"}),
            lambda r: r.update(exclusive_component_timings={}),
        ):
            report = deepcopy(original)
            mutate(report)
            with self.assertRaises(ValueError):
                validate_actor_component_profile(report, **expected)

    def test_calls_and_latency_must_be_real_finite_measurements(self):
        original, expected = self.artifact()
        for calls in (None, 0, -1, True, 1.5):
            report = deepcopy(original)
            report["exclusive_component_timings"]["policy"]["calls"] = calls
            with self.assertRaises(ValueError):
                validate_actor_component_profile(report, **expected)
        for mean in (None, float("nan"), float("inf"), -1., True):
            report = deepcopy(original)
            report["exclusive_component_timings"]["policy"]["mean_ms"] = mean
            with self.assertRaises(ValueError):
                validate_actor_component_profile(report, **expected)

    def test_heuristic_policy_and_missing_actor_checkpoint_cannot_receive_proxy(self):
        report, expected = self.artifact()
        expected["config"]["policy"]["long_write_policy"] = "heuristic"
        with self.assertRaisesRegex(ValueError, "only for the PPO"):
            validate_actor_component_profile(report, **expected)
        report, expected = self.artifact()
        expected["actor_checkpoint_sha256"] = None
        with self.assertRaisesRegex(ValueError, "actor checkpoint"):
            validate_actor_component_profile(report, **expected)

    def test_source_tree_is_verified_when_present(self):
        report, expected = self.artifact()
        expected["source_tree_sha256"] = None
        with self.assertRaisesRegex(ValueError, "source_tree_sha256"):
            validate_actor_component_profile(report, **expected)
        report.pop("source_provenance")
        result = validate_actor_component_profile(report, **expected)
        self.assertFalse(result["actor_source_tree_verified"])


if __name__ == "__main__":
    unittest.main()
