"""Renderer fixtures are explicitly synthetic, never reported as model metrics."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dust3r.experiments.results import join_reports, render_benchmark


def fixture_report(seed=0, experiment="hybrid_sampson", *, performance=False):
    config = {"model": "SYNTHETIC FIXTURE", "memory": {"backend": "math_reference"},
              "training": {"seed": seed, "learning_rate": 1e-4}}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "endo3r_performance_v1" if performance else "endo3r_benchmark_v1", "schema_version": 1,
            "experiment_id": experiment, "seed": seed, "config": config, "config_sha256": config_hash,
            "checkpoint_hashes": {"endo3r": "0" * 64}, "manifest_sha256": "1" * 64,
            "frame_indices_sha256": "2" * 64, "provenance_sha256": "3" * 64,
            "split": "test", "image_hw": [256, 320], "protocol": "synthetic fixture protocol",
            "scale_protocol": "synthetic scale", "prediction_camera_source": "synthetic camera",
            "pose_protocol": "synthetic pose", "actual_backends": {"short": "math_reference", "long": "torch_reference"},
            "metadata": {"is_fixture": True, "source": "synthetic unit test"},
            "aggregate": {"status": "complete", "frame_count": 3, "scored_frame_count": 3,
                          "failed_frame_count": 0, "pose_window_count": 0, "pose_failed_window_count": 0,
                          "depth_metrics": {"abs_rel": .1 + seed * .1, "sq_rel": .01, "rmse_m": .02,
                                            "rmse_log": .2, "delta1": .9}, "pose_metrics": {}},
            "timing": {"fps": 10 + seed * 2, "mean_latency_ms": 100., "p50_latency_ms": 90.,
                       "p95_latency_ms": 120., "peak_vram_mib": 1000., "write_rate": .5,
                       "raft_call_rate": .25, "kda_update_rate": .5, "actor_mean_latency_ms": None,
                       "hardware_id": "SYNTHETIC CPU FIXTURE", "timing_scope": "synthetic_fixture"} if performance else None}


def write_report(directory, name, report):
    path = Path(directory) / name
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    return path


def fixture_paper():
    return {"schema": "endo3r_paper_reference_v1", "source": "paper_reference_not_measured",
            "experiment_id": "Endo3R-paper", "citation_url": "https://arxiv.org/abs/2504.03198",
            "reported_units": {"rmse": "unspecified", "sq_rel": "unspecified"},
            "depth_metrics": {"abs_rel": .124, "sq_rel": .227, "rmse": 1.209,
                              "rmse_log": .135, "delta1": .839}, "pose_metrics": {},
            "timing": {"fps": 19.17}, "hardware": "unspecified"}


class ReportTests(unittest.TestCase):
    def test_join_quality_performance_both_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            quality = write_report(directory, "quality.json", fixture_report())
            timing = write_report(directory, "timing.json", fixture_report(performance=True))
            for paths in ((quality, timing), (timing, quality)):
                report = join_reports(paths)[0]
                self.assertEqual(report["aggregate"]["depth_metrics"]["abs_rel"], .1)
                self.assertEqual(report["timing"]["fps"], 10)
                self.assertEqual(len(report["artifact_paths"]), 2)

    def test_mismatched_checkpoint_selection_or_config_cannot_join(self):
        with tempfile.TemporaryDirectory() as directory:
            quality = write_report(directory, "quality.json", fixture_report())
            for key, replacement in (("checkpoint_hashes", {"endo3r": "changed"}),
                                     ("frame_indices_sha256", "changed"), ("manifest_sha256", "changed"),
                                     ("prediction_camera_source", "GT camera instead of predicted")):
                report = fixture_report(performance=True)
                report[key] = replacement
                timing = write_report(directory, "timing.json", report)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    join_reports((quality, timing))

    def test_duplicate_quality_never_selects_last_result(self):
        with tempfile.TemporaryDirectory() as directory:
            one = write_report(directory, "one.json", fixture_report())
            two = write_report(directory, "two.json", fixture_report())
            with self.assertRaisesRegex(ValueError, "Duplicate quality"):
                join_reports((one, two))

    def test_seed_means_sample_std_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = []
            for seed in (0, 1):
                report = fixture_report(seed)
                report["timing"] = fixture_report(seed, performance=True)["timing"]
                reports.append(write_report(directory, f"run{seed}.json", report))
            artifacts = render_benchmark(reports, Path(directory) / "table")
            summary = json.loads(Path(artifacts["artifacts"]["json"]).read_text())
            row = summary["rows"][0]
            self.assertAlmostEqual(row["metrics"]["abs_rel"]["mean"], .15)
            self.assertAlmostEqual(row["metrics"]["abs_rel"]["std"], .070710678)
            self.assertEqual(row["metrics"]["ate_mm"]["measured_n"], 0)
            self.assertIsNone(row["metrics"]["ate_mm"]["mean"])
            self.assertEqual(summary["measurement_count"], 0)
            self.assertEqual(summary["fixture_run_count"], 2)
            page = Path(artifacts["artifacts"]["html"]).read_text(encoding="utf-8")
            self.assertIn("SYNTHETIC FIXTURE OUTPUT", page)
            self.assertIn("N/A", page)
            self.assertIn("<svg", page)

    def test_only_verified_training_seed_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = [fixture_report(seed) for seed in (0, 1, 2)]
            paths = [write_report(directory, f"seed{report['seed']}.json", report) for report in reports]
            artifacts = render_benchmark(paths, Path(directory) / "table")
            row = json.loads(Path(artifacts["artifacts"]["json"]).read_text())["rows"][0]
            self.assertEqual(row["seeds"], [0, 1, 2])
            self.assertEqual(len(row["config_sha256"]), 3)
            self.assertEqual(row["metrics"]["abs_rel"]["measured_n"], 3)
            self.assertEqual(len(row["seed_normalized_config_sha256"]), 64)
            # Normalization must not mutate the embedded source configs.
            self.assertEqual([report["config"]["training"]["seed"] for report in reports], [0, 1, 2])

    def test_seed_normalization_rejects_other_hyperparameter_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            first = fixture_report()
            first_path = write_report(directory, "first.json", first)
            for mutate in (
                lambda cfg: cfg["training"].update(learning_rate=2e-4),
                lambda cfg: cfg["memory"].update(model_dim=128),
                lambda cfg: cfg.update(other_seed=99),
            ):
                second = fixture_report(1)
                mutate(second["config"])
                second["config_sha256"] = hashlib.sha256(json.dumps(second["config"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                second_path = write_report(directory, "second.json", second)
                with self.assertRaisesRegex(ValueError, "different configs"):
                    render_benchmark([first_path, second_path], Path(directory) / "table")

    def test_mismatched_embedded_training_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = fixture_report(1)
            report["config"]["training"]["seed"] = 42
            report["config_sha256"] = hashlib.sha256(json.dumps(report["config"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            path = write_report(directory, "wrong_seed.json", report)
            with self.assertRaisesRegex(ValueError, "training.seed must match"):
                render_benchmark([path], Path(directory) / "table")

    def test_pair_and_amortized_output_latency_are_named_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            report = fixture_report()
            report["timing"] = fixture_report(performance=True)["timing"]
            report["timing"]["amortized_output_ms"] = {"mean_ms": 75., "p50_ms": 70., "p95_ms": 95.}
            path = write_report(directory, "latency.json", report)
            artifacts = render_benchmark([path], Path(directory) / "table")
            row = json.loads(Path(artifacts["artifacts"]["json"]).read_text())["rows"][0]
            self.assertEqual(row["metrics"]["mean_latency_ms"]["mean"], 100.)
            self.assertEqual(row["metrics"]["amortized_output_mean_ms"]["mean"], 75.)
            markdown = Path(artifacts["artifacts"]["markdown"]).read_text(encoding="utf-8")
            self.assertIn("Decoded pair p95 (ms)", markdown)
            self.assertIn("Amortized output p95 (ms)", markdown)

    def test_pose_failures_keep_counts_and_omit_complete_plot_points(self):
        with tempfile.TemporaryDirectory() as directory:
            report = fixture_report()
            report["timing"] = fixture_report(performance=True)["timing"]
            report["aggregate"].update(pose_window_count=1, pose_failed_window_count=1)
            report["aggregate"]["pose_metrics"] = {"ate_mm": 2.}
            path = write_report(directory, "failed_pose.json", report)
            artifacts = render_benchmark([path], Path(directory) / "table")
            row = json.loads(Path(artifacts["artifacts"]["json"]).read_text())["rows"][0]
            self.assertIn("incomplete_pose", row["status"])
            self.assertEqual(row["failure_counts"]["0"]["pose_failed_window_count"], 1)
            self.assertEqual(row["metrics"]["ate_mm"]["measured_n"], 1)
            self.assertNotIn("<circle", Path(artifacts["artifacts"]["quality_fps_svg"]).read_text())

    def test_incomplete_quality_and_timing_never_appear_as_complete_points(self):
        with tempfile.TemporaryDirectory() as directory:
            report = fixture_report()
            report["aggregate"].update(status="incomplete", frame_count=4, scored_frame_count=3, failed_frame_count=1)
            report["timing"] = fixture_report(performance=True)["timing"]
            report["timing"]["coverage_complete"] = False
            path = write_report(directory, "failed.json", report)
            artifacts = render_benchmark([path], Path(directory) / "table")
            row = json.loads(Path(artifacts["artifacts"]["json"]).read_text())["rows"][0]
            self.assertIn("incomplete_quality", row["status"])
            self.assertIn("incomplete_timing", row["status"])
            self.assertEqual(row["metrics"]["abs_rel"]["measured_n"], 1)
            self.assertEqual(row["failure_counts"]["0"]["failed_frame_count"], 1)
            self.assertIsNone(row["metrics"]["fps"]["mean"])
            self.assertEqual(row["metrics"]["fps"]["measured_n"], 0)
            self.assertNotIn("<circle", Path(artifacts["artifacts"]["quality_fps_svg"]).read_text())

    def test_aggregation_rejects_hardware_or_precision_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            first = fixture_report()
            first["timing"] = fixture_report(performance=True)["timing"]
            second = fixture_report(1)
            second["timing"] = fixture_report(1, performance=True)["timing"]
            second["timing"]["hardware_id"] = "ANOTHER SYNTHETIC DEVICE"
            paths = [write_report(directory, "first.json", first), write_report(directory, "second.json", second)]
            with self.assertRaisesRegex(ValueError, "hardware/precision/scopes"):
                render_benchmark(paths, Path(directory) / "output")

    def test_nonfinite_and_invalid_rate_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = fixture_report()
            bad["aggregate"]["depth_metrics"]["abs_rel"] = float("nan")
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                render_benchmark([path], Path(directory) / "output")
            bad = fixture_report()
            bad["timing"] = fixture_report(performance=True)["timing"]
            bad["timing"]["write_rate"] = 1.1
            path = write_report(directory, "bad.json", bad)
            with self.assertRaisesRegex(ValueError, r"in \[0,1\]"):
                render_benchmark([path], Path(directory) / "output")

    def test_html_and_csv_are_sanitized(self):
        with tempfile.TemporaryDirectory() as directory:
            report = fixture_report(experiment="=formula<script>alert(1)</script>")
            report["timing"] = fixture_report(performance=True)["timing"]
            path = write_report(directory, "fixture.json", report)
            artifacts = render_benchmark([path], Path(directory) / "table")
            page = Path(artifacts["artifacts"]["html"]).read_text(encoding="utf-8")
            self.assertNotIn("<script>alert(1)</script>", page)
            self.assertIn("&lt;script&gt;", page)
            with Path(artifacts["artifacts"]["csv"]).open(encoding="utf-8", newline="") as handle:
                row = list(csv.DictReader(handle))[0]
            self.assertTrue(row["experiment_id"].startswith("'="))

    def test_catalog_pending_and_paper_do_not_create_measured_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = {"experiments": [{"id": "legacy_official", "label": "Legacy official"},
                                       {"id": "hybrid_sampson"}, {"id": "hybrid_ppo"}]}
            artifacts = render_benchmark([], directory, paper_reference=fixture_paper(), catalog=catalog)
            summary = json.loads(Path(artifacts["artifacts"]["json"]).read_text())
            self.assertEqual(summary["measurement_count"], 0)
            self.assertEqual(summary["fixture_run_count"], 0)
            self.assertEqual(len(summary["rows"]), 4)
            paper, pending = summary["rows"][0], summary["rows"][1:]
            self.assertEqual(paper["source"], "paper_reference_not_measured")
            self.assertIsNone(paper["metrics"]["rmse_m"]["mean"])
            self.assertEqual(paper["metrics"]["fps"]["mean"], 19.17)
            self.assertIsNone(summary["speedup_against_paper"])
            for row in pending:
                self.assertEqual(row["status"], "pending")
                self.assertTrue(all(value["mean"] is None for value in row["metrics"].values()))

    def test_synthetic_and_measured_artifacts_cannot_join(self):
        with tempfile.TemporaryDirectory() as directory:
            quality = write_report(directory, "quality.json", fixture_report())
            performance = fixture_report(performance=True)
            performance["metadata"] = {"source": "server measured"}
            timing = write_report(directory, "timing.json", performance)
            with self.assertRaisesRegex(ValueError, "synthetic"):
                join_reports([quality, timing])

    def test_embedded_config_hash_and_catalog_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = fixture_report()
            report["config"]["model"] = "tampered"
            path = write_report(directory, "changed.json", report)
            with self.assertRaisesRegex(ValueError, "config_sha256"):
                render_benchmark([path], Path(directory) / "output")
            path = write_report(directory, "changed.json", fixture_report())
            with self.assertRaisesRegex(ValueError, "absent"):
                render_benchmark([path], Path(directory) / "output", catalog={"experiments": [{"id": "other"}]})


if __name__ == "__main__":
    unittest.main()
