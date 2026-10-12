"""Join authentic run artifacts and render auditable benchmark tables.

Uses only the Python standard library. It consumes artifacts; it does not run
inference, install anything, acquire data or invent missing measurements. Paper
numbers remain reference-only and never become server speedup denominators.
"""
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics


SCHEMA = "endo3r_benchmark_v1"
JOIN_FIELDS = ("experiment_id", "seed", "config_sha256", "checkpoint_hashes",
               "manifest_sha256", "split", "image_hw", "frame_indices_sha256",
               "provenance_sha256", "protocol", "scale_protocol",
               "prediction_camera_source", "pose_protocol")
GROUP_FIELDS = ("manifest_sha256", "split", "image_hw", "frame_indices_sha256",
                "provenance_sha256", "protocol", "scale_protocol",
                "prediction_camera_source", "pose_protocol")
METRICS = (
    ("abs_rel", "AbsRel ↓", "depth_metrics", "abs_rel"),
    ("sq_rel_m", "SqRel (m) ↓", "depth_metrics", "sq_rel"),
    ("rmse_m", "RMSE (m) ↓", "depth_metrics", "rmse_m"),
    ("rmse_log", "RMSE-log ↓", "depth_metrics", "rmse_log"),
    ("delta1", "δ₁ ↑", "depth_metrics", "delta1"),
    ("ate_mm", "ATE (mm) ↓", "pose_metrics", "ate_mm"),
    ("rpe_translation_mm", "RPE-t (mm) ↓", "pose_metrics", "rpe_translation_mm"),
    ("rpe_rotation_deg", "RPE-r (°) ↓", "pose_metrics", "rpe_rotation_deg"),
    ("fps", "FPS ↑", "timing", "fps"),
    ("mean_latency_ms", "Decoded pair mean (ms) ↓", "timing", "mean_latency_ms"),
    ("p50_latency_ms", "Decoded pair p50 (ms) ↓", "timing", "p50_latency_ms"),
    ("p95_latency_ms", "Decoded pair p95 (ms) ↓", "timing", "p95_latency_ms"),
    ("amortized_output_mean_ms", "Amortized output mean (ms) ↓", "timing", "amortized_output_ms.mean_ms"),
    ("amortized_output_p50_ms", "Amortized output p50 (ms) ↓", "timing", "amortized_output_ms.p50_ms"),
    ("amortized_output_p95_ms", "Amortized output p95 (ms) ↓", "timing", "amortized_output_ms.p95_ms"),
    ("peak_vram_mib", "Peak VRAM (MiB) ↓", "timing", "peak_vram_mib"),
    ("write_rate", "WRITE rate ↓", "timing", "write_rate"),
    ("raft_call_rate", "RAFT rate ↓", "timing", "raft_call_rate"),
    ("kda_update_rate", "KDA rate ↓", "timing", "kda_update_rate"),
    ("actor_mean_latency_ms", "Actor act profile proxy (ms) ↓", "timing", "actor_mean_latency_ms"),
)


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value is forbidden: {value}")


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _seed_normalized_config(report):
    """Ignore only the verified training.seed value when grouping run seeds.

    Full config hashes still bind each quality/timing pair. Other random-seed
    fields and every hyperparameter remain part of the group identity.
    """
    config = report.get("config")
    if config is None:
        return {"unavailable_embedded_config_sha256": report["config_sha256"]}
    if not isinstance(config, dict):
        raise ValueError("Embedded config must be a mapping")
    config = json.loads(_canonical(config))
    training = config.get("training")
    if isinstance(training, dict) and "seed" in training:
        if isinstance(training["seed"], bool) or not isinstance(training["seed"], int) or training["seed"] != report["seed"]:
            raise ValueError("Embedded training.seed must match the report seed before aggregation")
        training["seed"] = "VERIFIED_RUN_SEED"
    return config


def _validate_report(report, path):
    if report.get("schema") not in {SCHEMA, "endo3r_performance_v1"} or report.get("schema_version", 1) != 1:
        raise ValueError(f"Unsupported benchmark report schema: {path}")
    for field in JOIN_FIELDS:
        if field not in report or report[field] is None:
            raise ValueError(f"Missing reproducibility identity {field}: {path}")
    if not isinstance(report["experiment_id"], str) or not report["experiment_id"]:
        raise ValueError("An explicit experiment_id is required; reports cannot be anonymously merged")
    if isinstance(report["seed"], bool) or not isinstance(report["seed"], int):
        raise ValueError("Seed must be an integer")
    if not isinstance(report["checkpoint_hashes"], dict) or not report["checkpoint_hashes"]:
        raise ValueError("Checkpoint fingerprints must identify the actually loaded model")
    if not isinstance(report["image_hw"], list) or len(report["image_hw"]) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in report["image_hw"]):
        raise ValueError("image_hw must be positive integer [H,W]")
    if report.get("config") is not None:
        actual = hashlib.sha256(_canonical(report["config"]).encode("utf-8")).hexdigest()
        if actual != report["config_sha256"]:
            raise ValueError(f"config_sha256 does not match the embedded config: {path}")
        _seed_normalized_config(report)
    # Check the whole structure for nonfinite values, including unused fields.
    _canonical(report)


def _fixture(report):
    metadata = report.get("metadata") or {}
    return metadata.get("is_fixture") is True or str(metadata.get("source", report.get("source", ""))).lower().startswith("synthetic")


def _same_identity(first, other):
    mismatches = [field for field in JOIN_FIELDS if first[field] != other[field]]
    if mismatches:
        raise ValueError(f"Cannot join quality/performance for {first['experiment_id']} seed={first['seed']}: mismatch {', '.join(mismatches)}")
    if _fixture(first) != _fixture(other):
        raise ValueError("Cannot join synthetic fixtures with measured artifacts")
    # Actual kernel provenance should agree whenever both reports contain it.
    if first.get("actual_backends") and other.get("actual_backends") and first["actual_backends"] != other["actual_backends"]:
        raise ValueError("Cannot join reports with different actual attention backends")
    first_source = (first.get('source_provenance') or {}).get('source_tree_sha256')
    other_source = (other.get('source_provenance') or {}).get('source_tree_sha256')
    if first_source or other_source:
        if not first_source or first_source != other_source:
            raise ValueError('Cannot join reports from different or unidentified source trees')


def join_reports(paths):
    """Join optional separate quality/timing reports, rejecting identity mismatches.

    A single report may contain both aggregate and timing. Separate reports must
    share every JOIN_FIELDS fingerprint. Supplying duplicate quality or timing
    for one run aborts; choosing whichever arrived last is never allowed.
    """
    runs = {}
    for path in paths:
        report = load_json(path)
        _validate_report(report, path)
        key = (report["experiment_id"], report["seed"])
        has_quality = report.get("schema") == SCHEMA and bool(report.get("aggregate"))
        has_timing = report.get("timing") is not None
        if not has_quality and not has_timing:
            raise ValueError(f"Report has neither quality nor timing results: {path}")
        if key not in runs:
            report = dict(report)
            if not has_quality:
                report["performance_aggregate"] = report.pop("aggregate", None)
            report["artifact_paths"] = [str(Path(path).resolve())]
            runs[key] = report
            continue
        existing = runs[key]
        _same_identity(existing, report)
        if has_quality and existing.get("aggregate"):
            raise ValueError(f"Duplicate quality result for {key}")
        if has_timing and existing.get("timing") is not None:
            raise ValueError(f"Duplicate timing result for {key}")
        if has_quality:
            existing["aggregate"] = report["aggregate"]
            existing["sequences"] = report.get("sequences", [])
        if has_timing:
            existing["timing"] = report["timing"]
            existing["performance_aggregate"] = report.get("aggregate") if not has_quality else None
            existing["hardware"] = report.get("hardware")
        existing["actual_backends"] = existing.get("actual_backends") or report.get("actual_backends", {})
        existing["artifact_paths"].append(str(Path(path).resolve()))
    return list(runs.values())


def _number(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Metric {name} must be finite, nonnegative numeric or null")
    if name in {"delta1", "write_rate", "raft_call_rate", "kda_update_rate"} and value > 1:
        raise ValueError(f"Metric {name} must be in [0,1]")
    return float(value)


def _metric(report, section, source_key, metric_name):
    aggregate = report.get("aggregate") or {}
    data = (report.get("timing") or {}) if section == "timing" else (aggregate.get(section) or {})
    if section == "timing" and metric_name == "fps" and data.get("coverage_complete") is False:
        return None
    for key in source_key.split("."):
        data = data.get(key) if isinstance(data, dict) else None
    return _number(data, metric_name)


def _run_statuses(report):
    aggregate = report.get("aggregate") or {}
    if not aggregate:
        statuses = ["timing_only"]
    else:
        status = aggregate.get("status", "unknown_quality_status")
        if aggregate.get("failed_frame_count", 0) > 0:
            status = "incomplete_quality"
        statuses = [status]
        if aggregate.get("pose_failed_window_count", 0) > 0:
            statuses.append("incomplete_pose")
    if (report.get("timing") or {}).get("coverage_complete") is False:
        statuses.append("incomplete_timing")
    return statuses


def _timing_context(report):
    timing = report.get("timing") or {}
    # Timing contexts need matching hardware, dtype and included work; they are
    # recorded even when no timings exist so quality-only artifacts remain useful.
    return {field: timing.get(field) for field in
            ("scope", "timing_scope", "hardware_id", "hardware", "device", "gpu_name",
             "dtype", "precision", "warmup_frames", "timing_method", "latency_unit", "actor_latency_source")}


def aggregate_runs(runs):
    """Aggregate only seeds of the same protocol/configuration and timing scope."""
    grouped = {}
    for run in runs:
        grouped.setdefault(run["experiment_id"], []).append(run)
    rows = []
    for experiment_id, group in sorted(grouped.items()):
        first = group[0]
        for other in group[1:]:
            bad = [key for key in GROUP_FIELDS if first[key] != other[key]]
            if bad or _fixture(first) != _fixture(other):
                raise ValueError(f"Cannot aggregate seeds for {experiment_id}: mismatched {bad or ['fixture provenance']}")
            if _seed_normalized_config(first) != _seed_normalized_config(other):
                raise ValueError(f"Cannot aggregate seeds of different configs for {experiment_id}")
            if first.get("actual_backends") != other.get("actual_backends"):
                raise ValueError(f"Cannot aggregate different actual kernels for {experiment_id}")
        # Quality-only seeds may coexist with timing seeds, but then each metric
        # reports its own measured_n rather than treating missing timings as zero.
        contexts = [_timing_context(r) for r in group if r.get("timing") is not None]
        if contexts and any(context != contexts[0] for context in contexts[1:]):
            raise ValueError(f"Cannot average timings from different hardware/precision/scopes for {experiment_id}")
        metric_values = {}
        for name, _, section, source in METRICS:
            values = [value for value in (_metric(r, section, source, name) for r in group) if value is not None]
            metric_values[name] = {"mean": statistics.mean(values) if values else None,
                                   "std": statistics.stdev(values) if len(values) > 1 else None,
                                   "measured_n": len(values)}
        statuses = [status for run in group for status in _run_statuses(run)]
        source = "synthetic_fixture_not_measured" if _fixture(first) else "run_artifact"
        rows.append({"experiment_id": experiment_id, "source": source,
                     "seeds": sorted(r["seed"] for r in group), "seed_count": len(group),
                     "status": statuses[0] if len(set(statuses)) == 1 else "mixed_" + "_".join(sorted(set(statuses))),
                     "metrics": metric_values,
                     "identity": {key: first[key] for key in GROUP_FIELDS},
                     "config_sha256": sorted(set(r["config_sha256"] for r in group)),
                     "seed_normalized_config_sha256": hashlib.sha256(_canonical(_seed_normalized_config(first)).encode()).hexdigest(),
                     "checkpoint_hashes_by_seed": {str(r["seed"]): r["checkpoint_hashes"] for r in group},
                     "actual_backends": first.get("actual_backends", {}),
                     "timing_context": contexts[0] if contexts else {},
                     "failure_counts": {str(r["seed"]): {key: r.get("aggregate", {}).get(key) for key in
                         ("frame_count", "scored_frame_count", "failed_frame_count", "pose_window_count", "pose_failed_window_count")} for r in group},
                     "artifact_paths": [path for r in group for path in r["artifact_paths"]]})
    return rows


def _fmt(value):
    if value is None:
        return "N/A"
    if value == 0:
        return "0"
    return f"{value:.5g}"


def _stat(stat):
    mean, std = stat["mean"], stat["std"]
    text = _fmt(mean)
    if std is not None:
        text += f" ± {_fmt(std)}"
    return text


def _md(value):
    return html.escape(str(value)).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _csv_safe(value):
    text = str(value)
    return "'" + text if text.startswith(("=", "+", "-", "@")) else text


def _paper(reference):
    if reference is None:
        return None
    if isinstance(reference, (str, Path)):
        reference = load_json(reference)
    _canonical(reference)
    if reference.get("source") != "paper_reference_not_measured":
        raise ValueError("Paper reference must explicitly declare source=paper_reference_not_measured")
    if not reference.get("citation_url"):
        raise ValueError("Paper reference requires an exact primary-source citation URL")
    for section in ("depth_metrics", "pose_metrics", "timing"):
        for key, value in (reference.get(section) or {}).items():
            if isinstance(value, (int, float)) or value is None:
                _number(value, key)
    # Published units and hardware can be unspecified. Never normalize them to
    # metres/millimetres or compute speedup from the paper's FPS.
    return reference


def _paper_metric_rows(paper):
    units = paper.get("reported_units") or {}
    depth, pose, timing = paper.get("depth_metrics") or {}, paper.get("pose_metrics") or {}, paper.get("timing") or {}
    return [
        ("AbsRel", depth.get("abs_rel"), "dimensionless"),
        ("SqRel", depth.get("sq_rel"), units.get("sq_rel", "unspecified")),
        ("RMSE", depth.get("rmse"), units.get("rmse", "unspecified")),
        ("RMSE-log", depth.get("rmse_log"), "dimensionless"),
        ("δ₁", depth.get("delta1"), "dimensionless"),
        ("ATE", pose.get("ate"), units.get("ate", "unspecified")),
        ("RPE-t", pose.get("rpe_translation"), units.get("rpe_translation", "unspecified")),
        ("RPE-r", pose.get("rpe_rotation"), units.get("rpe_rotation", "unspecified")),
        ("FPS", timing.get("fps"), "frames/s; paper hardware/scope not established"),
    ]


def _plot(rows, x_key, title, x_label):
    points = []
    for row in rows:
        x, y = row["metrics"][x_key]["mean"], row["metrics"]["abs_rel"]["mean"]
        if x is not None and y is not None and row["status"] == "complete":
            points.append((x, y, row["experiment_id"], row["source"]))
    width, height = 820, 400
    left, right, top, bottom = 80, 540, 60, 335
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title, quote=True)}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="24" y="28" font-family="sans-serif" font-size="17">{html.escape(title)}</text>']
    if not points:
        parts.append('<text x="50" y="180" font-family="sans-serif">No complete paired quality/timing measurements supplied.</text></svg>')
        return "".join(parts)
    x_max, y_max = max(x for x, *_ in points) * 1.12, max(y for _, y, *_ in points) * 1.12
    x_max, y_max = max(x_max, 1e-9), max(y_max, 1e-9)
    for i in range(6):
        fraction = i / 5
        xx, yy = left + fraction * (right - left), bottom - fraction * (bottom - top)
        parts.extend([f'<line x1="{xx}" y1="{top}" x2="{xx}" y2="{bottom}" stroke="#eee"/>',
                      f'<line x1="{left}" y1="{yy}" x2="{right}" y2="{yy}" stroke="#eee"/>',
                      f'<text x="{xx}" y="{bottom + 20}" text-anchor="middle" font-family="sans-serif" font-size="11">{_fmt(fraction*x_max)}</text>',
                      f'<text x="{left - 10}" y="{yy+4}" text-anchor="end" font-family="sans-serif" font-size="11">{_fmt(fraction*y_max)}</text>'])
    parts.extend([f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>',
                  f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
                  f'<text x="310" y="390" text-anchor="middle" font-family="sans-serif">{html.escape(x_label)}</text>',
                  '<text transform="translate(22,205) rotate(-90)" text-anchor="middle" font-family="sans-serif">AbsRel ↓</text>'])
    colors = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00")
    for i, (x, y, name, source) in enumerate(points):
        xx, yy, color = left + x / x_max * (right - left), bottom - y / y_max * (bottom - top), colors[i % len(colors)]
        label = name + (" [synthetic fixture]" if source.startswith("synthetic") else "")
        parts.extend([f'<circle cx="{xx}" cy="{yy}" r="6" fill="{color}"><title>{html.escape(label)}: {html.escape(x_label)}={_fmt(x)}, AbsRel={_fmt(y)}</title></circle>',
                      f'<circle cx="570" cy="{76+i*24}" r="5" fill="{color}"/>',
                      f'<text x="580" y="{80+i*24}" font-family="sans-serif" font-size="11">{html.escape(label)}</text>'])
    parts.append('</svg>')
    return "".join(parts)


def _empty_row(experiment_id, source, status):
    return {"experiment_id": experiment_id, "source": source, "seeds": [], "seed_count": 0,
            "status": status, "metrics": {name: {"mean": None, "std": None, "measured_n": 0} for name, *_ in METRICS},
            "identity": {}, "config_sha256": [], "checkpoint_hashes_by_seed": {},
            "seed_normalized_config_sha256": None,
            "actual_backends": {}, "timing_context": {}, "failure_counts": {}, "artifact_paths": []}


def _with_catalog(rows, catalog):
    if catalog is None:
        return rows
    catalog = load_json(catalog) if isinstance(catalog, (str, Path)) else catalog
    experiments = catalog.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("Experiment catalog needs an experiments array")
    selected, ids = [], set()
    for experiment in experiments:
        experiment_id = experiment.get("id", experiment.get("experiment_id"))
        if not isinstance(experiment_id, str) or not experiment_id or experiment_id in ids:
            raise ValueError("Catalog experiment ids must be unique nonempty strings")
        ids.add(experiment_id)
        if experiment.get("enabled", True) is not False:
            selected.append((experiment_id, experiment))
    if any(row["experiment_id"] not in ids for row in rows):
        raise ValueError("Result experiment_id is absent from the supplied study catalog")
    by_id = {row["experiment_id"]: row for row in rows}
    ordered = []
    for experiment_id, experiment in selected:
        row = by_id.get(experiment_id, _empty_row(experiment_id, "pending_server_run_not_measured", "pending"))
        row["label"] = experiment.get("label", experiment_id)
        ordered.append(row)
    return ordered


def render_benchmark(paths, output_dir, *, paper_reference=None, catalog=None):
    """Write summary JSON/CSV/Markdown/standalone HTML and SVG scatter plots."""
    runs = join_reports(paths)
    rows, paper = _with_catalog(aggregate_runs(runs), catalog), _paper(paper_reference)
    if paper:
        reference_row = _empty_row(paper.get("experiment_id", "Endo3R-paper"), "paper_reference_not_measured", "reference")
        # These columns are dimensionless; published RMSE/SqRel/pose units are
        # preserved separately instead of guessed or mixed into metric columns.
        for name in ("abs_rel", "rmse_log", "delta1"):
            reference_row["metrics"][name]["mean"] = _number((paper.get("depth_metrics") or {}).get(name), name)
        reference_row["metrics"]["fps"]["mean"] = _number((paper.get("timing") or {}).get("fps"), "fps")
        rows.insert(0, reference_row)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = [row for row in rows if row["source"] not in {"paper_reference_not_measured", "pending_server_run_not_measured"}]
    all_fixture = bool(run_rows) and all(row["source"].startswith("synthetic") for row in run_rows)
    banner = "SYNTHETIC FIXTURE OUTPUT — NOT MODEL PERFORMANCE" if all_fixture else "Benchmark artifacts; unavailable measurements are N/A."
    summary = {"schema": "endo3r_benchmark_table_v1", "schema_version": 1,
               "measurement_count": sum(row["seed_count"] for row in rows if not row["source"].startswith("synthetic")),
               "fixture_run_count": sum(row["seed_count"] for row in rows if row["source"].startswith("synthetic")),
               "rows": rows, "paper_reference": paper, "speedup_against_paper": None,
               "notes": ["Paper numbers are reference-only; GPU/timing scope and unspecified metric units are not inferred.",
                         "Std is sample standard deviation across available seeds; single-seed std is N/A.",
                         "N/A is never interpreted as zero; every metric records measured_n.",
                         "Complete and incomplete quality results are labelled separately.",
                         "Only matching verified training.seed values are normalized across seeds; all other config fields remain fixed.",
                         "Decoded-pair latency includes the terminal pair's two outputs; amortized output latency divides pair time by output count and is not observed frame-arrival latency.",
                         "Failed pose windows and incomplete timing mark rows incomplete and exclude them from quality/compute plots.",
                         "No FPS speedup is computed against the published paper."]}
    if summary["measurement_count"] == 0:
        summary["notes"].append("No measured server results have been supplied; pending rows contain no measured metrics.")
    files = {"json": output_dir / "benchmark.json", "csv": output_dir / "benchmark.csv",
             "markdown": output_dir / "benchmark.md", "html": output_dir / "benchmark.html",
             "quality_fps_svg": output_dir / "quality_vs_fps.svg",
             "quality_vram_svg": output_dir / "quality_vs_vram.svg"}
    files["json"].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_columns = ["experiment_id", "source", "seed_count", "seeds", "status"]
    csv_columns += [name + suffix for name, *_ in METRICS for suffix in ("_mean", "_std", "_measured_n")]
    with files["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            record = {key: _csv_safe(row[key]) for key in ("experiment_id", "source", "seed_count", "status")}
            record["seeds"] = ",".join(str(i) for i in row["seeds"])
            for name, *_ in METRICS:
                for suffix, value in row["metrics"][name].items():
                    record[name + "_" + suffix] = "N/A" if value is None else value
            writer.writerow(record)
    columns = ["Experiment", "Source", "Seeds", "Status"] + [label for _, label, *_ in METRICS]
    markdown = ["# Endo3R benchmark", "", banner, "",
                "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    body_rows = []
    for row in rows:
        cells = [row.get("label", row["experiment_id"]), row["source"], ",".join(str(i) for i in row["seeds"]) or "N/A", row["status"]]
        cells += [_stat(row["metrics"][name]) for name, *_ in METRICS]
        markdown.append("| " + " | ".join(_md(cell) for cell in cells) + " |")
        body_rows.append('<tr>' + ''.join('<td>' + html.escape(str(cell)) + '</td>' for cell in cells) + '</tr>')
    if not rows:
        markdown += ["", "No measured server results supplied."]
        body_rows.append(f'<tr><td colspan="{len(columns)}">No measured server results supplied.</td></tr>')
    metadata_sections = []
    for row in rows:
        detail = {key: row[key] for key in ("identity", "config_sha256", "seed_normalized_config_sha256", "checkpoint_hashes_by_seed", "actual_backends", "timing_context", "failure_counts", "artifact_paths")}
        markdown += ["", "## " + _md(row["experiment_id"]) + " provenance", "", "```json", json.dumps(detail, indent=2), "```"]
        metadata_sections.append('<details><summary>' + html.escape(row["experiment_id"]) + ' provenance</summary><pre>' + html.escape(json.dumps(detail, indent=2)) + '</pre></details>')
    paper_html = ""
    if paper:
        paper_info = {key: paper.get(key) for key in ("experiment_id", "source", "citation_url", "reported_units", "depth_metrics", "pose_metrics", "timing", "hardware", "notes")}
        published_rows = _paper_metric_rows(paper)
        markdown += ["", "## Published paper reference — not measured on this server", "",
                     "| Metric | Published value | Reported unit |", "| --- | --- | --- |"]
        markdown += ["| " + " | ".join((_md(metric), _fmt(value), _md(unit))) + " |" for metric, value, unit in published_rows]
        markdown += ["", "```json", json.dumps(paper_info, indent=2), "```", "", "Published metric units remain as reported; hardware/timing scope is unspecified. No paper FPS speedup is calculated."]
        published_table = '<table><thead><tr><th>Metric</th><th>Published value</th><th>Reported unit</th></tr></thead><tbody>'
        published_table += ''.join('<tr><td>' + html.escape(metric) + '</td><td>' + _fmt(value) + '</td><td>' + html.escape(str(unit)) + '</td></tr>' for metric, value, unit in published_rows)
        published_table += '</tbody></table>'
        paper_html = '<section><h2>Published paper reference — not measured on this server</h2>' + published_table + '<pre>' + html.escape(json.dumps(paper_info, indent=2)) + '</pre><p>Published metric units remain as reported. No paper FPS speedup is calculated.</p></section>'
    markdown += ["", *summary["notes"], ""]
    files["markdown"].write_text("\n".join(markdown), encoding="utf-8")
    fps_svg = _plot(rows, "fps", "Quality vs throughput (paper reference excluded)", "FPS ↑")
    vram_svg = _plot(rows, "peak_vram_mib", "Quality vs GPU memory (paper reference excluded)", "Peak VRAM (MiB) ↓")
    files["quality_fps_svg"].write_text(fps_svg, encoding="utf-8")
    files["quality_vram_svg"].write_text(vram_svg, encoding="utf-8")
    html_text = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Endo3R benchmark</title>'
    html_text += '<style>body{font:14px system-ui,sans-serif;margin:24px;color:#18202a}table{border-collapse:collapse;white-space:nowrap}th,td{padding:9px;border:1px solid #ccd3da;text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#edf3f8}.scroll{overflow:auto}.banner{background:#fff3cc;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:12px}svg{width:100%;max-width:950px}details{margin:12px 0}</style><body><h1>Endo3R benchmark</h1>'
    html_text += '<p class="banner">' + html.escape(banner) + '</p><div class="scroll"><table><thead><tr>'
    html_text += ''.join('<th>' + html.escape(label) + '</th>' for label in columns) + '</tr></thead><tbody>' + ''.join(body_rows) + '</tbody></table></div>'
    html_text += fps_svg + vram_svg + paper_html + ''.join(metadata_sections)
    html_text += '<ul>' + ''.join('<li>' + html.escape(note) + '</li>' for note in summary["notes"]) + '</ul></body></html>'
    files["html"].write_text(html_text, encoding="utf-8")
    return {"artifacts": {key: str(path.resolve()) for key, path in files.items()},
            "measurement_count": summary["measurement_count"], "fixture_run_count": summary["fixture_run_count"],
            "experiment_rows": len(rows)}
