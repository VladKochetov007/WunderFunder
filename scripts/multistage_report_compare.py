"""Build separate auditable tables and charts from primary reports and inner proxies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

MAX_REPORT_BYTES = 16 * 1024 * 1024
PRIMARY_SCHEMA = "wnn.multistage-primary-report.v1"
INNER_SCHEMA = "wnn.multistage-inner-objective.v1"
PRIMARY_SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
AUXILIARY_SPLIT_SHA256 = "26b448bad7f60612cea482f4abaf5f184e5a724228eef227d7e78a8155190075"
PRIMARY_SEQUENCE_COUNT = 1_873
PRIMARY_SCORED_ROWS = 3_528_273
PRIMARY_FIT_SEQUENCE_COUNT = 8_260
PRIMARY_CONTROL_SEQUENCE_COUNT = 2_065
INNER_CONTROL_SEQUENCE_COUNT = 2_065
TL3_REFERENCE_WP = 0.6923774245652725
WP_REDUCTION = "mean_of_two_pooled_clipped_weighted_Pearson_targets"
INNER_MASK = "all_after_warmup AND row_index>=99"


def _finite(value: object) -> bool:
    """Accept a finite JSON number and reject Booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sha256(value: object) -> bool:
    """Check one lowercase SHA-256 string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: object) -> str:
    """Hash one value with the project's sorted compact JSON encoding."""
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_report(path: Path) -> tuple[dict[str, Any], str]:
    """Read one bounded non-symlink report and return its raw-file SHA-256."""
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"report input must be a regular file: {path}")
    if info.st_size > MAX_REPORT_BYTES:
        raise ValueError(f"report input exceeds {MAX_REPORT_BYTES} bytes: {path}")
    raw = path.read_bytes()
    if len(raw) != info.st_size or len(raw) > MAX_REPORT_BYTES:
        raise ValueError(f"report changed or exceeded its size bound while reading: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"report root must be a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _required_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _primary_row(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a full official-valid report and extract only its recorded metrics."""
    report, report_sha = _read_report(path)
    if (
        report.get("schema") != PRIMARY_SCHEMA
        or report.get("report_type") != "primary_development_exposed_report_only"
        or report.get("objective_eligible") is not False
        or report.get("checkpoint_selection_allowed") is not False
    ):
        raise ValueError(f"not a report-only primary report: {path}")
    model = _required_mapping(report.get("model"), "primary model")
    fit = _required_mapping(report.get("fit"), "primary fit")
    evaluation = _required_mapping(report.get("evaluation"), "primary evaluation")
    metrics = _required_mapping(evaluation.get("metrics"), "primary metrics")
    config = _required_mapping(model.get("config"), "primary model config")
    model_config = _required_mapping(config.get("model"), "primary architecture config")
    reference = _required_mapping(report.get("comparison_reference"), "TL3 reference")
    family = model_config.get("architecture")
    if not isinstance(family, str) or not family.isascii() or not family.isalnum():
        raise ValueError(f"primary report has an invalid architecture label: {path}")
    checkpoint_sha = model.get("checkpoint_sha256")
    if not _sha256(checkpoint_sha) or not _sha256(model.get("config_sha256")):
        raise ValueError(f"primary report lacks checkpoint or config identity: {path}")
    if _canonical_sha256(config) != model.get("config_sha256"):
        raise ValueError(f"primary report config digest does not match its config: {path}")
    if (
        evaluation.get("role") != "primary_valid"
        or evaluation.get("source") != "valid"
        or evaluation.get("split_sha256") != PRIMARY_SPLIT_SHA256
        or evaluation.get("mask") != "is_scored AND row_index>=99"
        or evaluation.get("sequence_count") != PRIMARY_SEQUENCE_COUNT
        or evaluation.get("selected_rows") != PRIMARY_SCORED_ROWS
        or metrics.get("selected_rows") != PRIMARY_SCORED_ROWS
        or fit.get("split_sha256") != AUXILIARY_SPLIT_SHA256
        or fit.get("sequence_count") != PRIMARY_FIT_SEQUENCE_COUNT
        or fit.get("control_sequence_count") != PRIMARY_CONTROL_SEQUENCE_COUNT
    ):
        raise ValueError(f"primary report differs from the frozen B300 report protocol: {path}")
    if (
        reference.get("name") != "canonical fresh TL3"
        or reference.get("wp") != TL3_REFERENCE_WP
        or reference.get("measured_in_this_report") is not False
        or not isinstance(reference.get("exposure"), str)
    ):
        raise ValueError(f"primary report has an unsupported TL3 reference: {path}")
    if not all(
        _finite(metrics.get(name)) for name in ("wp", "t0", "t1", "wmse", "wmse_t0", "wmse_t1")
    ):
        raise ValueError(f"primary report has non-finite or missing metrics: {path}")
    run_identity = model.get("run_identity_sha256")
    producer_provenance = _required_mapping(
        _required_mapping(model.get("producer"), "primary producer").get("provenance"),
        "primary producer provenance",
    )
    identity = _required_mapping(producer_provenance.get("identity"), "primary run identity")
    if not _sha256(run_identity) or _canonical_sha256(identity) != run_identity:
        raise ValueError(f"primary report lacks its producer run identity: {path}")
    label = f"{family}:{path.parent.name}/{path.stem}"
    row = {
        "label": label,
        "family": family,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": model["config_sha256"],
        "run_identity_sha256": run_identity,
        "fit_sequences": fit["sequence_count"],
        "control_sequences": fit["control_sequence_count"],
        "primary_valid_sequences": evaluation["sequence_count"],
        "primary_valid_scored_rows": metrics["selected_rows"],
        "wp": float(metrics["wp"]),
        "t0": float(metrics["t0"]),
        "t1": float(metrics["t1"]),
        "wmse": float(metrics["wmse"]),
        "wmse_t0": float(metrics["wmse_t0"]),
        "wmse_t1": float(metrics["wmse_t1"]),
        "delta_wp_vs_tl3": float(metrics["wp"]) - TL3_REFERENCE_WP,
        "primary_split_sha256": evaluation["split_sha256"],
        "fit_split_sha256": fit["split_sha256"],
        "report_path": str(path.resolve()),
        "report_sha256": report_sha,
        "evidence_scope": "primary_development_exposed_report_only",
    }
    signature = {
        "checkpoint_sha256": checkpoint_sha,
        "primary_split_sha256": evaluation["split_sha256"],
        "report_sha256": report_sha,
    }
    return row, signature


def _inner_row(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one inner train-proxy objective without comparing it to primary valid."""
    report, report_sha = _read_report(path)
    if report.get("schema") != INNER_SCHEMA or report.get("status") != "complete":
        raise ValueError(f"not a completed inner objective: {path}")
    evaluation = _required_mapping(report.get("evaluation"), "inner evaluation")
    objective = _required_mapping(report.get("objective"), "inner objective metrics")
    producer = _required_mapping(report.get("producer"), "inner objective producer")
    producer_provenance = _required_mapping(producer.get("provenance"), "inner provenance")
    identity = _required_mapping(producer_provenance.get("identity"), "inner producer identity")
    prediction_producer = _required_mapping(
        report.get("prediction_producer"), "inner prediction producer"
    )
    config = _required_mapping(prediction_producer.get("config"), "inner model config")
    model_config = _required_mapping(config.get("model"), "inner architecture config")
    family = model_config.get("architecture")
    if not isinstance(family, str) or not family.isascii() or not family.isalnum():
        raise ValueError(f"inner objective has an invalid architecture label: {path}")
    checkpoint_sha = report.get("model_sha256")
    if (
        report.get("split_sha256") != AUXILIARY_SPLIT_SHA256
        or report.get("primary_split_sha256") != PRIMARY_SPLIT_SHA256
        or not _sha256(report.get("evaluation_members_sha256"))
        or not _sha256(report.get("fit_members_sha256"))
        or producer.get("checkpoint_sha256") != checkpoint_sha
        or producer.get("split_sha256") != AUXILIARY_SPLIT_SHA256
        or producer.get("fit_members_sha256") != report.get("fit_members_sha256")
        or not _sha256(checkpoint_sha)
        or not _sha256(report.get("prediction_manifest_sha256"))
        or not _sha256(report.get("fit_members_sha256"))
        or not _sha256(report.get("evaluation_members_sha256"))
        or report.get("sequence_count") != INNER_CONTROL_SEQUENCE_COUNT
        or type(report.get("selected_rows")) is not int
        or report["selected_rows"] <= 0
        or evaluation.get("role") != "control"
        or evaluation.get("scope") != "inner_train_proxy"
        or evaluation.get("source") != "train"
        or evaluation.get("row_mask") != INNER_MASK
        or evaluation.get("is_scored_used") is not False
        or objective.get("name") != "wp"
        or objective.get("reduction") != WP_REDUCTION
    ):
        raise ValueError(f"inner objective differs from the frozen train-proxy protocol: {path}")
    if not all(_finite(objective.get(name)) for name in ("value", "t0", "t1")):
        raise ValueError(f"inner objective has non-finite or missing metrics: {path}")
    identity_sha = producer_provenance.get("identity_sha256")
    if not _sha256(identity_sha) or identity_sha != _canonical_sha256(identity):
        raise ValueError(f"inner objective identity binding is missing or inconsistent: {path}")
    if _canonical_sha256(config) != report.get("config_sha256"):
        raise ValueError(f"inner objective config digest does not match its producer: {path}")
    label = f"{family}:{path.parent.name}/{path.stem}"
    row = {
        "label": label,
        "family": family,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": report.get("config_sha256"),
        "identity_sha256": identity_sha,
        "fit_members_sha256": report["fit_members_sha256"],
        "evaluation_members_sha256": report["evaluation_members_sha256"],
        "control_sequences": report["sequence_count"],
        "selected_rows": report["selected_rows"],
        "wp": float(objective["value"]),
        "t0": float(objective["t0"]),
        "t1": float(objective["t1"]),
        "split_sha256": report["split_sha256"],
        "primary_split_sha256": report["primary_split_sha256"],
        "report_path": str(path.resolve()),
        "report_sha256": report_sha,
        "evidence_scope": "inner_train_proxy_selection_only_not_chronological_validation",
    }
    signature = {
        "checkpoint_sha256": checkpoint_sha,
        "split_sha256": report["split_sha256"],
        "fit_members_sha256": report["fit_members_sha256"],
        "evaluation_members_sha256": report["evaluation_members_sha256"],
        "report_sha256": report_sha,
    }
    return row, signature


def _deduplicate_rows(
    rows: list[tuple[dict[str, Any], dict[str, Any]]], label: str
) -> list[dict[str, Any]]:
    """Collapse byte-identical endpoint reports and reject conflicting duplicates."""
    unique: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row, signature in rows:
        key = str(signature["checkpoint_sha256"])
        previous = unique.get(key)
        if previous is not None:
            if previous[1] != signature:
                raise ValueError(f"{label} repeats one checkpoint with conflicting report identity")
            continue
        unique[key] = row, signature
    return [pair[0] for pair in sorted(unique.values(), key=lambda pair: pair[0]["label"])]


def _write_table(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    """Write a deterministic CSV table from validated report records."""
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_rows(
    rows: list[dict[str, Any]],
    output_prefix: Path,
    *,
    title: str,
    primary: bool,
) -> None:
    """Export WP and per-target score panels while keeping evidence scopes separate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: float(row["wp"]))
    labels = [str(row["label"]) for row in ordered]
    wp = [float(row["wp"]) for row in ordered]
    t0 = [float(row["t0"]) for row in ordered]
    t1 = [float(row["t1"]) for row in ordered]
    figure, (wp_axis, target_axis) = plt.subplots(
        1, 2, figsize=(max(11.0, len(rows) * 0.8), max(4.5, len(rows) * 0.35))
    )
    y_positions = list(range(len(ordered)))
    bars = wp_axis.barh(y_positions, wp, color="#4c78a8")
    wp_axis.set_yticks(y_positions, labels)
    wp_axis.set_xlabel("Pooled clipped WP")
    wp_axis.set_title("WP")
    wp_axis.grid(axis="x", alpha=0.25)
    wp_axis.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    if primary:
        wp_axis.axvline(TL3_REFERENCE_WP, color="#e45756", linestyle="--", linewidth=1.5)
        wp_axis.text(
            TL3_REFERENCE_WP,
            1.01,
            "canonical fresh TL3 (reference; not measured here)",
            transform=wp_axis.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
        )
    width = 0.36
    target_axis.barh(
        [position - width / 2 for position in y_positions], t0, height=width, label="target 0"
    )
    target_axis.barh(
        [position + width / 2 for position in y_positions], t1, height=width, label="target 1"
    )
    target_axis.set_yticks(y_positions, labels)
    target_axis.set_xlabel("Pooled clipped weighted Pearson")
    target_axis.set_title("Per-target scores")
    target_axis.legend()
    target_axis.grid(axis="x", alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            output_prefix.with_suffix(f".{suffix}"),
            dpi=160 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def build_comparisons(
    primary_paths: list[Path], inner_paths: list[Path], output_dir: Path
) -> dict[str, object]:
    """Write separate primary and inner-proxy comparisons from immutable JSON inputs."""
    if not primary_paths and not inner_paths:
        raise ValueError("provide at least one --primary-report or --inner-objective")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stat = os.lstat(output_dir)
    if stat.S_ISLNK(output_stat.st_mode) or not stat.S_ISDIR(output_stat.st_mode):
        raise ValueError("comparison output directory must be a real directory")
    expected_outputs = [output_dir / "comparison_manifest.json"]
    if primary_paths:
        expected_outputs.extend(
            output_dir / f"primary_comparison.{suffix}" for suffix in ("csv", "json", "png", "svg")
        )
    if inner_paths:
        expected_outputs.extend(
            output_dir / f"inner_proxy_comparison.{suffix}"
            for suffix in ("csv", "json", "png", "svg")
        )
    if any(path.exists() or path.is_symlink() for path in expected_outputs):
        raise FileExistsError("comparison output already exists; choose a fresh output directory")
    created: dict[str, object] = {"schema": "wnn.multistage-report-comparison.v1"}
    if primary_paths:
        primary_rows = _deduplicate_rows(
            [_primary_row(path) for path in primary_paths], "primary report"
        )
        if not primary_rows:
            raise ValueError("primary report set has no unique endpoints")
        if {row["primary_split_sha256"] for row in primary_rows} != {PRIMARY_SPLIT_SHA256}:
            raise ValueError("primary reports do not share the frozen primary split")
        baseline = {
            "name": "canonical fresh TL3",
            "wp": TL3_REFERENCE_WP,
            "measured_in_this_report": False,
            "exposure": "development-exposed full official valid; no hidden/public forecast",
        }
        prefix = output_dir / "primary_comparison"
        columns = (
            "label",
            "family",
            "checkpoint_sha256",
            "wp",
            "delta_wp_vs_tl3",
            "t0",
            "t1",
            "wmse",
            "wmse_t0",
            "wmse_t1",
            "fit_sequences",
            "control_sequences",
            "primary_valid_sequences",
            "primary_valid_scored_rows",
            "primary_split_sha256",
            "fit_split_sha256",
            "report_path",
            "report_sha256",
            "evidence_scope",
        )
        _write_table(prefix.with_suffix(".csv"), primary_rows, columns)
        primary_result = {
            "schema": "wnn.multistage-primary-comparison.v1",
            "scope": "primary_development_exposed_report_only",
            "objective_eligible": False,
            "checkpoint_selection_allowed": False,
            "baseline": baseline,
            "models": primary_rows,
        }
        prefix.with_suffix(".json").write_text(
            json.dumps(primary_result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _plot_rows(
            primary_rows,
            prefix,
            title="Frozen endpoints on exposed official primary valid",
            primary=True,
        )
        created["primary"] = {
            "rows": len(primary_rows),
            "baseline_wp": TL3_REFERENCE_WP,
            "csv": str(prefix.with_suffix(".csv")),
            "json": str(prefix.with_suffix(".json")),
            "png": str(prefix.with_suffix(".png")),
            "svg": str(prefix.with_suffix(".svg")),
        }
    if inner_paths:
        inner_rows = _deduplicate_rows(
            [_inner_row(path) for path in inner_paths], "inner objective"
        )
        if not inner_rows:
            raise ValueError("inner objective set has no unique endpoints")
        scopes = {
            (row["split_sha256"], row["primary_split_sha256"], row["fit_members_sha256"])
            for row in inner_rows
        }
        if len(scopes) != 1:
            raise ValueError("inner objectives differ in split or fitting-member ancestry")
        prefix = output_dir / "inner_proxy_comparison"
        columns = (
            "label",
            "family",
            "checkpoint_sha256",
            "wp",
            "t0",
            "t1",
            "control_sequences",
            "selected_rows",
            "split_sha256",
            "primary_split_sha256",
            "fit_members_sha256",
            "evaluation_members_sha256",
            "report_path",
            "report_sha256",
            "evidence_scope",
        )
        _write_table(prefix.with_suffix(".csv"), inner_rows, columns)
        inner_result = {
            "schema": "wnn.multistage-inner-proxy-comparison.v1",
            "scope": "inner_train_proxy_selection_only_not_chronological_validation",
            "comparison_reference": None,
            "models": inner_rows,
        }
        prefix.with_suffix(".json").write_text(
            json.dumps(inner_result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _plot_rows(
            inner_rows,
            prefix,
            title="Auxiliary train-only control proxy (not chronological validation)",
            primary=False,
        )
        created["inner_proxy"] = {
            "rows": len(inner_rows),
            "baseline_wp": None,
            "csv": str(prefix.with_suffix(".csv")),
            "json": str(prefix.with_suffix(".json")),
            "png": str(prefix.with_suffix(".png")),
            "svg": str(prefix.with_suffix(".svg")),
        }
    summary_path = output_dir / "comparison_manifest.json"
    summary_path.write_text(
        json.dumps(created, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return created


def main() -> None:
    """Create a separate table and plot set for each report scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-report", action="append", type=Path, default=[])
    parser.add_argument("--inner-objective", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = build_comparisons(args.primary_report, args.inner_objective, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"comparison refused: {error}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
