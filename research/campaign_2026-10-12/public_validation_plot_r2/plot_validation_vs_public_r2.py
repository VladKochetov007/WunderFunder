"""Render R2 paired local validation and public outcomes without pooling protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# CSV rows are heterogeneous; Matplotlib's axis and artist interfaces are dynamic.
EXPECTED_PROTOCOL_COUNTS = {
    "exact_f1_receipt": 3,
    "package_reported_f1": 1,
    "legacy_source_reported_368": 5,
    "legacy_conditional_reconstruction_368": 2,
}
CATEGORY_ORDER = tuple(EXPECTED_PROTOCOL_COUNTS)
CATEGORY_LABELS = {
    "exact_f1_receipt": "Exact F1 receipt",
    "package_reported_f1": "Package-reported F1",
    "legacy_source_reported_368": "Legacy source-reported 368-group",
    "legacy_conditional_reconstruction_368": "Conditional EMA6 reconstruction",
}
CATEGORY_COLORS = {
    "exact_f1_receipt": "#2563eb",
    "package_reported_f1": "#ea580c",
    "legacy_source_reported_368": "#64748b",
    "legacy_conditional_reconstruction_368": "#7c3aed",
}
CATEGORY_MARKERS = {
    "exact_f1_receipt": "o",
    "package_reported_f1": "s",
    "legacy_source_reported_368": "^",
    "legacy_conditional_reconstruction_368": "D",
}
MODEL_LABELS_BY_ID = {
    "2ebpj2er": "a.9 student / e6",
    "0srim121": "scratch Muon E4",
    "0hikjxpr": "student/scratch 50:50",
    "19w7a96y": "Older student T256",
    "1ykwb95u": "EMA6 S3/S4",
    "246401vj": "mixed EMA6 / since S4",
    "1bmxngta": "Since2",
    "21svxfgi": "EMA4 LPFT",
    "05ytuk8y": "S4 LPFT",
    "1jdh5rtz": "Base35",
    "1x8nyiwi": "S4 FT",
}


def sha256_file(path: Path) -> str:
    """Hash a small tabular input or generated artifact incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path, project_root: Path) -> str:
    """Prefer repository-relative paths while supporting external output folders."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV with its header and reject an empty input."""
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def numeric_value(row: dict[str, str], key: str) -> float:
    """Parse a finite floating-point field from a CSV row."""
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric field {key!r} for {row.get('submission_id')}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key!r} for {row.get('submission_id')}")
    return value


def classify_protocol(protocol: str) -> str:
    """Map the four source protocols to visibly separate plot categories."""
    if protocol == "fold1_exact_receipt":
        return "exact_f1_receipt"
    if protocol == "fold1_package_reported":
        return "package_reported_f1"
    if protocol.startswith("source-reported 368-group metric"):
        return "legacy_source_reported_368"
    if protocol.startswith("conditional v2-CPU reconstruction"):
        return "legacy_conditional_reconstruction_368"
    raise ValueError(f"unregistered validation protocol: {protocol!r}")


def bind_rows(
    candidates: list[dict[str, str]], coverage: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Join numeric candidate rows to coverage records and verify both scores."""
    coverage_by_id: dict[str, dict[str, str]] = {}
    for row in coverage:
        submission_id = row.get("submission_id", "").strip()
        if submission_id:
            if submission_id in coverage_by_id:
                raise ValueError(f"duplicate submission in endpoint coverage: {submission_id}")
            coverage_by_id[submission_id] = row

    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        submission_id = candidate.get("submission_id", "").strip()
        if not submission_id or submission_id in seen_ids:
            raise ValueError(f"missing or duplicate candidate submission ID: {submission_id!r}")
        seen_ids.add(submission_id)
        observed = coverage_by_id.get(submission_id)
        if observed is None or observed.get("portal_status") != "SCORE":
            raise ValueError(f"candidate has no scored endpoint coverage row: {submission_id}")

        public_score = numeric_value(candidate, "public_score_outcome")
        local_score = numeric_value(candidate, "local_metric")
        if (
            not observed.get("public_score_outcome", "").strip()
            or not observed.get("local_metric", "").strip()
        ):
            raise ValueError(f"coverage row lacks paired scores for {submission_id}")
        if not math.isclose(
            public_score, numeric_value(observed, "public_score_outcome"), abs_tol=1e-12
        ):
            raise ValueError(f"public score differs between inputs for {submission_id}")
        if not math.isclose(local_score, numeric_value(observed, "local_metric"), abs_tol=1e-12):
            raise ValueError(f"local score differs between inputs for {submission_id}")
        if candidate["local_metric_protocol"] != observed["local_metric_protocol"]:
            raise ValueError(f"protocol differs between inputs for {submission_id}")

        category = classify_protocol(candidate["local_metric_protocol"])
        if submission_id not in MODEL_LABELS_BY_ID:
            raise ValueError(f"no reviewed display label for submission {submission_id}")
        result.append(
            {
                "submission_id": submission_id,
                "model_label": MODEL_LABELS_BY_ID[submission_id],
                "public_score": public_score,
                "local_metric": local_score,
                "public_minus_local": public_score - local_score,
                "protocol": candidate["local_metric_protocol"],
                "protocol_class": category,
                "protocol_label": CATEGORY_LABELS[category],
                "precision_scope": candidate["local_metric_precision_scope"],
                "ancestry_status": candidate["local_metric_ancestry_status"],
                "ancestry_cluster": candidate["ancestry_cluster_for_validation"],
                "member_specs_and_weights": candidate["member_specs_and_weights"],
                "package_alias": candidate["package_alias"],
                "source_evidence_paths": candidate["source_evidence_paths"],
            }
        )

    counts = Counter(str(row["protocol_class"]) for row in result)
    if dict(counts) != EXPECTED_PROTOCOL_COUNTS:
        raise ValueError(f"protocol counts changed: {dict(counts)}")
    return sorted(
        result, key=lambda row: (CATEGORY_ORDER.index(row["protocol_class"]), row["submission_id"])
    )


def write_plot_data(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact, auditable CSV with the exact plotted values."""
    columns = (
        "submission_id",
        "model_label",
        "public_score",
        "local_metric",
        "public_minus_local",
        "protocol",
        "protocol_class",
        "protocol_label",
        "precision_scope",
        "ancestry_status",
        "ancestry_cluster",
        "member_specs_and_weights",
        "package_alias",
        "source_evidence_paths",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_category(axis: Any, rows: list[dict[str, Any]], category: str, panel_number: int) -> None:
    """Draw one protocol-specific scatter facet with labeled submission IDs."""
    category_rows = [row for row in rows if row["protocol_class"] == category]
    marker = CATEGORY_MARKERS[category]
    hollow = category == "legacy_conditional_reconstruction_368"
    axis.scatter(
        [row["local_metric"] for row in category_rows],
        [row["public_score"] for row in category_rows],
        marker=marker,
        s=62,
        facecolors="none" if hollow else CATEGORY_COLORS[category],
        edgecolors=CATEGORY_COLORS[category],
        linewidths=1.5,
        zorder=3,
    )

    label_offsets = {
        "a.9 student / e6": (8, 11),
        "scratch Muon E4": (8, -17),
        "student/scratch 50:50": (8, -32),
        "Older student T256": (8, 8),
        "EMA6 S3/S4": (8, 12),
        "mixed EMA6 / since S4": (-112, -17),
        "Since2": (-30, 15),
        "EMA4 LPFT": (-43, -16),
        "S4 LPFT": (-50, -19),
        "Base35": (-42, 8),
        "S4 FT": (8, 10),
    }
    for row in category_rows:
        axis.annotate(
            row["model_label"],
            (row["local_metric"], row["public_score"]),
            xytext=label_offsets[row["model_label"]],
            textcoords="offset points",
            fontsize=7.5,
            color="#1f2937",
            arrowprops={
                "arrowstyle": "-",
                "color": "#94a3b8",
                "linewidth": 0.55,
                "shrinkA": 1.5,
                "shrinkB": 3.5,
            },
        )

    axis.plot(
        [0.655, 0.675],
        [0.655, 0.675],
        color="#94a3b8",
        linestyle="--",
        linewidth=1,
        zorder=1,
    )
    axis.set_xlim(0.655, 0.705)
    axis.set_ylim(0.655, 0.675)
    axis.xaxis.set_major_locator(MultipleLocator(0.01))
    axis.yaxis.set_major_locator(MultipleLocator(0.005))
    axis.grid(True, color="#e2e8f0", linewidth=0.7)
    axis.set_title(
        f"{CATEGORY_LABELS[category]}  (n={len(category_rows)})", loc="left", fontsize=10
    )
    axis.set_xlabel("Local validation score")
    if panel_number in (0, 2):
        axis.set_ylabel("Public score")
    else:
        axis.set_ylabel("")


def make_figure(path_png: Path, path_svg: Path, rows: list[dict[str, Any]]) -> None:
    """Create protocol-faceted scatter plots and the corresponding gap panel."""
    fig = plt.figure(figsize=(15.2, 9.2))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.06,
        right=0.985,
        top=0.865,
        bottom=0.12,
        wspace=0.34,
        hspace=0.34,
        width_ratios=(1.0, 1.0, 1.15),
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    for index, (axis, category) in enumerate(zip(axes, CATEGORY_ORDER, strict=True)):
        _plot_category(axis, rows, category, index)

    gap_axis = fig.add_subplot(grid[:, 2])
    ordered_rows = rows
    y_positions = list(range(len(ordered_rows)))
    for category in CATEGORY_ORDER:
        category_rows = [row for row in ordered_rows if row["protocol_class"] == category]
        hollow = category == "legacy_conditional_reconstruction_368"
        points = [
            (index, row)
            for index, row in enumerate(ordered_rows)
            if row["protocol_class"] == category
        ]
        gap_axis.scatter(
            [row["public_minus_local"] for _, row in points],
            [position for position, _ in points],
            marker=CATEGORY_MARKERS[category],
            s=50,
            facecolors="none" if hollow else CATEGORY_COLORS[category],
            edgecolors=CATEGORY_COLORS[category],
            linewidths=1.4,
            label=f"{CATEGORY_LABELS[category]} (n={len(category_rows)})",
            zorder=3,
        )

    gap_axis.axvline(0.0, color="#475569", linewidth=1.2)
    gap_axis.set_xlim(-0.047, 0.006)
    gap_axis.set_ylim(len(y_positions) - 0.5, -0.5)
    gap_axis.set_yticks(y_positions, [str(row["model_label"]) for row in ordered_rows])
    gap_axis.tick_params(axis="y", labelsize=7.5)
    for position, row in enumerate(ordered_rows):
        gap_axis.get_yticklabels()[position].set_color(CATEGORY_COLORS[row["protocol_class"]])
        gap_axis.annotate(
            f"{row['public_minus_local']:+.3f}",
            (row["public_minus_local"], position),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color="#334155",
        )
    previous_category = ordered_rows[0]["protocol_class"]
    for index, row in enumerate(ordered_rows[1:], start=1):
        if row["protocol_class"] != previous_category:
            gap_axis.axhline(index - 0.5, color="#cbd5e1", linewidth=0.8)
            previous_category = row["protocol_class"]
    gap_axis.xaxis.set_major_locator(MultipleLocator(0.01))
    gap_axis.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    gap_axis.set_xlabel("Public − local validation score")
    gap_axis.set_title("Paired gap by submission", loc="left", fontsize=10)
    fig.suptitle(
        "Local validation versus public outcome, separated by evidence protocol",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.935,
        "Latest C: public 0.6671 only; no matched local CV metric, so it is excluded from the points.",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#475569",
    )
    fig.text(
        0.5,
        0.025,
        "Gap = public − local. Row colors mark protocols; historical records share exposure/ancestry. "
        "The diagonal is equality, not a fit; gaps do not predict transfer.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#475569",
    )
    fig.savefig(path_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(
        path_svg,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": "Public outcomes versus local validation by protocol",
            "Description": "Eleven historical matched records, separated by validation protocol.",
        },
    )
    svg_contents = path_svg.read_text(encoding="utf-8")
    path_svg.write_text(
        "\n".join(line.rstrip() for line in svg_contents.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def build_receipt(
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    rows: list[dict[str, Any]],
    project_root: Path,
) -> dict[str, Any]:
    """Describe input binding, protocol counts, and limits of this plot."""
    protocol_summaries: dict[str, dict[str, float | int]] = {}
    for category in CATEGORY_ORDER:
        category_rows = [row for row in rows if row["protocol_class"] == category]
        gaps = [float(row["public_minus_local"]) for row in category_rows]
        protocol_summaries[category] = {
            "count": len(category_rows),
            "gap_min_public_minus_local": min(gaps),
            "gap_max_public_minus_local": max(gaps),
        }
    return {
        "schema": "wnn.public-validation-plot-receipt.r2",
        "created_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            key: {
                "path": display_path(path, project_root),
                "sha256": sha256_file(path),
            }
            for key, path in input_paths.items()
        },
        "included_rows": len(rows),
        "protocol_counts": {
            key: int(value)
            for key, value in Counter(str(row["protocol_class"]) for row in rows).items()
        },
        "protocol_gap_ranges": protocol_summaries,
        "latest_c": {
            "public_score": 0.6671,
            "local_metric": None,
            "plot_treatment": "annotated public-only outcome; omitted from paired scatter and gap panel",
            "source_status": "user-transcribed report; not present as a matched row in these CSVs",
        },
        "method": {
            "comparison": "paired local metric versus public score for rows with both numeric values",
            "facets": list(CATEGORY_ORDER),
            "pooled_regression_line": False,
            "equality_diagonal": True,
            "public_minus_local_gap": "public_score - local_metric",
            "scatter_label_leader_lines": "thin neutral-gray segments to each target point",
        },
        "limitations": [
            "All included validation protocols are retrospective and exposed.",
            "The three exact F1 receipts bind float checkpoint/pair predictions; exact native Q8 scores were not measured.",
            "The package-reported F1 row lacks established checkpoint digest and precision conversion.",
            "The two conditional v2-CPU rows reconstruct EMA6 member0 rather than the exact served ensemble.",
            "The five source-reported 368-group rows lack receipt-verified ancestry and native/package equivalence.",
            "Submission rows share ancestry and validation exposure; individual dots are not independent replications.",
            "No regression, calibration, or public-score forecast was fitted.",
        ],
        "outputs": {
            key: {
                "path": display_path(path, project_root),
                "sha256": sha256_file(path),
            }
            for key, path in output_paths.items()
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse project paths for the reproducible plot command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def main() -> None:
    """Validate the two source CSVs, then write the scatter, gap plot, and receipt."""
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = project_root / "research/campaign_2026-10-12/public_score_regression_r1"
    input_paths = {
        "candidate_features": data_root / "candidate_features_r1.csv",
        "endpoint_coverage": data_root / "endpoint_coverage_r1.csv",
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "plot_data_csv": output_dir / "PUBLIC_LOCAL_PLOT_DATA_R2.csv",
        "figure_png": output_dir / "PUBLIC_VS_LOCAL_BY_PROTOCOL_R2.png",
        "figure_svg": output_dir / "PUBLIC_VS_LOCAL_BY_PROTOCOL_R2.svg",
    }
    receipt_path = output_dir / "PLOT_RECEIPT_R2.json"

    if any(path.exists() for path in (*output_paths.values(), receipt_path)):
        raise FileExistsError("refusing to replace an existing frozen plot artifact")
    candidates = read_csv(input_paths["candidate_features"])
    coverage = read_csv(input_paths["endpoint_coverage"])
    rows = bind_rows(candidates, coverage)
    if len(rows) != 11:
        raise ValueError(f"expected 11 matched records, observed {len(rows)}")

    write_plot_data(output_paths["plot_data_csv"], rows)
    make_figure(output_paths["figure_png"], output_paths["figure_svg"], rows)
    receipt = build_receipt(input_paths, output_paths, rows, project_root)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "rows": len(rows),
                "protocol_counts": receipt["protocol_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
