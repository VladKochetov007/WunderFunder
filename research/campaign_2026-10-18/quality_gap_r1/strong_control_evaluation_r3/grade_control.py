"""Compute float64 endpoint grades and paired component-bootstrap intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
from input_bindings import _load_inputs

from wnn.evaluation import paired_component_bootstrap, paired_component_moments
from wnn.gmetric import METRIC_ID, global_wp

BOOTSTRAP_COUNT = 2_000
BOOTSTRAP_SEED = 271_828
SCORE_TOLERANCE = 5e-10
COMPARISONS = ("wide", "fp32_mix", "q8")


def sha256_file(path: Path) -> str:
    """Hash one file with bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_grade_result(path: Path, expected_sha256: str) -> dict[str, object]:
    """Read the existing diagnostic report only after checking its arrival hash."""
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("diagnostic result exceeds the configured byte bound")
    digest = sha256_file(path)
    if len(expected_sha256) != 64 or digest != expected_sha256:
        raise ValueError("diagnostic result SHA-256 differs from the supplied captured hash")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("diagnostic result must contain a JSON object")
    if (
        value.get("schema") != "wnn.quality_gap.residual_diagnostic.v1"
        or value.get("metric_id") != METRIC_ID
    ):
        raise ValueError("diagnostic result schema or metric does not match the frozen diagnostic")
    return cast(dict[str, object], value)


def _group_statistics(
    targets: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    row_group: np.ndarray,
    component: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build canonical centered Welford moments per row group for a paired bootstrap."""
    groups = np.unique(row_group)
    candidate_moments: list[np.ndarray] = []
    reference_moments: list[np.ndarray] = []
    components: list[int] = []
    for group in groups:
        selected = np.flatnonzero(row_group == group)
        group_components = np.unique(component[selected])
        if len(group_components) != 1:
            raise ValueError(f"row group {int(group)} maps to multiple components")
        y = targets[selected][None, :, :]
        candidate_rows = candidate[selected][None, :, :]
        reference_rows = reference[selected][None, :, :]
        mask = np.ones((1, len(selected)), dtype=bool)
        candidate_stats, reference_stats = paired_component_moments(
            y, candidate_rows, reference_rows, mask
        )
        candidate_moments.append(candidate_stats[0])
        reference_moments.append(reference_stats[0])
        components.append(int(group_components[0]))
    return (
        np.stack(candidate_moments),
        np.stack(reference_moments),
        np.asarray(components, dtype=np.int64),
    )


def evaluate(
    inputs_path: Path,
    diagnostic_result_path: Path,
    diagnostic_result_sha256: str,
    output_path: Path,
) -> dict[str, object]:
    """Check canonical float64 grades and compute fixed paired component intervals."""
    if output_path.exists():
        raise FileExistsError(f"grade output already exists: {output_path}")
    manifest, arrays, predictions, _ = _load_inputs(inputs_path)
    provenance = cast(dict[str, object], manifest.get("derived_comparison", {}))
    if provenance.get("status") != "prepared_from_sealed_same_row_alignment":
        raise ValueError("input manifest is not the prepared strong-control comparison")
    diagnostic = _read_grade_result(diagnostic_result_path, diagnostic_result_sha256)
    diagnostic_models = cast(dict[str, dict[str, object]], diagnostic["models"])
    if diagnostic.get("selected_rows") != len(arrays["target"]):
        raise ValueError("diagnostic result row count differs from validated input arrays")
    diagnostic_inputs = cast(dict[str, object], diagnostic["inputs"])
    if diagnostic_inputs.get("manifest_sha256") != sha256_file(inputs_path):
        raise ValueError("diagnostic result was generated from different derived inputs")
    bundle = cast(dict[str, object], manifest["bundle"])
    if diagnostic_inputs.get("bundle_sha256") != bundle.get("sha256"):
        raise ValueError("diagnostic result bundle differs from validated same-row inputs")
    model_hashes = cast(dict[str, str], diagnostic_inputs["model_hashes"])
    model_specs = cast(dict[str, dict[str, object]], manifest["models"])
    if model_hashes.get("strong_control") != model_specs["strong_control"].get("sha256"):
        raise ValueError("diagnostic result control model hash differs from its input manifest")
    targets = arrays["target"]
    row_group = arrays["row_group"]
    component = arrays["component"]
    mask = np.ones(len(targets), dtype=bool)
    grades = {
        name: global_wp(targets, values, mask)
        for name, values in predictions.items()
        if name in ("wide", "deep", "fp32_mix", "q8", "strong_control")
    }
    if "strong_control" not in grades:
        raise ValueError("validated inputs do not include the aligned strong control")
    for name, grade in grades.items():
        report_grade = cast(dict[str, object], diagnostic_models[name]["score"])
        for metric in ("t0", "t1", "wp"):
            if abs(float(grade[metric]) - float(report_grade[metric])) > SCORE_TOLERANCE:
                raise ValueError(f"canonical and diagnostic grades differ for {name}/{metric}")

    comparisons: dict[str, object] = {}
    candidate = predictions["strong_control"]
    for baseline in COMPARISONS:
        candidate_stats, baseline_stats, group_components = _group_statistics(
            targets,
            candidate,
            predictions[baseline],
            row_group,
            component,
        )
        result = paired_component_bootstrap(
            candidate_stats,
            baseline_stats,
            group_components,
            n_boot=BOOTSTRAP_COUNT,
            seed=BOOTSTRAP_SEED,
            centered=True,
        )
        expected_delta = float(grades["strong_control"]["wp"]) - float(grades[baseline]["wp"])
        if abs(float(result["mean_diff"]) - expected_delta) > SCORE_TOLERANCE:
            raise AssertionError(f"component bootstrap point differs from pooled grade: {baseline}")
        comparisons[baseline] = result

    report: dict[str, object] = {
        "schema": "wnn.quality_gap.strong_control_component_grade.v1",
        "status": "passed",
        "metric_id": METRIC_ID,
        "inputs_sha256": sha256_file(inputs_path),
        "diagnostic_result_sha256": diagnostic_result_sha256,
        "selected_rows": len(targets),
        "row_groups": int(len(np.unique(row_group))),
        "components": int(len(np.unique(component))),
        "grades_float64": grades,
        "paired_component_bootstrap": {
            "method": "paired component resampling of canonical centered weighted moments; recompute pooled WP for every resample",
            "resamples": BOOTSTRAP_COUNT,
            "seed": BOOTSTRAP_SEED,
            "conditional_on_fixed_predictions": True,
            "comparisons": comparisons,
        },
        "interpretation_limitations": [
            "historical strong-control pipeline, not the exact public incumbent",
            "one F0 evaluation panel with prior exposure",
            "bootstrap conditions on fixed predictions and does not measure training-seed or public-transfer variation",
            "comparisons are descriptive; no tuning or candidate selection is performed",
        ],
        "canonical_metric_source_sha256": sha256_file(
            Path(__file__).resolve().parents[4] / "src/wnn/gmetric.py"
        ),
        "canonical_evaluation_source_sha256": sha256_file(
            Path(__file__).resolve().parents[4] / "src/wnn/evaluation.py"
        ),
        "diagnostic_input_validation_source_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "diagnostics/input_bindings.py"
        ),
    }
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {"status": "passed", "output": str(output_path), "sha256": sha256_file(output_path)}


def main() -> None:
    """Run the independent paired grade after aligned inputs and diagnostic results exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--diagnostic-result", type=Path, required=True)
    parser.add_argument("--diagnostic-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.inputs.resolve(strict=True),
                args.diagnostic_result.resolve(strict=True),
                args.diagnostic_result_sha256,
                args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
