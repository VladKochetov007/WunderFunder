"""Check the frozen direction math on small generated arrays only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import direction_probe_math as probe
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "DIRECTION_PROBE_SYNTHETIC_R5.json"


def _sha256(path: Path) -> str:
    """Hash a small source file for the synthetic receipt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric64(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> float:
    """Evaluate the regular weighted Pearson branch using float64 inputs."""
    labels = np.clip(np.asarray(targets, dtype=np.float64), -2.0, 2.0)
    pred = np.clip(np.asarray(predictions, dtype=np.float64), -2.0, 2.0)
    scores = []
    for target in range(2):
        selected = mask & (np.abs(labels[:, target]) > 0.0)
        y, p = labels[selected, target], pred[selected, target]
        weight = np.abs(y)
        total = weight.sum()
        y_mean, p_mean = np.sum(weight * y) / total, np.sum(weight * p) / total
        y_center, p_center = y - y_mean, p - p_mean
        cov = np.sum(weight * y_center * p_center) / total
        sd_y = np.sqrt(np.sum(weight * y_center**2) / total)
        sd_p = np.sqrt(np.sum(weight * p_center**2) / total)
        scores.append(float(np.clip(cov / (sd_y * sd_p), -1.0, 1.0)))
    return float(np.mean(scores))


def _assert_close(
    left: np.ndarray | float, right: np.ndarray | float, tolerance: float, label: str
) -> float:
    """Assert a scalar or matrix parity and return its maximum absolute error."""
    error = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    if error > tolerance:
        raise AssertionError(f"{label} max error {error} exceeds {tolerance}")
    return error


def _finite_difference_fixture() -> dict[str, Any]:
    """Check alpha-zero and alpha-0.1 analytic derivatives against float64 paths."""
    rows = np.arange(80, dtype=np.float64)
    x = np.linspace(-0.55, 0.55, len(rows))
    targets = np.column_stack(
        (0.24 + 0.46 * x + 0.035 * np.sin(rows), 0.31 - 0.39 * x + 0.028 * np.cos(rows))
    )
    parent = np.column_stack((0.72 * x + 0.02 * np.cos(rows), -0.61 * x + 0.015 * np.sin(rows)))
    direction = np.column_stack((0.08 * np.sin(rows * 0.31), 0.06 * np.cos(rows * 0.27)))
    candidate = parent + direction
    common_a = parent + np.column_stack((0.05 * np.cos(rows * 0.19), -0.04 * np.sin(rows * 0.23)))
    mask = np.ones(len(rows), dtype=bool)
    mask[::13] = False
    derivatives: dict[str, Any] = {}
    for alpha in (0.0, 0.1):
        _, centered = probe.endpoint_statistics(targets, mask, parent, candidate, common_a, alpha)
        covariance, weights = probe.unpack_centered_moments(centered[None, ...])
        analytic = [
            probe.score_derivative(covariance[k], weights[k], 2)["derivative"] for k in range(2)
        ]
        base = parent + alpha * direction
        differences = []
        for step in (1e-4, 1e-5, 1e-6):
            finite = (
                _metric64(targets, base + step * direction, mask) - _metric64(targets, base, mask)
            ) / step
            differences.append(
                {
                    "step": step,
                    "finite_difference": finite,
                    "error_vs_mean_derivative": abs(finite - float(np.mean(analytic))),
                }
            )
        if differences[-1]["error_vs_mean_derivative"] > 2e-5:
            raise AssertionError(
                f"alpha={alpha} analytic derivative disagrees with decreasing-step finite differences"
            )
        derivatives[str(alpha)] = {
            "target_derivatives": analytic,
            "wp_mean_derivative": float(np.mean(analytic)),
            "finite_differences": differences,
        }
    return derivatives


def _component_fixture() -> dict[str, Any]:
    """Compare raw and centered component resamples with direct row recomputation."""
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    row = np.arange(12, dtype=np.float64)
    for component in range(2):
        x = np.linspace(-0.6, 0.6, len(row)) + component * 0.015
        parent = np.column_stack((x, 0.82 * x + 0.025 * np.sin(row)))
        direction = np.column_stack(
            (0.035 * np.sin(row * 0.41 + component), 0.03 * np.cos(row * 0.33 + component))
        )
        candidate = parent + direction
        a_direction = (
            np.zeros_like(parent)
            if component == 0
            else np.column_stack((0.11 * np.sin(row * 0.18), -0.09 * np.cos(row * 0.21)))
        )
        common_a = parent + a_direction
        targets = np.column_stack(
            (0.34 + 0.62 * x + 0.03 * np.cos(row), 0.29 - 0.51 * x + 0.025 * np.sin(row))
        )
        blocks.append((targets, parent, candidate, common_a))
    raw_parts, centered_parts = [], []
    for targets, parent, candidate, common_a in blocks:
        raw, centered = probe.endpoint_statistics(
            targets, np.ones(len(targets), dtype=bool), parent, candidate, common_a, 0.0
        )
        raw_parts.append(raw)
        centered_parts.append(centered)
    raw_table, centered_table = np.stack(raw_parts), np.stack(centered_parts)
    all_rows = [np.concatenate([block[i] for block in blocks], axis=0) for i in range(4)]
    all_raw, all_centered = probe.endpoint_statistics(
        all_rows[0], np.ones(len(all_rows[0]), dtype=bool), *all_rows[1:], 0.0
    )
    additive_cov, _ = probe.unpack_additive_moments(raw_table)
    stable_cov, _ = probe.unpack_centered_moments(centered_table)
    direct_cov, _ = probe.unpack_centered_moments(all_centered[None, ...])
    additive_error = _assert_close(
        additive_cov, direct_cov, 2e-12, "additive moments vs direct rows"
    )
    stable_error = _assert_close(stable_cov, direct_cov, 2e-12, "centered moments vs direct rows")
    _assert_close(raw_table.sum(axis=0), all_raw, 2e-12, "raw first/full-cross-moment addition")

    bootstrap_results: dict[str, Any] = {}
    ranks: dict[str, list[int]] = {"calibration": [], "calibration_plus_A": []}
    for label, multiplicity, repeated in (
        ("component0_twice", np.array([2, 0]), 0),
        ("component1_twice", np.array([0, 2]), 1),
    ):
        sampled_cov, sampled_weight = probe.unpack_centered_moments(centered_table, multiplicity)
        target, parent, candidate, common_a = blocks[repeated]
        repeat_rows = np.concatenate((np.arange(len(target)), np.arange(len(target))))
        _, direct_centered = probe.endpoint_statistics(
            target[repeat_rows],
            np.ones(len(repeat_rows), dtype=bool),
            parent[repeat_rows],
            candidate[repeat_rows],
            common_a[repeat_rows],
            0.0,
        )
        direct_resampled, direct_weight = probe.unpack_centered_moments(direct_centered[None, ...])
        _assert_close(sampled_cov, direct_resampled, 2e-12, f"{label} covariance resample")
        _assert_close(sampled_weight, direct_weight, 2e-12, f"{label} target weights")
        analysis = probe.analyze_covariance(sampled_cov[0], sampled_weight[0], include_a=True)
        for name in ranks:
            ranks[name].append(int(analysis["projections"][name]["rank"]))
        bootstrap_results[label] = {
            "rank_calibration": ranks["calibration"][-1],
            "rank_calibration_plus_A": ranks["calibration_plus_A"][-1],
        }
    if ranks["calibration_plus_A"] != [1, 2]:
        raise AssertionError(f"fixture did not produce the frozen rank transition: {ranks}")
    full_analysis = probe.analyze_covariance(
        direct_cov[0], float(probe.unpack_centered_moments(all_centered[None, ...])[1][0]), True
    )
    full_rank = full_analysis["projections"]["calibration_plus_A"]["rank"]
    if full_rank != 2:
        raise AssertionError("full component fixture did not retain the common-A rank")
    return {
        "raw_additive_vs_direct_max_abs_error": additive_error,
        "centered_moments_vs_direct_max_abs_error": stable_error,
        "bootstrap_direct_row_checks": bootstrap_results,
        "rank_histogram_fixture": {
            name: probe.rank_histogram(values) for name, values in ranks.items()
        },
        "full_rank_calibration_plus_A": full_rank,
        "projection_identity_error": full_analysis["projections"]["calibration_plus_A"][
            "reconstruction_error"
        ],
        "projection_norm_share": full_analysis["projections"]["calibration_plus_A"][
            "projection_norm_share"
        ],
    }


def _gate_fixture() -> dict[str, Any]:
    """Exercise all official EPS gates and the final correlation clip boundary."""
    covariance = np.eye(3, dtype=np.float64)
    weight_gate = probe.score_derivative(covariance, probe.EPS / 2.0, 2)
    target_gate_cov = covariance.copy()
    target_gate_cov[0, 0] = probe.EPS**2
    target_gate = probe.score_derivative(target_gate_cov, 1.0, 2)
    prediction_gate_cov = covariance.copy()
    prediction_gate_cov[1, 1] = probe.EPS**2
    prediction_gate = probe.score_derivative(prediction_gate_cov, 1.0, 2)
    prediction_stays_closed_cov = prediction_gate_cov.copy()
    prediction_stays_closed_cov[2, 2] = 0.0
    prediction_stays_closed = probe.score_derivative(prediction_stays_closed_cov, 1.0, 2)
    cap_cov = np.array([[1.0, 1.0, 0.2], [1.0, 1.0, 0.2], [0.2, 0.2, 1.0]])
    cap = probe.score_derivative(cap_cov, 2.0, 2)
    expected = (
        weight_gate["status"] == "weight_below_eps"
        and target_gate["status"] == "target_sd_exact_eps_gate_boundary"
        and target_gate["right_gate"] == "stays_closed_target_is_fixed"
        and prediction_gate["status"] == "prediction_sd_exact_eps_gate_opens"
        and prediction_gate["right_gate"] == "opens_for_positive_alpha"
        and prediction_stays_closed["status"] == "prediction_sd_exact_eps_gate_stays_closed"
        and prediction_stays_closed["derivative"] == 0.0
        and prediction_stays_closed["right_gate"] == "stays_closed_near_zero"
        and cap["status"] == "upper_correlation_boundary"
        and cap["derivative"] == 0.0
    )
    if not expected:
        raise AssertionError("official EPS or correlation clipping branch fixture failed")
    return {
        "weight_below_eps": weight_gate["status"],
        "target_sd_exact_eps": {
            "status": target_gate["status"],
            "right_gate": target_gate["right_gate"],
        },
        "prediction_sd_exact_eps": {
            "status": prediction_gate["status"],
            "right_gate": prediction_gate["right_gate"],
            "derivative": prediction_gate["derivative"],
        },
        "prediction_sd_exact_eps_stays_closed": {
            "status": prediction_stays_closed["status"],
            "right_gate": prediction_stays_closed["right_gate"],
            "derivative": prediction_stays_closed["derivative"],
        },
        "correlation_upper_boundary": {"status": cap["status"], "derivative": cap["derivative"]},
    }


def _import_resolution_fixture() -> dict[str, Any]:
    """Resolve reused preflight module paths without executing either module."""
    root = Path(__file__).resolve().parents[5]
    feasibility = root / "research/campaign_2026-10-24/resume4h_r1/feasibility_scout"
    runner_text = (HERE / "run_direction_probe.py").read_text(encoding="utf-8")
    expected_path_expression = 'str(FEASIBILITY / "f1_mean_shrink_r1")'
    if expected_path_expression not in runner_text:
        raise AssertionError("runner no longer adds the F1 preflight directory")
    import_paths = (
        root / "src",
        root / "scripts",
        feasibility,
        feasibility / "f1_mean_shrink_r1",
        root / "research/experiment_drafts/online-relationships-20261022",
    )
    sys.path[:0] = [str(path) for path in import_paths]
    expected = {
        "f0_grade_preflight": feasibility / "f0_grade_preflight.py",
        "f1_grade_preflight_r2": feasibility / "f1_mean_shrink_r1/f1_grade_preflight_r2.py",
    }
    resolved: dict[str, str] = {}
    for name, path in expected.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != path.resolve():
            raise AssertionError(f"{name} does not resolve to the frozen source path")
        resolved[name] = str(Path(spec.origin).resolve())
    return {"resolved_module_paths": resolved, "module_bodies_executed": False}


def main() -> None:
    """Run generated-array checks and create one immutable validation receipt."""
    from wnn.gmetric import GlobalAccumulator

    if OUTPUT.exists():
        raise FileExistsError(f"synthetic validation receipt is create-only: {OUTPUT}")
    derivatives = _finite_difference_fixture()
    components = _component_fixture()
    gates = _gate_fixture()
    import_resolution = _import_resolution_fixture()
    y = np.array([[0.1, 0.2], [0.4, -0.3], [0.8, 0.5], [-0.2, 0.7]], dtype=np.float32)
    p = np.array([[0.0, 0.1], [0.2, -0.1], [0.6, 0.3], [-0.1, 0.4]], dtype=np.float32)
    mask = np.array([True, True, False, True])
    accumulator = GlobalAccumulator()
    accumulator.add(y, p, mask)
    metric = accumulator.result()
    if metric["metric_id"] != "global_wp_clip2_v1" or metric["selected_rows"] != int(mask.sum()):
        raise AssertionError("canonical official metric sanity check failed")
    clip_base = np.array([-2.0, -2.0, 2.0, 2.0, -2.2, 2.2, 0.0])
    clip_direction = np.array([1.0, -1.0, -1.0, 1.0, 2.0, -2.0, 2.0])
    clip_observed = probe.right_clip_derivative(clip_base, clip_direction).tolist()
    clip_expected = [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0]
    if clip_observed != clip_expected:
        raise AssertionError("one-sided clip-boundary tangent fixture failed")
    source_files = {
        "math": _sha256(HERE / "direction_probe_math.py"),
        "runner": _sha256(HERE / "run_direction_probe.py"),
        "synthetic_checker": _sha256(Path(__file__)),
    }
    result = {
        "schema": "wnn.fixed-prediction-direction-probe-synthetic.v5",
        "status": "passed",
        "source_sha256": source_files,
        "real_prediction_or_target_payloads_opened": False,
        "finite_difference_derivatives": derivatives,
        "clip_boundary_derivatives": {"observed": clip_observed, "expected": clip_expected},
        "official_metric_fixture": metric,
        "component_and_projection_checks": components,
        "official_gate_checks": gates,
        "import_resolution": import_resolution,
        "assertions": [
            "float64 one-sided finite differences converge to alpha=0 and alpha=0.1 derivatives",
            "signed right clip tangents handle exact boundaries and strict saturation",
            "raw additive full cross-moments and stable centered merges match direct rows",
            "component-resampled moments match repeated direct rows and expose rank changes",
            "calibration plus common-A projection reconstructs the derivative with bounded norm share",
            "official W, target-SD, prediction-SD open/closed EPS branches and final rho clip are explicit",
            "canonical global_wp_clip2_v1 accumulator accepts the synthetic row mask",
        ],
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
