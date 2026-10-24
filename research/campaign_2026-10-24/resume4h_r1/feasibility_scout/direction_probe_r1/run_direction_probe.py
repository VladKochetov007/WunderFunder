"""Run the admitted fixed-prediction score-direction falsifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any  # Frozen producer receipts use runtime-defined fields.

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
FEASIBILITY = HERE.parent
ONLINE = ROOT / "research/experiment_drafts/online-relationships-20261022"
F0_ENDPOINTS = ("A_lag1", "B_lag1", "C_lag1", "B_lag16", "C_lag16")
F0_BC = {"B_lag1", "C_lag1", "B_lag16", "C_lag16"}
ROWS = 20_000
ALPHA = 0.1
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20_261_024


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load a small frozen JSON record and reject non-object roots."""
    if path.stat().st_size > 20_000_000:
        raise ValueError(f"metadata exceeds 20 MB: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def resolve(path_value: str) -> Path:
    """Resolve a repository-relative or absolute path."""
    path = Path(path_value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def verify_sources(recipe: dict[str, Any]) -> dict[str, str]:
    """Verify every frozen code, metric, review, and source-recipe pin."""
    manifest_record = recipe["source_manifest"]
    manifest_path = resolve(manifest_record["path"])
    if sha256_file(manifest_path) != manifest_record["sha256"]:
        raise ValueError("direction-probe source manifest SHA-256 changed")
    manifest = read_json(manifest_path)
    if manifest.get("files") != recipe["source_sha256"]:
        raise ValueError("source manifest and recipe pin maps disagree")
    observed: dict[str, str] = {}
    for relative, expected in recipe["source_sha256"].items():
        actual = sha256_file(resolve(relative))
        if actual != expected:
            raise ValueError(f"frozen direction-probe source changed: {relative}")
        observed[relative] = actual
    return dict(sorted(observed.items()))


def _verify_input_file_bindings(recipe: dict[str, Any]) -> None:
    """Require every recipe path/hash pair to match the independently pinned source map."""
    bindings = {
        "F0": (
            ("base_recipe_path", "base_recipe_sha256"),
            ("base_grade_path", "base_grade_sha256"),
            ("base_source_manifest_path", "base_source_manifest_sha256"),
            ("fixed_shrink_recipe_path", "fixed_shrink_recipe_sha256"),
            ("fixed_shrink_grade_path", "fixed_shrink_grade_sha256"),
            ("fixed_shrink_source_manifest_path", "fixed_shrink_source_manifest_sha256"),
            ("preflight_path", "preflight_sha256"),
        ),
        "F1": (
            ("fixed_shrink_recipe_path", "fixed_shrink_recipe_sha256"),
            ("fixed_shrink_grade_path", "fixed_shrink_grade_sha256"),
            ("fixed_shrink_source_manifest_path", "fixed_shrink_source_manifest_sha256"),
            ("preflight_path", "preflight_sha256"),
        ),
    }
    source_sha256 = recipe["source_sha256"]
    for fold, fields in bindings.items():
        spec = recipe["inputs"][fold]
        for path_key, sha_key in fields:
            path_value = spec[path_key]
            if source_sha256.get(path_value) != spec[sha_key]:
                raise ValueError(f"{fold} {path_key}/{sha_key} is not bound by the source map")
    expected_input_recipes = {
        "F0_base": recipe["inputs"]["F0"]["base_recipe_sha256"],
        "F0_fixed_shrink": recipe["inputs"]["F0"]["fixed_shrink_recipe_sha256"],
        "F1_fixed_shrink": recipe["inputs"]["F1"]["fixed_shrink_recipe_sha256"],
    }
    if recipe["input_recipe_sha256"] != expected_input_recipes:
        raise ValueError("input recipe digest summary differs from the F0/F1 input records")


def verify_admission(
    recipe: dict[str, Any], recipe_sha256: str, admission_sha256: str
) -> dict[str, Any]:
    """Require root admission to bind this exact recipe, inputs, and outputs."""
    path = resolve(recipe["admission_path"])
    if sha256_file(path) != admission_sha256:
        raise ValueError("direction-probe admission hash differs from the supplied pin")
    admission = read_json(path)
    expected = {
        "schema": "wnn.fixed-prediction-direction-probe-admission.v1",
        "status": "admitted",
        "recipe_sha256": recipe_sha256,
        "source_sha256": recipe["source_sha256"],
        "input_recipe_sha256": recipe["input_recipe_sha256"],
        "target_cache": recipe["target_cache"],
        "output_paths": {
            name: str(resolve(value)) for name, value in recipe["output_paths"].items()
        },
        "target_quality_reads_authorized": True,
        "new_fit_count": 0,
    }
    if any(admission.get(key) != value for key, value in expected.items()):
        raise ValueError("root admission does not bind the frozen probe contract")
    resources = admission.get("resources", {})
    if resources != recipe["resources"]:
        raise ValueError("root admission differs from the frozen worker resource contract")
    return {"path": str(path), "sha256": admission_sha256, "record": admission}


def _verify_saved_preflight(
    path_value: str, expected_sha256: str, observed: dict[str, Any], label: str
) -> dict[str, str]:
    """Require the reused label-free ancestry preflight to reproduce exactly."""
    path = resolve(path_value)
    if sha256_file(path) != expected_sha256 or read_json(path) != observed:
        raise ValueError(f"{label} source/ancestry preflight differs from its frozen receipt")
    return {"path": str(path), "sha256": expected_sha256}


def prepare_inputs(recipe: dict[str, Any]) -> dict[str, Any]:
    """Run the existing F0/F1 source-qualified preflights without target access."""
    sys.path[:0] = [
        str(ROOT / "src"),
        str(ROOT / "scripts"),
        str(FEASIBILITY),
        str(FEASIBILITY / "f1_mean_shrink_r1"),
        str(ONLINE),
    ]
    from f0_grade_preflight import prepare_f0_grade
    from f1_grade_preflight_r2 import prepare_f1_grade

    f0_spec, f1_spec = recipe["inputs"]["F0"], recipe["inputs"]["F1"]
    f0 = prepare_f0_grade(resolve(f0_spec["base_recipe_path"]), f0_spec["base_recipe_sha256"])
    f1 = prepare_f1_grade(
        resolve(f1_spec["fixed_shrink_recipe_path"]), f1_spec["fixed_shrink_recipe_sha256"]
    )
    f0_preflight = _verify_saved_preflight(
        f0_spec["preflight_path"], f0_spec["preflight_sha256"], f0.report, "F0"
    )
    f1_preflight = _verify_saved_preflight(
        f1_spec["preflight_path"], f1_spec["preflight_sha256"], f1.report, "F1"
    )
    f0_eval, f1_eval = f0_spec["evaluation"], f1_spec["evaluation"]
    f0_membership = f0.report.get("evaluation_membership_sha256")
    f1_membership = f1.report.get("evaluation", {}).get("row_refs_sha256")
    if (
        len(f0.evaluation_groups) != f0_eval["group_count"]
        or len(set(f0.components)) != f0_eval["component_count"]
        or f0_membership != f0_eval["membership_sha256"]
    ):
        raise ValueError("F0 group/component order differs from the source-qualified recipe")
    if (
        len(f1.evaluation_groups) != f1_eval["group_count"]
        or len(set(f1.components)) != f1_eval["component_count"]
        or f1_membership != f1_eval["membership_sha256"]
    ):
        raise ValueError("F1 group/component order differs from its source-qualified recipe")
    f0_shrink = read_json(resolve(f0_spec["fixed_shrink_recipe_path"]))
    if resolve(f0_spec["parent_prediction"]["path"]) != f0.parent_prediction_path:
        raise ValueError("F0 parent path differs from the frozen recipe")
    if f0_spec["parent_prediction"]["sha256"] != recipe["prediction_sha256"]["F0_parent"]:
        raise ValueError("F0 parent prediction pin differs from the probe recipe")
    for name in F0_ENDPOINTS:
        candidate = f0_shrink["candidate_predictions"][name]
        if resolve(candidate["prediction_path"]) != f0.candidate_prediction_paths[name]:
            raise ValueError(f"F0 {name} path differs between the R3 recipe and preflight")
        if candidate["prediction_sha256"] != recipe["prediction_sha256"][f"F0_{name}"]:
            raise ValueError(f"F0 {name} payload pin differs between the R3 recipe and probe")
    alias = f0_shrink["omitted_duplicate"]
    if (
        alias["prediction_sha256"]
        != f0_shrink["candidate_predictions"]["A_lag1"]["prediction_sha256"]
    ):
        raise ValueError("F0 A lag-16 is not the frozen duplicate of A lag-1")
    f0_grade = read_json(resolve(f0_spec["base_grade_path"]))
    f0_shrink_grade = read_json(resolve(f0_spec["fixed_shrink_grade_path"]))
    f1_grade = read_json(resolve(f1_spec["fixed_shrink_grade_path"]))
    if (
        resolve(f1_spec["parent_prediction"]["path"]) != f1.parent_prediction_path
        or f1_spec["parent_prediction"]["sha256"] != recipe["prediction_sha256"]["F1_parent"]
    ):
        raise ValueError("F1 parent prediction differs from its frozen source pin")
    if (
        resolve(f1_spec["candidate_prediction"]["path"]) != f1.candidate_prediction_path
        or f1_spec["candidate_prediction"]["sha256"] != recipe["prediction_sha256"]["F1_A_lag1"]
    ):
        raise ValueError("F1 candidate prediction differs from its frozen source pin")
    return {
        "f0": f0,
        "f1": f1,
        "f0_grade": f0_grade,
        "f0_shrink_grade": f0_shrink_grade,
        "f1_grade": f1_grade,
        "f0_shrink_recipe": f0_shrink,
        "preflights": {"F0": f0_preflight, "F1": f1_preflight},
    }


def _prediction_paths(prepared: dict[str, Any]) -> dict[str, Path]:
    """Return each distinct endpoint payload exactly once."""
    f0, f1 = prepared["f0"], prepared["f1"]
    paths = {
        "F0_parent": f0.parent_prediction_path,
        **{f"F0_{name}": f0.candidate_prediction_paths[name] for name in F0_ENDPOINTS},
        "F1_parent": f1.parent_prediction_path,
        "F1_A_lag1": f1.candidate_prediction_path,
    }
    return paths


def verify_prediction_headers_and_hashes(
    prepared: dict[str, Any], recipe: dict[str, Any]
) -> dict[str, str]:
    """Check source-pinned payload identity and NPY headers before target reads."""
    import numpy as np

    observed: dict[str, str] = {}
    for name, path in _prediction_paths(prepared).items():
        expected = recipe["prediction_sha256"][name]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{name} prediction payload differs from its frozen SHA-256")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_groups = 457 if name.startswith("F1_") else 468
        if array.shape != (expected_groups, ROWS, 2) or array.dtype.str != "<f4":
            raise ValueError(f"{name} prediction shape or dtype changed")
        observed[name] = actual
    return dict(sorted(observed.items()))


def _compare_score(
    observed: dict[str, Any], expected: dict[str, Any], label: str
) -> dict[str, Any]:
    """Require exact selected-row counts and 1e-12 parity with the old grade."""
    errors: dict[str, float] = {}
    if observed["selected_rows"] != expected["selected_rows"]:
        raise ValueError(f"{label} selected-row count differs from its frozen grade")
    for metric in ("wp", "t0", "t1"):
        errors[metric] = float(observed[metric] - expected[metric])
        if abs(errors[metric]) > 1e-12:
            raise ValueError(f"{label} {metric} differs from its frozen grade")
    return {"expected": expected, "observed": observed, "signed_error": errors}


def _score_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float | int]:
    """Return a matched difference between official score records."""
    return {
        "wp": float(left["wp"] - right["wp"]),
        "t0": float(left["t0"] - right["t0"]),
        "t1": float(left["t1"] - right["t1"]),
        "selected_rows": int(left["selected_rows"]),
    }


def _component_draws(rng: Any, count: int) -> Any:
    """Generate fixed paired connected-component bootstrap multiplicities."""
    import numpy as np

    choices = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
    return np.stack([np.bincount(row, minlength=count) for row in choices]).astype(np.int16)


def _finite_mean(left: float | None, right: float | None) -> float | None:
    """Average target derivatives only when both target branches are defined."""
    return None if left is None or right is None else 0.5 * (left + right)


def _bootstrap_endpoint(
    moments_zero: Any,
    moments_point_one: Any,
    draws: Any,
    include_a: bool,
    math_module: Any,
) -> dict[str, Any]:
    """Recompute derivatives, projections, and ranks from paired component moments."""
    import numpy as np

    projection_names = ["calibration", "calibration_plus_A"] if include_a else ["calibration"]
    sample_names = [
        "D_at_0_target0",
        "D_at_0_target1",
        "D_at_0_wp_mean",
        "D_at_0_1_target0",
        "D_at_0_1_target1",
        "D_at_0_1_wp_mean",
    ]
    for name in projection_names:
        sample_names.extend(
            f"{name}_{measure}_target{target}"
            for measure in ("projected_derivative", "remaining_derivative", "projection_norm_share")
            for target in (0, 1)
        )
        sample_names.extend(
            f"{name}_{measure}_wp_mean"
            for measure in ("projected_derivative", "remaining_derivative", "projection_norm_share")
        )
    samples: dict[str, list[float]] = {name: [] for name in sample_names}
    ranks = {name: [[], []] for name in projection_names}

    for multiplicity in draws:
        cov_zero, weight_zero = math_module.unpack_centered_moments(moments_zero, multiplicity)
        cov_point_one, weight_point_one = math_module.unpack_centered_moments(
            moments_point_one, multiplicity
        )
        analyses = [
            math_module.analyze_covariance(cov_zero[target], weight_zero[target], include_a)
            for target in range(2)
        ]
        derivative_zero = [item["total"]["derivative"] for item in analyses]
        derivative_point_one = [
            math_module.score_derivative(cov_point_one[target], weight_point_one[target], 2)[
                "derivative"
            ]
            for target in range(2)
        ]
        for key, value in zip(
            ("D_at_0_target0", "D_at_0_target1", "D_at_0_1_target0", "D_at_0_1_target1"),
            (*derivative_zero, *derivative_point_one),
            strict=True,
        ):
            samples[key].append(float("nan") if value is None else float(value))
        for name, values in (
            ("D_at_0_wp_mean", derivative_zero),
            ("D_at_0_1_wp_mean", derivative_point_one),
        ):
            mean = _finite_mean(values[0], values[1])
            samples[name].append(float("nan") if mean is None else mean)
        for projection_name in projection_names:
            target_values: dict[str, list[float | None]] = {
                measure: []
                for measure in (
                    "projected_derivative",
                    "remaining_derivative",
                    "projection_norm_share",
                )
            }
            for target, analysis in enumerate(analyses):
                projection = analysis["projections"].get(projection_name)
                if projection is None:
                    for measure in target_values:
                        target_values[measure].append(None)
                    continue
                ranks[projection_name][target].append(int(projection["rank"]))
                for measure, value in projection.items():
                    if measure in target_values:
                        target_values[measure].append(value)
            for measure, values in target_values.items():
                for target, value in enumerate(values):
                    samples[f"{projection_name}_{measure}_target{target}"].append(
                        float("nan") if value is None else float(value)
                    )
                mean = _finite_mean(values[0], values[1])
                samples[f"{projection_name}_{measure}_wp_mean"].append(
                    float("nan") if mean is None else mean
                )
    return {
        "draws": int(len(draws)),
        "seed": BOOTSTRAP_SEED,
        "unit": "connected component, paired within endpoint and target",
        "intervals": {
            name: math_module.summarize(np.asarray(values)) for name, values in samples.items()
        },
        "rank_histograms_by_target": {
            name: [math_module.rank_histogram(values) for values in target_ranks]
            for name, target_ranks in ranks.items()
        },
    }


def derivative_report(
    moments_zero: Any,
    moments_point_one: Any,
    draws: Any,
    include_a: bool,
    math_module: Any,
) -> dict[str, Any]:
    """Summarize two-target real-path derivatives and alpha-zero projections."""
    cov_zero, weight_zero = math_module.unpack_centered_moments(moments_zero)
    cov_point_one, weight_point_one = math_module.unpack_centered_moments(moments_point_one)
    per_target: list[dict[str, Any]] = []
    for target in range(2):
        at_zero = math_module.analyze_covariance(cov_zero[target], weight_zero[target], include_a)
        at_point_one = math_module.score_derivative(
            cov_point_one[target], weight_point_one[target], 2
        )
        per_target.append(
            {
                "target": target,
                "D_at_0": at_zero["total"],
                "alpha0_projection": at_zero["projections"],
                "projection_status": at_zero["projection_status"],
                "calibration_control_derivatives_at_0": at_zero["control_derivatives"],
                "D_at_0_1": at_point_one,
            }
        )
    d0 = [item["D_at_0"]["derivative"] for item in per_target]
    d01 = [item["D_at_0_1"]["derivative"] for item in per_target]
    return {
        "per_target": per_target,
        "D_at_0_wp_mean": _finite_mean(d0[0], d0[1]),
        "D_at_0_1_wp_mean": _finite_mean(d01[0], d01[1]),
        "bootstrap": _bootstrap_endpoint(
            moments_zero, moments_point_one, draws, include_a, math_module
        ),
        "alpha0_projection_only": True,
        "derivatives_use_real_arithmetic": True,
        "fixed_alpha0_1_prediction_uses_float32_recipe_arithmetic": True,
        "coefficients": "evaluation-label-weighted diagnostic only; never export or serve",
    }


def _reproduction_checks(
    fold: str,
    observed: dict[str, Any],
    prepared: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Check parent, raw, and fixed-alpha scores against prior frozen grades."""
    checks: dict[str, Any] = {}
    if fold == "F0":
        parent_expected = prepared["f0_grade"]["endpoint_scores"]["parent"]["overall"]
        checks["parent"] = _compare_score(observed["parent"], parent_expected, "F0 parent")
        for name in candidate_names:
            checks[name] = {
                "raw_alpha1": _compare_score(
                    observed[f"raw_{name}"],
                    prepared["f0_shrink_grade"]["baseline_raw_scores"][name]["overall"],
                    f"F0 {name} raw",
                ),
                "fixed_alpha0_1": _compare_score(
                    observed[f"shrink_{name}"],
                    prepared["f0_shrink_grade"]["shrunken_scores"][name]["overall"],
                    f"F0 {name} fixed alpha",
                ),
            }
        return checks
    grade = prepared["f1_grade"]["endpoints"]
    checks["parent"] = _compare_score(observed["parent"], grade["parent"]["overall"], "F1 parent")
    name = candidate_names[0]
    checks[name] = {
        "raw_alpha1": _compare_score(
            observed[f"raw_{name}"], grade["raw_A_lag1"]["overall"], "F1 raw A"
        ),
        "fixed_alpha0_1": _compare_score(
            observed[f"shrink_{name}"], grade["shrink_A_lag1"]["overall"], "F1 fixed alpha A"
        ),
    }
    return checks


def _score_fold(
    fold: str,
    prepared_endpoint: Any,
    candidate_arrays: dict[str, Any],
    targets: Any,
    masks: Any,
    target_positions: dict[int, int],
    draws: Any,
    prepared: dict[str, Any],
    math_module: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score one source-qualified fold and accumulate complete component moments."""
    import numpy as np
    from f0_grade_scoring import _official_mask

    from wnn.gmetric import GlobalAccumulator

    groups = [int(group) for group in prepared_endpoint.evaluation_groups]
    components = np.asarray(prepared_endpoint.components, dtype=np.int64)
    component_ids = np.unique(components)
    component_position = {int(component): i for i, component in enumerate(component_ids)}
    if len(groups) != len(components) or not groups:
        raise ValueError(f"{fold} evaluation groups and component labels are misaligned")
    parent = np.load(prepared_endpoint.parent_prediction_path, mmap_mode="r", allow_pickle=False)
    if parent.shape != (len(groups), ROWS, 2) or parent.dtype.str != "<f4":
        raise ValueError(f"{fold} parent prediction header changed after preflight")
    scores = {"parent": GlobalAccumulator()}
    for name in candidate_arrays:
        scores[f"raw_{name}"] = GlobalAccumulator()
        scores[f"shrink_{name}"] = GlobalAccumulator()
    moments = {
        name: {
            "alpha0_raw": np.zeros((len(component_ids), 2, 73), dtype=np.float64),
            "alpha0_centered": np.zeros((len(component_ids), 2, 73), dtype=np.float64),
            "alpha0_1_raw": np.zeros((len(component_ids), 2, 13), dtype=np.float64),
            "alpha0_1_centered": np.zeros((len(component_ids), 2, 13), dtype=np.float64),
        }
        for name in candidate_arrays
    }
    clipping: dict[str, dict[str, Any]] = {
        name: {"alpha0_1": None, "alpha1": None} for name in candidate_arrays
    }
    common_a = candidate_arrays["A_lag1"]
    alpha_f32 = np.float32(ALPHA)
    for group_index, group in enumerate(groups):
        if group not in target_positions:
            raise ValueError(f"{fold} evaluation group missing from target row-group index")
        target_row = targets[target_positions[group]]
        mask_row = np.asarray(masks[target_positions[group]], dtype=bool)
        official = _official_mask(mask_row, 99, ROWS)
        parent_row = parent[group_index]
        scores["parent"].add(target_row, parent_row, official)
        component_index = component_position[int(components[group_index])]
        for name, candidate in candidate_arrays.items():
            candidate_row = candidate[group_index]
            # This matches the previously graded float32 operation order exactly.
            shrink_row = parent_row + alpha_f32 * (candidate_row - parent_row)
            scores[f"raw_{name}"].add(target_row, candidate_row, official)
            scores[f"shrink_{name}"].add(target_row, shrink_row, official)
            raw_zero, centered_zero = math_module.endpoint_statistics(
                target_row, official, parent_row, candidate_row, common_a[group_index], 0.0
            )
            raw_one, centered_one = math_module.endpoint_statistics(
                target_row, official, parent_row, candidate_row, common_a[group_index], ALPHA
            )
            moments[name]["alpha0_raw"][component_index] += raw_zero
            moments[name]["alpha0_centered"][component_index] = math_module.merge_centered_sequence(
                moments[name]["alpha0_centered"][component_index], centered_zero
            )
            moments[name]["alpha0_1_raw"][component_index] += raw_one
            moments[name]["alpha0_1_centered"][component_index] = (
                math_module.merge_centered_sequence(
                    moments[name]["alpha0_1_centered"][component_index], centered_one
                )
            )
            clipping[name]["alpha0_1"] = math_module.merge_clipping(
                clipping[name]["alpha0_1"],
                math_module.clipping_diagnostics(target_row, official, parent_row, shrink_row),
            )
            clipping[name]["alpha1"] = math_module.merge_clipping(
                clipping[name]["alpha1"],
                math_module.clipping_diagnostics(target_row, official, parent_row, candidate_row),
            )
    endpoint_scores = {name: accumulator.result() for name, accumulator in scores.items()}
    reproduction = _reproduction_checks(fold, endpoint_scores, prepared, tuple(candidate_arrays))
    endpoint_reports: dict[str, Any] = {}
    moment_arrays: dict[str, Any] = {}
    for name in candidate_arrays:
        include_a = fold == "F0" and name in F0_BC
        raw_score, shrink_score = endpoint_scores[f"raw_{name}"], endpoint_scores[f"shrink_{name}"]
        endpoint_reports[name] = {
            "scores": {
                "parent": endpoint_scores["parent"],
                "raw_alpha1": raw_score,
                "fixed_alpha0_1": shrink_score,
                "raw_alpha1_minus_parent": _score_delta(raw_score, endpoint_scores["parent"]),
                "fixed_alpha0_1_minus_parent": _score_delta(
                    shrink_score, endpoint_scores["parent"]
                ),
            },
            "reproduces_existing_grade": reproduction[name],
            "derivatives": derivative_report(
                moments[name]["alpha0_centered"],
                moments[name]["alpha0_1_centered"],
                draws,
                include_a,
                math_module,
            ),
            "clipping": {
                "parent_to_fixed_alpha0_1": clipping[name]["alpha0_1"],
                "parent_to_raw_alpha1": clipping[name]["alpha1"],
            },
        }
        for key, table in moments[name].items():
            moment_arrays[f"{fold}_{name}_{key}"] = table
    return (
        {
            "fold": fold,
            "evaluation_group_count": len(groups),
            "component_count": int(len(component_ids)),
            "source_qualified_membership_sha256": (
                prepared_endpoint.report.get("evaluation_membership_sha256")
                or prepared_endpoint.report["evaluation"]["row_refs_sha256"]
            ),
            "f0_unique_A_lag16_alias": {"A_lag16": "A_lag1"} if fold == "F0" else None,
            "endpoint_scores": endpoint_scores,
            "endpoints": endpoint_reports,
            "reproduction_checks": reproduction,
        },
        {"component_ids": component_ids, **moment_arrays},
    )


def run_probe(
    recipe: dict[str, Any],
    recipe_sha256: str,
    admission: dict[str, Any],
    prepared: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Stream targets once, reproduce frozen scores, and save fixed moment tables."""
    import direction_probe_math as math_module
    import numpy as np
    from f0_grade_scoring import _materialize_target_maps

    outputs = {name: resolve(value) for name, value in recipe["output_paths"].items()}
    if any(path.exists() or path.is_symlink() for path in outputs.values()):
        raise FileExistsError("direction-probe results are create-only")
    target_path = resolve(recipe["target_cache"]["path"])
    target_before = sha256_file(target_path)
    if target_before != recipe["target_cache"]["sha256"]:
        raise ValueError("official target cache differs from its pre-label SHA-256")
    prediction_before = verify_prediction_headers_and_hashes(prepared, recipe)

    f0, f1 = prepared["f0"], prepared["f1"]
    groups = [*map(int, f0.evaluation_groups), *map(int, f1.evaluation_groups)]
    target_prepared = SimpleNamespace(
        target_cache_path=target_path,
        evaluation_groups=groups,
        parent_recipe={"target_cache": recipe["target_cache"]},
    )
    f0_arrays = {
        name: np.load(f0.candidate_prediction_paths[name], mmap_mode="r", allow_pickle=False)
        for name in F0_ENDPOINTS
    }
    f1_arrays = {"A_lag1": np.load(f1.candidate_prediction_path, mmap_mode="r", allow_pickle=False)}
    scratch = resolve(recipe["resources"]["scratch"])
    if not scratch.is_dir():
        raise ValueError("declared physical-disk scratch directory is unavailable")
    rng_f0 = np.random.default_rng(BOOTSTRAP_SEED)
    rng_f1 = np.random.default_rng(BOOTSTRAP_SEED)
    draws_f0 = _component_draws(rng_f0, len(np.unique(f0.components)))
    draws_f1 = _component_draws(rng_f1, len(np.unique(f1.components)))
    with tempfile.TemporaryDirectory(prefix="direction-probe-targets-", dir=scratch) as temporary:
        targets, masks, row_groups, target_positions, _ = _materialize_target_maps(
            target_prepared, Path(temporary)
        )
        del row_groups
        f0_result, f0_moments = _score_fold(
            "F0", f0, f0_arrays, targets, masks, target_positions, draws_f0, prepared, math_module
        )
        f1_result, f1_moments = _score_fold(
            "F1", f1, f1_arrays, targets, masks, target_positions, draws_f1, prepared, math_module
        )
        del targets, masks
    target_after = sha256_file(target_path)
    prediction_after = {
        name: sha256_file(path) for name, path in _prediction_paths(prepared).items()
    }
    if target_after != target_before:
        raise ValueError("target archive changed during the score pass")
    if prediction_after != prediction_before:
        raise ValueError("one or more prediction payloads changed during the score pass")

    arrays: dict[str, Any] = {
        "F0_component_ids": np.asarray(f0_moments.pop("component_ids"), dtype=np.int64),
        "F1_component_ids": np.asarray(f1_moments.pop("component_ids"), dtype=np.int64),
        **f0_moments,
        **f1_moments,
    }
    with outputs["moments"].open("xb") as stream:
        np.savez(stream, **arrays)
    moment_receipt = {
        "path": str(outputs["moments"]),
        "sha256": sha256_file(outputs["moments"]),
        "bytes": outputs["moments"].stat().st_size,
        "keys": sorted(arrays),
        "storage": "complete per-component weighted first moments and full cross-product matrices",
    }
    result = {
        "schema": "wnn.fixed-prediction-direction-probe-result.v1",
        "status": "graded_exploratory_exposed_diagnostic",
        "recipe_sha256": recipe_sha256,
        "source_sha256": source_hashes,
        "input_recipe_sha256": recipe["input_recipe_sha256"],
        "preflight_receipts": prepared["preflights"],
        "admission": admission,
        "metric": recipe["metric"],
        "estimand": recipe["estimand"],
        "folds": {"F0": f0_result, "F1": f1_result},
        "moment_artifact": moment_receipt,
        "moment_table_schema": {
            "raw_packing": "[total_weight, weighted_first_moments[d], row_major_sum_w_x_outer_x[d,d]]",
            "centered_packing": "[total_weight, weighted_mean[d], row_major_centered_sum_w_x_outer_x[d,d]]",
            "alpha0_variables": list(math_module.ALPHA_ZERO_NAMES),
            "alpha0_1_variables": list(math_module.ALPHA_POINT_ONE_NAMES),
            "target_axis": "two target-specific tables; average score gradients only after calculation",
        },
        "prediction_sha256_before_score": prediction_before,
        "prediction_sha256_after_score": prediction_after,
        "target_cache_identity": {
            "path": str(target_path),
            "expected_sha256": recipe["target_cache"]["sha256"],
            "pre_score_sha256": target_before,
            "post_score_sha256": target_after,
            "stable_during_score": True,
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "connected component",
        },
        "exposure": recipe["exposure"],
        "interpretation_limits": recipe["interpretation_limits"],
        "new_fit_count": 0,
        "alpha_search": False,
        "endpoint_selection": False,
        "coefficient_export": False,
        "competition_upload": False,
        "expected_public_gain": None,
    }
    with outputs["results"].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    """Run only with an exact recipe and separately pinned root admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify frozen inputs and print a label-free receipt without scoring",
    )
    args = parser.parse_args()
    recipe_path = args.recipe.resolve(strict=True)
    recipe_sha256 = sha256_file(recipe_path)
    if recipe_sha256 != args.recipe_sha256:
        raise ValueError("direction-probe recipe differs from the supplied SHA-256")
    recipe = read_json(recipe_path)
    if recipe.get("status") != "ready_for_root_admission":
        raise ValueError("direction-probe recipe is not frozen and awaiting root admission")
    source_hashes = verify_sources(recipe)
    _verify_input_file_bindings(recipe)
    admission = verify_admission(recipe, recipe_sha256, args.admission_sha256)
    prepared = prepare_inputs(recipe)
    if args.preflight_only:
        import numpy as np

        prediction_hashes = verify_prediction_headers_and_hashes(prepared, recipe)
        target_path = resolve(recipe["target_cache"]["path"])
        target_sha256 = sha256_file(target_path)
        target_bytes = target_path.stat().st_size
        if (
            target_sha256 != recipe["target_cache"]["sha256"]
            or target_bytes != recipe["target_cache"]["file_bytes"]
        ):
            raise ValueError("target cache identity differs from the frozen preflight recipe")
        if any(prepared[fold].target_cache_path != target_path for fold in ("f0", "f1")):
            raise ValueError("F0/F1 preflights do not use the pinned target cache")
        receipt = {
            "schema": "wnn.fixed-prediction-direction-probe-preflight.v2",
            "status": "passed_label_free_preflight",
            "recipe_sha256": recipe_sha256,
            "source_sha256": source_hashes,
            "admission_sha256": args.admission_sha256,
            "preflight_receipts": prepared["preflights"],
            "folds": {
                "F0": {
                    "evaluation_group_count": len(prepared["f0"].evaluation_groups),
                    "component_count": int(np.unique(prepared["f0"].components).size),
                    "membership_sha256": prepared["f0"].report["evaluation_membership_sha256"],
                    "parent_prediction_path": str(prepared["f0"].parent_prediction_path),
                    "candidate_prediction_paths": {
                        name: str(prepared["f0"].candidate_prediction_paths[name])
                        for name in F0_ENDPOINTS
                    },
                },
                "F1": {
                    "evaluation_group_count": len(prepared["f1"].evaluation_groups),
                    "component_count": int(np.unique(prepared["f1"].components).size),
                    "membership_sha256": prepared["f1"].report["evaluation"]["row_refs_sha256"],
                    "parent_prediction_path": str(prepared["f1"].parent_prediction_path),
                    "candidate_prediction_path": str(prepared["f1"].candidate_prediction_path),
                },
            },
            "prediction_headers_and_sha256": prediction_hashes,
            "target_cache_identity": {
                "path": str(target_path),
                "file_bytes": target_bytes,
                "sha256": target_sha256,
                "target_values_read": False,
            },
            "prediction_values_read": False,
            "score_function_called": False,
        }
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
        return
    run_probe(recipe, recipe_sha256, admission, prepared, source_hashes)


if __name__ == "__main__":
    main()
