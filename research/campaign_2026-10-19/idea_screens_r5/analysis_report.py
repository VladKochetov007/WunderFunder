"""Create-only archives and compact source-bound R5 result records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .analysis_fit import ALPHA, BASE_VIEW, MOMENT_NAMES, MomentFit
from .analysis_score import DECODERS

REFERENCE_CASE = "full64_raw_signed_ema_hl32"


def digest(path: Path) -> str:
    """Hash a bounded file incrementally for evidence bindings."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write finite, create-only JSON evidence."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def save_model(path: Path, fit: MomentFit) -> dict:
    """Save only numeric fitted quantities in a create-only NPZ archive."""
    with path.open("xb") as stream:
        np.savez(
            stream,
            base_mean=fit.mean,
            base_scale=fit.scale,
            moment_coefficients=fit.coefficients,
            moment_intercepts=fit.intercepts,
            mu_b=fit.mu_b,
            epsilon=fit.epsilon,
        )
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}


def _array_metadata(value: np.ndarray, include_range: bool = True) -> dict:
    """Describe model parameters without copying their values into JSON."""
    array = np.ascontiguousarray(value)
    metadata = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "byte_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    if include_range:
        metadata["min"] = float(array.min())
        metadata["max"] = float(array.max())
    return metadata


def reference_comparison(reference: dict, scoring: dict, reference_sha256: str) -> dict:
    """Compare fixed R5 endpoints with stored R3 selected-alpha component moments."""
    try:
        case = reference["cases"][REFERENCE_CASE]
        endpoint = case["outer"]["selected"]
        metric = endpoint["official_metric"]
        components = endpoint["component_moments"]
    except (KeyError, TypeError) as exc:
        raise ValueError("frozen R3 RESULT lacks its selected endpoint/component moments") from exc
    if int(case["selected_alpha"]) != 100:
        raise ValueError("frozen R3 reference is not the fixed selected-alpha-100 endpoint")
    if metric.get("metric_id") != "global_wp_clip2_v1":
        raise ValueError("frozen R3 reference metric differs")
    reference_by_component = {int(row["component"]): row for row in components}
    if len(reference_by_component) != len(components):
        raise ValueError("frozen R3 endpoint contains duplicate component moments")
    reference_rows = sum(int(row["scored_rows"]) for row in components)
    if int(metric.get("selected_rows", -1)) != reference_rows:
        raise ValueError("frozen R3 pooled row count differs from its component moments")
    current = scoring["eval"]["by_decoder"]
    current_rows = {int(row["component"]): row for row in current[DECODERS[0]]["component_moments"]}
    if set(current_rows) != set(reference_by_component):
        raise ValueError("R5 and frozen R3 evaluation components differ")
    paired_components = []
    for component in sorted(reference_by_component):
        old = reference_by_component[component]
        now = current_rows[component]
        if int(old["scored_rows"]) != int(now["scored_rows"]):
            raise ValueError(f"R5 and frozen R3 scored mask differs for component {component}")
        paired_components.append(
            {
                "component": component,
                "scored_rows": int(old["scored_rows"]),
                "r3_official_metric": old["official_metric"],
                "r3_additive_raw_sums_W_WY_WP_WYY_WPP_WYP": old[
                    "additive_raw_sums_W_WY_WP_WYY_WPP_WYP"
                ],
                "r5_wp_delta_by_decoder": {
                    decoder: _component_wp(current[decoder], component)
                    - float(old["official_metric"]["wp"])
                    if old["official_metric"] is not None
                    and _component_wp(current[decoder], component) is not None
                    else None
                    for decoder in DECODERS
                },
            }
        )
    deltas = {}
    for decoder in DECODERS:
        current_metric = current[decoder]["official_metric"]
        deltas[decoder] = {
            "r5_official_metric": current_metric,
            "r5_minus_r3_wp": (
                None if current_metric is None else float(current_metric["wp"] - metric["wp"])
            ),
        }
    return {
        "reference_result_sha256": reference_sha256,
        "case": REFERENCE_CASE,
        "alpha": 100,
        "metric_id": metric["metric_id"],
        "r3_official_metric": metric,
        "decoder_comparisons": deltas,
        "component_comparisons": paired_components,
        "interpretation": "descriptive recipe comparison; no R3 refit or regrade",
    }


def _component_wp(scored: dict, component: int) -> float | None:
    """Find one retained R5 official component metric."""
    for row in scored["component_moments"]:
        if int(row["component"]) == component:
            metric = row["official_metric"]
            return None if metric is None else float(metric["wp"])
    raise ValueError(f"R5 result lacks component {component}")


def build_result(
    output: Path,
    fit: MomentFit,
    fit_counts: dict,
    scoring: dict,
    diagnostics: dict,
    reference: dict,
    reference_sha256: str,
    sample_sha256: str,
    source_manifest_sha256: str,
    model_archive: dict,
    cost: dict,
) -> dict:
    """Write the fixed decoder comparison with counts, diagnostics and lineage."""
    reference_rows = reference_comparison(reference, scoring, reference_sha256)
    result = {
        "schema": "bounded_r5_conditional_moment_decoder_v1",
        "metric_id": "global_wp_clip2_v1",
        "sample_manifest_sha256": sample_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "reference_result_sha256": reference_sha256,
        "fit": {
            "base_view": BASE_VIEW,
            "precision": "R2 moment-head prediction outputs are float32; fit and decoder arithmetic are float64; canonical scorer applies its existing float32 input path",
            "moment_target_order": [
                f"{target}_{name}" for target in ("t0", "t1") for name in MOMENT_NAMES
            ],
            "alpha": ALPHA,
            "fit_calls": 1,
            "target_columns": 6,
            "decoder_count": 2,
            "fit_rule": "unweighted centered six-column ridge; intercepts unpenalized",
            "model_parameter_metadata": {
                "normalizer_mean": {
                    **_array_metadata(fit.mean),
                    "l2_norm": float(np.linalg.norm(fit.mean)),
                },
                "normalizer_scale": {
                    **_array_metadata(fit.scale),
                    "l2_norm": float(np.linalg.norm(fit.scale)),
                },
                "moment_coefficients": {
                    **_array_metadata(fit.coefficients, include_range=False),
                    "frobenius_norm": float(np.linalg.norm(fit.coefficients)),
                },
                "moment_intercepts": {
                    **_array_metadata(fit.intercepts),
                    "l2_norm": float(np.linalg.norm(fit.intercepts)),
                },
            },
            "moment_coefficient_l2_norm_by_column": np.linalg.norm(
                fit.coefficients, axis=0
            ).tolist(),
            "mu_b_by_target": fit.mu_b.tolist(),
            "epsilon_by_target": fit.epsilon.tolist(),
            "model_archive": model_archive,
        },
        "counts": {
            **fit_counts,
            "eval_sequences": scoring["eval"]["sequence_count"],
            "eval_components": scoring["eval"]["component_count"],
            "eval_scored_rows": scoring["eval"]["scored_rows"],
            "eval_scored_rows_by_component": scoring["mask_rows_by_component"]["eval"],
            "fit_scored_rows_by_component": scoring["mask_rows_by_component"]["fit"],
            "fit_full_mask_score_rows": scoring["fit"]["scored_rows"],
            "fit_full_mask_components": scoring["fit"]["component_count"],
            "fit_stride_rows": fit_counts["fit_sampled_rows"],
            "failures": [],
        },
        "scoring": {"fit": scoring["fit"], "eval": scoring["eval"]},
        "diagnostics": diagnostics,
        "r3_reference_comparison": reference_rows,
        "cost": cost,
        "interpretation_limits": [
            "The F0 panel is exposed development data, not a fresh holdout.",
            "The two decoders share the same fitted heads; only the denominator rule differs.",
            "Linear moment estimates are unconstrained and may violate conditional-moment bounds.",
            "The R3 comparison changes the fitted functional and loss; it is descriptive, not an isolated decoder effect.",
            "The row-dependent ratio assumption can fail under sparse hidden evaluation selection.",
            "No R3 fit, inference, or regrade is performed by this probe.",
        ],
    }
    result_path = output / "RESULT.json"
    write_json(result_path, result)
    write_json(
        output / "DONE.json",
        {
            "result_sha256": digest(result_path),
            "result_bytes": result_path.stat().st_size,
            "source_manifest_sha256": source_manifest_sha256,
            "sample_manifest_sha256": sample_sha256,
            "reference_result_sha256": reference_sha256,
            "model_archive": model_archive,
            "failures": [],
        },
    )
    return result
