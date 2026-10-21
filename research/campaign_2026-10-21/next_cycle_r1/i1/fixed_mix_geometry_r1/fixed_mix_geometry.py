"""Describe fixed F0 i1 mixture geometry without searching coefficients."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", Path(__file__).resolve().parents[5])).resolve()
PACKET = Path(__file__).resolve().parents[1]
ABC = PACKET / "interim_f0_abc"
AB = PACKET / "interim_f0_ab"
DIAG = PACKET.parent / "endpoint_diagnosis"
SCRATCH = Path("/home/vlad/development/wunder/research_tmp").resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(PACKET), str(ABC), str(AB), str(DIAG)]

import diagnose_native_h512 as moments  # noqa: E402
import grade_i1_f0_ab as interim  # noqa: E402
import grade_i1_f0_abc as frozen  # noqa: E402
import grade_i1_panel as panel  # noqa: E402

from wnn import gmetric  # noqa: E402

FROZEN_RESULT = ABC / "RESULT_R1.json"
FROZEN_RESULT_SHA256 = "d28a0c4fff38adac931ada71a970dc99536e1549094efee8cb304cc542997419"
DIAGNOSIS_SHA256 = "7259a35a34a789745a1c10672f105867f57435619c3119df5ef1228b375ae3f2"
TARGET_SHA256 = frozen.TARGET_SHA256
RUNS = frozen.RUNS
ARMS = ("A", "B", "C")
MODEL_NAMES = ("B", "C", "B90_C10", "B90_A10")
SCOPE_SLICES = {
    "overall": slice(99, panel.ROWS),
    **{f"rows_{start}_{stop}": slice(start, stop) for start, stop in panel.TIME_BINS},
}
MEMORY_LIMIT_BYTES = interim.MEMORY_LIMIT_BYTES


def _new_scope() -> dict[str, Any]:
    """Allocate small per-scope accumulators, never evaluation-sized arrays."""
    return {
        "rows": 0,
        "scores": {name: gmetric.GlobalAccumulator() for name in MODEL_NAMES},
        "pairs": {
            target: {
                kind: {name: np.zeros(6, dtype=np.float64) for name in MODEL_NAMES}
                for kind in ("raw", "clipped")
            }
            for target in range(2)
        },
        "raw_unweighted": {
            target: {name: np.zeros(6, dtype=np.float64) for name in MODEL_NAMES}
            for target in range(2)
        },
        "deltas": {
            target: {
                kind: {
                    candidate: {
                        part: np.zeros(6, dtype=np.float64)
                        for part in ("target_delta", "baseline_delta", "delta_variance")
                    }
                    for candidate in ("B90_C10", "B90_A10")
                }
                for kind in ("raw", "clipped")
            }
            for target in range(2)
        },
        "clip_counts": {name: np.zeros(2, dtype=np.int64) for name in MODEL_NAMES},
    }


def _stat(state: np.ndarray) -> dict[str, float] | None:
    """Summarize weighted centered moments, retaining empty targets as null."""
    if state[0] <= 0.0:
        return None
    value = moments.summary(state)
    value["target_variance"] = value["target_std"] ** 2
    value["prediction_variance"] = value["prediction_std"] ** 2
    return value


def _preflight() -> tuple[
    dict[str, Any], dict[str, np.ndarray], dict[str, Any], list[int], np.ndarray, dict[str, str]
]:
    """Verify frozen F0 identities and complete NPY coverage before labels load."""
    source_hashes = frozen._verify_sources()
    diagnosis_path = DIAG / "diagnose_native_h512.py"
    if panel.sha256(diagnosis_path) != DIAGNOSIS_SHA256:
        raise ValueError("weighted-moment helper differs from its inspected source")
    result_sha = panel.sha256(FROZEN_RESULT)
    if result_sha != FROZEN_RESULT_SHA256:
        raise ValueError("frozen F0 A/B/C grade bytes changed")
    result = json.loads(FROZEN_RESULT.read_text(encoding="utf-8"))
    if (
        result.get("schema") != "wnn.i1-f0-abc-interim-grade.v1"
        or result.get("status") != "interim_f0_abc_cell_only"
    ):
        raise ValueError("F0 result is not the expected interim A/B/C cell")
    for name, digest in source_hashes.items():
        if result.get("source_sha256", {}).get(name) != digest:
            raise ValueError(f"F0 grade source pin differs for {name}")
    target_path = ROOT / "cache/valid_targets.npz"
    if panel.sha256(target_path) != TARGET_SHA256:
        raise ValueError("official target cache differs from its frozen bytes")

    fold = 0
    split_path, subset_path = panel.SPLITS[fold], panel.SUBSETS[fold]
    split_sha, subset_sha = panel.file_sha256(split_path), panel.file_sha256(subset_path)
    if split_sha != frozen.SPLIT_SHA256 or subset_sha != frozen.SUBSET_SHA256:
        raise ValueError("F0 split or fit-subset bytes differ from the frozen cell")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    if subset.get("split_manifest_sha256") != split_sha:
        raise ValueError("F0 fitting subset does not reference the exact evaluation split")
    groups, components = panel._expected_membership(fold, split)
    if len(groups) != 468 or len(np.unique(components)) != 141:
        raise ValueError("F0 membership differs from the complete frozen fold")

    predictions: dict[str, np.ndarray] = {}
    identities: dict[str, dict[str, Any]] = {}
    frozen_identities = result["evaluation"]["prediction_identity_and_npy_coverage"]
    for arm in ARMS:
        prediction, identity = panel._verify_run(fold, arm, RUNS[arm], groups, split_sha, subset)
        identity.update(frozen._verify_trainer_sources(RUNS[arm]))
        if identity != frozen_identities.get(arm):
            raise ValueError(f"{arm}: current prediction identity differs from the frozen grade")
        predictions[arm], identities[arm] = prediction, identity
    panel._verify_paired_fit_identities(identities, fold)
    frozen._verify_source_cell(identities)
    if result["evaluation"].get("groups") != len(groups) or result["evaluation"].get(
        "components"
    ) != len(np.unique(components)):
        raise ValueError("frozen A/B/C grade does not certify complete F0 coverage")
    estimate = interim._memory_estimate_bytes(target_path, len(groups), identities)
    if estimate > MEMORY_LIMIT_BYTES:
        raise MemoryError(f"preflight estimate {estimate} exceeds {MEMORY_LIMIT_BYTES} bytes")
    paths: dict[str, Path] = {
        "frozen_f0_abc_grade": FROZEN_RESULT,
        "target_cache": target_path,
        "split": split_path,
        "fit_subset": subset_path,
        "diagnosis_helper": diagnosis_path,
        "adapter": Path(__file__).resolve(),
        **{f"source_{name}": path for name, path in frozen._source_paths().items()},
    }
    for arm, run in RUNS.items():
        paths.update(
            {
                f"{arm}_prediction": ROOT / "cache" / f"pred_{run}_valid.npy",
                f"{arm}_sidecar": ROOT / "cache" / f"pred_{run}_valid.json",
                f"{arm}_checkpoint": ROOT / "artifacts/runs" / run / "model_ep2.pt",
                f"{arm}_provenance": ROOT / "artifacts/runs" / run / "provenance.json",
            }
        )
    evidence: dict[str, Any] = {
        "result_sha256": result_sha,
        "source_hashes": source_hashes,
        "target_sha256": TARGET_SHA256,
        "split_sha256": split_sha,
        "subset_sha256": subset_sha,
        "memory_estimate_bytes": estimate,
        "identities": identities,
        "groups": groups,
        "components": components,
        "paths": paths,
    }
    evidence["input_hashes_before_labels"] = {
        name: panel.sha256(path) for name, path in paths.items()
    }
    targets, mask = interim._load_targets(target_path, groups)
    if targets.shape != (468, panel.ROWS, 2) or mask.shape != (468, panel.ROWS):
        raise ValueError("official target arrays do not cover every complete F0 sequence")
    if int(mask.sum()) != result["evaluation"].get("scored_rows"):
        raise ValueError("official target mask differs from the frozen F0 scored-row count")
    evidence["targets"], evidence["mask"] = targets, mask
    return evidence, predictions, identities, groups, components, source_hashes


def _consume_scope(
    state: dict[str, Any],
    targets: np.ndarray,
    mask: np.ndarray,
    raw_predictions: dict[str, np.ndarray],
) -> None:
    """Accumulate one sequence and time interval in bounded memory."""
    state["rows"] += int(mask.sum())
    for name in MODEL_NAMES:
        state["scores"][name].add(targets[None], raw_predictions[name][None], mask[None])
    if not mask.any():
        return
    clipped_targets = np.clip(targets, -2.0, 2.0)
    weights = np.abs(clipped_targets)
    for target in range(2):
        y, w = clipped_targets[mask, target], weights[mask, target]
        for name in MODEL_NAMES:
            raw = raw_predictions[name][mask, target]
            clipped = np.clip(raw, -2.0, 2.0)
            state["pairs"][target]["raw"][name] = moments.pair(
                state["pairs"][target]["raw"][name], y, raw, w
            )
            state["pairs"][target]["clipped"][name] = moments.pair(
                state["pairs"][target]["clipped"][name], y, clipped, w
            )
            state["raw_unweighted"][target][name] = moments.pair(
                state["raw_unweighted"][target][name], raw, raw, np.ones(raw.size)
            )
            state["clip_counts"][name][target] += int(np.count_nonzero(np.abs(raw) > 2.0))
        base = raw_predictions["B"][mask, target]
        for kind in ("raw", "clipped"):
            base_values = np.clip(base, -2.0, 2.0) if kind == "clipped" else base
            for candidate in ("B90_C10", "B90_A10"):
                mixture = raw_predictions[candidate][mask, target]
                mixture_values = np.clip(mixture, -2.0, 2.0) if kind == "clipped" else mixture
                delta = mixture_values - base_values
                part_states = state["deltas"][target][kind][candidate]
                part_states["target_delta"] = moments.pair(part_states["target_delta"], y, delta, w)
                part_states["baseline_delta"] = moments.pair(
                    part_states["baseline_delta"], base_values, delta, w
                )
                part_states["delta_variance"] = moments.pair(
                    part_states["delta_variance"], delta, delta, w
                )


def _render_scope(state: dict[str, Any]) -> dict[str, Any]:
    """Render official scores, weighted geometry, clipping rates and identities."""
    scores = {
        name: (state["scores"][name].result() if state["rows"] else None) for name in MODEL_NAMES
    }
    geometry: dict[str, Any] = {}
    max_score_error = max_cov_error = max_var_error = 0.0
    for target in range(2):
        key = f"t{target}"
        geometry[key] = {"prediction": {}, "delta_from_B": {}}
        for kind in ("raw", "clipped"):
            stats = {name: _stat(state["pairs"][target][kind][name]) for name in MODEL_NAMES}
            geometry[key]["prediction"][kind] = stats
            if kind == "clipped":
                official = {
                    name: (None if scores[name] is None else float(scores[name][key]))
                    for name in MODEL_NAMES
                }
                reconstructed = {
                    name: float(gmetric._wp_from_moments(state["pairs"][target][kind][name]))
                    for name in MODEL_NAMES
                }
                errors = {
                    name: (
                        None
                        if official[name] is None
                        else abs(official[name] - reconstructed[name])
                    )
                    for name in MODEL_NAMES
                }
                max_score_error = max(
                    max_score_error, max((v for v in errors.values() if v is not None), default=0.0)
                )
                geometry[key]["score_reconstruction"] = {
                    "official": official,
                    "from_clipped_weighted_moments": reconstructed,
                    "absolute_error": errors,
                }
            if kind == "raw":
                geometry[key]["unweighted_raw_output_moments"] = {
                    name: _stat(state["raw_unweighted"][target][name]) for name in MODEL_NAMES
                }
                geometry[key]["prediction_clipped_fraction"] = {
                    name: (
                        None
                        if state["rows"] == 0
                        else float(state["clip_counts"][name][target] / state["rows"])
                    )
                    for name in MODEL_NAMES
                }
            geometry[key]["delta_from_B"][kind] = {}
            for candidate in ("B90_C10", "B90_A10"):
                d = state["deltas"][target][kind][candidate]
                y_b, y_m = stats["B"], stats[candidate]
                y_d, b_d, d_d = (
                    _stat(d["target_delta"]),
                    _stat(d["baseline_delta"]),
                    _stat(d["delta_variance"]),
                )
                if any(value is None for value in (y_b, y_m, y_d, b_d, d_d)):
                    delta_result = None
                else:
                    var_b = y_b["prediction_std"] ** 2
                    cov_y_b, cov_y_m = y_b["covariance"], y_m["covariance"]
                    cov_y_d, cov_b_d = y_d["covariance"], b_d["covariance"]
                    var_d, var_m = d_d["prediction_std"] ** 2, y_m["prediction_std"] ** 2
                    conditional = cov_y_d - cov_y_b * cov_b_d / var_b if var_b > 1e-24 else None
                    cov_error = cov_y_m - cov_y_b - cov_y_d
                    var_error = var_m - var_b - 2.0 * cov_b_d - var_d
                    max_cov_error = max(max_cov_error, abs(cov_error))
                    max_var_error = max(max_var_error, abs(var_error))
                    delta_result = {
                        "delta_definition": (
                            f"raw: {candidate}-B; clipped: clip({candidate})-clip(B)"
                        ),
                        "cov_y_delta": cov_y_d,
                        "cov_y_baseline": cov_y_b,
                        "cov_baseline_delta": cov_b_d,
                        "var_baseline": var_b,
                        "var_delta": var_d,
                        "var_mixture": var_m,
                        "residual_cov_y_delta_given_baseline_linear_projection": conditional,
                        "covariance_identity_error": cov_error,
                        "variance_identity_error": var_error,
                    }
                geometry[key]["delta_from_B"][kind][candidate] = delta_result
    score_deltas = {
        candidate: (
            None
            if scores["B"] is None or scores[candidate] is None
            else {
                metric: float(scores[candidate][metric] - scores["B"][metric])
                for metric in ("wp", "t0", "t1")
            }
        )
        for candidate in ("C", "B90_C10", "B90_A10")
    }
    return {
        "scored_rows": state["rows"],
        "scores": scores,
        "official_score_deltas_vs_B": score_deltas,
        "weighted_geometry": geometry,
        "max_abs_score_reconstruction_error": max_score_error,
        "max_abs_covariance_identity_error": max_cov_error,
        "max_abs_variance_identity_error": max_var_error,
    }


def _diagnose(evidence: dict[str, Any], predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    """Stream F0 groups for the fixed 90/10 candidate and A-control mixtures."""
    targets, mask = evidence["targets"], evidence["mask"]
    states = {name: _new_scope() for name in SCOPE_SLICES}
    max_abs_prediction = {name: 0.0 for name in ARMS}
    for index, _group in enumerate(evidence["groups"]):
        y, valid = targets[index], mask[index]
        arm_predictions = {name: np.asarray(predictions[name][index]) for name in ARMS}
        for name, prediction in arm_predictions.items():
            if not np.isfinite(prediction).all():
                raise ValueError(f"nonfinite {name} prediction in F0 sequence index {index}")
            max_abs_prediction[name] = max(
                max_abs_prediction[name], float(np.max(np.abs(prediction)))
            )
        arm_predictions["B90_C10"] = panel._mixture(arm_predictions["B"], arm_predictions["C"])
        arm_predictions["B90_A10"] = panel._mixture(arm_predictions["B"], arm_predictions["A"])
        for name, interval in SCOPE_SLICES.items():
            _consume_scope(
                states[name],
                y[interval],
                valid[interval],
                {
                    model: value[interval]
                    for model, value in arm_predictions.items()
                    if model in MODEL_NAMES
                },
            )
    rendered = {name: _render_scope(state) for name, state in states.items()}
    if max(value["max_abs_score_reconstruction_error"] for value in rendered.values()) > 1e-7:
        raise ValueError("official score differs from the clipped weighted-moment reconstruction")
    return {
        "fold": "F0",
        "seed": 1001,
        "groups": len(evidence["groups"]),
        "components": int(len(np.unique(evidence["components"]))),
        "scope": "previously exposed development fold; posthoc geometry conditional on these fits",
        "mixtures": {"B90_C10": {"B": 0.9, "C": 0.1}, "B90_A10": {"B": 0.9, "A": 0.1}},
        "official_metric": "global_wp_clip2_v1; clipping occurs after the fixed blend",
        "scopes": rendered,
        "max_abs_raw_prediction_by_fit": max_abs_prediction,
        "interpretation_limit": (
            "Descriptive covariance geometry on exposed fitted endpoints. No coefficient search, "
            "causal claim, independent-fold claim, or public-transfer claim is made."
        ),
    }


def main() -> None:
    """Write a create-only F0 geometry report after exact source preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if SCRATCH not in output.parents:
        raise ValueError(f"output must be under physical scratch {SCRATCH}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    evidence, predictions, identities, groups, components, source_hashes = _preflight()
    report = _diagnose(evidence, predictions)
    after_hashes = {name: panel.sha256(path) for name, path in evidence["paths"].items()}
    if after_hashes != evidence["input_hashes_before_labels"]:
        raise ValueError(
            "a source, prediction, checkpoint, manifest, or target cache changed during analysis"
        )
    if panel.sha256(FROZEN_RESULT) != FROZEN_RESULT_SHA256:
        raise ValueError("frozen F0 A/B/C grade changed during analysis")
    report.update(
        {
            "schema": "wnn.i1-fixed-mixture-geometry.v1",
            "experiment_id": "E-PLATEAU-I1-20261021-R1",
            "status": "posthoc_f0_fixed_mixture_geometry",
            "source_sha256": {
                **source_hashes,
                "endpoint_diagnosis_moment_helper": DIAGNOSIS_SHA256,
                "frozen_f0_abc_result": FROZEN_RESULT_SHA256,
                "target_cache": TARGET_SHA256,
                "split": evidence["split_sha256"],
                "fit_subset": evidence["subset_sha256"],
                "adapter": panel.sha256(Path(__file__).resolve()),
            },
            "evaluation": {
                "groups": len(groups),
                "components": int(len(np.unique(components))),
                "scored_rows": int(evidence["mask"].sum()),
                "prediction_identity_and_npy_coverage": identities,
            },
            "memory_estimate": {
                "estimated_peak_bytes": evidence["memory_estimate_bytes"],
                "admission_limit_bytes": MEMORY_LIMIT_BYTES,
                "prediction_access": "full-fold arrays remain read-only memory maps; one sequence streams at a time",
            },
            "input_hashes_before_labels": evidence["input_hashes_before_labels"],
            "input_hashes_after_analysis": after_hashes,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {name: values["scores"]["B90_C10"] for name, values in report["scopes"].items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
