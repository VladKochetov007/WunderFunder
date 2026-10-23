"""Offline weighted geometry for five frozen, exposed-F0 prediction caches."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any  # JSON metadata has heterogeneous nested values.

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from wnn.gmetric import GlobalAccumulator  # noqa: E402

SPLIT = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
TARGETS = ROOT / "cache/valid_targets.npz"
GRADE = ROOT / "research/campaign_2026-10-18/geometry_variations_r3/geometry_r3_grade.json"
OUTPUT = Path(__file__).resolve().parent / "receipt.json"
STEPS = 20_000
BINS = {
    "pos_99_1000": (99, 1_000),
    "pos_1000_5000": (1_000, 5_000),
    "pos_5000_10000": (5_000, 10_000),
    "pos_10000_20000": (10_000, 20_000),
}
SCOPES = {"pooled_scored": (99, STEPS), **BINS}
MODELS = {
    "parent": "warm_mechanism_f0_s3_c_r3",
    "r2_full_geometry": "geometry_adapter_f0_s3_r2",
    "r2_scale_m0p50": "geometry_adapter_f0_s3_r2_scale_a_m0p50",
    "r3_moments_only": "geometry_variation_f0_s3_r3_moments_only",
    "r3_dictionary_only": "geometry_variation_f0_s3_r3_dictionary_only",
}


def sha256(path: Path) -> str:
    """Hash a small manifest or receipt in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_pair(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> dict[str, float | None]:
    """Summarize weighted covariance and correlation over positive-weight rows."""
    selected = weight > 0
    if not selected.any():
        return {"weight": 0.0, "covariance": None, "correlation": None}
    x = np.asarray(x[selected], dtype=np.float64)
    y = np.asarray(y[selected], dtype=np.float64)
    w = np.asarray(weight[selected], dtype=np.float64)
    total = float(w.sum())
    mean_x, mean_y = float(np.sum(w * x) / total), float(np.sum(w * y) / total)
    dx, dy = x - mean_x, y - mean_y
    var_x = float(np.sum(w * dx * dx) / total)
    var_y = float(np.sum(w * dy * dy) / total)
    covariance = float(np.sum(w * dx * dy) / total)
    denominator = np.sqrt(var_x * var_y)
    correlation = None if denominator <= 1e-15 else float(covariance / denominator)
    return {
        "weight": total,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "variance_x": var_x,
        "variance_y": var_y,
        "covariance": covariance,
        "correlation": correlation,
    }


def projection(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> dict[str, float | None]:
    """Fit a weighted intercept and slope for one descriptive fixed projection."""
    pair = weighted_pair(x, y, weight)
    var_x = float(pair.get("variance_x") or 0.0)
    var_y = float(pair.get("variance_y") or 0.0)
    covariance = float(pair.get("covariance") or 0.0)
    mean_x = float(pair.get("mean_x") or 0.0)
    mean_y = float(pair.get("mean_y") or 0.0)
    slope = covariance / var_x if var_x > 0 else 0.0
    return {
        "intercept": mean_y - slope * mean_x,
        "slope": slope,
        "weighted_r2": covariance * covariance / (var_x * var_y)
        if var_x > 0 and var_y > 0
        else None,
        "weight": pair["weight"],
    }


def main() -> int:
    split = json.loads(SPLIT.read_text())
    groups = [int(group) for group in split["evaluation"]["valid"]]
    components_all = {
        int(row["row_group"]): int(row["component"]) for row in split["sequences"]["valid"]
    }
    components = np.asarray([components_all[group] for group in groups], dtype=np.int64)
    split_sha = sha256(SPLIT)
    if int(split["fold"]) != 0 or len(groups) != 468 or len(set(groups)) != 468:
        raise ValueError("fold0 evaluation group identity or size changed")
    if len(np.unique(components)) != 141:
        raise ValueError("fold0 evaluation component count changed")

    grade = json.loads(GRADE.read_text())
    if grade.get("split_manifest_sha256") != split_sha:
        raise ValueError("frozen grade and fold0 split hashes disagree")
    with np.load(TARGETS, allow_pickle=False) as archive:
        target_groups = np.asarray(archive["row_group"], dtype=np.int64)
        target_position = {int(group): i for i, group in enumerate(target_groups)}
        if len(target_position) != len(target_groups):
            raise ValueError("duplicate target-cache row_group")
        target_indices = [target_position[group] for group in groups]
        full_y = archive["y"]
        full_mask = archive["mask"]
        y = np.asarray(full_y[target_indices], dtype=np.float32)
        mask = np.asarray(full_mask[target_indices], dtype=bool)
        del full_y, full_mask
    if y.shape != (len(groups), STEPS, 2) or mask.shape != (len(groups), STEPS):
        raise ValueError(f"unexpected target/mask shapes: {y.shape}/{mask.shape}")
    mask &= np.arange(STEPS)[None, :] >= 99
    y_clipped = np.clip(y, -2.0, 2.0)
    weights = np.abs(y_clipped)

    predictions: dict[str, np.ndarray] = {}
    identities: dict[str, dict[str, Any]] = {}
    for name, run in MODELS.items():
        model = grade["candidates"][name]
        path = ROOT / "cache" / f"pred_{run}_valid.npy"
        sidecar = json.loads(path.with_suffix(".json").read_text())
        if sidecar.get("run") != run or sidecar.get("split_manifest_sha256") != split_sha:
            raise ValueError(f"{name}: prediction run or split identity mismatch")
        declared = [int(group) for group in sidecar["row_groups"]]
        if len(set(declared)) != len(declared) or len(declared) not in (468, 1873):
            raise ValueError(f"{name}: unexpected sidecar row groups")
        position = {group: index for index, group in enumerate(declared)}
        if any(group not in position for group in groups):
            raise ValueError(f"{name}: prediction cache misses evaluation groups")
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
        if mapped.shape != (len(declared), STEPS, 2) or mapped.dtype != np.float32:
            raise ValueError(f"{name}: unexpected prediction shape or dtype")
        if sidecar.get("checkpoint_sha256") != model["producer_checkpoint_sha256"]:
            raise ValueError(f"{name}: producer checkpoint differs from frozen grade")
        source_sha = model.get("source_checkpoint_sha256")
        if source_sha and sidecar.get("source_checkpoint_sha256") not in (None, source_sha):
            raise ValueError(f"{name}: source checkpoint differs from frozen grade")
        predictions[name] = np.clip(
            np.asarray(mapped[[position[group] for group in groups]], dtype=np.float32), -2.0, 2.0
        )
        identities[name] = {
            "run": run,
            "role": model["role"],
            "checkpoint_sha256": model["producer_checkpoint_sha256"],
            "source_checkpoint_sha256": source_sha,
            "prediction_sha256_from_frozen_grade": model["prediction_sha256"],
            "sidecar_sha256_from_frozen_grade": model["sidecar_sha256"],
            "aligned_evaluation_groups": model["aligned_evaluation_groups"],
        }

    parent = predictions["parent"]
    target_fits = {
        f"t{target}": projection(
            parent[..., target][mask], y_clipped[..., target][mask], weights[..., target][mask]
        )
        for target in range(2)
    }
    correction_fits = {}
    for name, prediction in predictions.items():
        delta = prediction - parent
        correction_fits[name] = {
            f"t{target}": projection(
                parent[..., target][mask], delta[..., target][mask], weights[..., target][mask]
            )
            for target in range(2)
        }

    frozen_scores = {row["name"]: row for row in grade["ranking"]}
    results: dict[str, Any] = {}
    for name, prediction in predictions.items():
        scopes = {}
        delta = prediction - parent
        for scope, (start, stop) in SCOPES.items():
            scope_mask = mask[:, start:stop]
            accumulator = GlobalAccumulator()
            accumulator.add(y[:, start:stop], prediction[:, start:stop], scope_mask)
            score = accumulator.result()
            if scope == "pooled_scored":
                prior = frozen_scores[name]
                if (
                    max(abs(float(score[key]) - float(prior[key])) for key in ("wp", "t0", "t1"))
                    > 2e-6
                ):
                    raise ValueError(f"{name}: pooled score differs from frozen grade")
            present = np.flatnonzero(scope_mask.any(axis=1))
            panel = {
                "selected_rows": int(scope_mask.sum()),
                "sequence_count": int(len(present)),
                "component_count": int(len(np.unique(components[present]))),
                "weighted_mass_t0": float(
                    np.sum(weights[:, start:stop, 0] * scope_mask, dtype=np.float64)
                ),
                "weighted_mass_t1": float(
                    np.sum(weights[:, start:stop, 1] * scope_mask, dtype=np.float64)
                ),
            }
            panel["weighted_mass_sum_targets"] = (
                panel["weighted_mass_t0"] + panel["weighted_mass_t1"]
            )
            geometry = {}
            for target in range(2):
                selected = scope_mask.reshape(-1)
                yt = y_clipped[:, start:stop, target].reshape(-1)[selected]
                wt = weights[:, start:stop, target].reshape(-1)[selected]
                pt = parent[:, start:stop, target].reshape(-1)[selected]
                ct = prediction[:, start:stop, target].reshape(-1)[selected]
                dt = delta[:, start:stop, target].reshape(-1)[selected]
                yfit = target_fits[f"t{target}"]
                dfit = correction_fits[name][f"t{target}"]
                y_residual = yt - (float(yfit["intercept"]) + float(yfit["slope"]) * pt)
                delta_residual = dt - (float(dfit["intercept"]) + float(dfit["slope"]) * pt)
                geometry[f"t{target}"] = {
                    "clipped_prediction_vs_parent": weighted_pair(ct, pt, wt),
                    "correction_vs_parent": weighted_pair(dt, pt, wt),
                    "correction_vs_target": weighted_pair(dt, yt, wt),
                    "correction_orthogonal_vs_target_orthogonal": weighted_pair(
                        delta_residual, y_residual, wt
                    )
                    if name != "parent"
                    else None,
                }
            scopes[scope] = {"official_score": score, "panel": panel, "weighted_geometry": geometry}
        results[name] = {**identities[name], "scopes": scopes}

    receipt = {
        "schema": "wnn.online_features.prediction_geometry.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "offline prediction geometry only; no model execution, fitting, or coefficient search",
        "metric": "global_wp_clip2_v1",
        "clip": [-2.0, 2.0],
        "weights": "absolute clipped official labels",
        "mask": "official target-cache mask AND positions >= 99; labels/mask used only by offline grader",
        "panel": {
            "fold": 0,
            "evaluation_groups": len(groups),
            "components": int(len(np.unique(components))),
            "total_selected_rows": int(mask.sum()),
            "exposure": "repeatedly inspected/exposed F0; not a fresh holdout",
        },
        "position_bins": {
            key: {"start_inclusive": lo, "stop_exclusive": hi} for key, (lo, hi) in BINS.items()
        },
        "alignment": "fold0 evaluation.valid row_group IDs matched to target and prediction sidecar row_group IDs",
        "split_manifest_sha256": split_sha,
        "target_cache": {
            "path": str(TARGETS.relative_to(ROOT)),
            "size_bytes": TARGETS.stat().st_size,
            "row_groups": len(target_groups),
            "evaluation_y_shape": list(y.shape),
            "evaluation_mask_shape": list(mask.shape),
        },
        "fixed_parent_projection": {
            "scope": "pooled selected exposed-F0 rows; same label-magnitude weights as metric",
            "target_fits": target_fits,
            "candidate_correction_fits": correction_fits,
            "interpretation": "descriptive exposed-F0 fitting diagnostic; coefficients and residual associations are not deployable or generalization evidence",
        },
        "models": results,
        "limitations": [
            "No bootstrap, model fit, prediction generation, or coefficient search was performed.",
            "One repeatedly exposed F0 panel; fold/seed uncertainty and transfer are not estimated.",
            "Bin residual diagnostics reuse pooled fixed-parent projections and are descriptive.",
            "Position patterns do not establish cause, gradient effect, clock, event, venue, or generalization mechanism.",
            "The R2 scale alias is a fixed inference derivative, not an independent fitted endpoint.",
        ],
        "provenance": {
            "split": str(SPLIT.relative_to(ROOT)),
            "frozen_grade": str(GRADE.relative_to(ROOT)),
            "frozen_grade_sha256": sha256(GRADE),
            "script_sha256": sha256(Path(__file__)),
            "prediction_hashes": "reused from geometry_r3_grade.json; not recomputed",
        },
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                name: value["scopes"]["pooled_scored"]["official_score"]
                for name, value in results.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
