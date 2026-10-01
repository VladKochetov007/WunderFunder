"""Exploratory OOF residual screen for the causal event-memory proxy.

Fold 1 rows fit the correction; fold 0 rows evaluate. The historical backbone
OOF pair is since/base, not the public since/since pair, and historical backbone
training was not chain-excluded. These scores are only a cheap falsifier.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pyarrow.parquet as pq
from memory_falsifier import fit_projection, forecast
from study import HERE, PRICE, ROOT, class_from_labels, columns_numpy

from wnn.gmetric import global_wp

CORRECTION_SCALE = 0.2
RIDGE = 100.0
VARIANTS = {
    "output_only": [0, 1],
    "level_age": [0, 1, 2, 3, 4],
    "last_event": [0, 1, 2, 3, 4, 5, 6],
    "position": [0, 1, 2, 3, 4, 7, 8],
    "content": [0, 1, 2, 3, 4, 9, 10],
    "reversion": [0, 1, 2, 3, 4, 11, 12],
}


def freeze_selection() -> dict:
    initial = json.loads((HERE / "selection.json").read_text())
    raw = (ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json").read_bytes()
    fold1 = json.loads(raw)
    rng = np.random.default_rng(20260930)
    fit = sorted(rng.choice(fold1["evaluation"]["valid"], 32, replace=False).tolist())
    evaluation = initial["row_groups"]["eval_valid"]
    components = {row["row_group"]: row["component"] for row in fold1["sequences"]["valid"]}
    if {components[g] for g in fit} & {components[g] for g in evaluation}:
        raise RuntimeError("Correction-fit and evaluation components overlap")
    selection = {
        "feature_projection_train_groups": initial["row_groups"]["fit_train"],
        "correction_fit_fold1_valid_groups": fit,
        "evaluation_fold0_valid_groups": evaluation,
        "fold0_selection_sha256": hashlib.sha256((HERE / "selection.json").read_bytes()).hexdigest(),
        "fold1_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "feature_derived_components_disjoint": True,
        "selection_before_correction_target_access": True,
    }
    path = HERE / "residual_selection.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise RuntimeError("Frozen residual selection differs")
    path.write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def row_design(p: np.ndarray, projection: np.ndarray, scale: float) -> np.ndarray:
    event = np.any(np.diff(p, axis=0) != 0, axis=1)
    positions = np.r_[0, np.flatnonzero(event) + 1]
    state = p[positions]
    memory = forecast(state, projection, scale)
    event_ix = np.searchsorted(positions, np.arange(len(p)), side="right") - 1
    age = np.log1p(np.arange(len(p)) - positions[event_ix])
    level = (p - p[:1]) @ projection
    memory_columns = []
    for name in ("last", "position", "content", "reversion"):
        memory_columns.append(memory[name][event_ix])
    return np.column_stack([level, age, *memory_columns])


def read_one(
    pf: pq.ParquetFile, group: int, fold: int, projection: np.ndarray, scale: float,
    cache: dict[int, tuple[np.memmap, np.memmap]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    tab = pf.read_row_group(group, columns=PRICE + ["t0", "t1", "is_scored"], use_threads=False)
    p = columns_numpy(tab, PRICE).astype(np.float64, copy=False)
    y_all = columns_numpy(tab, ["t0", "t1"]).astype(np.float64, copy=False)
    label_class = class_from_labels(y_all)
    mask = tab.column("is_scored").to_numpy().astype(bool)
    mask[:99] = False
    base = 0.5 * (cache[fold][0][group].astype(np.float64) + cache[fold][1][group].astype(np.float64))
    base = np.clip(base, -2, 2)
    design = row_design(p, projection, scale)
    return y_all[mask], base[mask], design[mask], mask, label_class


def full_design(base: np.ndarray, extra: np.ndarray, target: int) -> np.ndarray:
    return np.column_stack([np.ones(len(base)), base[:, target], extra])


def fit_models(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    fitted = {}
    for target in range(2):
        for name, columns in VARIANTS.items():
            normal = np.zeros((len(columns), len(columns)), np.float64)
            rhs = np.zeros(len(columns), np.float64)
            for y, base, extra in rows:
                x = full_design(base, extra, target)[:, columns]
                truth = np.clip(y[:, target], -2, 2)
                w = np.abs(truth)
                residual = truth - base[:, target]
                normal += x.T @ (w[:, None] * x)
                rhs += x.T @ (w * residual)
            penalty = np.eye(len(columns)) * RIDGE
            penalty[0, 0] = 0
            fitted[f"t{target}_{name}"] = np.linalg.solve(normal + penalty, rhs).tolist()
    return fitted


def evaluate(
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]], fitted: dict
) -> dict:
    targets = np.concatenate([r[0] for r in rows])
    base = np.concatenate([r[1] for r in rows])
    extra = np.concatenate([r[2] for r in rows])
    classes = np.concatenate([np.full(len(r[0]), r[3], np.int8) for r in rows])
    scores = {}
    for name in ["unchanged", *VARIANTS]:
        predicted = base.copy()
        if name != "unchanged":
            for target in range(2):
                cols = VARIANTS[name]
                predicted[:, target] += CORRECTION_SCALE * (
                    full_design(base, extra, target)[:, cols]
                    @ np.asarray(fitted[f"t{target}_{name}"])
                )
        predicted = np.clip(predicted, -2, 2)
        all_score = global_wp(targets, predicted, np.ones(len(targets), bool))["wp"]
        scores[name] = {
            "global_wp": all_score,
            "delta_vs_unchanged": None,
            "by_label_derived_class_diagnostic": {
                str(k): global_wp(targets[classes == k], predicted[classes == k],
                                  np.ones(np.sum(classes == k), bool))["wp"] for k in range(4)
            },
        }
    reference = scores["unchanged"]["global_wp"]
    for row in scores.values():
        row["delta_vs_unchanged"] = row["global_wp"] - reference
    return {"n_selected_rows": len(targets), "scores": scores}


def main() -> None:
    start = time.monotonic()
    selection = freeze_selection()
    projection, _, scale = fit_projection(selection["feature_projection_train_groups"])
    cache = {
        fold: tuple(
            np.load(ROOT / "cache" / f"pred_{name}_f{fold}h_valid.npy", mmap_mode="r")
            for name in ("ab82_ft_since4s3", "ab82_ft_base4s4")
        ) for fold in (0, 1)
    }
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    fit_rows = []
    for group in selection["correction_fit_fold1_valid_groups"]:
        y, base, extra, _, _ = read_one(pf, group, 1, projection, scale, cache)
        fit_rows.append((y, base, extra))
    fitted = fit_models(fit_rows)
    eval_rows = []
    for group in selection["evaluation_fold0_valid_groups"]:
        y, base, extra, _, label_class = read_one(pf, group, 0, projection, scale, cache)
        eval_rows.append((y, base, extra, label_class))
    report = {
        "selection_sha256": hashlib.sha256((HERE / "residual_selection.json").read_bytes()).hexdigest(),
        "reference": "equal since/base historical OOF recipe analog; not public since/since incumbent",
        "historical_backbone_chain_exclusions": False,
        "correction_fit_vs_eval_components_disjoint": True,
        "train_mask": "exact is_scored on fold1 only",
        "eval_mask": "exact is_scored on fold0 only",
        "clip": 2,
        "ridge": RIDGE,
        "correction_scale": CORRECTION_SCALE,
        "features_by_variant": VARIANTS,
        "fitted_coefficients": fitted,
        "result": evaluate(eval_rows, fitted),
        "wall_seconds": time.monotonic() - start,
    }
    (HERE / "residual_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
