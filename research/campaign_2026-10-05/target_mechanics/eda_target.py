"""Bounded, component-separated target mechanics audit on exposed validation."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from utils import weighted_pearson
from wnn.splits import connected_components, is_holdout

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
FEATURES = [
    "i0_p0",
    "i0_p11",
    "i0_v0",
    "i0_v11",
    "i0_dp0",
    "i0_dv0",
    "i1_p0",
    "i1_p11",
    "i1_v0",
    "i1_v11",
    "i1_dp0",
    "i1_dv0",
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
]


def digest(value: str) -> str:
    """Give selection a stable seed without using target amplitude."""
    return hashlib.sha256(value.encode()).hexdigest()


def quantiles(values: np.ndarray) -> list[float]:
    """Summarize an array with fixed quantiles."""
    return np.quantile(values, [0, 0.25, 0.5, 0.75, 0.95, 1]).tolist()


def design(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create current/past-only state, differences and event age features."""
    blocks = [raw]
    events = np.r_[False, np.any(np.diff(raw[:, :2], axis=0) != 0, axis=1)]
    last = np.maximum.accumulate(np.where(events, np.arange(len(raw)), 0))
    for lag in (1, 8, 32, 128):
        past = raw[np.maximum(np.arange(len(raw)) - lag, 0)]
        blocks.append(raw - past)
    age = np.log1p(np.arange(len(raw)) - last)[:, None]
    full = np.column_stack([*blocks, age, events]).astype(np.float32)
    event_only = np.column_stack([*blocks[1:], age, events]).astype(np.float32)
    return np.nan_to_num(full, nan=0, posinf=0, neginf=0), np.nan_to_num(
        event_only, nan=0, posinf=0, neginf=0
    )


def structure(y: np.ndarray, raw: np.ndarray) -> dict[str, object]:
    """Audit target runs and event alignment; future offsets are diagnostic."""
    event = np.r_[False, np.any(np.diff(raw[:, :2], axis=0) != 0, axis=1)]
    target_event = np.any(np.diff(y, axis=0) != 0, axis=1)
    result: dict[str, object] = {
        "target_pair_correlation": float(np.corrcoef(y.T)[0, 1]),
        "clipped_target_pair_correlation": float(np.corrcoef(np.clip(y, -2, 2).T)[0, 1]),
        "same_sign": float((np.sign(y[:, 0]) == np.sign(y[:, 1])).mean()),
        "equal_targets": float((y[:, 0] == y[:, 1]).mean()),
        "observed_column_event_rate": float(event.mean()),
        "joint_label_change_rate": float(target_event.mean()),
    }
    for k in range(2):
        changes = np.r_[0, np.flatnonzero(np.diff(y[:, k]) != 0) + 1, len(y)]
        value, count = np.unique(y[:, k], return_counts=True)
        order = np.argsort(count)[-12:][::-1]
        result[f"t{k}"] = {
            "unique_values": len(value),
            "top_support": [[float(value[j]), int(count[j])] for j in order],
            "zero_fraction": float((y[:, k] == 0).mean()),
            "clipped_fraction": float((np.abs(y[:, k]) > 2).mean()),
            "absolute_quantiles": quantiles(np.abs(y[:, k])),
            "run_length_quantiles": quantiles(np.diff(changes)),
            "change_rate": float((np.diff(y[:, k]) != 0).mean()),
        }
    offsets = [-128, -32, -8, -1, 0, 1, 8, 32, 128]
    changes = np.r_[False, target_event]
    alignment = {}
    for lag in offsets:
        indexes = np.arange(len(y))
        usable = (indexes + lag >= 0) & (indexes + lag < len(y))
        aligned = event[indexes[usable] + lag]
        alignment[str(lag)] = {
            "p_label_change_given_event_at_offset": float(changes[usable][aligned].mean())
            if aligned.any()
            else None,
            "p_event_at_offset_given_label_change": float(aligned[changes[usable]].mean())
            if changes[usable].any()
            else None,
        }
    result["event_alignment_offset_positive_is_future_nondeployable"] = alignment
    return result


def main() -> None:
    """Select, record, read and evaluate the bounded predeclared audit."""
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "cache/meta_valid.npz") as meta:
        classes = meta["market_class"].astype(int)
        seq = meta["seq_ix"].astype(int)
    with np.load(ROOT / "cache/chains.npz") as graph:
        components = connected_components(graph["next"])
        component_map = {
            int(group): int(component)
            for split, group, component in zip(
                graph["split"], graph["row_group"], components, strict=True
            )
            if split == "valid"
        }
    selected = []
    used = set()
    for cls in range(4):
        eligible = [g for g in range(len(seq)) if classes[g] == cls and not is_holdout(int(seq[g]))]
        eligible.sort(key=lambda g: digest(f"target-mechanics-v1:{component_map[g]}:{g}"))
        for role, size in (("fit", 6), ("test", 4)):
            count = 0
            for group in eligible:
                comp = component_map[group]
                if comp in used:
                    continue
                used.add(comp)
                selected.append(
                    {
                        "row_group": group,
                        "seq_ix": int(seq[group]),
                        "component": comp,
                        "label_class_diagnostic_only": cls,
                        "role": role,
                    }
                )
                count += 1
                if count == size:
                    break
    selection = {
        "cohort": selected,
        "selection_rule": "6 fit + 4 test distinct components per cached label-derived class; SHA256 target-mechanics-v1 component/group order; legacy protected holdout excluded",
        "component_overlap": [],
        "historical_exposure": "Validation labels and panels have extensive historical use. This is descriptive, not a fresh holdout or generalization claim. Class is label-derived and never an inference input.",
        "metric": "global_wp_clip2_v1",
        "fit_subsample": "every twentieth row >=128; no is_scored fit filtering",
        "frozen_models": "ridge alpha=100; direct histogram 60 trees/12 leaves; factor sign and magnitude 30 trees each/12 leaves; shared full design; no search",
    }
    (OUT / "cohort.json").write_text(json.dumps(selection, indent=2))
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    evidence = {
        "parquet_row_groups": pf.num_row_groups,
        "first_group_rows": pf.metadata.row_group(0).num_rows,
        "cohort": [],
        "metric": "global_wp_clip2_v1",
    }
    fitted_x, fitted_event, fitted_y, tests = [], [], [], []
    for node in selected:
        table = pf.read_row_group(
            node["row_group"],
            columns=[*FEATURES, "t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        raw = np.column_stack([table[c].to_numpy() for c in FEATURES])
        y = np.column_stack([table[c].to_numpy() for c in ("t0", "t1")])
        mask = table["need_prediction"].to_numpy() & table["is_scored"].to_numpy()
        x, event_x = design(raw)
        evidence["cohort"].append(
            {**node, "scored_rows": int(mask.sum()), "structure": structure(y, raw)}
        )
        if node["role"] == "fit":
            fitted_x.append(x[128::20])
            fitted_event.append(event_x[128::20])
            fitted_y.append(np.clip(y[128::20], -2, 2))
        else:
            tests.append((node, x, event_x, y, mask))
    (OUT / "raw_evidence.json").write_text(json.dumps(evidence, indent=2))
    logging.info("Raw evidence saved before modeling")
    xfit, efit, yfit = (
        np.concatenate(fitted_x),
        np.concatenate(fitted_event),
        np.concatenate(fitted_y),
    )
    center, scale = xfit.mean(0), np.maximum(xfit.std(0), 1e-4)
    ecenter, escale = efit.mean(0), np.maximum(efit.std(0), 1e-4)
    direct, sign, mag, ridge, event = [], [], [], [], []
    for k in range(2):
        weight = np.abs(yfit[:, k]) + 1e-3
        kwargs = {
            "max_leaf_nodes": 12,
            "learning_rate": 0.08,
            "l2_regularization": 10.0,
            "random_state": 417,
            "early_stopping": False,
        }
        direct.append(
            HistGradientBoostingRegressor(max_iter=60, **kwargs).fit(
                xfit, yfit[:, k], sample_weight=weight
            )
        )
        sign.append(
            HistGradientBoostingRegressor(max_iter=30, **kwargs).fit(
                xfit, np.sign(yfit[:, k]), sample_weight=weight
            )
        )
        mag.append(
            HistGradientBoostingRegressor(max_iter=30, **kwargs).fit(
                xfit, np.abs(yfit[:, k]), sample_weight=weight
            )
        )
        ridge.append(
            Ridge(alpha=100).fit((xfit - center) / scale, yfit[:, k], sample_weight=weight)
        )
        event.append(
            Ridge(alpha=100).fit((efit - ecenter) / escale, yfit[:, k], sample_weight=weight)
        )
    pools: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "direct_histogram",
            "factor_sign_magnitude",
            "ridge_current_past",
            "ridge_changes_age",
            "oracle_previous_target_nondeployable",
            "direct_histogram_causal_event_hold",
        )
    }
    all_y, all_cls = [], []
    for node, x, ex, y, mask in tests:
        predictions = {
            "direct_histogram": np.column_stack([m.predict(x) for m in direct]),
            "factor_sign_magnitude": np.column_stack(
                [sign[k].predict(x) * mag[k].predict(x) for k in range(2)]
            ),
            "ridge_current_past": np.column_stack([m.predict((x - center) / scale) for m in ridge]),
            "ridge_changes_age": np.column_stack(
                [m.predict((ex - ecenter) / escale) for m in event]
            ),
            "oracle_previous_target_nondeployable": y[np.maximum(np.arange(len(y)) - 1, 0)],
        }
        last_event = np.maximum.accumulate(np.where(x[:, -1] > 0, np.arange(len(x)), 0))
        predictions["direct_histogram_causal_event_hold"] = predictions["direct_histogram"][
            last_event
        ]
        for name, pred in predictions.items():
            pools[name].append(pred[mask])
        all_y.append(y[mask])
        all_cls.append(np.full(mask.sum(), node["label_class_diagnostic_only"]))
    y, cls = np.concatenate(all_y), np.concatenate(all_cls)
    scores = {}
    for name, blocks in pools.items():
        p = np.concatenate(blocks).astype(np.float32)
        scores[name] = {}
        for subset in ("global", "class0", "class1", "class2", "class3"):
            take = np.ones(len(y), bool) if subset == "global" else cls == int(subset[-1])
            sides = [weighted_pearson(y[take, k], p[take, k]) for k in range(2)]
            scores[name][subset] = {
                "t0": sides[0],
                "t1": sides[1],
                "mean": float(np.mean(sides)),
                "rows": int(take.sum()),
            }
    (OUT / "predictor_scores.json").write_text(
        json.dumps(
            {
                "metric": "global_wp_clip2_v1",
                "fit_rows": len(yfit),
                "test_scored_rows": len(y),
                "scores": scores,
            },
            indent=2,
        )
    )
    logging.info("Finished scores: %s", {name: result["global"] for name, result in scores.items()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
