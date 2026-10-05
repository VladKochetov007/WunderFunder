"""Fixed complete-predictor residual diagnostic on the original bounded cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import time
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

from utils import weighted_pearson
from wnn.splits import connected_components

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "strong_baseline"
STEMS = {
    "parent": "pred_c20261004_combo_s4_final_fp32_valid",
    "head": "pred_c20261004_combo_head_s4_final_fp32_valid",
}
AGE_LABELS = ["0", "1", "2-7", "8-31", "32-127", "128+"]


def summarize(
    y: np.ndarray, predictions: dict[str, np.ndarray], take: np.ndarray
) -> dict[str, object]:
    """Report official domain correlations and symmetric-mode residual diagnostics."""
    yt = np.clip(y[take], -2, 2).astype(float)
    weight = np.abs(yt)
    joint_weight = weight.mean(axis=1)
    result: dict[str, object] = {
        "rows": int(take.sum()),
        "target_weight": weight.sum(axis=0).tolist(),
    }
    if not len(yt):
        return result
    for name, full in predictions.items():
        pt = np.clip(full[take], -2, 2).astype(float)
        error = pt - yt
        mode_error = np.column_stack([error.sum(axis=1), error[:, 0] - error[:, 1]]) / np.sqrt(2)
        sides = [weighted_pearson(y[take, k], full[take, k]) for k in range(2)]
        total = np.maximum(weight.sum(axis=0), 1e-12)
        ym = np.sum(weight * yt, axis=0) / total
        pm = np.sum(weight * pt, axis=0) / total
        wrong_sign = np.sign(pt) != np.sign(yt)
        result[name] = {
            "wp_t0": sides[0],
            "wp_t1": sides[1],
            "wp_mean": float(np.mean(sides)),
            "target_wmse": (
                np.sum(weight * error**2, axis=0) / np.maximum(weight.sum(axis=0), 1e-12)
            ).tolist(),
            "sum_difference_mode_mse_joint_absolute_weight": (
                np.sum(joint_weight[:, None] * mode_error**2, axis=0)
                / max(joint_weight.sum(), 1e-12)
            ).tolist(),
            "mean_prediction": pt.mean(axis=0).tolist(),
            "target_weighted_mean": ym.tolist(),
            "prediction_weighted_mean": pm.tolist(),
            "target_weighted_std": np.sqrt(
                np.sum(weight * (yt - ym) ** 2, axis=0) / total
            ).tolist(),
            "prediction_weighted_std": np.sqrt(
                np.sum(weight * (pt - pm) ** 2, axis=0) / total
            ).tolist(),
            "absolute_target_weighted_sign_accuracy": (
                1 - np.sum(weight * wrong_sign, axis=0) / total
            ).tolist(),
            "weighted_squared_error_fraction_from_wrong_sign": (
                np.sum(weight * error**2 * wrong_sign, axis=0)
                / np.maximum(np.sum(weight * error**2, axis=0), 1e-12)
            ).tolist(),
        }
    return result


def select_excluded(out: Path) -> list[dict[str, object]]:
    """Freeze 16 deterministic distinct components before reading arrays or rows."""
    provenance = json.loads(
        (ROOT / "artifacts/runs/c20261004_combo_s4_final/provenance.json").read_text()
    )
    fitting = set(provenance["fit_row_groups"]["valid"])
    with np.load(ROOT / "cache/chains.npz") as graph:
        component = connected_components(graph["next"])
        mapping = {
            int(group): int(comp)
            for split, group, comp in zip(
                graph["split"], graph["row_group"], component, strict=True
            )
            if split == "valid"
        }
    eligible = [group for group in range(1873) if group not in fitting]
    eligible.sort(
        key=lambda group: hashlib.sha256(
            f"strong-residual-excluded-v1:{mapping[group]}:{group}".encode()
        ).hexdigest()
    )
    chosen, seen = [], set()
    for group in eligible:
        if mapping[group] not in seen:
            seen.add(mapping[group])
            chosen.append(group)
        if len(chosen) == 16:
            break
    with np.load(ROOT / "cache/meta_valid.npz") as metadata:
        cohort = [
            {
                "row_group": group,
                "seq_ix": int(metadata["seq_ix"][group]),
                "component": mapping[group],
                "label_class_diagnostic_only": int(metadata["market_class"][group]),
                "role": "historically_exposed_baseline_excluded",
            }
            for group in chosen
        ]
    (out / "cohort_fixed_before_array_reads.json").write_text(
        json.dumps(
            {
                "eligible_groups": len(eligible),
                "selection": "SHA256 strong-residual-excluded-v1 component/group order; one group per component; first16; no class stratification or score selection",
                "cohort": cohort,
                "historical_exposure": "Legacy excluded validation is extensively exposed; this is not a fresh holdout. Component links to fitted train data are separately disclosed.",
            },
            indent=2,
        )
    )
    return cohort


def main(panel: str) -> None:
    """Read only fixed row groups via mmap and preserve every diagnostic."""
    start = time.monotonic()
    out = OUT if panel == "fitted" else OUT.with_name("strong_baseline_excluded")
    out.mkdir(exist_ok=True)
    cohort = (
        select_excluded(out)
        if panel == "excluded"
        else [
            node
            for node in json.loads((OUT.parent / "cohort.json").read_text())["cohort"]
            if node["role"] == "test"
        ]
    )
    models, identities = {}, {}
    with np.load(ROOT / "cache/chains.npz") as graph:
        components = connected_components(graph["next"])
        component_map = {
            (str(split), int(group)): int(comp)
            for split, group, comp in zip(
                graph["split"], graph["row_group"], components, strict=True
            )
        }
    for name, stem in STEMS.items():
        models[name] = np.load(ROOT / "cache" / f"{stem}.npy", mmap_mode="r")
        metadata = json.loads((ROOT / "cache" / f"{stem}.json").read_text())
        if models[name].shape != (1873, 20000, 2) or metadata["row_groups"] != list(range(1873)):
            raise ValueError("prediction order/shape mismatch")
        provenance = json.loads(
            (ROOT / "artifacts/runs" / metadata["run"] / "provenance.json").read_text()
        )
        fit_groups = set(provenance["fit_row_groups"]["valid"])
        fit_components = {
            component_map[(split, int(group))]
            for split, groups in provenance["fit_row_groups"].items()
            for group in groups
        }
        identities[name] = {
            "path": f"cache/{stem}.npy",
            "prediction_sha256": metadata["prediction_sha256"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "epoch": metadata["epoch"],
            "fit_valid_sequences": len(fit_groups),
            "test_cohort_fit_overlap": [
                node["row_group"] for node in cohort if node["row_group"] in fit_groups
            ],
            "test_cohort_feature_component_overlap_with_any_fit": [
                node["row_group"] for node in cohort if node["component"] in fit_components
            ],
        }
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    ys, prediction_blocks, classes, ages, events, label_changes = [], {}, [], [], [], []
    updates = []
    sequence_results, scored_alignment = [], []
    for node in cohort:
        group = node["row_group"]
        table = pf.read_row_group(
            group,
            columns=["t0", "t1", "i0_p0", "i0_p11", "need_prediction", "is_scored"],
            use_threads=False,
        )
        y = np.column_stack([table[c].to_numpy() for c in ("t0", "t1")])
        raw = np.column_stack([table[c].to_numpy() for c in ("i0_p0", "i0_p11")])
        event = np.r_[False, np.any(np.diff(raw, axis=0) != 0, axis=1)]
        last = np.maximum.accumulate(np.where(event, np.arange(len(y)), 0))
        age = np.arange(len(y)) - last
        change = np.r_[False, np.any(np.diff(y, axis=0) != 0, axis=1)]
        mask = table["need_prediction"].to_numpy() & table["is_scored"].to_numpy()
        destination_changes = mask & change
        next_available = np.arange(len(y)) < len(y) - 1
        next_event = np.r_[event[1:], False]
        scored_alignment.append(
            {
                **node,
                "scored_destination_label_changes": int(destination_changes.sum()),
                "current_event_at_scored_destination_label_change": int(
                    (destination_changes & event).sum()
                ),
                "next_event_at_scored_destination_label_change_nondeployable": int(
                    (destination_changes & next_event & next_available).sum()
                ),
                "next_event_available_scored_destination_label_changes": int(
                    (destination_changes & next_available).sum()
                ),
                "definition": "Transition(t-1→t) selected iff need_prediction[t] AND is_scored[t]; next event[t+1] is future oracle audit only.",
            }
        )
        predictions = {
            name: np.array(model[group], dtype=np.float32) for name, model in models.items()
        }
        predictions["complete"] = (
            np.float32(0.5) * predictions["parent"] + np.float32(0.5) * predictions["head"]
        )
        predictions["complete_causal_event_hold"] = predictions["complete"][last]
        predictions["head_causal_event_hold"] = predictions["head"][last]
        for name, pred in predictions.items():
            if not np.isfinite(pred).all():
                raise ValueError("invalid required predictions")
            prediction_blocks.setdefault(name, []).append(pred[mask])
        sequence_results.append({**node, "scores": summarize(y, predictions, mask)})
        yc, pc = (
            np.clip(y, -2, 2).astype(float),
            np.clip(predictions["complete"], -2, 2).astype(float),
        )
        previous = pc[np.maximum(np.arange(len(y)) - 1, 0)]
        weight = np.abs(yc)
        loss_before = weight * (previous - yc) ** 2
        loss_after = weight * (pc - yc) ** 2
        # This label-based condition is an oracle diagnostic, never an input gate.
        stable = mask & ~event & ~change
        updates.append(
            {
                **node,
                "same_label_no_observed_event_scored_rows": int(stable.sum()),
                "target_weight": weight[stable].sum(axis=0).tolist(),
                "weighted_loss_reduction_sum": (loss_before[stable] - loss_after[stable])
                .sum(axis=0)
                .tolist(),
                "fraction_update_helps": ((loss_before[stable] - loss_after[stable]) > 0)
                .mean(axis=0)
                .tolist(),
                "prediction_delta_rms": np.sqrt(
                    ((pc[stable] - previous[stable]) ** 2).mean(axis=0)
                ).tolist(),
                "no_event_label_changes": int((mask & ~event & change).sum()),
            }
        )
        ys.append(y[mask])
        classes.append(np.full(mask.sum(), node["label_class_diagnostic_only"], dtype=int))
        ages.append(age[mask])
        events.append(event[mask])
        label_changes.append(change[mask])
    y = np.concatenate(ys)
    predictions = {name: np.concatenate(blocks) for name, blocks in prediction_blocks.items()}
    cls, age, event, change = map(np.concatenate, (classes, ages, events, label_changes))
    report = {
        "metric": "global_wp_clip2_v1",
        "scope": f"Structural diagnostic panel={panel}; fitted panel has16/16 supervised baseline fitting overlap; excluded panel is historically exposed and may have fitted train component links; no generalization/promotion/public0.7000 claim",
        "identities": identities,
        "combination": "float32 0.5parent_epoch3 +0.5head_epoch0, clip only inside metric/residual diagnostics",
        "global": summarize(y, predictions, np.ones(len(y), bool)),
        "classes": {},
        "event_age": {},
        "within_run_updates_oracle_label_condition": updates,
        "sequence_results": sequence_results,
        "scored_transition_alignment": scored_alignment,
    }
    bounds = np.array([0, 1, 2, 8, 32, 128, np.inf])
    for k in range(4):
        report["classes"][str(k)] = summarize(y, predictions, cls == k)
    for lower, upper, label in zip(bounds[:-1], bounds[1:], AGE_LABELS, strict=True):
        report["event_age"][label] = summarize(y, predictions, (age >= lower) & (age < upper))
    report["event_label_conditions"] = {
        name: summarize(y, predictions, take)
        for name, take in {
            "observed_event": event,
            "no_observed_event": ~event,
            "label_changed_nondeployable_condition": change,
            "label_unchanged_nondeployable_condition": ~change,
        }.items()
    }
    report["class_event_age"] = {
        f"{k}/{label}": summarize(y, predictions, (cls == k) & (age >= lower) & (age < upper))
        for k in range(4)
        for lower, upper, label in zip(bounds[:-1], bounds[1:], AGE_LABELS, strict=True)
    }
    report["resources"] = {
        "wall_seconds": time.monotonic() - start,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
    }
    (out / "residual_report.json").write_text(json.dumps(report, indent=2))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("parent", "head", "complete", "complete_causal_event_hold"):
        axes[0].plot(
            AGE_LABELS,
            [report["event_age"][label][name]["target_wmse"][0] for label in AGE_LABELS],
            marker="o",
            label=name,
        )
        axes[1].plot(
            AGE_LABELS,
            [report["event_age"][label][name]["target_wmse"][1] for label in AGE_LABELS],
            marker="o",
            label=name,
        )
    for k, ax in enumerate(axes):
        ax.set_title(f"t{k}: {panel} cohort residual diagnostic")
        ax.set_xlabel("rows since observed-column event")
        ax.set_ylabel("clipped absolute-target weighted MSE")
        ax.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(out / "event_age_residuals.png", dpi=150)
    report["resources"] = {
        "wall_seconds": time.monotonic() - start,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
    }
    (out / "residual_report.json").write_text(json.dumps(report, indent=2))
    logging.info(
        "Global complete/hold: %s / %s",
        report["global"]["complete"],
        report["global"]["complete_causal_event_hold"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("fitted", "excluded"), default="fitted")
    main(parser.parse_args().panel)
