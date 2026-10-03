from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import lfilter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SEED = 20261003
ARMS = ("raw", "row16", "event16", "rate_adjusted16")
SCALES = np.array([16, 32, 64, 128, 256, 512, 1024])


def save_json(name: str, data: dict) -> None:
    """Persist a human-readable experiment artifact in the owned directory."""
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def exposure() -> dict[str, set[int]]:
    """Conservatively include all groups named by saved exploratory scripts."""
    return {
        "train": set(range(0, 10607, 265))
        | set(range(0, 10607, 212))
        | set(range(3, 10607, 208))
        | set(range(1, 10607, 106))
        | {5},
        "valid": set(range(40))
        | set(range(0, 1873, 47))
        | set(range(0, 1873, 37))
        | set(range(0, 1873, 39))
        | set(range(5, 1873, 38))
        | set(range(7, 1873, 31))
        | set(range(5, 1873, 19))
        | set(range(2, 1873, 19)),
    }


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Compute a causal EMA initialized from the first observed row."""
    result, _ = lfilter(
        [alpha], [1, -(1 - alpha)], values, axis=0, zi=((1 - alpha) * values[0])[None, :]
    )
    return result


def build(values: np.ndarray, price_indices: list[int]) -> list[np.ndarray]:
    """Create matched causal histories from features alone, resetting each call."""
    events = np.r_[False, np.any(np.diff(values[:, price_indices], axis=0) != 0, axis=1)]
    row_mean = ema(values, 2 / 17)
    positions = np.r_[0, np.flatnonzero(events)]
    event_mean = ema(values[positions], 2 / 17)
    event_index = np.cumsum(events)
    rates = ema(events.astype(float)[:, None], 2 / 65)[:, 0]
    adjusted = np.empty_like(values)
    state = values[0].copy()
    for row in range(len(values)):
        alpha = 1 - (1 - 2 / 17) ** rates[row]
        state += alpha * (values[row] - state)
        adjusted[row] = state
    ones = np.ones((len(values), 1))
    return [
        np.c_[values, ones],
        np.c_[values, values - row_mean, ones],
        np.c_[values, values - event_mean[event_index], ones],
        np.c_[values, values - adjusted, ones],
    ]


def moments(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Store additive sufficient statistics for exact pooled clipped WP."""
    target = np.clip(target, -2, 2)
    prediction = np.clip(prediction, -2, 2)
    weights = np.abs(target)
    return np.array(
        [
            weights.sum(),
            weights @ target,
            weights @ prediction,
            weights @ (target * target),
            weights @ (prediction * prediction),
            weights @ (target * prediction),
        ],
        dtype=float,
    )


def wp(stats: np.ndarray) -> np.ndarray:
    """Evaluate WP over the last axis of pooled sufficient statistics."""
    w, y, p, yy, pp, yp = np.moveaxis(stats, -1, 0)
    return (yp - y * p / w) / np.sqrt((yy - y * y / w) * (pp - p * p / w))


def dfa(values: np.ndarray) -> np.ndarray:
    """DFA1 using nonoverlapping forward windows and least-squares detrending."""
    profile = np.cumsum(values - values.mean())
    fluctuation = []
    for scale in SCALES:
        count = len(profile) // scale
        if count < 4:
            fluctuation.append(np.nan)
            continue
        segments = profile[: count * scale].reshape(count, scale)
        grid = np.arange(scale) - (scale - 1) / 2
        centered = segments - segments.mean(axis=1, keepdims=True)
        slopes = centered @ grid / (grid @ grid)
        residual = centered - slopes[:, None] * grid
        fluctuation.append(np.sqrt(np.mean(residual**2)))
    return np.array(fluctuation)


def scaling_slope(fluctuation: np.ndarray) -> float:
    """Fit only the frozen short-scale range; avoid silent zero floors."""
    selected = (SCALES <= 256) & np.isfinite(fluctuation) & (fluctuation > 0)
    if selected.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(SCALES[selected]), np.log(fluctuation[selected]), 1)[0])


def main() -> None:
    """Run the frozen CPU screen, saving IDs, fit outputs and raw summaries."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.monotonic()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be unavailable"
    hypotheses = json.loads((OUT / "HYPOTHESES.json").read_text())
    rng = np.random.default_rng(SEED)
    files = {
        name: pq.ParquetFile(ROOT / "datasets" / f"{name}.parquet") for name in ("train", "valid")
    }
    exposed = exposure()
    groups = {
        name: sorted(
            rng.choice(
                [g for g in range(files[name].num_row_groups) if g not in exposed[name]],
                size=size,
                replace=False,
            ).tolist()
        )
        for name, size in (("train", 32), ("valid", 24))
    }
    manifest = {
        "selection_seed": SEED,
        "own_prior_exposure": {},
        "new_sample": {},
        "campaign_prior_exposure": "unknown",
        "component_disjointness": "unknown",
    }
    for name, parquet in files.items():
        for key, selected in (
            ("own_prior_exposure", sorted(exposed[name])),
            ("new_sample", groups[name]),
        ):
            manifest[key][name] = [
                {
                    "row_group": group,
                    "seq_ix": int(
                        parquet.read_row_group(group, columns=["seq_ix"], use_threads=False)
                        .column("seq_ix")[0]
                        .as_py()
                    ),
                }
                for group in selected
            ]
        manifest[f"{name}_file"] = {
            "size": (ROOT / "datasets" / f"{name}.parquet").stat().st_size,
            "row_groups": parquet.num_row_groups,
        }
    save_json("sample_manifest.json", manifest)
    state = {
        "phase": "running_frozen_probe",
        "commands": [
            "research/campaign_2026-10-03/continuation/run_cpu.sh eda-fractal .venv/bin/python research/campaign_2026-10-03/eda/fractal/continuation_probe.py"
        ],
        "artifacts": ["RECOVERY_SUMMARY.txt", "HYPOTHESES.json", "sample_manifest.json"],
        "inspected_sequence_ids": manifest["new_sample"],
        "last_failure": None,
        "pending_work": ["clock fit and scoring", "scaling control", "figures", "report"],
    }
    save_json("STATE.json", state)
    features = files["valid"].schema_arrow.names[4:116]
    price_indices = [features.index(f"i0_p{k}") for k in range(22)]
    dimensions = [113, 225, 225, 225]
    gram = [np.zeros((2, dim, dim)) for dim in dimensions]
    rhs = [np.zeros((2, dim)) for dim in dimensions]
    causality_errors = []
    for index, group in enumerate(groups["train"]):
        table = files["train"].read_row_group(
            group, columns=features + ["t0", "t1", "need_prediction"], use_threads=False
        )
        values = np.column_stack([table[c].to_numpy() for c in features]).astype(float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite inputs require an explicitly frozen imputation rule")
        designs = build(values, price_indices)
        if index == 0:
            for length in (99, 500, 4096):
                prefix = build(values[:length], price_indices)
                causality_errors.extend(
                    float(np.max(np.abs(a[:length] - b)))
                    for a, b in zip(designs, prefix, strict=True)
                )
            perturbed = values.copy()
            perturbed[500:] += 123
            changed = build(perturbed, price_indices)
            causality_errors.extend(
                float(np.max(np.abs(a[:500] - b[:500])))
                for a, b in zip(designs, changed, strict=True)
            )
            assert max(causality_errors) < 1e-10
        mask = table["need_prediction"].to_numpy().astype(bool)
        for target_index in range(2):
            target = np.clip(table[f"t{target_index}"].to_numpy()[mask].astype(float), -2, 2)
            if not np.isfinite(target).all():
                raise ValueError("Nonfinite training targets")
            weights = np.abs(target)
            for arm, design in enumerate(designs):
                selected = design[mask]
                gram[arm][target_index] += selected.T @ (weights[:, None] * selected)
                rhs[arm][target_index] += selected.T @ (weights * target)
        logging.info("fit group %s (%s/32)", group, index + 1)
    coefficients = []
    penalties = []
    for arm, dim in enumerate(dimensions):
        arm_coef = []
        arm_penalties = []
        for target_index in range(2):
            penalty = 0.001 * np.trace(gram[arm][target_index][:-1, :-1]) / (dim - 1)
            regularizer = np.eye(dim) * penalty
            regularizer[-1, -1] = 0
            arm_coef.append(
                np.linalg.solve(gram[arm][target_index] + regularizer, rhs[arm][target_index])
            )
            arm_penalties.append(float(penalty))
        coefficients.append(np.column_stack(arm_coef))
        penalties.append(arm_penalties)
    np.savez(OUT / "clock_coefficients.npz", **dict(zip(ARMS, coefficients, strict=True)))
    pooled = []
    scaling = []
    curves = []
    for index, group in enumerate(groups["valid"]):
        table = files["valid"].read_row_group(
            group,
            columns=features + ["t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        values = np.column_stack([table[c].to_numpy() for c in features]).astype(float)
        designs = build(values, price_indices)
        mask = table["need_prediction"].to_numpy().astype(bool) & table[
            "is_scored"
        ].to_numpy().astype(bool)
        stats = np.empty((4, 2, 6))
        for arm, design in enumerate(designs):
            predictions = design @ coefficients[arm]
            for target_index in range(2):
                target = table[f"t{target_index}"].to_numpy()[mask].astype(float)
                predicted = predictions[mask, target_index]
                if not np.isfinite(target).all() or not np.isfinite(predicted).all():
                    raise ValueError("Nonfinite scoring values")
                stats[arm, target_index] = moments(target, predicted)
        pooled.append(stats)
        if index < 12:
            events = np.any(np.diff(values[:, price_indices], axis=0) != 0, axis=1)
            increments = np.diff(values[:, price_indices[0]])[events]
            original = dfa(increments)
            shuffled = np.array([dfa(rng.permutation(increments)) for _ in range(16)])
            slopes = [scaling_slope(curve) for curve in shuffled]
            scaling.append(
                {
                    "row_group": group,
                    "n_events": len(increments),
                    "original_slope": scaling_slope(original),
                    "shuffled_slopes": slopes,
                    "difference": scaling_slope(original) - float(np.nanmedian(slopes)),
                }
            )
            curves.append(np.vstack([original, np.nanmedian(shuffled, axis=0)]))
        logging.info("scored group %s (%s/24)", group, index + 1)
    stats_array = np.array(pooled)
    scores = wp(stats_array.sum(axis=0))
    panels = [wp(panel.sum(axis=0)) for panel in np.array_split(stats_array, 2)]
    event_delta = scores[2].mean() - scores[1].mean()
    rate_delta = scores[2].mean() - scores[3].mean()
    survives = (
        event_delta >= 0.002
        and rate_delta >= 0.002
        and all(panel[2].mean() > panel[1].mean() for panel in panels)
    )
    differences = np.array([row["difference"] for row in scaling])
    result = {
        "arms": ARMS,
        "pooled_wp": scores.tolist(),
        "panel_wp": [panel.tolist() for panel in panels],
        "event_minus_row": float(event_delta),
        "event_minus_rate": float(rate_delta),
        "clock_survives": bool(survives),
        "penalties": penalties,
        "scaling": scaling,
        "scaling_median_delta": float(np.nanmedian(differences)),
        "scaling_positive_sequences": int((differences > 0).sum()),
        "scaling_survives": bool(np.nanmedian(differences) > 0.05 and (differences > 0).sum() >= 9),
        "causality_max_abs_error": max(causality_errors),
        "runtime_seconds": time.monotonic() - started,
        "hypotheses_sha256": hashlib.sha256((OUT / "HYPOTHESES.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "uncertainty": hypotheses["uncertainty"],
    }
    save_json("continuation_results.json", result)
    np.savez(
        OUT / "continuation_raw.npz",
        per_sequence_moments=stats_array,
        scaling_curves=np.array(curves),
        scales=SCALES,
        groups=groups["valid"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for target_index in range(2):
        axes[0].plot(ARMS, scores[:, target_index], "o-", label=f"t{target_index}")
    axes[0].set_ylabel("Globally pooled clipped WP")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend()
    for panel_index, panel in enumerate(panels):
        axes[1].plot(ARMS, panel.mean(axis=1), "o-", label=f"fixed panel {panel_index + 1}")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylabel("Mean target WP")
    axes[1].legend()
    figure.suptitle("Frozen causal clock screen: same training and evaluation rows")
    figure.tight_layout()
    figure.savefig(OUT / "fig8_causal_clock.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for curve in curves:
        axes[0].loglog(SCALES, curve[0], color="tab:blue", alpha=0.2)
        axes[0].loglog(SCALES, curve[1], color="tab:orange", alpha=0.2)
    axes[0].set_title("Signed event increments: original blue, shuffled orange")
    axes[0].set_xlabel("event scale")
    axes[0].set_ylabel("DFA1 fluctuation")
    axes[1].bar(np.arange(12), differences)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axhline(0.05, color="red", linestyle="--", label="frozen median threshold")
    axes[1].set_xlabel("frozen validation sequence index")
    axes[1].set_ylabel("original minus shuffled DFA slope (16..256)")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUT / "fig9_shuffle_falsifier.png", dpi=150)
    plt.close(figure)
    state.update(
        phase="probe_complete",
        artifacts=state["artifacts"]
        + [
            "continuation_results.json",
            "continuation_raw.npz",
            "clock_coefficients.npz",
            "fig8_causal_clock.png",
            "fig9_shuffle_falsifier.png",
        ],
        pending_work=["inspect figures", "headless notebook", "report"],
    )
    save_json("STATE.json", state)
    logging.info("RESULT %s", json.dumps(result))


if __name__ == "__main__":
    main()
