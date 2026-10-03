from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from continuation_clock_probe import (
    ALPHA,
    OUT,
    ROOT,
    correlation,
    fingerprint,
    histories,
    moments,
    read_sequence,
    save_json,
)

ARMS = ("raw", "row_clock", "event_clock", "hazard_clock", "stale_event")


def hazard_displacement(raw: np.ndarray) -> np.ndarray:
    """Match expected event-time decay using only lagged prefix event frequency."""
    price = raw[:, :22].astype(np.float64)
    events = np.r_[False, np.any(price[1:] != price[:-1], axis=1)]
    prior_events = np.r_[0, np.cumsum(events)[:-1]]
    rate = (24.75 + prior_events) / (99 + np.arange(len(raw)))
    alpha = 1 - (1 - ALPHA) ** rate
    ema = price[0].copy()
    result = np.empty_like(price, dtype=np.float32)
    for step in range(len(price)):
        ema += alpha[step] * (price[step] - ema)
        result[step] = price[step] - ema
    return result


def designs(raw: np.ndarray) -> list[np.ndarray]:
    """Use identical raw columns and equal-width history blocks."""
    row, event, stale = histories(raw)
    return [
        raw,
        np.column_stack((raw, row)),
        np.column_stack((raw, event)),
        np.column_stack((raw, hazard_displacement(raw))),
        np.column_stack((raw, stale)),
    ]


def main() -> None:
    """Execute the frozen timescale/shrinkage followup, preserving first scores."""
    started = time.monotonic()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    manifest = json.loads((OUT / "cache/continuation_manifest.json").read_text())
    hypothesis = (OUT / "HYPOTHESES_followup_frozen.json").read_bytes()
    state = json.loads((OUT / "STATE.json").read_text())
    command = "research/campaign_2026-10-03/continuation/run_cpu.sh eda-econometrics .venv/bin/python research/campaign_2026-10-03/eda/econometrics/scripts/continuation_hazard_probe.py"
    state["commands"].append(command)
    state["phase"] = "hazard_probe_running"
    save_json("STATE.json", state)
    train = pq.ParquetFile(ROOT / "datasets/train.parquet")
    valid = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    sums = [np.zeros(112 if arm == 0 else 134) for arm in range(5)]
    squares = [np.zeros_like(value) for value in sums]
    count = 0
    fingerprints: dict[bytes, tuple[int, int]] = {}
    for g in manifest["train_groups"]:
        seq, raw, _, required, _ = read_sequence(train, g)
        idx = np.flatnonzero(required)[::10]
        count += len(idx)
        for arm, values in enumerate(designs(raw)):
            sample = values[idx].astype(np.float64)
            sums[arm] += sample.sum(axis=0)
            squares[arm] += (sample * sample).sum(axis=0)
        for step, raw_row in enumerate(raw):
            fingerprints[fingerprint(raw_row)] = (seq, step)
        if g == manifest["train_groups"][0]:
            prefix_error = float(
                np.max(np.abs(hazard_displacement(raw)[:777] - hazard_displacement(raw[:777])))
            )
            assert prefix_error == 0
    means = [value / count for value in sums]
    scales = [
        np.maximum(np.sqrt(np.maximum(square / count - mean * mean, 0)), 1e-3)
        for square, mean in zip(squares, means, strict=True)
    ]
    grams = [np.zeros((2, len(mean) + 1, len(mean) + 1)) for mean in means]
    rhs = [np.zeros((2, len(mean) + 1)) for mean in means]
    for g in manifest["train_groups"]:
        _, raw, target, required, _ = read_sequence(train, g)
        idx = np.flatnonzero(required)[::10]
        for arm, values in enumerate(designs(raw)):
            design = np.column_stack((np.ones(len(idx)), (values[idx] - means[arm]) / scales[arm]))
            for j in range(2):
                weight = np.abs(target[idx, j])
                grams[arm][j] += design.T @ (weight[:, None] * design)
                rhs[arm][j] += design.T @ (weight * target[idx, j])
    coefficients = []
    for arm in range(5):
        fitted = []
        for j in range(2):
            matrix = grams[arm][j].copy()
            penalty = 0.01 * np.diag(matrix)[1:].mean()
            matrix += np.diag(np.r_[0, np.full(len(matrix) - 1, penalty)])
            fitted.append(np.linalg.solve(matrix, rhs[arm][j]))
        coefficients.append(np.stack(fitted, axis=1))
    pooled = np.zeros((2, 5, 2, 6))
    records = []
    overlaps = []
    scored_count = 0
    for g in manifest["valid_groups"]:
        seq, raw, target, required, mask = read_sequence(valid, g)
        matches: dict[int, int] = {}
        for raw_row in raw:
            hit = fingerprints.get(fingerprint(raw_row))
            if hit is not None:
                matches[hit[0]] = matches.get(hit[0], 0) + 1
        if matches:
            overlaps.append({"valid_seq": seq, "matched_train_sequences": matches})
        scored_count += int(mask.sum())
        for arm, values in enumerate(designs(raw)):
            design = np.column_stack((np.ones(len(raw)), (values - means[arm]) / scales[arm]))
            prediction = design @ coefficients[arm]
            scores = []
            for j in range(2):
                stats = moments(target[mask, j], prediction[mask, j])
                pooled[0, arm, j] += stats
                pooled[1, arm, j] += moments(target[required, j], prediction[required, j])
                scores.append(correlation(stats))
            records.append(
                {
                    "seq": seq,
                    "group": g,
                    "arm": ARMS[arm],
                    "wp0": scores[0],
                    "wp1": scores[1],
                    "mean_wp": float(np.mean(scores)),
                }
            )
    scores = {
        name: {
            "scored_wp": [correlation(pooled[0, arm, j]) for j in range(2)],
            "all_required_wp": [correlation(pooled[1, arm, j]) for j in range(2)],
        }
        for arm, name in enumerate(ARMS)
    }
    for score in scores.values():
        score["mean_scored_wp"] = float(np.mean(score["scored_wp"]))
        score["mean_all_required_wp"] = float(np.mean(score["all_required_wp"]))
    deltas = np.subtract(scores["event_clock"]["scored_wp"], scores["hazard_clock"]["scored_wp"])
    frame = pd.DataFrame(records)
    pivot = frame.pivot(index="seq", columns="arm", values="mean_wp")
    result = {
        "test_id": "ECO_CLOCK_02",
        "hypothesis_sha256": hashlib.sha256(hypothesis).hexdigest(),
        "scores": scores,
        "event_minus_hazard_targets": deltas.tolist(),
        "event_minus_hazard_mean": float(deltas.mean()),
        "frozen_gate_passed": bool(deltas.mean() > 0.002 and np.all(deltas > 0)),
        "positive_sequence_event_minus_hazard": int((pivot.event_clock > pivot.hazard_clock).sum()),
        "prefix_parity_max_abs": prefix_error,
        "exact_raw_row_cross_split_matches": overlaps,
        "train_fingerprint_unique_rows": len(fingerprints),
        "train_rows_screened": 48 * 20000,
        "valid_rows_screened": 24 * 20000,
        "valid_scored_rows": scored_count,
        "train_sample_rows": count,
        "provenance_limit": "full selected raw-row identity screen; transformation/unsampled linked chains unknown",
        "elapsed_seconds": time.monotonic() - started,
        "adaptive_followup": True,
        "independent_replication": False,
    }
    save_json("cache/continuation_hazard_results.json", result)
    frame.to_csv(OUT / "cache/continuation_hazard_sequence_scores.csv", index=False)
    np.savez(
        OUT / "cache/continuation_hazard_models.npz",
        **{
            f"{name}_{kind}": value
            for name, values in zip(
                ARMS, zip(means, scales, coefficients, strict=True), strict=True
            )
            for kind, value in zip(("mean", "scale", "coefficients"), values, strict=True)
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    positions = np.arange(5)
    for j in range(2):
        axes[0].bar(
            positions + (j - 0.5) * 0.35,
            [scores[name]["scored_wp"][j] for name in ARMS],
            width=0.35,
            label=f"t{j}",
        )
    axes[0].set_xticks(positions, ARMS, rotation=25)
    axes[0].set_ylabel("Scored pooled WP; arm-specific training scaling")
    axes[0].legend()
    axes[1].bar(np.arange(len(pivot)), pivot.event_clock - pivot.hazard_clock)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Selected validation sequence (sorted ID)")
    axes[1].set_ylabel("Event minus hazard WP; descriptive")
    fig.suptitle("Adaptive frozen followup: discrete events versus expected event rate")
    fig.tight_layout()
    fig.savefig(OUT / "fig/continuation_hazard_scores.png", dpi=130)
    plt.close(fig)
    state["phase"] = "hazard_probe_completed"
    state["artifacts"].extend(
        [
            "HYPOTHESES_followup_frozen.json",
            "cache/continuation_hazard_results.json",
            "cache/continuation_hazard_sequence_scores.csv",
            "cache/continuation_hazard_models.npz",
            "fig/continuation_hazard_scores.png",
        ]
    )
    save_json("STATE.json", state)
    logging.info("Completed hazard probe: %s", json.dumps(result))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        main()
    except Exception:
        logging.exception("Hazard probe failed; preserve traceback and stop for review")
        raise
