from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import lfilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[3]
FEATURES = [
    f"i{i}_{group}{j}"
    for i in range(2)
    for group, count in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
    for j in range(count)
] + [f"a{j}" for j in range(8)]
ARMS = ("raw", "row_clock", "event_clock", "stale_event")
ALPHA = 2 / 33


def save_json(name: str, data: object) -> None:
    """Persist a reviewable artifact inside the analyst directory."""
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def histories(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build causal EMA displacement, resetting all state for each sequence."""
    price = raw[:, :22].astype(np.float64)
    row_ema = lfilter(
        [ALPHA], [1, -(1 - ALPHA)], price, axis=0, zi=((1 - ALPHA) * price[0])[None, :]
    )[0]
    event = np.r_[True, np.any(price[1:] != price[:-1], axis=1)]
    compressed = price[event]
    event_ema = lfilter(
        [ALPHA], [1, -(1 - ALPHA)], compressed, axis=0, zi=((1 - ALPHA) * compressed[0])[None, :]
    )[0]
    event_ema = event_ema[np.cumsum(event) - 1]
    displacement = price - event_ema
    stale = np.zeros_like(displacement)
    stale[4096:] = displacement[:-4096]
    return (
        (price - row_ema).astype(np.float32),
        displacement.astype(np.float32),
        stale.astype(np.float32),
    )


def read_sequence(
    pf: pq.ParquetFile, group: int
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one selected row group, never the complete Parquet file."""
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURES, "t0", "t1"]
    has_mask = "is_scored" in pf.schema_arrow.names
    if has_mask:
        columns.append("is_scored")
    table = pf.read_row_group(group, columns=columns, use_threads=False)
    frame = table.to_pandas(use_threads=False)
    assert np.array_equal(frame.step_in_seq.to_numpy(), np.arange(len(frame)))
    raw = frame[FEATURES].to_numpy(dtype=np.float32)
    targets = np.clip(frame[["t0", "t1"]].to_numpy(dtype=np.float64), -2, 2)
    required = frame.need_prediction.to_numpy(dtype=bool)
    mask = frame.is_scored.to_numpy(dtype=bool) & required if has_mask else required
    assert np.isfinite(raw).all() and np.isfinite(targets).all()
    return int(frame.seq_ix.iloc[0]), raw, targets, required, mask


def moments(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return additive sufficient statistics for exact pooled weighted Pearson."""
    pred = np.clip(prediction, -2, 2)
    weight = np.abs(target)
    return np.array(
        [
            weight.sum(),
            weight @ target,
            weight @ pred,
            weight @ (target * target),
            weight @ (pred * pred),
            weight @ (target * pred),
        ]
    )


def correlation(stats: np.ndarray) -> float:
    """Compute pooled correlation without averaging sequence correlations."""
    w, y, p, yy, pp, yp = stats
    variance = (yy - y * y / w) * (pp - p * p / w)
    return float((yp - y * p / w) / np.sqrt(max(variance, 1e-30)))


def fingerprint(row: np.ndarray) -> bytes:
    """Hash all 112 exact raw float32 values, excluding labels and IDs."""
    return hashlib.blake2b(row.tobytes(), digest_size=16).digest()


def main() -> None:
    """Fit and score the preregistered small causal clock falsifier."""
    started = time.monotonic()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    hypothesis_bytes = (OUT / "HYPOTHESES_clock_frozen.json").read_bytes()
    train = pq.ParquetFile(ROOT / "datasets/train.parquet")
    valid = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    prior = set(pd.read_csv(OUT / "cache/train_lattice300.csv").g.astype(int))
    acf = np.load(OUT / "cache/target_acf.npz")
    prior_acf_count = sum(acf[key].shape[0] for key in acf.files if key != "lags")
    assert prior_acf_count == 120
    prior.update(np.random.default_rng(2).choice(train.num_row_groups, 120, replace=False).tolist())
    prior.add(0)
    rng = np.random.default_rng(610103)
    available = np.array([i for i in range(train.num_row_groups) if i not in prior])
    training_groups = sorted(rng.choice(available, 48, replace=False).tolist())
    validation_groups = sorted(rng.choice(valid.num_row_groups, 24, replace=False).tolist())
    manifest: dict[str, object] = {
        "hypothesis_sha256": hashlib.sha256(hypothesis_bytes).hexdigest(),
        "train_groups": training_groups,
        "valid_groups": validation_groups,
        "prior_train_groups_excluded": sorted(prior),
        "prior_acf_ids": "inferred seed2/n120; verified NPZ sequence count, not original saved IDs",
        "valid_prior_exposure": "all sequences in own valid_seq_summary.csv",
        "train_ids": [],
        "valid_ids": [],
        "feature_order": FEATURES,
        "dataset_metadata": {},
        "component_provenance": "unknown; exact raw-row screening incomplete",
    }
    for name, pf in (("train", train), ("valid", valid)):
        path = ROOT / f"datasets/{name}.parquet"
        manifest["dataset_metadata"][name] = {
            "path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "groups": pf.num_row_groups,
            "rows": pf.metadata.num_rows,
            "schema_sha256": hashlib.sha256(str(pf.schema_arrow).encode()).hexdigest(),
        }
    command = "research/campaign_2026-10-03/continuation/run_cpu.sh eda-econometrics .venv/bin/python research/campaign_2026-10-03/eda/econometrics/scripts/continuation_clock_probe.py"
    state = {
        "phase": "clock_probe_running",
        "commands": [command],
        "artifacts": ["RECOVERY_SUMMARY.txt", "HYPOTHESES.json"],
        "inspected_sequence_ids": manifest,
        "pending_work": ["score", "inspect figures", "notebook", "report"],
        "last_failure": None,
    }
    save_json("STATE.json", state)
    save_json("cache/continuation_manifest.json", manifest)
    # At most one raw sequence plus small sufficient statistics are live.
    sums = np.zeros(134)
    squares = np.zeros(134)
    sample_count = 0
    fingerprints: dict[bytes, tuple[int, int]] = {}
    prefix_error = 0.0
    for g in training_groups:
        seq, raw, _, required, _ = read_sequence(train, g)
        manifest["train_ids"].append(seq)
        row, event, _ = histories(raw)
        idx = np.flatnonzero(required)[::10]
        # Shared scaling for both history arms prevents different regularization budgets.
        values = np.column_stack((raw[idx], row[idx])).astype(np.float64)
        sums += values.sum(axis=0)
        squares += (values * values).sum(axis=0)
        sample_count += len(idx)
        for step in idx:
            fingerprints[fingerprint(raw[step])] = (seq, int(step))
        if g == training_groups[0]:
            short = histories(raw[:777])
            prefix_error = max(
                float(np.max(np.abs(a[:777] - b)))
                for a, b in zip(histories(raw), short, strict=True)
            )
            assert prefix_error == 0
    mean = sums / sample_count
    scale = np.maximum(np.sqrt(np.maximum(squares / sample_count - mean * mean, 0)), 1e-3)
    gram = np.zeros((4, 2, 135, 135))
    rhs = np.zeros((4, 2, 135))
    weight_sum = np.zeros(2)
    for g in training_groups:
        _, raw, targets, required, _ = read_sequence(train, g)
        history = histories(raw)
        idx = np.flatnonzero(required)[::10]
        target = targets[idx]
        weight_sum += np.abs(target).sum(axis=0)
        for arm in range(4):
            values = raw[idx] if arm == 0 else np.column_stack((raw[idx], history[arm - 1][idx]))
            width = values.shape[1]
            design = np.column_stack((np.ones(len(idx)), (values - mean[:width]) / scale[:width]))
            for target_ix in range(2):
                weights = np.abs(target[:, target_ix])
                gram[arm, target_ix, : width + 1, : width + 1] += design.T @ (
                    weights[:, None] * design
                )
                rhs[arm, target_ix, : width + 1] += design.T @ (weights * target[:, target_ix])
    coefficients = []
    for arm in range(4):
        width = 113 if arm == 0 else 135
        fitted = []
        for target_ix in range(2):
            matrix = gram[arm, target_ix, :width, :width].copy()
            penalty = 0.01 * np.mean(np.diag(matrix)[1:])
            matrix += np.diag(np.r_[0, np.full(width - 1, penalty)])
            fitted.append(np.linalg.solve(matrix, rhs[arm, target_ix, :width]))
        coefficients.append(np.stack(fitted, axis=1))
    save_json("cache/continuation_manifest.json", manifest)
    np.savez(
        OUT / "cache/continuation_models.npz",
        mean=mean,
        scale=scale,
        **dict(zip(ARMS, coefficients, strict=True)),
    )
    pooled = np.zeros((2, 4, 2, 6))
    per_sequence = []
    overlap = []
    clipped = np.zeros((4, 2), dtype=np.int64)
    scored_count = 0
    for g in validation_groups:
        seq, raw, targets, required, mask = read_sequence(valid, g)
        manifest["valid_ids"].append(seq)
        history = histories(raw)
        matches = 0
        examples = []
        for step, raw_row in enumerate(raw):
            hit = fingerprints.get(fingerprint(raw_row))
            if hit is not None:
                matches += 1
                if len(examples) < 4:
                    examples.append({"valid_step": step, "train_seq": hit[0], "train_step": hit[1]})
        if matches:
            overlap.append({"valid_seq": seq, "matched_rows": matches, "examples": examples})
        scored_count += int(mask.sum())
        for arm, name in enumerate(ARMS):
            values = raw if arm == 0 else np.column_stack((raw, history[arm - 1]))
            width = values.shape[1]
            design = np.column_stack((np.ones(len(raw)), (values - mean[:width]) / scale[:width]))
            prediction = design @ coefficients[arm]
            clipped[arm] += (np.abs(prediction[mask]) > 2).sum(axis=0)
            scores = []
            for target_ix in range(2):
                stats = moments(targets[mask, target_ix], prediction[mask, target_ix])
                pooled[0, arm, target_ix] += stats
                pooled[1, arm, target_ix] += moments(
                    targets[required, target_ix], prediction[required, target_ix]
                )
                scores.append(correlation(stats))
            per_sequence.append(
                {
                    "group": g,
                    "seq": seq,
                    "arm": name,
                    "scored_rows": int(mask.sum()),
                    "wp0": scores[0],
                    "wp1": scores[1],
                    "mean_wp": float(np.mean(scores)),
                }
            )
        if g == validation_groups[0]:
            fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            window = slice(4000, 4600)
            axes[0].plot(raw[window, 0], label="current i0_p0")
            axes[0].set_title(f"Causal history example: validation seq {seq}, steps 4000–4599")
            axes[1].plot(history[0][window, 0], label="row-clock displacement")
            axes[1].plot(history[1][window, 0], label="event-clock displacement")
            for axis in axes:
                axis.legend()
                axis.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(OUT / "fig/continuation_clock_window.png", dpi=130)
            plt.close(fig)
    scores = {
        name: {
            "scored_wp": [correlation(pooled[0, arm, j]) for j in range(2)],
            "all_required_wp": [correlation(pooled[1, arm, j]) for j in range(2)],
            "clipped_prediction_fraction": (clipped[arm] / scored_count).tolist(),
        }
        for arm, name in enumerate(ARMS)
    }
    for score in scores.values():
        score["mean_scored_wp"] = float(np.mean(score["scored_wp"]))
        score["mean_all_required_wp"] = float(np.mean(score["all_required_wp"]))
    event_delta = scores["event_clock"]["mean_scored_wp"] - scores["row_clock"]["mean_scored_wp"]
    target_deltas = np.subtract(
        scores["event_clock"]["scored_wp"], scores["row_clock"]["scored_wp"]
    )
    result = {
        "test_id": "ECO_CLOCK_01",
        "scores": scores,
        "event_minus_row": event_delta,
        "event_minus_row_targets": target_deltas.tolist(),
        "frozen_gate_passed": bool(event_delta > 0.002 and np.all(target_deltas > 0)),
        "prefix_parity_max_abs": prefix_error,
        "train_sample_rows": sample_count,
        "valid_scored_rows": scored_count,
        "fit_weight_sums": weight_sum.tolist(),
        "exact_raw_row_overlap": overlap,
        "fingerprint_train_rows": len(fingerprints),
        "overlap_screen_limit": "only every tenth required training row; no campaign chain mapping",
        "elapsed_seconds": time.monotonic() - started,
        "interpretation": "exploratory sample-exposed local ridge; no strong-baseline or causal-effect claim",
    }
    frame = pd.DataFrame(per_sequence)
    frame.to_csv(OUT / "cache/continuation_sequence_scores.csv", index=False)
    pivot = frame.pivot(index="seq", columns="arm", values="mean_wp")
    result["positive_sequence_event_minus_row"] = int((pivot.event_clock > pivot.row_clock).sum())
    result["mean_sequence_wp_diagnostic"] = frame.groupby("arm").mean_wp.mean().to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    positions = np.arange(4)
    for j in range(2):
        axes[0].bar(
            positions + (j - 0.5) * 0.35,
            [scores[name]["scored_wp"][j] for name in ARMS],
            width=0.35,
            label=f"t{j}",
        )
    axes[0].set_xticks(positions, ARMS, rotation=20)
    axes[0].set_ylabel("Pooled scored weighted Pearson")
    axes[0].legend()
    axes[1].bar(np.arange(len(pivot)), pivot.event_clock - pivot.row_clock)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Selected validation sequence (sorted ID)")
    axes[1].set_ylabel("Event minus row WP, descriptive")
    fig.suptitle("Matched causal clocks; 24 previously summary-exposed sequences")
    fig.tight_layout()
    fig.savefig(OUT / "fig/continuation_clock_scores.png", dpi=130)
    plt.close(fig)
    save_json("cache/continuation_results.json", result)
    save_json("cache/continuation_manifest.json", manifest)
    state.update(
        phase="clock_probe_completed",
        inspected_sequence_ids=manifest,
        pending_work=["inspect figures", "headless notebook", "report"],
        artifacts=[
            "RECOVERY_SUMMARY.txt",
            "HYPOTHESES.json",
            "cache/continuation_manifest.json",
            "cache/continuation_results.json",
            "cache/continuation_sequence_scores.csv",
            "cache/continuation_models.npz",
            "fig/continuation_clock_scores.png",
            "fig/continuation_clock_window.png",
        ],
    )
    save_json("STATE.json", state)
    logging.info("Completed clock probe: %s", json.dumps(result))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        main()
    except Exception:
        logging.exception("Clock probe failed; preserve traceback and stop for review")
        raise
