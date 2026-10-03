from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from s30_anchor_probe import (
    FEATURES,
    ROOT,
    design,
    features,
    indices,
    moments,
    read_group,
    save_json,
    score_record,
    selected,
)

LOGGER = logging.getLogger(__name__)
MODES = ["anchor_p", "anchor_dp", "anchor_pdp", "current_held_pdp", "current"]
P = [FEATURES.index(f"i{i}_{c}") for i in range(2) for c in ["p0", "p11"]]
DP = [FEATURES.index(f"i{i}_dp0") for i in range(2)]


def choose_features(x: np.ndarray, anchor: np.ndarray, mode: str) -> np.ndarray:
    """Change input timestamps for frozen p/dp blocks while preserving dimension."""
    if mode == "current":
        return x
    if mode == "current_held_pdp":
        result = x.copy()
        chosen = P + DP
        columns = chosen + [i + len(FEATURES) for i in chosen]
        result[:, columns] = x[anchor][:, columns]
        return result
    result = x[anchor].copy()
    chosen = {"anchor_p": P, "anchor_dp": DP, "anchor_pdp": P + DP}[mode]
    columns = chosen + [i + len(FEATURES) for i in chosen]
    result[:, columns] = x[:, columns]
    return result


def main() -> None:
    """Measure necessity and sufficiency of current p/dp in the frozen ridge probe."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run through continuation/run_cpu.sh with CUDA disabled")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    sample = json.loads((ROOT / "HYPOTHESES.json").read_text())["sample"]
    previous = json.loads((ROOT / "s30_results.json").read_text())
    fitted = np.load(ROOT / "s30_coefficients.npz")
    mean, scale = fitted["mean"], fitted["scale"]
    dim = len(mean) + 1
    grams = {mode: np.zeros((2, dim, dim)) for mode in MODES}
    rhs = {mode: np.zeros((dim, 2)) for mode in MODES}
    weights = np.zeros(2)
    train = pq.ParquetFile("datasets/train.parquet")
    valid = pq.ParquetFile("datasets/valid.parquet")
    for rg in sample["train_rowgroups"]:
        data = read_group(train, rg, "train")
        idx = selected(data, "train")[::8]
        x = features(data)
        anchor = indices(data, rg)["a6"]
        y = np.clip(np.column_stack([data["t0"], data["t1"]])[idx], -2, 2)
        w = np.abs(y)
        weights += w.sum(axis=0)
        for mode in MODES:
            d = design(choose_features(x, anchor, mode)[idx], mean, scale)
            for target in range(2):
                wd = d * w[:, target, None]
                grams[mode][target] += d.T @ wd
                rhs[mode][:, target] += wd.T @ y[:, target]
    penalty = np.eye(dim)
    penalty[0, 0] = 0
    coefficients = {
        mode: np.column_stack(
            [
                np.linalg.solve(grams[mode][i] + 0.001 * weights[i] * penalty, rhs[mode][:, i])
                for i in range(2)
            ]
        )
        for mode in MODES
    }
    pooled = {mode: np.zeros((2, 6)) for mode in MODES}
    rows = []
    for rg in sample["valid_rowgroups"]:
        data = read_group(valid, rg, "valid")
        idx = selected(data, "valid")
        x = features(data)
        anchor = indices(data, rg)["a6"]
        y = np.column_stack([data["t0"], data["t1"]])[idx]
        for mode in MODES:
            prediction = (
                design(choose_features(x, anchor, mode)[idx], mean, scale) @ coefficients[mode]
            )
            stats = moments(y, prediction)
            pooled[mode] += stats
            rows.append(
                {
                    "rowgroup": rg,
                    "seq_ix": int(data["seq_ix"][0]),
                    "mode": mode,
                    **score_record(stats),
                }
            )
    scores = {mode: score_record(pooled[mode]) for mode in MODES}
    control_difference = abs(
        scores["current"]["mean_wp"] - previous["scores"]["current"]["mean_wp"]
    )
    if control_difference > 1e-10:
        raise AssertionError("Current-input control failed exact reproduction")
    anchor_score = previous["scores"]["anchor_fit"]["mean_wp"]
    gap = scores["current"]["mean_wp"] - anchor_score
    fraction = (scores["anchor_pdp"]["mean_wp"] - anchor_score) / gap
    loss = scores["current"]["mean_wp"] - scores["current_held_pdp"]["mean_wp"]
    result = {
        "classification": "adaptive exposed-sample p/dp feature-time ablation",
        "scores": scores,
        "anchor_pdp_recovery_fraction": fraction,
        "holding_pdp_loss": loss,
        "hypothesis_rejected": fraction < 0.8 or loss < 0.02,
        "current_control_difference": control_difference,
    }
    save_json(ROOT / "s32_results.json", result)
    np.savez(ROOT / "s32_coefficients.npz", mean=mean, scale=scale, **coefficients)
    with (ROOT / "s32_sequence_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        ROOT / "s32_provenance.json",
        {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "hypotheses_sha256": hashlib.sha256(
                (ROOT / "HYPOTHESES_s32.json").read_bytes()
            ).hexdigest(),
            "parent_script_sha256": hashlib.sha256(
                (ROOT / "s30_anchor_probe.py").read_bytes()
            ).hexdigest(),
            "sample_and_ids": "s30_provenance.json, unchanged",
            "adaptive_after": "s31_results.json",
        },
    )
    fig, axis = plt.subplots(figsize=(11, 5), layout="constrained")
    axis.bar(MODES, [scores[mode]["mean_wp"] for mode in MODES], color="#0072B2")
    axis.axhline(anchor_score, color="gray", linestyle="--", label="anchor-only model")
    axis.tick_params(axis="x", rotation=20)
    axis.set_ylabel("Mean clipped target-weighted correlation")
    axis.set_title("p/dp necessity and sufficiency: adaptive, same exposed sample")
    axis.legend()
    fig.savefig(ROOT / "f09_price_revision.png", dpi=130)
    plt.close(fig)
    state = json.loads((ROOT / "STATE.json").read_text())
    state["phase"] = "price_revision_complete"
    state["artifacts"] += [
        "HYPOTHESES_s32.json",
        "s32_results.json",
        "s32_coefficients.npz",
        "s32_sequence_scores.csv",
        "s32_provenance.json",
        "f09_price_revision.png",
    ]
    state["pending"] = ["inspect f09", "headless notebook", "REPORT.txt"]
    save_json(ROOT / "STATE.json", state)
    LOGGER.info("Results %s", json.dumps(result))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except Exception:
        LOGGER.exception("Price-revision probe failed")
        raise
