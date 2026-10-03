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
MODES = [
    "anchor",
    "anchor_clocks",
    "anchor_flow",
    "anchor_aux",
    "current_held_clocks",
    "current",
]
CLOCKS = [FEATURES.index(f"a{i}") for i in [5, 6, 7]]
FLOW = [FEATURES.index(f"i{i}_{c}") for i in range(2) for c in ["v0", "v11", "dv0"]]
AUX = [FEATURES.index(f"a{i}") for i in range(8)]


def block_features(x: np.ndarray, anchor: np.ndarray, mode: str) -> np.ndarray:
    """Choose current or cached input blocks, without targets or future rows."""
    if mode == "current":
        return x
    if mode == "current_held_clocks":
        result = x.copy()
        columns = CLOCKS + [c + len(FEATURES) for c in CLOCKS]
        result[:, columns] = x[anchor][:, columns]
        return result
    result = x[anchor].copy()
    groups = {"anchor": [], "anchor_clocks": CLOCKS, "anchor_flow": FLOW, "anchor_aux": AUX}
    chosen = groups[mode]
    columns = chosen + [c + len(FEATURES) for c in chosen]
    result[:, columns] = x[:, columns]
    return result


def main() -> None:
    """Run the preregistered block-time ablation on the original bounded sample."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run through continuation/run_cpu.sh with CUDA disabled")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    hypotheses = json.loads((ROOT / "HYPOTHESES.json").read_text())
    fitted = np.load(ROOT / "s30_coefficients.npz")
    mean, scale = fitted["mean"], fitted["scale"]
    dim = len(mean) + 1
    grams = {mode: np.zeros((2, dim, dim)) for mode in MODES}
    rhs = {mode: np.zeros((dim, 2)) for mode in MODES}
    weights = np.zeros(2)
    for rg in hypotheses["sample"]["train_rowgroups"]:
        data = read_group(pq.ParquetFile("datasets/train.parquet"), rg, "train")
        idx = selected(data, "train")[::8]
        x = features(data)
        anchor = indices(data, rg)["a6"]
        y = np.clip(np.column_stack([data["t0"], data["t1"]])[idx], -2, 2)
        w = np.abs(y)
        weights += w.sum(axis=0)
        for mode in MODES:
            d = design(block_features(x, anchor, mode)[idx], mean, scale)
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
    prefix_checks = []
    for split, rg in [("train", 104), ("valid", 6)]:
        data = read_group(pq.ParquetFile(f"datasets/{split}.parquet"), rg, split)
        full = indices(data, rg)
        for cutoff in [99, 641, 1501]:
            prefix = indices({c: x[:cutoff] for c, x in data.items()}, rg)
            for name in full:
                np.testing.assert_array_equal(full[name][:cutoff], prefix[name])
            prefix_checks.append(
                {"split": split, "rowgroup": rg, "prefix_length": cutoff, "passed": True}
            )
    for rg in hypotheses["sample"]["valid_rowgroups"]:
        data = read_group(pq.ParquetFile("datasets/valid.parquet"), rg, "valid")
        idx = selected(data, "valid")
        x = features(data)
        anchor = indices(data, rg)["a6"]
        y = np.column_stack([data["t0"], data["t1"]])[idx]
        for mode in MODES:
            prediction = (
                design(block_features(x, anchor, mode)[idx], mean, scale) @ coefficients[mode]
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
    anchor_score = scores["anchor"]["mean_wp"]
    gap = scores["current"]["mean_wp"] - anchor_score
    clock_fraction = (scores["anchor_clocks"]["mean_wp"] - anchor_score) / gap
    flow_gain = scores["anchor_flow"]["mean_wp"] - anchor_score
    previous = json.loads((ROOT / "s30_results.json").read_text())
    checks = {
        "current_score_absolute_difference": abs(
            scores["current"]["mean_wp"] - previous["scores"]["current"]["mean_wp"]
        ),
        "anchor_score_absolute_difference": abs(
            anchor_score - previous["scores"]["anchor_fit"]["mean_wp"]
        ),
    }
    if max(checks.values()) > 1e-10:
        raise AssertionError(f"Matched anchor/current fits failed reproduction: {checks}")
    result = {
        "classification": "adaptive exploratory block-time ablation; same exposed sample",
        "scores": scores,
        "clock_recovery_fraction": clock_fraction,
        "flow_gain_over_anchor": flow_gain,
        "h3_rejected": clock_fraction < 0.8,
        "h4_rejected": flow_gain < 0.02,
        "prefix_checks": prefix_checks,
        "s30_reproduction_checks": checks,
    }
    save_json(ROOT / "s31_results.json", result)
    np.savez(ROOT / "s31_coefficients.npz", mean=mean, scale=scale, **coefficients)
    with (ROOT / "s31_sequence_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        ROOT / "s31_provenance.json",
        {
            "hypotheses_sha256": hashlib.sha256(
                (ROOT / "HYPOTHESES_s31.json").read_bytes()
            ).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parent_script_sha256": hashlib.sha256(
                (ROOT / "s30_anchor_probe.py").read_bytes()
            ).hexdigest(),
            "sample_and_sequence_ids": "s30_provenance.json, unchanged",
            "adaptive_after": "s30_results.json",
            "target_features": False,
            "future_inputs": False,
        },
    )
    fig, axis = plt.subplots(figsize=(11, 5), layout="constrained")
    values = [scores[mode]["mean_wp"] for mode in MODES]
    axis.bar(
        MODES, values, color=["#999999", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#0072B2"]
    )
    axis.tick_params(axis="x", rotation=20)
    axis.set_ylabel("Mean clipped target-weighted correlation")
    axis.set_title("Feature time ablation: identical ridge/sample; adaptive, previously exposed")
    axis.axhline(anchor_score, color="gray", linestyle="--", lw=0.8)
    fig.savefig(ROOT / "f08_revision_blocks.png", dpi=130)
    plt.close(fig)
    state = json.loads((ROOT / "STATE.json").read_text())
    state["phase"] = "revision_blocks_complete"
    state["artifacts"] += [
        "HYPOTHESES_s31.json",
        "s31_results.json",
        "s31_sequence_scores.csv",
        "s31_coefficients.npz",
        "s31_provenance.json",
        "f08_revision_blocks.png",
    ]
    state["pending"] = ["inspect f08", "headless notebook", "REPORT.txt"]
    save_json(ROOT / "STATE.json", state)
    LOGGER.info("Results %s", json.dumps(result))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except Exception:
        LOGGER.exception("Revision-block probe failed")
        raise
