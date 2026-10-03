from __future__ import annotations

import hashlib
import json
import logging
import resource
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from continuation_probe import OUT, RAW, ROOT, SEED, correlation, features, moments


def ablation_features(raw: np.ndarray, offset: float) -> dict[str, np.ndarray]:
    """Separate age saturation from phase and recurrence on the same inputs."""
    original = features(raw, offset)
    full = original["period4"]
    base = original["age"]
    cap = np.column_stack((base, full[:, -1]))
    sham = original["sham6"]
    angle = 2 * np.pi * (np.arange(len(raw)) + offset) / 4
    sine, cosine = np.sin(angle), np.cos(angle)
    contrast = sham[:, 30]
    return {
        "cap4": cap,
        "phase4": np.column_stack((cap, full[:, 28:30])),
        "recurrence4": np.column_stack((cap, full[:, 30])),
        "full4": full,
        "sham4": np.column_stack(
            (base, sine, cosine, contrast, contrast * sine, contrast * cosine, full[:, -1])
        ),
    }


def main() -> None:
    """Execute the separately frozen conditional ablation and audit artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.monotonic()
    manifest = json.loads((OUT / "continuation_manifest.json").read_text())
    state = json.loads((OUT / "STATE.json").read_text())
    state.update(
        phase="running_frozen_ablation",
        commands=[
            "run_cpu.sh eda-puzzle-b .venv/bin/python scripts/continuation_probe.py",
            "run_cpu.sh eda-puzzle-b .venv/bin/python scripts/clock_ablation.py",
            "run_cpu.sh eda-puzzle-b .venv/bin/marimo export html eda_puzzle_b.py -o continuation_notebook.html",
        ],
    )
    (OUT / "STATE.json").write_text(json.dumps(state, indent=2))
    accumulators = {}
    models = {}
    pooled = {}
    sequence_rows = []
    saved_stats = {}
    for split in ("train", "valid"):
        parquet = pq.ParquetFile(ROOT / "datasets" / f"{split}.parquet")
        for rg in manifest["sample"][split]:
            columns = RAW + ["t0", "t1", "step_in_seq"]
            if split == "valid":
                columns.append("is_scored")
            table = parquet.read_row_group(rg, columns=columns).to_pandas()
            raw = table[RAW].to_numpy(dtype=float)
            offset = float(np.random.default_rng(SEED + rg).uniform(0, 6))
            arms = ablation_features(raw, offset)
            target = np.clip(table[["t0", "t1"]].to_numpy(dtype=float), -2, 2)
            if split == "train":
                rows = np.arange(99, len(raw), 7)
                rows = rows[np.isfinite(target[rows]).all(1)]
                y = target[rows]
                for arm, values in arms.items():
                    x = values[rows]
                    d = x.shape[1]
                    if arm not in accumulators:
                        accumulators[arm] = {
                            "n": 0,
                            "sum": np.zeros(d),
                            "xx": np.zeros((d, d)),
                            "xy": np.zeros((d, 2)),
                            "y": np.zeros(2),
                        }
                    acc = accumulators[arm]
                    acc["n"] += len(x)
                    acc["sum"] += x.sum(0)
                    acc["xx"] += x.T @ x
                    acc["xy"] += x.T @ y
                    acc["y"] += y.sum(0)
            else:
                mask = (
                    table.is_scored.to_numpy(dtype=bool)
                    & (table.step_in_seq.to_numpy() >= 99)
                    & np.isfinite(target).all(1)
                )
                for arm, values in arms.items():
                    mean, sd, beta, ym = models[arm]
                    pred = ((values[mask] - mean) / sd) @ beta + ym
                    stats = moments(pred, target[mask])
                    pooled[arm] += stats
                    saved_stats[rg, arm] = stats
                    score = correlation(stats)
                    sequence_rows.append(
                        {"rg": rg, "arm": arm, "t0": score[0], "t1": score[1], "mean": score.mean()}
                    )
            logging.info("%s rg=%s", split, rg)
        if split == "train":
            models = {}
            for arm, acc in accumulators.items():
                n = acc["n"]
                mean, ym = acc["sum"] / n, acc["y"] / n
                centered = acc["xx"] - n * np.outer(mean, mean)
                sd = np.sqrt(np.maximum(np.diag(centered) / n, 0)) + 1e-6
                xx = centered / np.outer(sd, sd)
                xy = (acc["xy"] - n * mean[:, None] * ym) / sd[:, None]
                beta = np.linalg.solve(xx + 0.01 * n * np.eye(len(mean)), xy)
                models[arm] = mean, sd, beta, ym
            pooled = {arm: np.zeros((6, 2)) for arm in models}
            sequence_rows = []
    sequences = pd.DataFrame(sequence_rows)
    pairs = sequences.pivot(index="rg", columns="arm", values="mean")
    scores = []
    for arm, stats in pooled.items():
        score = correlation(stats)
        scores.append(
            {
                "arm": arm,
                "t0": score[0],
                "t1": score[1],
                "mean": score.mean(),
                "delta_cap4": (score - correlation(pooled["cap4"])).mean(),
                "wins_vs_cap4": int((pairs[arm] > pairs.cap4).sum()),
            }
        )
    result = pd.DataFrame(scores)
    original = pd.read_csv(OUT / "continuation_scores.csv")
    reference = original[(original.stride == 7) & (original.arm == "period4")][
        ["t0", "t1"]
    ].to_numpy()[0]
    difference = np.max(np.abs(correlation(pooled["full4"]) - reference))
    assert difference < 1e-10, difference
    result.to_csv(OUT / "ablation_scores.csv", index=False)
    sequences.to_csv(OUT / "ablation_sequence_scores.csv", index=False)
    audit = {
        "protocol_sha256": hashlib.sha256(
            (OUT / "HYPOTHESES_ABLATION.json").read_bytes()
        ).hexdigest(),
        "full4_reproduction_max_abs_difference": float(difference),
        "wall_seconds": time.monotonic() - started,
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "n_scored_rows": int(
            pd.read_csv(OUT / "continuation_sequence_scores.csv")
            .query("stride==7 and arm=='raw'")
            .n.sum()
        ),
        "worst_full4_minus_cap4": (pairs.full4 - pairs.cap4).nsmallest(5).to_dict(),
        "best_full4_minus_cap4": (pairs.full4 - pairs.cap4).nlargest(5).to_dict(),
    }
    sensitivity = []
    for excluded in ([], [54], [1385], [54, 1385]):
        control = sum(
            value
            for (rg, arm), value in saved_stats.items()
            if arm == "cap4" and rg not in excluded
        )
        candidate = sum(
            value
            for (rg, arm), value in saved_stats.items()
            if arm == "full4" and rg not in excluded
        )
        sensitivity.append(
            {
                "excluded_rg": excluded,
                "cap4": correlation(control).tolist(),
                "full4": correlation(candidate).tolist(),
                "delta_mean": float((correlation(candidate) - correlation(control)).mean()),
            }
        )
    audit["posthoc_concentration_sensitivity"] = sensitivity
    np.savez(
        OUT / "ablation_sequence_moments.npz",
        **{f"rg{rg}_{arm}": value for (rg, arm), value in saved_stats.items()},
    )
    (OUT / "ablation_audit.json").write_text(json.dumps(audit, indent=2))
    state.update(
        phase="ablation_complete", pending_work=["finish notebook", "inspect figures", "report"]
    )
    (OUT / "STATE.json").write_text(json.dumps(state, indent=2))
    logging.info("complete seconds=%.2f", audit["wall_seconds"])


if __name__ == "__main__":
    main()
