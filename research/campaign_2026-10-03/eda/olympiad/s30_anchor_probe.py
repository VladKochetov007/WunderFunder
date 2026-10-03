from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FEATURES = [
    f"i{i}_{name}" for i in range(2) for name in ("p0", "p11", "dp0", "v0", "v11", "dv0")
] + [f"a{i}" for i in range(8)]
AGE_EDGES = np.array([0, 1, 8, 32, 128, 20001])
MODES = (
    ["current", "carry_a6", "carry_price"]
    + [f"carry_delay_{i}" for i in range(5)]
    + ["anchor_fit", "delay_fit"]
)
LOGGER = logging.getLogger(__name__)


def save_json(path: Path, payload: dict) -> None:
    """Save small structured probe artifacts within the owned directory."""
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def read_group(pf: pq.ParquetFile, rg: int, split: str) -> dict[str, np.ndarray]:
    """Read one row group and only probe columns, leaving targets out of features."""
    flag = "need_prediction" if split == "train" else "is_scored"
    columns = FEATURES + ["t0", "t1", "seq_ix", "step_in_seq", flag]
    table = pf.read_row_group(rg, columns=columns, use_threads=False)
    return {c: table.column(c).to_numpy(zero_copy_only=False) for c in columns}


def features(data: dict[str, np.ndarray]) -> np.ndarray:
    """Separate finite input values from the frozen sentinel indicators."""
    raw = np.column_stack([data[c] for c in FEATURES]).astype(np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite input encountered; frozen probe has no imputer")
    sentinel = raw < -5.19
    return np.column_stack([np.where(sentinel, 0, raw), sentinel])


def anchor_indices(reset: np.ndarray) -> np.ndarray:
    """Return the most recent input-defined reset index using only prefixes."""
    reset = reset.copy()
    reset[0] = True
    return np.maximum.accumulate(np.where(reset, np.arange(len(reset)), 0))


def delayed_indices(reset: np.ndarray, seed: int) -> np.ndarray:
    """Queue each arriving reset with a random delay, never reading future events."""
    rng = np.random.default_rng(seed)
    delayed = np.zeros(len(reset), dtype=bool)
    delayed[0] = True
    last_due = 0
    # An event's due time depends only on that event and previously queued events.
    for event in np.flatnonzero(reset)[1:]:
        due = max(int(event) + int(rng.integers(1, 8)), last_due + 1)
        last_due = due
        if due < len(reset):
            delayed[due] = True
    return anchor_indices(delayed)


def indices(data: dict[str, np.ndarray], rg: int) -> dict[str, np.ndarray]:
    """Construct true, price and delayed reset streams without target inputs."""
    reset = data["a6"] < -5.19
    reset = reset.copy()
    reset[0] = True
    price = np.r_[
        True,
        (np.diff(data["i0_p0"]) != 0) | (np.diff(data["i0_p11"]) != 0),
    ]
    result = {"a6": anchor_indices(reset), "price": anchor_indices(price)}
    for replicate in range(5):
        seed = 7300 + rg + 100000 * replicate
        result[f"delay_{replicate}"] = delayed_indices(reset, seed)
    return result


def selected(data: dict[str, np.ndarray], split: str) -> np.ndarray:
    """Select scoring rows, keeping the entire input prefix available."""
    flag = "need_prediction" if split == "train" else "is_scored"
    return np.flatnonzero(data[flag] & (data["step_in_seq"] >= 99))


def design(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply training-only normalization and add an intercept."""
    return np.column_stack([np.ones(len(x)), (x - mean) / scale])


def moments(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Compute additive moments for clipped, target-weighted correlation."""
    y = np.clip(y, -2, 2)
    prediction = np.clip(prediction, -2, 2)
    weight = np.abs(y)
    return np.stack(
        [
            weight.sum(axis=0),
            (weight * y).sum(axis=0),
            (weight * prediction).sum(axis=0),
            (weight * y * y).sum(axis=0),
            (weight * prediction * prediction).sum(axis=0),
            (weight * y * prediction).sum(axis=0),
        ],
        axis=1,
    )


def correlation(stats: np.ndarray) -> np.ndarray:
    """Calculate per-target correlations, returning NaN for constant predictions."""
    weight, sy, sp, syy, spp, syp = stats.T
    covariance = syp - sy * sp / np.maximum(weight, 1e-30)
    denominator = np.sqrt(
        np.maximum(syy - sy * sy / np.maximum(weight, 1e-30), 0)
        * np.maximum(spp - sp * sp / np.maximum(weight, 1e-30), 0)
    )
    return np.divide(
        covariance,
        denominator,
        out=np.full(2, np.nan),
        where=denominator > 1e-14,
    )


def score_record(stats: np.ndarray) -> dict:
    """Represent undefined correlations explicitly in JSON."""
    scores = correlation(stats)
    return {
        "t0": float(scores[0]) if np.isfinite(scores[0]) else None,
        "t1": float(scores[1]) if np.isfinite(scores[1]) else None,
        "mean_wp": float(scores.mean()) if np.isfinite(scores).all() else None,
    }


def audit(data: dict[str, np.ndarray], rg: int, split: str) -> dict:
    """Audit transitions with targets offline, never using them for inference."""
    change = np.r_[
        False,
        (np.diff(data["t0"]) != 0) | (np.diff(data["t1"]) != 0),
    ]
    reset = data["a6"] < -5.19
    reset = reset.copy()
    reset[0] = False
    price = np.r_[
        False,
        (np.diff(data["i0_p0"]) != 0) | (np.diff(data["i0_p11"]) != 0),
    ]
    return {
        "split": split,
        "rowgroup": rg,
        "seq_ix": int(data["seq_ix"][0]),
        "rows": len(change),
        "eligible_rows": len(selected(data, split)),
        "target_changes": int(change.sum()),
        "a6_resets": int(reset.sum()),
        "a6_misses": int((change & ~reset).sum()),
        "a6_extras": int((reset & ~change).sum()),
        "price_resets": int(price.sum()),
        "price_misses": int((change & ~price).sum()),
        "sentinel_values": np.unique(data["a6"][reset]).tolist(),
        "prior_exposure": "predecessor exhaustive checks; exploratory reanalysis",
    }


def main() -> None:
    """Run the frozen bounded probe and preserve scores, provenance and figures."""
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use continuation/run_cpu.sh; CUDA must be unavailable")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    hypotheses = json.loads((ROOT / "HYPOTHESES.json").read_text())
    train_rgs = hypotheses["sample"]["train_rowgroups"]
    valid_rgs = hypotheses["sample"]["valid_rowgroups"]
    train = pq.ParquetFile("datasets/train.parquet")
    valid = pq.ParquetFile("datasets/valid.parquet")
    audits: list[dict] = []
    total = 0
    sums = np.zeros(2 * len(FEATURES))
    squares = np.zeros_like(sums)
    for rg in train_rgs:
        data = read_group(train, rg, "train")
        audits.append(audit(data, rg, "train"))
        x = features(data)[selected(data, "train")[::8]]
        total += len(x)
        sums += x.sum(axis=0)
        squares += (x * x).sum(axis=0)
    mean = sums / total
    scale = np.sqrt(np.maximum(squares / total - mean * mean, 0))
    scale[scale < 1e-8] = 1
    dim = len(mean) + 1
    fit_names = ["current", "anchor_fit", "delay_fit"]
    grams = {name: np.zeros((2, dim, dim)) for name in fit_names}
    rhs = {name: np.zeros((dim, 2)) for name in fit_names}
    weights = np.zeros(2)
    for rg in train_rgs:
        data = read_group(train, rg, "train")
        idx = selected(data, "train")[::8]
        x = features(data)
        events = indices(data, rg)
        y = np.clip(np.column_stack([data["t0"], data["t1"]])[idx], -2, 2)
        w = np.abs(y)
        weights += w.sum(axis=0)
        sources = {
            "current": idx,
            "anchor_fit": events["a6"][idx],
            "delay_fit": events["delay_0"][idx],
        }
        for name, source in sources.items():
            d = design(x[source], mean, scale)
            for target in range(2):
                wd = d * w[:, target, None]
                grams[name][target] += d.T @ wd
                rhs[name][:, target] += wd.T @ y[:, target]
    penalty = np.eye(dim)
    penalty[0, 0] = 0
    coefficients = {}
    for name in fit_names:
        coefficients[name] = np.column_stack(
            [
                np.linalg.solve(
                    grams[name][target] + 0.001 * weights[target] * penalty,
                    rhs[name][:, target],
                )
                for target in range(2)
            ]
        )
    np.savez(
        ROOT / "s30_coefficients.npz",
        mean=mean,
        scale=scale,
        feature_names=np.array(FEATURES),
        **coefficients,
    )
    pooled = {name: np.zeros((2, 6)) for name in MODES}
    age_stats = {name: np.zeros((len(AGE_EDGES) - 1, 2, 6)) for name in MODES}
    sequence_stats = []
    per_sequence = []
    age_counts = np.zeros(len(AGE_EDGES) - 1, dtype=int)
    example = {}
    scored_rows = 0
    for rg in valid_rgs:
        data = read_group(valid, rg, "valid")
        record = audit(data, rg, "valid")
        audits.append(record)
        idx = selected(data, "valid")
        scored_rows += len(idx)
        x = features(data)
        d = design(x, mean, scale)
        events = indices(data, rg)
        age = np.arange(len(x)) - events["a6"]
        bins = np.searchsorted(AGE_EDGES[1:], age[idx], side="right")
        y_all = np.column_stack([data["t0"], data["t1"]])
        y = y_all[idx]
        raw_prediction = d @ coefficients["current"]
        predictions = {
            "current": raw_prediction,
            "carry_a6": raw_prediction[events["a6"]],
            "carry_price": raw_prediction[events["price"]],
            "anchor_fit": (d @ coefficients["anchor_fit"])[events["a6"]],
            "delay_fit": (d @ coefficients["delay_fit"])[events["delay_0"]],
        }
        for replicate in range(5):
            predictions[f"carry_delay_{replicate}"] = raw_prediction[events[f"delay_{replicate}"]]
            record[f"delay_{replicate}_resets"] = int(
                np.sum(np.diff(events[f"delay_{replicate}"]) != 0)
            )
        local_stats = np.zeros((len(MODES), 2, 6))
        for mode_i, name in enumerate(MODES):
            stats = moments(y, predictions[name][idx])
            local_stats[mode_i] = stats
            pooled[name] += stats
            per_sequence.append(
                {
                    "rowgroup": rg,
                    "seq_ix": record["seq_ix"],
                    "mode": name,
                    "n_scored": len(idx),
                    **score_record(stats),
                }
            )
            for age_i in range(len(AGE_EDGES) - 1):
                mask = bins == age_i
                age_stats[name][age_i] += moments(y[mask], predictions[name][idx][mask])
        sequence_stats.append(local_stats)
        for age_i in range(len(age_counts)):
            age_counts[age_i] += int(np.sum(bins == age_i))
        if rg == valid_rgs[0]:
            example = {
                "step": data["step_in_seq"].copy(),
                "a6": data["a6"].copy(),
                "target": y_all.copy(),
                "current": predictions["current"].copy(),
                "anchor_fit": predictions["anchor_fit"].copy(),
                "carry_a6": predictions["carry_a6"].copy(),
                "seq_ix": record["seq_ix"],
            }
        LOGGER.info("Evaluated valid rg=%s seq=%s rows=%s", rg, record["seq_ix"], len(idx))
    seq_stats = np.stack(sequence_stats)
    np.save(ROOT / "s30_sequence_moments.npy", seq_stats)
    current_i = MODES.index("current")
    anchor_i = MODES.index("carry_a6")
    differences = []
    for stats in seq_stats:
        differences.append(
            float(correlation(stats[current_i]).mean() - correlation(stats[anchor_i]).mean())
        )
    differences = np.array(differences)
    evaluable = np.isfinite(differences)
    win_fraction = float(np.mean(differences[evaluable] > 0))
    mean_delta = float(
        correlation(pooled["current"]).mean() - correlation(pooled["carry_a6"]).mean()
    )
    bootstrap = []
    rng = np.random.default_rng(301)
    for _ in range(500):
        sample = seq_stats[rng.integers(0, len(seq_stats), len(seq_stats))].sum(axis=0)
        bootstrap.append(
            float(correlation(sample[current_i]).mean() - correlation(sample[anchor_i]).mean())
        )
    result = {
        "classification": "exploratory bounded exposed-sample causal-input probe",
        "train_fit_rows": total,
        "valid_scored_rows": scored_rows,
        "scores": {name: score_record(pooled[name]) for name in MODES},
        "current_minus_carry_a6": mean_delta,
        "sequence_win_fraction": win_fraction,
        "sequence_evaluable": int(evaluable.sum()),
        "sequence_bootstrap_descriptive_delta_quantiles": np.quantile(
            bootstrap, [0.025, 0.5, 0.975]
        ).tolist(),
        "bootstrap_caveat": "sequence resampling is descriptive; no connected-component or campaign-independent inference",
        "h1_rejected_for_probe": mean_delta > 0.01 and win_fraction > 0.60,
        "h2_threshold_misses": sum(r["a6_misses"] for r in audits),
        "h2_threshold_extras": sum(r["a6_extras"] for r in audits),
        "price_anchor_misses": sum(r["price_misses"] for r in audits),
        "age_bins": [
            {
                "lo": int(AGE_EDGES[i]),
                "hi_exclusive": int(AGE_EDGES[i + 1]),
                "rows": int(age_counts[i]),
                "scores": {name: score_record(age_stats[name][i]) for name in MODES},
            }
            for i in range(len(age_counts))
        ],
        "seconds": time.monotonic() - started,
    }
    save_json(ROOT / "s30_results.json", result)
    save_json(ROOT / "s30_audit.json", {"sequences": audits})
    with (ROOT / "s30_sequence_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sequence[0]))
        writer.writeheader()
        writer.writerows(per_sequence)
    provenance = {
        "hypotheses_sha256": hashlib.sha256((ROOT / "HYPOTHESES.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "features": FEATURES,
        "sample": hypotheses["sample"],
        "sequence_ids": [
            {"split": r["split"], "rowgroup": r["rowgroup"], "seq_ix": r["seq_ix"]} for r in audits
        ],
        "datasets": {
            split: {
                "path": f"datasets/{split}.parquet",
                "file_size": Path(f"datasets/{split}.parquet").stat().st_size,
                "rowgroups": pf.num_row_groups,
                "rows": pf.metadata.num_rows,
                "schema_sha256": hashlib.sha256(str(pf.schema_arrow).encode()).hexdigest(),
            }
            for split, pf in [("train", train), ("valid", valid)]
        },
        "environment": {
            key: os.environ.get(key)
            for key in ["CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]
        },
        "peak_array_bound": "one sequence x 40 features, 10 predictions; well under 100 MB arrays",
        "causality": "no target-derived regimes, future minimum, terminal run duration, or target inputs; delayed control queues depend only on arriving resets",
        "limitations": "all sampled sequences previously exposed; separate splits do not establish independent discovery; weak probe is not strong baseline comparison",
    }
    save_json(ROOT / "s30_provenance.json", provenance)
    plot_results(result, example)
    state = json.loads((ROOT / "STATE.json").read_text())
    state.update(
        phase="anchor_probe_complete",
        inspected_sequence_ids=provenance["sequence_ids"],
        artifacts=state["artifacts"]
        + [
            "s30_results.json",
            "s30_audit.json",
            "s30_sequence_scores.csv",
            "s30_sequence_moments.npy",
            "s30_coefficients.npz",
            "s30_provenance.json",
            "f06_anchor_probe.png",
            "f07_anchor_predictions.png",
        ],
        pending=["inspect figures", "run marimo notebook headlessly", "write REPORT.txt"],
    )
    save_json(ROOT / "STATE.json", state)
    LOGGER.info("Result %s", json.dumps(result["scores"]))
    LOGGER.info(
        "H1 rejected=%s delta=%.5f wins=%.3f seconds=%.1f",
        result["h1_rejected_for_probe"],
        mean_delta,
        win_fraction,
        result["seconds"],
    )


def plot_results(result: dict, example: dict) -> None:
    """Plot measured contrasts and a fixed sequence trace, with exposure caveats."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    names = ["current", "carry_a6", "carry_price", "anchor_fit", "delay_fit"]
    values = [result["scores"][name]["mean_wp"] for name in names]
    axes[0].bar(names, values, color=["#0072B2", "#D55E00", "#999999", "#009E73", "#CC79A7"])
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Mean clipped target-weighted correlation")
    delays = [result["scores"][f"carry_delay_{i}"]["mean_wp"] for i in range(5)]
    axes[0].axhspan(
        min(delays), max(delays), alpha=0.15, color="gray", label="five causal delayed controls"
    )
    axes[0].legend(fontsize=8)
    for name in ["current", "carry_a6", "anchor_fit"]:
        ys = [record["scores"][name]["mean_wp"] for record in result["age_bins"]]
        axes[1].plot(range(len(ys)), ys, "o-", label=name)
    labels = [f"{r['lo']}–{r['hi_exclusive'] - 1}\nn={r['rows']}" for r in result["age_bins"]]
    axes[1].set_xticks(range(len(labels)), labels, fontsize=8)
    axes[1].set_xlabel("Rows since current a6 reset (prefix-only)")
    axes[1].set_ylabel("Mean WP within age stratum")
    axes[1].legend()
    fig.suptitle(
        "Frozen olympiad probe: 20 previously exposed valid sequences; no baseline comparison"
    )
    fig.savefig(ROOT / "f06_anchor_probe.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, layout="constrained")
    view = slice(100, 900)
    for name, label in [
        ("target", "t0 observed, offline"),
        ("current", "current inputs"),
        ("anchor_fit", "anchor-trained / carried"),
    ]:
        axes[0].plot(example["step"][view], example[name][view, 0], lw=1, label=label)
    axes[0].legend()
    axes[0].set_ylabel("t0 / predictions")
    axes[1].plot(example["step"][view], example["a6"][view], color="black", lw=0.7)
    axes[1].axhline(-5.19, color="red", linestyle="--", label="fixed reset threshold")
    axes[1].legend()
    axes[1].set_ylabel("a6")
    axes[1].set_xlabel("step_in_seq")
    fig.suptitle(f"Fixed example valid rowgroup 6 / seq {example['seq_ix']}; exploratory")
    fig.savefig(ROOT / "f07_anchor_predictions.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except Exception:
        LOGGER.exception("Probe failed; preserve traceback and return to controller")
        raise
