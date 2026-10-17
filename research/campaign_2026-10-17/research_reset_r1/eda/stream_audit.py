"""Descriptive, stratified no-fit audit; one parquet sequence in memory at a time.

Market classes are retrospective label strata, never proposed inference inputs.
Every statistic is exploratory on already exposed data. No model is fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def moments(y: np.ndarray, mask: np.ndarray) -> list:
    """Return target joint descriptive moments on explicitly selected rows."""
    z = np.clip(y[mask].astype(np.float64), -2, 2)
    return [len(z), *z.sum(0), *(z * z).sum(0), (z[:, 0] * z[:, 1]).sum()]


def audit_sequence(x: np.ndarray, y: np.ndarray, need: np.ndarray, scored: np.ndarray) -> dict:
    """Describe changes, event ages, selected-label mass and fixed lag correlations."""
    ne = x[1:] != x[:-1]
    events = {
        "i0_price_and": ne[:, 0] & ne[:, 11],
        "i0_price_any": ne[:, :22].any(1),
        "i0_volume": ne[:, 22:44].any(1),
        "i0_trade": ne[:, 44:52].any(1),
        "i1_price": ne[:, 52:74].any(1),
        "i1_volume": ne[:, 74:96].any(1),
        "i1_trade": ne[:, 96:104].any(1),
        "extras": ne[:, 104:].any(1),
        "any": ne.any(1),
    }
    flags = {k: np.r_[False, v] for k, v in events.items()}
    age = np.arange(len(x)) - np.maximum.accumulate(
        np.where(flags["i0_price_and"], np.arange(len(x)), 0)
    )
    bins = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 20_001])
    group = np.searchsorted(bins, age, side="right") - 1
    mass = np.abs(np.clip(y.astype(np.float64), -2, 2))
    age_stats = []
    for j in range(len(bins) - 1):
        m = (group == j) & need
        s = m & scored
        age_stats.append([int(m.sum()), int(s.sum()), *mass[m].sum(0), *mass[s].sum(0)])
    target_change = np.any(y[1:] != y[:-1], axis=1)
    eqrun = np.diff(np.r_[0, np.flatnonzero(events["any"]) + 1, len(x)])
    target_run = np.diff(np.r_[0, np.flatnonzero(target_change) + 1, len(x)])
    lag_stats = {}
    for lag in [-64, -16, -4, -1, 0, 1, 4, 16, 64]:
        a, b = flags["i1_price"], flags["i0_price_and"]
        # Positive lag: i1 at t precedes i0 at t+lag. Negative lag is a control.
        if lag > 0:
            a, b = a[:-lag], b[lag:]
        elif lag < 0:
            a, b = a[-lag:], b[:lag]
        lag_stats[str(lag)] = [len(a), int(a.sum()), int(b.sum()), int((a & b).sum())]
    selected = need & scored
    z = np.clip(y, -2, 2)
    return {
        "rows": len(x), "need": int(need.sum()), "scored": int(selected.sum()),
        "nonfinite_x": int((~np.isfinite(x)).sum()),
        "nonfinite_y": int((~np.isfinite(y)).sum()),
        "events": {k: int(v.sum()) for k, v in events.items()},
        "target_changes": int(target_change.sum()),
        "target_changes_without_i0_and": int((target_change & ~events["i0_price_and"]).sum()),
        "i0_and_without_target_change": int((~target_change & events["i0_price_and"]).sum()),
        "age_bins": bins.tolist(), "age_stats": age_stats,
        "raw_identical_run_quantiles": np.quantile(eqrun, [0.5, 0.9, 0.99, 1]).tolist(),
        "target_run_quantiles": np.quantile(target_run, [0.5, 0.9, 0.99, 1]).tolist(),
        "joint_all": moments(y, need), "joint_scored": moments(y, selected),
        "zero_fraction_all": (y[need] == 0).mean(0).tolist(),
        "clipped_fraction_all": (np.abs(y[need]) >= 2).mean(0).tolist(),
        "clipped_mass_scored": (np.abs(z[selected]) * (np.abs(y[selected]) >= 2)).sum(0).tolist(),
        "mass_scored": np.abs(z[selected]).sum(0).tolist(),
        "lag_events": lag_stats,
    }


def aggregate(rows: list[dict]) -> dict:
    """Aggregate exact sufficient statistics, retaining sequence-level variability."""
    out = {k: sum(r[k] for r in rows) for k in [
        "rows", "need", "scored", "nonfinite_x", "nonfinite_y", "target_changes",
        "target_changes_without_i0_and", "i0_and_without_target_change",
    ]}
    out["events"] = {k: sum(r["events"][k] for r in rows) for k in rows[0]["events"]}
    for k in ["age_stats", "joint_all", "joint_scored", "clipped_mass_scored", "mass_scored"]:
        out[k] = np.array([r[k] for r in rows]).sum(0).tolist()
    out["sequence_scored_rate_quantiles"] = np.quantile(
        [r["scored"] / r["need"] for r in rows], [0, .1, .5, .9, 1]
    ).tolist()
    for k in ["zero_fraction_all", "clipped_fraction_all", "raw_identical_run_quantiles", "target_run_quantiles"]:
        out[f"mean_sequence_{k}"] = np.array([r[k] for r in rows]).mean(0).tolist()
    out["lag_events"] = {k: np.array([r["lag_events"][k] for r in rows]).sum(0).tolist()
                         for k in rows[0]["lag_events"]}
    return out


def drift(a: np.ndarray, b: np.ndarray, names: list[str]) -> list[dict]:
    """Compute train-anchored marginal histogram JS, with no learned predictor."""
    out = []
    for j, name in enumerate(names):
        edges = np.unique(np.quantile(a[:, j], np.linspace(0, 1, 21)))
        edges = np.r_[-np.inf, edges[1:-1], np.inf]
        p = np.histogram(a[:, j], bins=edges)[0].astype(float) + 1e-10
        q = np.histogram(b[:, j], bins=edges)[0].astype(float) + 1e-10
        p, q = p / p.sum(), q / q.sum()
        m = (p + q) / 2
        js = ((p * np.log(p / m)).sum() + (q * np.log(q / m)).sum()) / 2
        out.append({"feature": name, "js_nats": float(js), "bins": len(p),
                    "train_quantiles": np.quantile(a[:, j], [.01, .5, .99]).tolist(),
                    "valid_quantiles": np.quantile(b[:, j], [.01, .5, .99]).tolist()})
    return sorted(out, key=lambda r: r["js_nats"], reverse=True)


def main() -> None:
    """Sample each known class uniformly without replacement and write aggregates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    start = time.time()
    rng = np.random.default_rng(20261017)
    names = [f"i{i}_{g}{j}" for i in range(2) for g, n in
             [("p", 22), ("v", 22), ("dp", 4), ("dv", 4)] for j in range(n)]
    names += [f"a{i}" for i in range(8)]
    report = {"seed": 20261017, "per_class": args.per_class, "training_steps": 0,
              "selection": "uniform sequences within cached retrospective market class; equal strata, not natural pooled mixture",
              "age_bins": [0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 20_001],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "splits": {}}
    samples = {}
    for split in ["train", "valid"]:
        cache = np.load(args.cache / f"{split}_targets.npz")
        groups, classes = cache["row_group"], cache["market_class"]
        path = args.data / f"{split}.parquet"
        pf = pq.ParquetFile(path)
        split_out = {"path": str(path), "bytes": path.stat().st_size,
                     "mtime_ns": path.stat().st_mtime_ns, "strata": {}}
        for cls in range(4):
            eligible = groups[classes == cls]
            chosen = np.sort(rng.choice(eligible, args.per_class, replace=False))
            rows, feats = [], []
            for group in chosen:
                cols = [*names, "seq_ix", "need_prediction", "t0", "t1"]
                if split == "valid":
                    cols += ["is_scored"]
                table = pf.read_row_group(int(group), columns=cols, use_threads=False)
                x = np.column_stack([table[c].to_numpy() for c in names]).astype(np.float32)
                y = np.column_stack([table[c].to_numpy() for c in ["t0", "t1"]])
                need = table["need_prediction"].to_numpy().astype(bool)
                scored = table["is_scored"].to_numpy().astype(bool) if split == "valid" else need
                assert len(x) == 20_000 and not need[:99].any()
                assert np.isfinite(x).all() and np.isfinite(y).all()
                row = audit_sequence(x, y, need, scored)
                row.update(row_group=int(group), seq_ix=int(table["seq_ix"][0].as_py()))
                rows.append(row)
                feats.append(x[rng.choice(np.flatnonzero(need), 128, replace=False)])
            samples[(split, cls)] = np.concatenate(feats)
            split_out["strata"][str(cls)] = {"population_sequences": len(eligible),
                "sample_sequences": len(rows), "aggregate": aggregate(rows), "sequences": rows}
            print(split, cls, "finished", len(rows), "sequences", flush=True)
        report["splits"][split] = split_out
    report["marginal_drift_by_class"] = {
        str(c): drift(samples[("train", c)], samples[("valid", c)], names) for c in range(4)
    }
    report["elapsed_seconds"] = time.time() - start
    report["maxrss_linux_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("wrote", args.output, "seconds", report["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
