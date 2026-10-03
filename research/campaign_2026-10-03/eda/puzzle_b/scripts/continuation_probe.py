from __future__ import annotations

import hashlib
import json
import logging
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import lfilter

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[3]
RAW = [f"i0_p{k}" for k in range(22)] + [f"i0_dp{k}" for k in range(4)]
ARMS = ("raw", "age", "period4", "period6", "sham6")
SEED = 20261003


def ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    """Compute a causal, zero-initialized exponential average."""
    return lfilter([alpha], [1, -(1 - alpha)], values)


def recurrence(events: np.ndarray, lag: int) -> np.ndarray:
    """Return prefix-only recurrence contrast at a fixed lag."""
    products = []
    for shift in (lag - 1, lag, lag + 1):
        product = np.zeros(len(events))
        product[shift:] = events[shift:] * events[:-shift]
        products.append(product)
    return ewma(products[1] - 0.5 * (products[0] + products[2]), 0.002)


def features(raw: np.ndarray, sham_offset: float) -> dict[str, np.ndarray]:
    """Build feature arms exclusively from current and earlier input rows."""
    index = np.arange(len(raw))
    event = np.r_[True, np.any(raw[1:, :22] != raw[:-1, :22], axis=1)]
    age = index - np.maximum.accumulate(np.where(event, index, 0))
    event = event.astype(float)
    base = np.column_stack((raw, np.log1p(age), ewma(event, 0.01)))
    result = {"raw": raw, "age": base}
    for arm, period in (("period4", 4), ("period6", 6), ("sham6", 6)):
        clock = index + sham_offset if arm == "sham6" else age
        angle = 2 * np.pi * clock / period
        contrast = recurrence(event, 7 if arm == "sham6" else period)
        sine, cosine = np.sin(angle), np.cos(angle)
        extra = np.column_stack(
            (
                sine,
                cosine,
                contrast,
                contrast * sine,
                contrast * cosine,
                np.minimum(age / period, 2),
            )
        )
        result[arm] = np.column_stack((base, extra))
    return result


def moments(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Accumulate exact pooled clipped weighted-correlation statistics."""
    pred, target = np.clip(pred, -2, 2), np.clip(target, -2, 2)
    weight = np.abs(target)
    return np.stack(
        (
            weight.sum(0),
            (weight * pred).sum(0),
            (weight * target).sum(0),
            (weight * pred**2).sum(0),
            (weight * target**2).sum(0),
            (weight * pred * target).sum(0),
        ),
        axis=0,
    )


def correlation(stats: np.ndarray) -> np.ndarray:
    """Evaluate correlation after pooling sufficient statistics."""
    weight, sp, sy, spp, syy, spy = stats
    return (spy - sp * sy / weight) / np.sqrt((spp - sp**2 / weight) * (syy - sy**2 / weight))


def source_manifest(path: Path, parquet: pq.ParquetFile) -> dict:
    """Identify large parquet sources without loading or hashing whole files."""
    stat = path.stat()
    with path.open("rb") as handle:
        handle.seek(-8, 2)
        tail = handle.read(8)
        size = int.from_bytes(tail[:4], "little")
        handle.seek(-(size + 8), 2)
        footer = handle.read(size + 8)
    return {
        "path": str(path.relative_to(ROOT)),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "row_groups": parquet.num_row_groups,
        "footer_sha256": hashlib.sha256(footer).hexdigest(),
    }


def main() -> None:
    """Run the frozen matched probe, preserving outputs and provenance."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.monotonic()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    protocol = OUT / "HYPOTHESES.json"
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    state = json.loads((OUT / "STATE.json").read_text())
    state.update(phase="running_frozen_probe", last_failure=None)
    (OUT / "STATE.json").write_text(json.dumps(state, indent=2))
    paths = {name: ROOT / "datasets" / f"{name}.parquet" for name in ("train", "valid")}
    files = {name: pq.ParquetFile(path) for name, path in paths.items()}
    prior = {
        name: set(pd.read_csv(OUT / f"seqscan_{name}.csv", usecols=["rg"]).rg)
        for name in ("train", "valid")
    }
    prior["train"].update([0, 1, 2, 3, 500, 5000, 10000])
    rng = np.random.default_rng(SEED)
    train_start = 2100 if files["train"].num_row_groups >= 2132 else 0
    selected = {
        "train": sorted(
            rng.choice(
                np.arange(train_start, files["train"].num_row_groups), 32, replace=False
            ).tolist()
        ),
        "valid": sorted(rng.choice(files["valid"].num_row_groups, 24, replace=False).tolist()),
    }
    manifest = {
        "seed": SEED,
        "protocol_sha256": protocol_hash,
        "sources": {k: source_manifest(paths[k], v) for k, v in files.items()},
        "sample": selected,
        "sequences": [],
        "exposure": "Own prior scans: 2000 random train row groups and all1873 valid; explicit train notebook rows also exposed; actual per-sequence flags use recorded CSV rg sets. Campaign-wide novelty unknown. Frozen protocol initial range assumption was erroneous, corrected after audit.",
        "sham": "rng(seed+rg) fixed phase offset for control only; seq_ix never used by features",
    }
    (OUT / "continuation_manifest.json").write_text(json.dumps(manifest, indent=2))
    state["inspected_sequence_ids"]["continuation_row_groups"] = selected
    (OUT / "STATE.json").write_text(json.dumps(state, indent=2))
    accumulators = {}
    for stride in (7, 6):
        for arm in ARMS:
            dimension = 26 if arm == "raw" else 28 if arm == "age" else 34
            accumulators[stride, arm] = {
                "n": 0,
                "sum": np.zeros(dimension),
                "xx": np.zeros((dimension, dimension)),
                "xy": np.zeros((dimension, 2)),
                "y": np.zeros(2),
            }
    checks = {}
    for split in ("train",):
        for rg in selected[split]:
            columns = RAW + ["t0", "t1", "seq_ix", "step_in_seq"]
            table = files[split].read_row_group(rg, columns=columns).to_pandas()
            raw = table[RAW].to_numpy(dtype=np.float64)
            assert np.isfinite(raw).all()
            assert np.array_equal(table.step_in_seq.to_numpy(), np.arange(len(raw)))
            offset = float(np.random.default_rng(SEED + rg).uniform(0, 6))
            arms = features(raw, offset)
            if rg == selected[split][0]:
                cut = 431
                truncated = features(raw[:cut], offset)
                mutated = raw.copy()
                mutated[cut:] = 123.45
                altered = features(mutated, offset)
                checks["prefix_exact"] = all(
                    np.array_equal(arms[a][:cut], truncated[a]) for a in ARMS
                )
                checks["future_suffix_invariant"] = all(
                    np.array_equal(arms[a][:cut], altered[a][:cut]) for a in ARMS
                )
                changed_targets = table.copy()
                changed_targets[["t0", "t1"]] = -999
                checks["target_invariant"] = all(
                    np.array_equal(
                        arms[a], features(changed_targets[RAW].to_numpy(dtype=float), offset)[a]
                    )
                    for a in ARMS
                )
                checks["metric_reference"] = bool(
                    np.allclose(
                        correlation(
                            moments(
                                np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]),
                                np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]),
                            )
                        ),
                        1,
                    )
                )
                assert all(checks.values()), checks
            target = np.clip(table[["t0", "t1"]].to_numpy(dtype=float), -2, 2)
            for stride in (7, 6):
                rows = np.arange(99, len(raw), stride)
                mask = np.isfinite(target[rows]).all(1)
                rows = rows[mask]
                y = target[rows]
                for arm, values in arms.items():
                    x = values[rows]
                    acc = accumulators[stride, arm]
                    acc["n"] += len(x)
                    acc["sum"] += x.sum(0)
                    acc["xx"] += x.T @ x
                    acc["xy"] += x.T @ y
                    acc["y"] += y.sum(0)
            manifest["sequences"].append(
                {
                    "split": split,
                    "rg": rg,
                    "seq_ix": int(table.seq_ix.iloc[0]),
                    "rows": len(raw),
                    "input_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                    "prior_own_exposure": rg in prior[split],
                }
            )
            logging.info("fit rg=%s", rg)
    models = {}
    coefficients = {}
    for (stride, arm), acc in accumulators.items():
        n = acc["n"]
        mean, ym = acc["sum"] / n, acc["y"] / n
        centered = acc["xx"] - n * np.outer(mean, mean)
        sd = np.sqrt(np.maximum(np.diag(centered) / n, 0)) + 1e-6
        xx = centered / np.outer(sd, sd)
        xy = (acc["xy"] - n * mean[:, None] * ym) / sd[:, None]
        beta = np.linalg.solve(xx + 0.01 * n * np.eye(len(mean)), xy)
        models[stride, arm] = mean, sd, beta, ym
        key = f"s{stride}_{arm}"
        coefficients.update(
            {f"{key}_mean": mean, f"{key}_sd": sd, f"{key}_beta": beta, f"{key}_ymean": ym}
        )
    np.savez(OUT / "continuation_models.npz", **coefficients)
    pooled = {key: np.zeros((6, 2)) for key in models}
    sequence_rows, recurrence_rows = [], []
    for rg in selected["valid"]:
        table = (
            files["valid"]
            .read_row_group(rg, columns=RAW + ["t0", "t1", "seq_ix", "step_in_seq", "is_scored"])
            .to_pandas()
        )
        raw = table[RAW].to_numpy(dtype=float)
        assert np.isfinite(raw).all()
        assert np.array_equal(table.step_in_seq.to_numpy(), np.arange(len(raw)))
        offset = float(np.random.default_rng(SEED + rg).uniform(0, 6))
        arms = features(raw, offset)
        target = table[["t0", "t1"]].to_numpy(dtype=float)
        mask = (
            table.is_scored.to_numpy(dtype=bool)
            & (table.step_in_seq.to_numpy() >= 99)
            & np.isfinite(target).all(1)
        )
        for (stride, arm), (mean, sd, beta, ym) in models.items():
            pred = ((arms[arm][mask] - mean) / sd) @ beta + ym
            stats = moments(pred, target[mask])
            pooled[stride, arm] += stats
            score = correlation(stats)
            sequence_rows.append(
                {
                    "rg": rg,
                    "seq_ix": int(table.seq_ix.iloc[0]),
                    "stride": stride,
                    "arm": arm,
                    "n": int(mask.sum()),
                    "t0": score[0],
                    "t1": score[1],
                    "mean": score.mean(),
                }
            )
        event = np.r_[False, np.any(raw[1:, :22] != raw[:-1, :22], axis=1)].astype(float)
        for lag in (3, 4, 5, 6, 7):
            recurrence_rows.append(
                {
                    "rg": rg,
                    "lag": lag,
                    "event_rate": event[99:].mean(),
                    "repeat_rate": (event[lag:] * event[:-lag]).mean(),
                }
            )
        manifest["sequences"].append(
            {
                "split": "valid",
                "rg": rg,
                "seq_ix": int(table.seq_ix.iloc[0]),
                "rows": len(raw),
                "scored_rows": int(mask.sum()),
                "scored_steps_sha256": hashlib.sha256(
                    table.step_in_seq.to_numpy()[mask].tobytes()
                ).hexdigest(),
                "input_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                "prior_own_exposure": rg in prior["valid"],
            }
        )
        logging.info("score rg=%s n=%s", rg, mask.sum())
    scores = []
    per_sequence = pd.DataFrame(sequence_rows)
    for (stride, arm), stats in pooled.items():
        score = correlation(stats)
        age = correlation(pooled[stride, "age"])
        frame = per_sequence[per_sequence.stride == stride].pivot(
            index="rg", columns="arm", values="mean"
        )
        scores.append(
            {
                "stride": stride,
                "arm": arm,
                "t0": score[0],
                "t1": score[1],
                "mean": score.mean(),
                "delta_age": (score - age).mean(),
                "paired_sequence_wins_vs_age": int((frame[arm] > frame.age).sum()),
                "train_rows": accumulators[stride, arm]["n"],
            }
        )
    pd.DataFrame(scores).to_csv(OUT / "continuation_scores.csv", index=False)
    per_sequence.to_csv(OUT / "continuation_sequence_scores.csv", index=False)
    pd.DataFrame(recurrence_rows).to_csv(OUT / "continuation_recurrence.csv", index=False)
    (OUT / "continuation_checks.json").write_text(json.dumps(checks, indent=2))
    train_ids = {s["seq_ix"] for s in manifest["sequences"] if s["split"] == "train"}
    valid_ids = {s["seq_ix"] for s in manifest["sequences"] if s["split"] == "valid"}
    manifest.update(
        wall_seconds=time.monotonic() - started,
        maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        direct_seq_ix_overlap=sorted(train_ids & valid_ids),
        component_independence="unknown; seq_ix nonoverlap is insufficient",
        versions={"numpy": np.__version__, "pandas": pd.__version__},
    )
    (OUT / "continuation_manifest.json").write_text(json.dumps(manifest, indent=2))
    state.update(
        phase="probe_complete",
        artifacts=state["artifacts"]
        + [
            "continuation_scores.csv",
            "continuation_manifest.json",
            "continuation_checks.json",
            "continuation_models.npz",
        ],
        pending_work=["headless notebook", "inspect figures", "report"],
    )
    (OUT / "STATE.json").write_text(json.dumps(state, indent=2))
    logging.info(
        "complete seconds=%.2f maxrss_kib=%s", manifest["wall_seconds"], manifest["maxrss_kib"]
    )


if __name__ == "__main__":
    main()
