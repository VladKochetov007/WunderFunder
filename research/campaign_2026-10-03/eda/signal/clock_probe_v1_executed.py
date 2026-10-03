from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import time

import numba as nb
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3] / "datasets"
CONFIGS = ("shared", "own", "row", "prefix_rate")
GROUPS = ("i0p", "i0v", "i0d", "i1p", "i1v", "i1d", "a")
LOG = logging.getLogger(__name__)


def save_json(name: str, value: dict) -> None:
    """Write probe metadata within this analyst's directory."""
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


def select_samples() -> dict:
    """Exclude recoverable predecessor subsets, without claiming independence."""
    cat = np.load(HERE / "catalogue.npz")
    excluded = {s: set() for s in ("train", "valid")}
    rng = np.random.default_rng(7)
    for split, count in (("train", 150), ("valid", 100)):
        excluded[split].update(rng.choice(len(cat[split]), count, replace=False))
    for split, count in (("train", 1500), ("valid", 600)):
        a = cat[split]
        keep = np.where(np.isin(np.round(a[:, 2], 4), [.1254, .2413, .4685, .6569]))[0]
        excluded[split].update(np.random.default_rng(14).choice(keep, count, False))
    for seed, count in ((1, 20), (4, 40), (5, 40), (6, 6), (8, 40),
                        (9, 60), (10, 60), (11, 40), (12, 60), (16, 60)):
        excluded["train"].update(np.random.default_rng(seed).choice(len(cat["train"]), count, False))
    rng = np.random.default_rng(15)
    for quantum in (.1254, .2413):
        keep = np.where(np.round(cat["train"][:, 2], 4) == quantum)[0]
        excluded["train"].update(rng.choice(keep, 40, False))
    excluded["train"].update((0, 6, 98))
    rng = np.random.default_rng(2603103)
    result = {"seed": 2603103, "exposure": "All selected IDs were summarized in prior catalogue; exclusions use predecessor defaults and do not prove lack of deeper exposure."}
    for split in ("train", "valid"):
        a = cat[split]
        eligible = [i for i in range(len(a)) if i not in excluded[split] and np.round(a[i, 2], 4) in (.1254, .2413, .4685, .6569)]
        rows = rng.choice(eligible, 64, False)
        result[split] = [{"rg": int(r), "seq_ix": int(a[r, 1]), "prior_target_spacing": float(np.round(a[r, 2], 4)), "prior_exposure": "catalogue"} for r in rows]
        result[f"{split}_excluded_count"] = len(excluded[split])
    return result


def read_inputs(pf: pq.ParquetFile, rg: int, columns: list[str]) -> np.ndarray:
    """Read only input columns from one sequence."""
    table = pf.read_row_group(rg, columns=columns, use_threads=False)
    x = np.column_stack([table[c].to_numpy() for c in columns]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite input in row group {rg}")
    return x


def changed(x: np.ndarray) -> np.ndarray:
    """Mark current-row changes; row zero initializes the state."""
    return np.r_[True, np.any(x[1:] != x[:-1], axis=1)]


@nb.njit(cache=False)
def ema_by_group(x: np.ndarray, ticks: np.ndarray, group: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    state = x[0].copy()
    for row in range(len(x)):
        for col in range(x.shape[1]):
            if ticks[row, group[col]]:
                state[col] += .25 * (x[row, col] - state[col])
            out[row, col] = state[col]
    return out


@nb.njit(cache=False)
def clock_covariates(ticks: np.ndarray) -> np.ndarray:
    out = np.empty((len(ticks), 6), np.float32)
    age = np.zeros(3, np.float32)
    rate = np.zeros(3, np.float32)
    for row in range(len(ticks)):
        for j in range(3):
            age[j] = 0 if ticks[row, j] else age[j] + 1
            rate[j] += ((1.0 if ticks[row, j] else 0.0) - rate[j]) / 64
            out[row, j] = np.log1p(age[j])
            out[row, j + 3] = rate[j]
    return out


def prepare(x: np.ndarray, columns: list[str]) -> tuple:
    """Build causal group events and shared covariates from unmodified inputs."""
    index = np.array([GROUPS.index(c[:2] + c[3] if c.startswith("i") else "a") for c in columns])
    own = np.column_stack([changed(x[:, index == j]) for j in range(len(GROUPS))])
    shared = own[:, 0]
    base = clock_covariates(np.column_stack((shared, shared | own[:, 1], own[:, 3] | own[:, 4])))
    sentinel_cols = [columns.index(c) for c in ("a5", "a6", "a7")]
    indicators = (x[:, sentinel_cols] < -5).astype(np.float32)
    clean = x.copy()
    for j in sentinel_cols:
        clean[:, j] = np.where(clean[:, j] < -5, 0, clean[:, j])
    return clean, indicators, base, own, index


def features(x: np.ndarray, columns: list[str], config: str) -> np.ndarray:
    """Generate features using only input rows at or before each prediction."""
    clean, indicators, base, own, index = prepare(x, columns)
    if config == "own":
        ticks = own
    elif config == "row":
        ticks = np.ones_like(own)
    else:
        ticks = np.repeat(own[:, :1], len(GROUPS), axis=1)
        if config == "prefix_rate" and len(x) > 99:
            rates = own[1:99].mean(0)
            steps = np.arange(1, len(x) - 98)[:, None]
            ticks[99:] = np.floor(steps * rates) > np.floor((steps - 1) * rates)
    band = clean - ema_by_group(clean, ticks, index)
    return np.column_stack((clean, band, indicators, base, np.ones(len(x)))).astype(np.float32)


def sufficient_stats(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Accumulate pooled clipped-target weighted Pearson moments in float64."""
    pred = np.clip(pred.astype(np.float64), -2, 2)
    target = np.clip(target.astype(np.float64), -2, 2)
    w = np.abs(target)
    return np.stack([w.sum(0), (w * pred).sum(0), (w * target).sum(0), (w * pred * pred).sum(0), (w * target * target).sum(0), (w * pred * target).sum(0)], axis=-1)


def scores(stats: np.ndarray) -> np.ndarray:
    """Compute separate target WP from additive sufficient statistics."""
    sw, sp, sy, spp, syy, spy = np.moveaxis(stats, -1, 0)
    cov = spy - sp * sy / sw
    return cov / np.sqrt(np.maximum((spp - sp * sp / sw) * (syy - sy * sy / sw), 1e-30))


def main() -> None:
    """Run the frozen four-clock probe on streamed sequence groups."""
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA must be unavailable")
    frozen = json.loads((HERE / "HYPOTHESES.json").read_text())
    manifest = select_samples()
    manifest["hypotheses_sha256"] = hashlib.sha256((HERE / "HYPOTHESES.json").read_bytes()).hexdigest()
    manifest["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save_json("clock_probe_manifest.json", manifest)
    state = json.loads((HERE / "STATE.json").read_text())
    state["phase"] = "running_frozen_clock_probe"
    state["inspected_sequence_ids"]["new"] = manifest
    save_json("STATE.json", state)
    pf_train = pq.ParquetFile(DATA / "train.parquet")
    columns = [c for c in pf_train.schema_arrow.names if c not in ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")]
    nf = 2 * len(columns) + 10
    gram = np.zeros((4, 2, nf, nf), np.float64)
    rhs = np.zeros((4, 2, nf), np.float64)
    checks = []
    for i, sample in enumerate(manifest["train"]):
        x = read_inputs(pf_train, sample["rg"], columns)
        y = np.column_stack([pf_train.read_row_group(sample["rg"], columns=[t], use_threads=False)[t].to_numpy() for t in ("t0", "t1")]).astype(np.float64)
        weights = np.abs(np.clip(y[99::4], -2, 2))
        for k, config in enumerate(CONFIGS):
            full = features(x, columns, config)
            if i < 2:
                for end in (98, 99, 255, 1023):
                    error = float(np.max(np.abs(full[:end + 1] - features(x[:end + 1], columns, config))))
                    checks.append({"rg": sample["rg"], "config": config, "end_row": end, "max_abs_error": error})
                    if error != 0:
                        raise AssertionError(f"Prefix causality failed: {checks[-1]}")
            a = full[99::4].astype(np.float64)
            for j in range(2):
                aw = a * weights[:, j, None]
                gram[k, j] += a.T @ aw
                rhs[k, j] += aw.T @ y[99::4, j]
        if i % 16 == 0:
            LOG.info("train %s/64 elapsed %.1fs", i, time.monotonic() - started)
    coefficients = np.empty((4, nf, 2), np.float64)
    for k in range(4):
        for j in range(2):
            diagonal = np.diag(gram[k, j]).copy()
            diagonal[-1] = 0
            coefficients[k, :, j] = np.linalg.solve(gram[k, j] + 1e-3 * np.diag(diagonal + 1e-9) + 1e-6 * np.eye(nf), rhs[k, j])
    np.savez(HERE / "clock_probe_weights.npz", coefficients=coefficients, columns=columns, configs=CONFIGS)
    pf_valid = pq.ParquetFile(DATA / "valid.parquet")
    stats, activity = [], []
    for i, sample in enumerate(manifest["valid"]):
        x = read_inputs(pf_valid, sample["rg"], columns)
        table = pf_valid.read_row_group(sample["rg"], columns=["t0", "t1", "is_scored"], use_threads=False)
        y = np.column_stack([table[t].to_numpy() for t in ("t0", "t1")])
        mask = table["is_scored"].to_numpy().astype(bool) & (np.arange(len(x)) >= 99)
        own = prepare(x, columns)[3]
        activity.append(np.stack([own[99:].mean(0), (own[99:] & ~own[99:, :1]).mean(0), own[1:99].mean(0)]))
        row_stats = []
        for k, config in enumerate(CONFIGS):
            a = features(x, columns, config)[mask].astype(np.float64)
            row_stats.append(sufficient_stats(a @ coefficients[k], y[mask]))
        stats.append(row_stats)
        if i % 16 == 0:
            LOG.info("valid %s/64 elapsed %.1fs", i, time.monotonic() - started)
    stats = np.array(stats)
    pooled = scores(stats.sum(0))
    halves = [scores(part.sum(0)) for part in np.split(stats, 2)]
    rng = np.random.default_rng(3103)
    boot = np.array([scores(stats[rng.integers(0, 64, 64)].sum(0)).mean(-1) for _ in range(250)])
    delta = float(pooled[1].mean() - pooled[0].mean())
    half_deltas = [float(h[1].mean() - h[0].mean()) for h in halves]
    timing_delta = float(pooled[1].mean() - pooled[3].mean())
    result = {"test_id": frozen["test_id"], "configs": {c: {"t0": float(pooled[k, 0]), "t1": float(pooled[k, 1]), "mean": float(pooled[k].mean()), "halves": [h[k].tolist() for h in halves]} for k, c in enumerate(CONFIGS)}, "own_minus_shared": delta, "own_minus_shared_halves": half_deltas, "own_minus_prefix_rate": timing_delta, "own_minus_shared_bootstrap_95_descriptive": np.quantile(boot[:, 1] - boot[:, 0], [.025, .975]).tolist(), "supports_own_clock": bool(delta > .002 and min(half_deltas) > 0), "supports_timing_specific": bool(delta > .002 and min(half_deltas) > 0 and timing_delta > .002), "causality_checks": checks, "target_isolation": "features accepts inputs only; neither scoring masks nor targets are passed to features", "uncertainty_caveat": "Sequence bootstrap is descriptive; connected-component dependencies and previous EDA exposure are unresolved.", "n_features_including_intercept": nf, "elapsed_seconds": time.monotonic() - started}
    np.savez(HERE / "clock_probe_stats.npz", stats=stats, activity=activity, configs=CONFIGS, groups=GROUPS)
    save_json("clock_probe_results.json", result)
    state["phase"] = "clock_probe_complete"
    state["pending_work"] = ["inspect figures", "update notebook", "freeze report"]
    state["artifacts"] += ["clock_probe.py", "clock_probe_manifest.json", "clock_probe_results.json", "clock_probe_weights.npz", "clock_probe_stats.npz"]
    save_json("STATE.json", state)
    LOG.info("results %s", json.dumps(result["configs"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
