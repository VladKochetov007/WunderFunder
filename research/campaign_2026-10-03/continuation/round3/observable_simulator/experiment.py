from __future__ import annotations

import hashlib
import json
import logging
import os
import resource
import signal
import time
from pathlib import Path

import numpy as np
from representation import Ridge, features
from simulator import ORIGINS, scalar_label, simulate, stopping_labels

OWN = Path(__file__).resolve().parent
ARMS = ("row", "reset", "history")
LOGGER = logging.getLogger(__name__)


def save(name: str, value: object) -> None:
    """Write strict own-branch artifacts."""
    (OWN / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fit_timeout(signum: int, frame: object) -> None:
    """Terminate a slow fit without retrying it."""
    raise TimeoutError("Hard 120-second fit limit")


def checks(sample: dict[str, np.ndarray]) -> dict[str, int]:
    """Prefix causality and independent target specification checks."""
    checked = 0
    for arm in ARMS:
        full = features(sample, arm)
        for cut in (64, 129, 241):
            prefix = {k: v[:, :cut] for k, v in sample.items()}
            assert np.array_equal(full[:, :cut], features(prefix, arm))
            surgery = {k: v.copy() for k, v in sample.items()}
            surgery["values"][:, cut:] += 999
            assert np.array_equal(full[:, :cut], features(surgery, arm)[:, :cut])
            checked += 2
        hidden = {k: v.copy() for k, v in sample.items()}
        hidden["mid"] += 37
        hidden["factor"] *= -53
        assert np.array_equal(full, features(hidden, arm))
        checked += 1
    y, _, _ = stopping_labels(sample["mid"])
    for p in range(len(sample["mid"])):
        for k, origin in enumerate(ORIGINS):
            assert np.array_equal(y[p, k], scalar_label(sample["mid"][p], int(origin)))
    # A direct hand-constructed first-hit and a no-hit forced-exit positive.
    fixture = np.tile(np.linspace(0, 1, 320), (2, 1))
    fixture[0, 49:] += 3
    fixture_y, fixture_hit, fixture_lag = stopping_labels(fixture)
    assert fixture_hit[0, 0] and fixture_lag[0, 0] == 1
    assert not fixture_hit[1, 0] and fixture_lag[1, 0] == 32
    assert np.allclose(fixture_y.sum(axis=2), -0.4)
    return {
        "prefix_and_latent_checks": checked,
        "scalar_target_checks": len(sample["mid"]) * 30,
        "hand_target_positives": 2,
    }


def metric(model: Ridge, x: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    prediction = model.predict(x.reshape(-1, x.shape[-1])).reshape(y.shape)
    errors = ((prediction - y) ** 2).mean(axis=2)
    mse = errors.mean()
    constant = ((y - model.intercept) ** 2).mean()
    return {"pair_mse": float(mse), "R2_vs_fit_mean": float(1 - mse / constant)}, errors.mean(
        axis=1
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    start, cpu = time.monotonic(), time.process_time()
    prereg = (OWN / "PREREG.json").read_bytes()
    signal.signal(signal.SIGALRM, fit_timeout)
    records, hashes, fits = [], {}, []
    path_losses, coefficients = {}, {}
    invariants = checks(simulate(990001, 8, "nominal"))
    worlds = {"nominal": 0, "absent": 100000, "unidentifiable": 200000}
    for world, offset in worlds.items():
        for rep, base in enumerate((11001, 22001, 33001, 44001)):
            seed = base + offset
            train = simulate(seed, 96, world)
            y_train, hits_train, _ = stopping_labels(train["mid"])
            models = {}
            for arm in ARMS:
                x = features(train, arm)[:, ORIGINS]
                model = Ridge()
                signal.alarm(120)
                try:
                    model.fit(x.reshape(-1, x.shape[-1]), y_train.reshape(-1, 2))
                finally:
                    signal.alarm(0)
                models[arm] = model
                fits.append(
                    {
                        "world": world,
                        "replication": rep,
                        "arm": arm,
                        "seconds": model.fit_seconds,
                        "features": x.shape[-1],
                    }
                )
                key = f"{world}_{rep}_{arm}"
                for name in ("mean", "scale", "beta", "intercept"):
                    coefficients[f"{key}_{name}"] = getattr(model, name)
            hashes[f"{world}_{rep}_fit"] = {k: digest_array(v) for k, v in train.items()}
            domains = ("nominal", "sign_flip", "stale") if world == "nominal" else ("nominal",)
            paired_mid = None
            for domain in domains:
                test = simulate(seed + 1000, 48, world, domain)
                y_test, hits, lags = stopping_labels(test["mid"])
                if paired_mid is None:
                    paired_mid = test["mid"].copy()
                else:
                    assert np.array_equal(paired_mid, test["mid"])
                sample_hash = {k: digest_array(v) for k, v in test.items()}
                hashes[f"{world}_{rep}_{domain}_test"] = sample_hash
                record = {
                    "world": world,
                    "replication": rep,
                    "domain": domain,
                    "fit_seed": seed,
                    "test_seed": seed + 1000,
                    "fit_origins": 2880,
                    "test_origins": 1440,
                    "fit_barrier_hits": int(hits_train.sum()),
                    "test_barrier_hits": int(hits.sum()),
                    "test_forced_cap_exits": int((~hits).sum()),
                    "mean_exit_lag": float(lags.mean()),
                    "arms": {},
                }
                for arm in ARMS:
                    x = features(test, arm)[:, ORIGINS]
                    summary, loss = metric(models[arm], x, y_test)
                    record["arms"][arm] = summary
                    path_losses[f"{world}_{rep}_{domain}_{arm}"] = loss
                hist = features(test, "history")[:, ORIGINS, 25]
                reset = features(test, "reset")[:, ORIGINS, 25]
                latent = test["factor"][:, ORIGINS]
                record["factor_probe_correlation"] = {
                    name: float(np.corrcoef(value.ravel(), latent.ravel())[0, 1])
                    if latent.std()
                    else None
                    for name, value in (("history_pool", hist), ("row_pool", reset))
                }
                records.append(record)
                LOGGER.info(
                    "%s rep%d %s mse=%s",
                    world,
                    rep,
                    domain,
                    {a: round(record["arms"][a]["pair_mse"], 4) for a in ARMS},
                )
                if world == "nominal" and rep == 0 and domain == "nominal":
                    np.savez_compressed(
                        OWN / "witness.npz",
                        **{k: v[:2] for k, v in test.items()},
                        targets=y_test[:2],
                        lags=lags[:2],
                    )
    rng = np.random.default_rng(601003)
    aggregates = {}
    for world, domain in (
        ("nominal", "nominal"),
        ("nominal", "sign_flip"),
        ("nominal", "stale"),
        ("absent", "nominal"),
        ("unidentifiable", "nominal"),
    ):
        subset = [r for r in records if r["world"] == world and r["domain"] == domain]
        aggregates[f"{world}/{domain}"] = {
            "arm_mean_mse": {
                a: float(np.mean([r["arms"][a]["pair_mse"] for r in subset])) for a in ARMS
            },
            "arm_mean_R2": {
                a: float(np.mean([r["arms"][a]["R2_vs_fit_mean"] for r in subset])) for a in ARMS
            },
            "comparisons": {},
        }
        for baseline in ("row", "reset"):
            relative = [
                1 - r["arms"]["history"]["pair_mse"] / r["arms"][baseline]["pair_mse"]
                for r in subset
            ]
            diffs = np.stack(
                [
                    path_losses[f"{world}_{rep}_{domain}_{baseline}"]
                    - path_losses[f"{world}_{rep}_{domain}_history"]
                    for rep in range(4)
                ]
            )
            boot = []
            for _ in range(1000):
                seeds = rng.integers(0, 4, 4)
                boot.append(np.mean([diffs[s, rng.integers(0, 48, 48)].mean() for s in seeds]))
            aggregates[f"{world}/{domain}"]["comparisons"][baseline] = {
                "relative_MSE_improvements": relative,
                "mean_relative_improvement": float(np.mean(relative)),
                "positive_replications": sum(v > 0 for v in relative),
                "two_percent_replications": sum(v >= 0.02 for v in relative),
                "paired_mean_MSE_reduction": float(diffs.mean()),
                "hierarchical_seed_path_bootstrap_95pct": np.quantile(
                    boot, [0.025, 0.975]
                ).tolist(),
            }
    nominal_pass = all(
        c["mean_relative_improvement"] >= 0.02 and c["positive_replications"] >= 3
        for c in aggregates["nominal/nominal"]["comparisons"].values()
    )
    negative_pass = all(
        c["two_percent_replications"] < 3
        for w in ("absent", "unidentifiable")
        for c in aggregates[f"{w}/nominal"]["comparisons"].values()
    )
    result = {
        "nominal_efficacy_gate": nominal_pass,
        "negative_controls_gate": negative_pass,
        "aggregates": aggregates,
        "records": records,
        "invariants": invariants,
        "fit_costs": fits,
        "data_hashes": hashes,
        "prereg_sha256": hashlib.sha256(prereg).hexdigest(),
        "IDs": {
            "fit": "world/replication/fit-seed/path0..95/origin48..280step8",
            "test": "world/replication/test-seed/path0..47/origin48..280step8",
            "domain_tests_are_paired_same_test_paths": True,
            "raw_train_origins": 0,
        },
        "cost": {
            "wall_seconds": time.monotonic() - start,
            "cpu_seconds": time.process_time() - cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "sum_fit_seconds": sum(f["seconds"] for f in fits),
            "max_fit_seconds": max(f["seconds"] for f in fits),
            "fits": len(fits),
            "GPU": False,
        },
    }
    np.savez_compressed(OWN / "models.npz", **coefficients)
    np.savez_compressed(OWN / "path_losses.npz", **path_losses)
    save("results.json", result)
    save(
        "STATE.json",
        {
            "phase": "scored",
            "raw_data_reads": 0,
            "representation_count": 1,
            "nominal_gate": nominal_pass,
            "negative_gate": negative_pass,
            "cost": result["cost"],
            "independent_review": "pending",
        },
    )
    LOGGER.info(
        "GATES nominal=%s negatives=%s COST=%s", nominal_pass, negative_pass, result["cost"]
    )


if __name__ == "__main__":
    main()
