from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
AUTHOR = ROOT.parents[1] / "round3/observable_simulator"
sys.path.insert(0, str(AUTHOR))
from representation import features
from simulator import ORIGINS, simulate, stopping_labels


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ahash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    return ((x - model["mean"]) / model["scale"]) @ model["beta"] + model["intercept"]


def oracle(mid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.empty((len(mid), 30, 2))
    lags = np.empty((len(mid), 30), dtype=int)
    hits = np.zeros((len(mid), 30), dtype=bool)
    for p, path in enumerate(mid):
        for k, t in enumerate(ORIGINS):
            exitlag = 32
            for lag in range(1, 33):
                if abs(path[t + lag] - path[t]) >= 2:
                    exitlag = lag
                    hits[p, k] = True
                    break
            d = path[t + exitlag] - path[t]
            labels[p, k] = [d - 0.2, -d - 0.2]
            lags[p, k] = exitlag
    return labels, hits, lags


def main() -> None:
    started = time.monotonic()
    declared = json.loads((AUTHOR / "results.json").read_text())
    manifest = json.loads((AUTHOR / "PROVENANCE.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        assert sha(AUTHOR / name) == expected, name
    models = np.load(AUTHOR / "models.npz")
    savedloss = np.load(AUTHOR / "path_losses.npz")
    worlds = {"nominal": 0, "absent": 100000, "unidentifiable": 200000}
    prng_states = set()
    fingerprints = {"fit": set(), "test": set()}
    ids = {"fit": [], "test": []}
    records = []
    maxloss = maxmoment = maxnormal = 0.0
    ranks = {}
    maxoracle = 0
    canary_checks = 0
    future_label_changes = 0
    cap_counts = {}
    scales = {}
    for world, offset in worlds.items():
        for rep, base in enumerate([11001, 22001, 33001, 44001]):
            seed = base + offset
            for part, count, s in [("fit", 96, seed), ("test", 48, seed + 1000)]:
                state = json.dumps(np.random.default_rng(s).bit_generator.state, sort_keys=True)
                assert state not in prng_states
                prng_states.add(state)
                data = simulate(s, count, world)
                for name, a in data.items():
                    assert (
                        ahash(a)
                        == declared["data_hashes"][
                            f"{world}_{rep}_" + ("fit" if part == "fit" else "nominal_test")
                        ][name]
                    )
                for p in range(count):
                    fingerprint = ahash(
                        np.concatenate([data["dt"][p], data["mid"][p], data["values"][p].ravel()])
                    )
                    assert fingerprint not in fingerprints[part]
                    fingerprints[part].add(fingerprint)
                    ids[part].append([world, rep, s, p, fingerprint])
                if part == "fit":
                    fit = data
                else:
                    nominal = data
            fy, _, _ = oracle(fit["mid"])
            for arm in ["row", "reset", "history"]:
                key = f"{world}_{rep}_{arm}"
                model = {
                    name: models[f"{key}_{name}"] for name in ["mean", "scale", "beta", "intercept"]
                }
                x = features(fit, arm)[:, ORIGINS].reshape(2880, -1)
                independent_mean = x.mean(0)
                independent_scale = np.maximum(x.std(0), 1e-6)
                independent_intercept = fy.reshape(-1, 2).mean(0)
                error = max(
                    np.abs(independent_mean - model["mean"]).max(),
                    np.abs(independent_scale - model["scale"]).max(),
                    np.abs(independent_intercept - model["intercept"]).max(),
                )
                maxmoment = max(maxmoment, error)
                assert error <= 1e-12
                scaled = (x - model["mean"]) / model["scale"]
                lhs = (scaled.T @ scaled + 10 * np.eye(x.shape[1])) @ model["beta"]
                rhs = scaled.T @ (fy.reshape(-1, 2) - model["intercept"])
                residual = np.linalg.norm(lhs - rhs) / max(1, np.linalg.norm(rhs))
                maxnormal = max(maxnormal, float(residual))
                assert residual <= 1e-10
                if world == "nominal" and rep == 0:
                    ranks[arm] = dict(
                        columns=x.shape[1], centered_rank=int(np.linalg.matrix_rank(scaled))
                    )
                scales[key] = dict(
                    mean_sha256=ahash(model["mean"]), scale_sha256=ahash(model["scale"])
                )
            for domain in ["nominal", "sign_flip", "stale"] if world == "nominal" else ["nominal"]:
                test = nominal if domain == "nominal" else simulate(seed + 1000, 48, world, domain)
                for name, a in test.items():
                    assert ahash(a) == declared["data_hashes"][f"{world}_{rep}_{domain}_test"][name]
                assert np.array_equal(test["mid"], nominal["mid"])
                assert np.array_equal(test["factor"], nominal["factor"])
                y, hits, lags = oracle(test["mid"])
                their_y, their_hits, their_lags = stopping_labels(test["mid"])
                assert (
                    np.array_equal(y, their_y)
                    and np.array_equal(hits, their_hits)
                    and np.array_equal(lags, their_lags)
                )
                maxoracle += y.shape[0] * y.shape[1]
                cap_counts[f"{world}_{rep}_{domain}"] = dict(
                    hits=int(hits.sum()), forced=int((~hits).sum()), maxlag=int(lags.max())
                )
                rec = dict(world=world, rep=rep, domain=domain, arms={})
                for arm in ["row", "reset", "history"]:
                    model = {
                        name: models[f"{world}_{rep}_{arm}_{name}"]
                        for name in ["mean", "scale", "beta", "intercept"]
                    }
                    x = features(test, arm)[:, ORIGINS]
                    prediction = predict(model, x)
                    pathloss = ((prediction - y) ** 2).mean(axis=(1, 2))
                    error = float(
                        np.abs(pathloss - savedloss[f"{world}_{rep}_{domain}_{arm}"]).max()
                    )
                    maxloss = max(maxloss, error)
                    assert error <= 1e-10
                    rec["arms"][arm] = dict(
                        pair_mse=float(pathloss.mean()),
                        R2_vs_fit_mean=float(
                            1 - pathloss.mean() / np.mean((y - model["intercept"]) ** 2)
                        ),
                    )
                records.append(rec)
            if rep == 0:
                sample = {k: v[:2].copy() for k, v in nominal.items()}
                for arm in ["row", "reset", "history"]:
                    full = features(sample, arm)
                    for cut in [49, 129, 281]:
                        truncated = {k: v[:, :cut].copy() for k, v in sample.items()}
                        assert np.array_equal(full[:, :cut], features(truncated, arm))
                        canary_checks += 1
                        surgery = {k: v.copy() for k, v in sample.items()}
                        surgery["values"][:, cut:] += 555
                        surgery["ages"][:, cut:] += 777
                        surgery["updates"][:, cut:] = ~surgery["updates"][:, cut:]
                        surgery["dt"][:, cut:] *= 31
                        assert np.array_equal(full[:, :cut], features(surgery, arm)[:, :cut])
                        canary_checks += 1
                    surgery = {k: v.copy() for k, v in sample.items()}
                    surgery["mid"][:, 49:] += 100
                    surgery["factor"][:, 49:] *= -99
                    assert np.array_equal(full, features(surgery, arm))
                    canary_checks += 1
                    a, _, _ = oracle(sample["mid"])
                    b, _, _ = oracle(surgery["mid"])
                    future_label_changes += int(np.any(a != b, axis=-1).sum())
    assert fingerprints["fit"].isdisjoint(fingerprints["test"])
    # Boundary semantics independent of generated path statistics.
    fixture = np.zeros((3, 320))
    fixture[0, 80:] = 2.0
    fixture[1, 49] = 3.0
    fixture[2, 81:] = 10.0
    y, h, l = oracle(fixture)
    assert h[0, 0] and l[0, 0] == 32 and y[0, 0, 0] == 1.8
    assert h[1, 0] and l[1, 0] == 1 and y[1, 0, 0] == 2.8
    assert not h[2, 0] and l[2, 0] == 32 and y[2, 0, 0] == -0.2
    rng = np.random.default_rng(601003)
    boots = {}
    maxboot = 0
    for world, domain in [
        ("nominal", "nominal"),
        ("nominal", "sign_flip"),
        ("nominal", "stale"),
        ("absent", "nominal"),
        ("unidentifiable", "nominal"),
    ]:
        for baseline in ["row", "reset"]:
            diffs = np.stack(
                [
                    savedloss[f"{world}_{rep}_{domain}_{baseline}"]
                    - savedloss[f"{world}_{rep}_{domain}_history"]
                    for rep in range(4)
                ]
            )
            values = []
            for _ in range(1000):
                seeds = rng.integers(0, 4, 4)
                values.append(np.mean([diffs[s, rng.integers(0, 48, 48)].mean() for s in seeds]))
            interval = np.quantile(values, [0.025, 0.975])
            expected = declared["aggregates"][f"{world}/{domain}"]["comparisons"][baseline][
                "hierarchical_seed_path_bootstrap_95pct"
            ]
            discrepancy = float(np.abs(interval - expected).max())
            maxboot = max(maxboot, discrepancy)
            assert discrepancy <= 1e-12
            boots[f"{world}/{domain}:{baseline}"] = dict(
                mean_mse_reduction=float(diffs.mean()), interval=interval.tolist()
            )
    noisevar = 2.8**2
    output = dict(
        passed=True,
        fits_executed=0,
        author_artifacts_preserved=len(manifest["files_sha256"]),
        generator_prng_states_unique=len(prng_states),
        fit_paths=len(fingerprints["fit"]),
        test_paths=len(fingerprints["test"]),
        canonical_fit_test_path_overlap=0,
        seed_path_ids=ids,
        perpath_mse_max_abs=maxloss,
        fit_moments_max_abs=maxmoment,
        ridge_equation_relative_residual=maxnormal,
        scalar_oracle_origins=maxoracle,
        oracle_boundary_fixtures_pass=True,
        prefix_future_latent_checks=canary_checks,
        changed_target_origins_under_future_latent_surgery=future_label_changes,
        capacity_ranks=ranks,
        reset_current_value_duplicate_columns=6,
        regularization_note="Duplicate standardized reset columns split coefficients, changing effective ridge penalty; equal28columns is width matching, not rank/functional-capacity matching.",
        negative_control_semantics="absent removes ALL drift; unidentifiable removes factor sensor loading AND observed I0 drift. By latent F and price-noise sign symmetry, every observable-conditional mean stopping displacement is zero; volatility/clock information need not be absent.",
        ownstream_only_analytic_counterexample=dict(
            no_common_factor=True,
            current_posterior_variance=noisevar / (noisevar + 1),
            two_observation_posterior_variance=noisevar / (noisevar + 2),
            improvement=noisevar / (noisevar + 1) - noisevar / (noisevar + 2),
            scope="constant Gaussian ownstream latent signal plus independent noise; no fit; shows history denoising does not establish crossinstrument necessity",
        ),
        bootstrap=boots,
        records=records,
        cap_counts=cap_counts,
        cost=dict(
            seconds=time.monotonic() - started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        ),
    )
    (ROOT / "results.json").write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in output.items()
                if k not in ["seed_path_ids", "records", "cap_counts"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
