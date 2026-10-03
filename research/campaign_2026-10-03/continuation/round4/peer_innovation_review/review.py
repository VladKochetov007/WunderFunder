from __future__ import annotations

import hashlib
import json
import logging
import os
import resource
import time
from pathlib import Path

import numpy as np

OWN = Path(__file__).resolve().parent
REPO = OWN.parents[4]
ACTOR = OWN.parent / "peer_innovation"
PRIOR = OWN.parent.parent / "round3" / "matched_usage"
NAMES = ("base", "base_plus_peer", "base_plus_shifted_peer")
LOG = logging.getLogger(__name__)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    """Stream artifact hashes, keeping modest live arrays."""
    h = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1048576):
            h.update(block)
    return h.hexdigest()


def save(name: str, value: object) -> None:
    (OWN / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def independent_ema(values: np.ndarray, half_life: int) -> np.ndarray:
    """Prefix recurrence from zero, separate implementation from actor helpers."""
    gain = -np.expm1(-np.log(2) / half_life)
    result = np.empty(len(values))
    last = 0.0
    for t, value in enumerate(values):
        last = (1 - gain) * last + gain * value
        result[t] = last
    return result


def gate_from_summary(summary: dict) -> bool:
    """Evaluate the actual preregistered conjunction, not a nominal-gain shortcut."""
    contrasts = summary["contrasts"]
    return all(
        c["relative_improvement"] >= 0.02 and c["paired_component95_interval"][0] > 0
        for c in contrasts.values()
    ) and all(
        s["adequate"] and min(s["peer_mse_reductions"]) >= 0
        for s in summary["input_freshness_strata"]
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    started, cpu = time.monotonic(), time.process_time()
    manifest = read(ACTOR / "artifact_provenance.json")
    paths = [ACTOR / name for name in manifest["artifacts"]]
    panel = read(ACTOR / "samples.json")
    control = panel["control"]
    paths += [REPO / p for p in control["source_sha256"]]
    paths += [
        REPO / control["checkpoint"],
        REPO / control["split_manifest"],
        PRIOR / "samples.json",
        OWN / "PREREG.json",
        OWN / "review.py",
    ]
    paths += [PRIOR / f"seq_{i:02d}.npz" for i in range(24)]
    before = {str(p): sha(p) for p in paths}
    for name, entry in manifest["artifacts"].items():
        assert before[str(ACTOR / name)] == entry["sha256"], name
    for name in ("frozen_sources.json", "frozen_fit_sources.json"):
        for file_name, expected in read(ACTOR / name).items():
            assert before[str(ACTOR / file_name)] == expected
    for name, expected in control["source_sha256"].items():
        assert before[str(REPO / name)] == expected
    assert before[str(REPO / control["checkpoint"])] == control["checkpoint_sha256"]
    assert before[str(REPO / control["split_manifest"])] == control["split_sha256"]
    assert before[str(PRIOR / "samples.json")] == panel["prior_sample_sha256"]
    old_panel = read(PRIOR / "samples.json")
    assert old_panel["samples"] == panel["samples"]
    split = read(REPO / control["split_manifest"])
    sorted_components = sorted(
        [s["component"] for s in panel["samples"]],
        key=lambda c: hashlib.sha256(f"peer-innovation-split-v1:{c}".encode()).digest(),
    )
    assert len(set(sorted_components)) == 24
    assert sorted_components[:16] == panel["fit_components"]
    assert sorted_components[16:] == panel["test_components"]
    assert set(panel["fit_components"]).isdisjoint(panel["test_components"])
    ledger, chunks, y64_parts = [], [], []
    maxima = {"own_ema": 0.0, "peer_ema": 0.0, "shift_ema": 0.0, "scalar_label_vs_float64": 0.0}
    inherited_forwards = identity_max = canaries = 0
    for i, sample in enumerate(panel["samples"]):
        assert sample["row_group"] in split["evaluation"]["valid"]
        assert sample["query_pi"] == 12 / sample["n_eligible"]
        meta = read(ACTOR / f"seq_{i:02d}.json")
        assert before[str(PRIOR / f"seq_{i:02d}.npz")] == meta["source_array_sha256"]
        assert sample["queries"] == [r["query"] for r in meta["ledger"]]
        ledger.extend(meta["ledger"])
        inherited_forwards += meta["state_replay_queries"]
        identity_max = max(identity_max, meta["identity_max_abs"])
        canaries += meta["future_canaries"]
        with (
            np.load(PRIOR / f"seq_{i:02d}.npz") as original,
            np.load(ACTOR / f"seq_{i:02d}.npz") as saved,
        ):
            raw, packed, std = original["raw"], original["packed"], original["std"]
            qs = saved["queries"]
            assert np.array_equal(qs, [r["query"] for r in meta["ledger"] if r["eligible"]])
            assert np.array_equal(saved["base"][:, :211], packed[qs])
            assert np.array_equal(saved["base"][:, 214:], saved["gru_state"].reshape(len(qs), 512))
            assert all(t - 288 in original["state_boundaries"] for t in qs)
            assert std[0] >= 1e-6 and std[52] >= 1e-6
            own_diff = np.r_[0.0, np.diff(raw[:, 0].astype(float))] / float(std[0])
            peer_diff = np.r_[0.0, np.diff(raw[:, 52].astype(float))] / float(std[52])
            own = np.stack([independent_ema(own_diff, h)[qs] for h in (4, 16, 64)], axis=1)
            peer_series = independent_ema(peer_diff, 16)
            for name, actual, expected in (
                ("own_ema", own, saved["base"][:, 211:214]),
                ("peer_ema", peer_series[qs], saved["peer"]),
                ("shift_ema", peer_series[qs - 256], saved["shift"]),
            ):
                error = float(np.abs(actual - expected).max())
                maxima[name] = max(maxima[name], error)
                assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)
            updates = np.r_[False, np.any(np.diff(raw[:, 52:74], axis=0) != 0, axis=1)]
            age = np.arange(len(raw)) - np.maximum.accumulate(
                np.where(updates, np.arange(len(raw)), 0)
            )
            assert np.array_equal(age[qs], saved["peer_age"])
            scalar_y = np.array(
                [float((float(raw[t + 32, 0]) - float(raw[t, 0])) / std[0]) for t in qs]
            )
            y64 = (raw[qs + 32, 0].astype(float) - raw[qs, 0].astype(float)) / float(std[0])
            assert np.array_equal(scalar_y, saved["target"])
            maxima["scalar_label_vs_float64"] = max(
                maxima["scalar_label_vs_float64"], float(np.abs(y64 - scalar_y).max())
            )
            y64_parts.append(y64)
            chunks.append(
                {
                    name: saved[name].copy()
                    for name in ("base", "peer", "shift", "peer_age", "target", "queries")
                }
            )
    arrays = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
    components = np.concatenate(
        [
            np.full(len(c["target"]), s["component"])
            for c, s in zip(chunks, panel["samples"], strict=True)
        ]
    )
    models = np.load(ACTOR / "panel_models.npz")
    for name, array in arrays.items():
        assert np.array_equal(array, models[name])
    assert np.array_equal(components, models["component"])
    fit = np.isin(components, panel["fit_components"])
    test = np.isin(components, panel["test_components"])
    assert np.array_equal(fit, models["fit"]) and np.array_equal(test, models["test"])
    weight = np.array([fit.sum() / (16 * np.sum(components[fit] == c)) for c in components[fit]])
    assert np.isclose(weight.sum(), 191)
    assert all(
        np.isclose(weight[components[fit] == c].sum(), 191 / 16) for c in panel["fit_components"]
    )
    matrices = [
        arrays["base"],
        np.c_[arrays["base"], arrays["peer"]],
        np.c_[arrays["base"], arrays["shift"]],
    ]
    geometry, predictions, hats = [], [], []
    results = read(ACTOR / "results.json")
    for name, x, reported in zip(NAMES, matrices, results["geometry"], strict=True):
        mean = np.average(x[fit], axis=0, weights=weight)
        scale = np.maximum(np.sqrt(np.average((x[fit] - mean) ** 2, axis=0, weights=weight)), 1e-6)
        block = np.r_[
            np.full(211, 1 / np.sqrt(211)),
            np.full(3, 1 / np.sqrt(3)),
            np.full(512, 1 / np.sqrt(512)),
            np.ones(x.shape[1] - 726),
        ]
        assert np.array_equal(mean, models[f"{name}_mean"])
        assert np.array_equal(scale, models[f"{name}_scale"])
        assert np.array_equal(block, models[f"{name}_block"])
        intercept = float(np.average(arrays["target"][fit], weights=weight))
        assert intercept == models[f"{name}_intercept"]
        z = (x[fit] - mean) / scale * block
        zw = z * np.sqrt(weight[:, None])
        gram = zw @ zw.T
        lam = float(models[f"{name}_lambda_"])
        eigen = np.linalg.eigvalsh(gram).clip(min=0)
        df = float((eigen / (eigen + lam)).sum())
        rank = int((eigen > max(eigen.max() * 1e-10, 1e-12)).sum())
        assert abs(df - 16) < 1e-10 and rank == 190
        beta = models[f"{name}_beta"]
        rhs = (arrays["target"][fit] - intercept) * np.sqrt(weight)
        residual = zw.T @ (zw @ beta - rhs) + lam * beta
        assert np.max(np.abs(residual)) < 1e-9
        hat = np.linalg.solve(gram + lam * np.eye(len(gram)), gram)
        hats.append(hat)
        pred = (x[test] - mean) / scale * block @ beta + intercept
        predictions.append(pred)
        geometry.append(
            {
                "arm": name,
                "fit_only_mean_scale_block_intercept": True,
                "rank": rank,
                "EDF_excluding_intercept": df,
                "lambda": lam,
                "normal_equation_max_abs": float(np.abs(residual).max()),
                "hat_frobenius_from_base": float(np.linalg.norm(hat - hats[0])),
            }
        )
        assert (
            abs(geometry[-1]["hat_frobenius_from_base"] - reported["hat_frobenius_from_base"])
            < 1e-10
        )
    prediction = np.stack(predictions, axis=1)
    assert np.array_equal(prediction, models["prediction"])
    loss = (prediction - arrays["target"][test, None]) ** 2
    assert np.array_equal(loss, models["loss"])
    test_components = panel["test_components"]
    per = np.array([loss[components[test] == c].mean(axis=0) for c in test_components])
    mse = per.mean(axis=0)
    assert np.array_equal(per, models["percomponent_loss"])
    draws = np.random.default_rng(614405).integers(0, 8, (1000, 8))
    assert np.array_equal(draws, models["bootstrap_indices"])
    boot = per[draws].mean(axis=1)
    failures = []
    contrasts = {}
    for column in (0, 2):
        name = NAMES[column]
        relative = float(1 - mse[1] / mse[column])
        reduction = float(mse[column] - mse[1])
        interval = np.quantile(boot[:, column] - boot[:, 1], [0.025, 0.975]).tolist()
        positive = int((per[:, column] > per[:, 1]).sum())
        contrasts[name] = {
            "relative_improvement": relative,
            "MSE_reduction": reduction,
            "CI95": interval,
            "positive_components": positive,
        }
        expected = results["contrasts"][name]
        assert relative == expected["relative_improvement"]
        assert np.array_equal(interval, expected["paired_component95_interval"])
        if relative < 0.02:
            failures.append(f"{name}: relative gain below2%")
        if interval[0] <= 0:
            failures.append(f"{name}: CI lower bound notpositive")
    strata = []
    for label, mask in (
        ("peer_age0", arrays["peer_age"][test] == 0),
        ("peer_age1+", arrays["peer_age"][test] > 0),
    ):
        used = [c for c in test_components if (mask & (components[test] == c)).any()]
        smse = np.array([loss[mask & (components[test] == c)].mean(axis=0) for c in used]).mean(
            axis=0
        )
        adequate = int(mask.sum()) >= 12 and len(used) >= 2
        reductions = [float(smse[j] - smse[1]) for j in (0, 2)]
        strata.append(
            {
                "name": label,
                "queries": int(mask.sum()),
                "components": len(used),
                "adequate": adequate,
                "reductions_base_shift": reductions,
            }
        )
        if not adequate:
            failures.append(f"{label}: inadequate coverage")
        for j, reduction in zip((0, 2), reductions, strict=True):
            if reduction < 0:
                failures.append(f"{label}: worse than{NAMES[j]}")
    assert not gate_from_summary(results) and not results["gate_passed"]
    controls = read(ACTOR / "controls.json")
    assert gate_from_summary(controls["positive"])
    assert not gate_from_summary(controls["negative"])
    original_ledger = read(ACTOR / "origins.json")["origins"]
    assert ledger == original_ledger
    weights = {
        "selected": len(ledger),
        "eligible": sum(r["eligible"] for r in ledger),
        "missing_future": sum(r["reason"] == "missing_future_end" for r in ledger),
        "selected_design_mass": sum(r["original_inverse_query_probability"] for r in ledger),
        "fit_design_mass": sum(
            r["original_inverse_query_probability"]
            for r in ledger
            if r["eligible"] and r["split"] == "fit"
        ),
        "test_design_mass": sum(
            r["original_inverse_query_probability"]
            for r in ledger
            if r["eligible"] and r["split"] == "test"
        ),
        "rejected_design_mass": sum(
            r["original_inverse_query_probability"] for r in ledger if not r["eligible"]
        ),
        "fit_queries": int(fit.sum()),
        "test_queries": int(test.sum()),
    }
    for r in ledger:
        sample = next(s for s in panel["samples"] if s["component"] == r["component"])
        assert r["original_inverse_query_probability"] == 1 / sample["query_pi"]
        assert r["split"] == ("fit" if r["component"] in panel["fit_components"] else "test")
    y64 = np.concatenate(y64_parts)
    ideal_loss = (prediction - y64[test, None]) ** 2
    ideal_mse = np.array(
        [ideal_loss[components[test] == c].mean(axis=0) for c in test_components]
    ).mean(axis=0)
    assert all(1 - ideal_mse[1] / ideal_mse[j] < 0.02 for j in (0, 2))
    assert before == {str(p): sha(p) for p in paths}
    review = {
        "verdict": "REJECTION_SUPPORTED_FOR_FIXED_PROXY_PROTOCOL_NOT_INFORMATION_ABSENCE",
        "independent_gru_forwards": 0,
        "new_fits": 0,
        "actor_helpers_imported": False,
        "artifacts_verified": len(manifest["artifacts"]),
        "feature_recreation_maxima": maxima,
        "geometry": geometry,
        "matched_df_not_equal_penalty_geometry": True,
        "MSE": dict(zip(NAMES, mse.tolist(), strict=True)),
        "contrasts": contrasts,
        "strata": strata,
        "failed_gate_clauses": failures,
        "weights": weights,
        "missing_origin_records": [r for r in ledger if not r["eligible"]],
        "all_float64_target_MSE": ideal_mse.tolist(),
        "target_precision_does_not_rescue_threshold": True,
        "inherited_actual_forward_checks": {
            "identity_queries": inherited_forwards,
            "future_canaries": canaries,
            "identity_max_abs": identity_max,
            "not_reexecuted": True,
        },
        "toy_control_specification_discrepancy": {
            "prereg": "shift independently generated",
            "implemented": "same causal EMA of one iid sequence at t-256",
            "iid_EMA_population_lag_correlation": 2 ** (-256 / 16),
            "severity": "minor control-specification deviation; real-data comparator declared same-prefix lag and unaffected; toy pass not arbitrary-noise power calibration",
        },
        "input_sha256": before,
        "cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time() - cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "GPU": False,
        },
    }
    save("results.json", review)
    save(
        "STATE.json",
        {
            "phase": "review_scored",
            "verdict": review["verdict"],
            "gru_forwards": 0,
            "new_fits": 0,
            "actor_helpers_imported": False,
        },
    )
    LOG.info(
        "REVIEW %s", json.dumps({k: v for k, v in review.items() if k not in ("input_sha256",)})
    )


if __name__ == "__main__":
    main()
