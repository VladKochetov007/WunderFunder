from __future__ import annotations

import json
import time

import numpy as np
from common import PRIOR, REPO, ROOT, dump, sha


def main() -> None:
    started = time.monotonic()
    for manifest in ["frozen_sources.json", "frozen_fit_sources.json"]:
        for name, expected in json.loads((ROOT / manifest).read_text()).items():
            assert sha(ROOT / name) == expected, name
    panel = json.loads((ROOT / "samples.json").read_text())
    prior = json.loads((PRIOR / "artifact_provenance.json").read_text())
    for name, item in prior["artifacts"].items():
        assert sha(PRIOR / name) == item["sha256"], name
    control = panel["control"]
    for name, expected in control["source_sha256"].items():
        assert sha(REPO / name) == expected, name
    assert sha(REPO / control["checkpoint"]) == control["checkpoint_sha256"]
    metadata = [json.loads((ROOT / f"seq_{i:02d}.json").read_text()) for i in range(24)]
    identity = max(m["identity_max_abs"] for m in metadata)
    assert identity == 0
    assert sum(m["future_canaries"] for m in metadata) == 24
    targets_checked = 0
    target_float64_difference = 0.0
    for i in range(24):
        with np.load(PRIOR / f"seq_{i:02d}.npz") as original, np.load(ROOT / f"seq_{i:02d}.npz") as saved:
            raw, std = original["raw"], original["std"]
            assert std[0] >= 1e-6 and std[52] >= 1e-6
            queries = saved["queries"]
            # Recreate extraction's scalar dtype promotion exactly. Separately
            # disclose the difference from an all-float64 reference calculation.
            expected = np.array([float((float(raw[t + 32, 0]) - float(raw[t, 0])) / std[0]) for t in queries])
            assert np.array_equal(expected, saved["target"])
            ideal = (raw[queries + 32, 0].astype(float) - raw[queries, 0].astype(float)) / float(std[0])
            target_float64_difference = max(target_float64_difference, float(np.abs(ideal - expected).max()))
            assert np.array_equal(original["packed"][queries], saved["base"][:, :211])
            assert np.array_equal(saved["gru_state"].reshape(len(queries), 512), saved["base"][:, 214:])
            targets_checked += len(queries)
    results = json.loads((ROOT / "results.json").read_text())
    with np.load(ROOT / "panel_models.npz") as saved:
        base, fit, test = saved["base"], saved["fit"], saved["test"]
        assert not set(saved["component"][fit]) & set(saved["component"][test])
        predictions = []
        for name, x in zip(["base", "base_plus_peer", "base_plus_shifted_peer"],
                           [base, np.c_[base, saved["peer"]], np.c_[base, saved["shift"]]], strict=True):
            p = ((x[test] - saved[name + "_mean"]) / saved[name + "_scale"] * saved[name + "_block"]) @ saved[name + "_beta"] + saved[name + "_intercept"]
            predictions.append(p)
        predicted = np.column_stack(predictions)
        prediction_error = float(np.abs(predicted - saved["prediction"]).max())
        assert prediction_error == 0
        loss = (predicted - saved["target"][test, None]) ** 2
        assert np.array_equal(loss, saved["loss"])
        percomponent = np.array([loss[saved["test_component"] == c].mean(0) for c in results["test_components"]])
        assert np.array_equal(percomponent, saved["percomponent_loss"])
        mse = percomponent.mean(0)
        assert np.array_equal(mse, list(results["arm_test_component_mse"].values()))
        draws = np.random.default_rng(614405).integers(0, len(percomponent), size=(1000, len(percomponent)))
        assert np.array_equal(draws, saved["bootstrap_indices"])
        bootstrap = percomponent[draws].mean(1)
        for column, name in [(0, "base"), (2, "base_plus_shifted_peer")]:
            ci = np.quantile(bootstrap[:, column] - bootstrap[:, 1], [0.025, 0.975])
            assert np.array_equal(ci, results["contrasts"][name]["paired_component95_interval"])
    dump("verification.json", dict(passed=True, prior_artifacts_unchanged=len(prior["artifacts"]),
         frozen_sources_unchanged=True, live_checkpoint_sources_unchanged=True,
         targets_recreated=targets_checked, identity_max_abs=identity, future_canaries=24,
         target_scalar_vs_float64_max_abs=target_float64_difference,
         verification_initial_failure="vector float64 recreation differs from frozen scalar NumPy float32 division; original failed log preserved; no fit/data/tolerance change",
         model_prefix_forwards=0, prediction_recreation_max_abs=prediction_error,
         loss_component_bootstrap_exact=True, seconds=time.monotonic() - started))
    print((ROOT / "verification.json").read_text())


if __name__ == "__main__":
    main()
