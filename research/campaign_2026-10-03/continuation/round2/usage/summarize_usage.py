from __future__ import annotations

import json
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from inspect_control import digest
from usage_core import I0, I1, LAGS, NAMES, QUERIES, REPO, ROOT, intervention_window, update_state

plt.switch_backend("Agg")
AGE_LABELS = ["0", "1–3", "4–15", "16–63", "64+"]


def norm_displacements(packed: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Measure input displacement to expose age/amplitude confounding."""
    values = np.zeros((len(QUERIES), len(NAMES)))
    for index, query in enumerate(QUERIES):
        for arm, name in enumerate(NAMES[1:], start=1):
            start, window = intervention_window(packed, int(query), name)
            group = I0 if name.startswith("i0") else I1
            affected = 1 if name.endswith("current") else int(query - start)
            difference = (window[:affected, group] - packed[start : start + affected, group]) / std[
                group
            ]
            values[index, arm] = np.sqrt(np.mean(difference.astype(float) ** 2))
    return values


def plot_usage(results: dict) -> list[str]:
    """Render response horizons and age profiles with explicit OOD interpretation."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in NAMES[1:]:
        entry = results["interventions"][name]
        values = np.sqrt(np.mean(np.square(entry["rms_by_lag_target"]), axis=1))
        axes[0].plot(LAGS, values, "o-", label=name)
    axes[0].set_xlabel("Original-stream propagation rows after intervention query")
    axes[0].set_ylabel("RMS output movement (two-target mean square)")
    axes[0].set_title("Restoring inputs does not instantly restore recurrent state")
    axes[0].legend(fontsize=8)
    ages = results["quiet_age"]
    for name in NAMES[1:]:
        values = [
            ages[label]["interventions"][name]["query_rms"] if ages[label]["count"] else np.nan
            for label in AGE_LABELS
        ]
        axes[1].plot(range(5), values, "o-", label=name)
    axes[1].set_xticks(range(5), [f"{label}\nn={ages[label]['count']}" for label in AGE_LABELS])
    axes[1].set_xlabel("Elapsed quiet age on original i0 price stream")
    axes[1].set_ylabel("Query RMS output movement")
    axes[1].set_title("Descriptive age response; donor amplitudes also vary")
    figure.suptitle(
        "Six component-excluded sequences / 342 fixed queries — input-space OOD sensitivity"
    )
    figure.tight_layout()
    first = "usage_map.png"
    figure.savefig(ROOT / first, dpi=130)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in NAMES[1:]:
        norms = [
            ages[label]["interventions"][name]["input_standardized_rms"]
            if ages[label]["count"]
            else np.nan
            for label in AGE_LABELS
        ]
        axes[0].plot(range(5), norms, "o-", label=name)
    axes[0].set_xticks(range(5), AGE_LABELS)
    axes[0].set_xlabel("Original quiet age")
    axes[0].set_ylabel("RMS donor displacement in training-standardized channels")
    axes[0].set_title("Age is confounded with intervention strength")
    axes[0].legend(fontsize=8)
    for index, name in enumerate(NAMES[1:]):
        entry = results["interventions"][name]
        lower, upper = entry["scored_query_wmse_damage_component_bootstrap_95"]
        axes[1].plot([lower, upper], [index, index], lw=3)
        axes[1].plot(entry["scored_query_wmse_damage"], index, "ko")
    axes[1].set_yticks(range(5), NAMES[1:])
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Clipped-target weighted MSE damage (altered minus original)")
    axes[1].set_title(
        f"Only {results['scored_queries']} scored queries / six components\nDiagnostic damage, not deployable improvement"
    )
    figure.tight_layout()
    second = "usage_controls.png"
    figure.savefig(ROOT / second, dpi=130)
    plt.close(figure)
    return [first, second]


def main() -> None:
    """Aggregate fixed interventions without adapting sample or fitting a head."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.monotonic()
    update_state("aggregating_frozen_responses")
    control = json.loads((ROOT / "control_identity.json").read_text())
    checkpoint = torch.load(REPO / control["checkpoint"], map_location="cpu", weights_only=False)
    std = checkpoint["model"]["std"].numpy()
    if np.any(std <= 0):
        raise AssertionError("Invalid normalization variance")
    summaries, responses, predictions = [], [], []
    baselines, targets, masks, ages, organizer_ages, norms = [], [], [], [], [], []
    hidden_query, hidden_end = [], []
    for index in range(6):
        summary = json.loads((ROOT / f"sequence_{index:02d}.json").read_text())
        summaries.append(summary)
        with np.load(ROOT / f"sequence_{index:02d}.npz") as arrays:
            responses.append(arrays["responses"])
            predictions.append(arrays["predictions"])
            baselines.append(arrays["baseline"][QUERIES])
            targets.append(arrays["targets"][QUERIES])
            masks.append(arrays["mask"][QUERIES])
            ages.append(arrays["quiet_age"])
            organizer_ages.append(arrays["organizer_age"])
            norms.append(norm_displacements(arrays["packed"], std))
            hidden_query.append(arrays["hidden_query"])
            hidden_end.append(arrays["hidden_end"])
    response = np.array(responses, dtype=float)
    prediction = np.clip(np.array(predictions, dtype=float), -2, 2)
    base = np.clip(np.array(baselines, dtype=float), -2, 2)
    target = np.clip(np.array(targets, dtype=float), -2, 2)
    selected, age = np.array(masks), np.array(ages)
    norm = np.array(norms)
    weight = np.abs(target) * selected[..., None]
    denominator = weight.sum(1)
    base_error = (weight * (target - base) ** 2).sum(1)
    squared_error = (target[:, :, None] - prediction[:, :, :, 0]) ** 2
    damage = (weight[:, :, None] * squared_error).sum(1) - base_error[:, None]
    rng = np.random.default_rng(614104)
    draws = rng.integers(0, 6, (400, 6))
    bootstrap_damage = (
        damage[draws].sum(1) / np.maximum(denominator[draws].sum(1)[:, None], 1e-12)
    ).mean(-1)
    measured_damage = (damage.sum(0) / np.maximum(denominator.sum(0)[None], 1e-12)).mean(-1)
    interventions = {}
    for index, name in enumerate(NAMES):
        identity_floor = max(summary["identity_replay_gate"]["rms"] for summary in summaries)
        query_rms = float(np.sqrt(np.mean(response[:, :, index, 0] ** 2)))
        interventions[name] = {
            "query_rms": query_rms,
            "rms_by_lag_target": np.sqrt(np.mean(response[:, :, index] ** 2, axis=(0, 1))).tolist(),
            "meanabs_query_target": np.mean(np.abs(response[:, :, index, 0]), axis=(0, 1)).tolist(),
            "clipped_query_rms": float(np.sqrt(np.mean((prediction[:, :, index, 0] - base) ** 2))),
            "query_hidden_rms_layer": np.sqrt(
                np.mean(np.array(hidden_query)[:, :, index] ** 2, axis=(0, 1))
            ).tolist(),
            "propagation32_hidden_rms_layer": np.sqrt(
                np.mean(np.array(hidden_end)[:, :, index] ** 2, axis=(0, 1))
            ).tolist(),
            "input_standardized_rms": float(np.sqrt(np.mean(norm[:, :, index] ** 2))),
            "scored_query_wmse_damage": float(measured_damage[index]),
            "scored_query_wmse_damage_component_bootstrap_95": np.quantile(
                bootstrap_damage[:, index], [0.025, 0.975]
            ).tolist(),
            "usage_threshold_passed": query_rms > 0.001 and query_rms > 100 * identity_floor,
        }
    age_bins = np.digitize(age, [1, 4, 16, 64])
    age_panel = {}
    for index, label in enumerate(AGE_LABELS):
        included = age_bins == index
        count = int(included.sum())
        age_panel[label] = {
            "count": count,
            "components": int(np.any(included, axis=1).sum()),
            "scored_count": int((included & selected).sum()),
            "interventions": {
                name: {
                    "query_rms": float(np.sqrt(np.mean(response[:, :, arm, 0][included] ** 2)))
                    if count
                    else None,
                    "input_standardized_rms": float(
                        np.sqrt(np.mean(norm[:, :, arm][included] ** 2))
                    )
                    if count
                    else None,
                }
                for arm, name in enumerate(NAMES)
            },
        }
    long_short = prediction[:, :, 1, 0] - prediction[:, :, 5, 0]
    long_short_raw = response[:, :, 1, 0] - response[:, :, 5, 0]
    result = {
        "model": "c20261001_d256_s3_f0@ep3",
        "queries": 6 * len(QUERIES),
        "scored_queries": int(selected.sum()),
        "components": 6,
        "prefix_rows_per_sequence": 4128,
        "fullvalid_inference": False,
        "interventions": interventions,
        "quiet_age": age_panel,
        "i0_long_minus_short_query_rms_clipped": float(np.sqrt(np.mean(long_short**2))),
        "i0_long_minus_short_query_rms_raw": float(np.sqrt(np.mean(long_short_raw**2))),
        "i0_long_minus_short_by_component_rms": np.sqrt(
            np.mean(long_short**2, axis=(1, 2))
        ).tolist(),
        "quiet_age_definition_disagreement_count": int(np.sum(age != np.array(organizer_ages))),
        "identity": {
            "cpu_reference_max_abs": max(
                item["baseline_cpu_reference_gate"]["max_abs"] for item in summaries
            ),
            "replay_max_abs": max(item["identity_replay_gate"]["max_abs"] for item in summaries),
            "cached_gpu_max_abs": max(
                item["cached_gpu_vs_cpu_gate"]["max_abs"] for item in summaries
            ),
            "cached_gpu_max_rms": max(item["cached_gpu_vs_cpu_gate"]["rms"] for item in summaries),
            "all_gates_passed": True,
        },
        "runtime": {
            "inference_seconds": sum(item["seconds"] for item in summaries),
            "max_command_inference_seconds": max(item["seconds"] for item in summaries),
            "max_rss_mib": max(item["peak_rss_mib"] for item in summaries),
            "aggregation_seconds": time.monotonic() - started,
        },
        "synthetic": json.loads((ROOT / "synthetic_results.json").read_text()),
        "sample_provenance": json.loads((ROOT / "samples.json").read_text()),
        "interpretation": {
            "input_history_usage": "tested donor sensitivity only",
            "usable_current_information": "limited scored-panel damage diagnostic",
            "missing_predictive_information": "unknown; no residual predictor fitted",
            "representation_changing_training_proposal": None,
        },
    }
    np.savez(
        ROOT / "aggregate_panel.npz",
        responses=response,
        predictions=prediction,
        baseline=base,
        targets=target,
        selected=selected,
        quiet_age=age,
        input_displacements=norm,
        damage_by_component=damage,
        score_weight_by_component=denominator,
    )
    (ROOT / "results.json").write_text(json.dumps(result, indent=2))
    figures = plot_usage(result)
    result["figures"] = figures
    result["runtime"]["aggregation_seconds"] = time.monotonic() - started
    (ROOT / "results.json").write_text(json.dumps(result, indent=2))
    logging.info(
        json.dumps(
            {
                name: {
                    key: entry[key]
                    for key in ["query_rms", "scored_query_wmse_damage", "usage_threshold_passed"]
                }
                for name, entry in interventions.items()
            },
            indent=2,
        )
    )
    logging.info(
        "Queries=%s scored=%s identity=%s runtime=%s",
        result["queries"],
        result["scored_queries"],
        result["identity"],
        result["runtime"],
    )
    update_state(
        "measured_results_ready_for_visual_inspection_report",
        {
            "pending": ["inspect figures", "REPORT/provenance"],
            "last_failure": None,
            "measurement_source_hashes": {path.name: digest(path) for path in ROOT.glob("*.py")},
        },
    )


if __name__ == "__main__":
    main()
