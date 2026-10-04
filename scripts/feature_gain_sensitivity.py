"""Calibrate the E422 gain probe with causal synthetic signals and nulls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from feature_gain import FUTURE_GAP, LABEL_END, MATURITY_EVENTS, input_summary, realised_gain, run
from feature_screen import digest_file
from feature_v2_screen import compare
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

SEQUENCE_LENGTH = 8
STATE_WIDTH = 64
SUMMARY_WIDTH = 336
EARLIER_ROWS = 128
FUTURE_ROWS = 32


def _make_component_cache(
    output: Path,
    *,
    count: int,
    component_start: int,
    sequence_start: int,
    condition: str,
    beta: float,
    seed: int,
) -> None:
    """Build one fit or evaluation cache using only causal anchor information."""
    rng = np.random.default_rng(seed)
    latent = rng.choice(np.array([-1.0, 1.0]), size=count)
    raw = rng.normal(scale=0.03, size=(count, SEQUENCE_LENGTH, 112)).astype(np.float32)
    state = rng.normal(size=(count, STATE_WIDTH)).astype(np.float32)
    independent_gain = rng.normal(size=count)
    cue_noise = rng.normal(scale=0.1, size=(count, 32)).astype(np.float32)

    state_order = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float32)
    unrelated_order = np.array([1, 1, -1, 1, 1, -1, -1, -1], dtype=np.float32)
    prototypes = np.zeros((2, SEQUENCE_LENGTH, 112), dtype=np.float32)
    prototypes[0, :, 0], prototypes[1, :, 0] = state_order, unrelated_order
    if not np.array_equal(
        input_summary(prototypes[0], SEQUENCE_LENGTH - 1),
        input_summary(prototypes[1], SEQUENCE_LENGTH - 1),
    ):
        raise ValueError("State-only order patterns leak through the input summary")
    if condition.startswith("state"):
        order = np.where(latent[:, None] > 0, state_order, unrelated_order)
        raw[:, :, 0] = order
        weighted_order_sum = order @ np.arange(1, SEQUENCE_LENGTH + 1, dtype=np.float32)
        state_signal = (weighted_order_sum + 10.0) / 2.0
        state[:, :32] = state_signal[:, None] + cue_noise
        gain_driver = latent
    else:
        raw[:, :, 0] = state_order
        gain_driver = latent if condition == "summary_only" else independent_gain

    if condition == "summary_only":
        raw[:, :, :56] += (latent[:, None, None] * 0.8).astype(np.float32)

    summaries = np.stack([input_summary(sequence, SEQUENCE_LENGTH - 1) for sequence in raw])
    if summaries.shape != (count, SUMMARY_WIDTH):
        raise ValueError("Synthetic E422 summary has an unexpected feature width")

    true_gain = np.column_stack(
        (
            1.0 + beta * np.tanh(gain_driver),
            1.0 - 0.8 * beta * np.tanh(gain_driver),
        )
    ).astype(np.float32)
    earlier_prediction = rng.normal(scale=0.35, size=(count, EARLIER_ROWS, 2)).astype(np.float32)
    earlier_target = earlier_prediction * true_gain[:, None, :]
    earlier_target += rng.normal(scale=0.04, size=earlier_target.shape).astype(np.float32)
    estimated_gain = np.stack(
        [
            realised_gain(target, prediction)
            for target, prediction in zip(earlier_target, earlier_prediction, strict=True)
        ]
    )

    future_prediction = rng.normal(scale=0.35, size=(count, FUTURE_ROWS, 2)).astype(np.float32)
    future_target = future_prediction * true_gain[:, None, :]
    future_target += rng.normal(scale=0.04, size=future_target.shape).astype(np.float32)

    sequence_ids = np.arange(sequence_start, sequence_start + count)
    components = component_start + np.arange(count) // 8
    anchor = LABEL_END + MATURITY_EVENTS - 1
    row_ids = np.arange(anchor + FUTURE_GAP, anchor + FUTURE_GAP + FUTURE_ROWS)
    rows = np.tile(row_ids, count)
    sequence_rows = np.repeat(sequence_ids, FUTURE_ROWS)
    component_rows = np.repeat(components, FUTURE_ROWS)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        state=state,
        summary=summaries,
        gain=estimated_gain,
        anchor=np.full(count, anchor, dtype=np.int64),
        seq_id=sequence_ids,
        seq_component=components,
        pred=future_prediction.reshape(-1, 2),
        y=future_target.reshape(-1, 2),
        row=rows,
        sequence=sequence_rows,
        component=component_rows,
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": 1,
                "sha256": digest_file(output),
                "checkpoint_sha256": "a" * 64,
                "component_graph_sha256": "b" * 64,
                "fit_components": [900_000],
                "outer_fold": 0,
                "rows": len(rows),
                "sequences": count,
                "label_episode": [1024, LABEL_END],
                "maturity_events": MATURITY_EVENTS,
                "future_gap_rows": FUTURE_GAP,
            },
            indent=2,
        )
        + "\n"
    )


def gain_mse_intervals(values: dict, components: np.ndarray, bootstrap: int) -> dict:
    """Bootstrap paired gain-MSE improvements over the fitted constant by component."""
    labels = values["target"]
    baseline = (labels - values["constant"]) ** 2
    groups = np.unique(components)
    draws = np.random.default_rng(271_832).integers(len(groups), size=(bootstrap, len(groups)))
    result = {}
    for name in ("candidate", "input_summary", "shuffled_gain"):
        improvement = baseline - (labels - values[name]) ** 2
        component_means = np.stack([improvement[components == group].mean(0) for group in groups])
        samples = component_means[draws].mean(1)
        result[name] = {
            "mean_mse_improvement_per_target": component_means.mean(0).tolist(),
            "ci95_per_target": np.quantile(samples, [0.025, 0.975], axis=0).T.tolist(),
        }
    return result


def run_calibration(output: Path, bootstrap: int = 2000) -> dict:
    """Run all frozen synthetic conditions through the production gain screen."""
    if output.exists():
        raise ValueError(f"E428 output already exists: {output}")
    output.mkdir(parents=True)
    cases = (
        ("state_strong", 0.55),
        ("state_weak", 0.25),
        ("summary_only", 0.55),
        ("null", 0.55),
    )
    reports = {}
    for condition, beta in cases:
        case_dir = output / condition
        fit_cache = case_dir / "fit.npz"
        evaluation_cache = case_dir / "evaluation.npz"
        _make_component_cache(
            fit_cache,
            count=256,
            component_start=0,
            sequence_start=0,
            condition=condition,
            beta=beta,
            seed=712_000,
        )
        _make_component_cache(
            evaluation_cache,
            count=256,
            component_start=32,
            sequence_start=10_000,
            condition=condition,
            beta=beta,
            seed=713_000,
        )
        run(
            fit_cache,
            evaluation_cache,
            case_dir / "screen",
            backend="numpy",
            bootstrap=bootstrap,
        )
        report = json.loads((case_dir / "screen/report.json").read_text())
        with np.load(case_dir / "screen/predictions.npz", allow_pickle=False) as stored:
            summary_predictions = {
                name: stored[name]
                for name in stored.files
                if name not in {"row", "sequence", "component"}
            }
        summary_predictions["state_probe"] = summary_predictions.pop("candidate")
        summary_predictions["candidate"] = summary_predictions.pop("input_summary")
        with np.load(evaluation_cache, allow_pickle=False) as stored:
            evaluation = {name: stored[name] for name in ("y", "sequence", "component")}
        summary_report = compare(evaluation, summary_predictions, bootstrap)
        (case_dir / "summary_contrasts.json").write_text(
            json.dumps(summary_report, indent=2) + "\n"
        )
        with np.load(case_dir / "screen/gain_predictions.npz", allow_pickle=False) as values:
            mse_intervals = gain_mse_intervals(
                {name: values[name] for name in values.files}, values["seq_component"], bootstrap
            )
            reports[condition] = {
                "beta": beta,
                "candidate_r2": report["recoverability"]["candidate"]["r2_vs_fit_mean_per_target"],
                "summary_r2": report["recoverability"]["input_summary"][
                    "r2_vs_fit_mean_per_target"
                ],
                "shuffled_r2": report["recoverability"]["shuffled_gain"][
                    "r2_vs_fit_mean_per_target"
                ],
                "candidate_minus_unchanged": report["reports"]["candidate_minus_unchanged"],
                "candidate_minus_summary": report["reports"]["candidate_minus_input_summary"],
                "candidate_minus_shuffled": report["reports"]["candidate_minus_shuffled_gain"],
                "summary_minus_unchanged": summary_report["candidate_minus_unchanged"],
                "summary_minus_state": summary_report["candidate_minus_state_probe"],
                "summary_minus_shuffled": summary_report["candidate_minus_shuffled_gain"],
                "gain_mse_improvement": mse_intervals,
                "evaluation_sequences": int(len(values["seq_id"])),
            }

    strong = reports["state_strong"]
    summary = reports["summary_only"]
    null = reports["null"]
    checks = {
        "state_signal_recovered": bool(
            min(strong["candidate_r2"]) > 0
            and np.mean(strong["candidate_r2"])
            > max(np.mean(strong["summary_r2"]), np.mean(strong["shuffled_r2"]))
            and all(
                strong[key]["ci95"][0] > 0
                for key in (
                    "candidate_minus_unchanged",
                    "candidate_minus_summary",
                    "candidate_minus_shuffled",
                )
            )
        ),
        "summary_signal_recovered": bool(
            min(summary["summary_r2"]) > 0
            and np.mean(summary["summary_r2"])
            > max(np.mean(summary["candidate_r2"]), np.mean(summary["shuffled_r2"]))
            and all(
                summary[key]["ci95"][0] > 0
                for key in (
                    "summary_minus_unchanged",
                    "summary_minus_state",
                    "summary_minus_shuffled",
                )
            )
        ),
        "null_has_no_recovery_or_significant_wp_gain": bool(
            all(
                interval[0] <= 0
                for result in null["gain_mse_improvement"].values()
                for interval in result["ci95_per_target"]
            )
            and null["candidate_minus_unchanged"]["ci95"][0] <= 0
            and null["summary_minus_unchanged"]["ci95"][0] <= 0
        ),
    }
    (output / "report.json").write_text(
        json.dumps(
            {
                "experiment": "E428",
                "purpose": "synthetic assay calibration only",
                "conditions": reports,
                "calibration_checks": checks,
                "calibration_passed": all(checks.values()),
                "interpretation": "Does not estimate real-data predictive benefit or close H-095.",
            },
            indent=2,
        )
        + "\n"
    )
    return reports


def plot_calibration(reports: dict, path: Path) -> None:
    """Save the prespecified recovery and later-WP calibration figure."""
    names = list(reports)
    x = np.arange(len(names))
    candidate = [np.mean(reports[name]["candidate_r2"]) for name in names]
    summary = [np.mean(reports[name]["summary_r2"]) for name in names]
    figure = Figure(figsize=(11, 4.2))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    axes[0].bar(x - 0.18, candidate, width=0.36, label="State")
    axes[0].bar(x + 0.18, summary, width=0.36, label="Input summary")
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_xticks(x, names, rotation=20, ha="right")
    axes[0].set_ylabel("Held-out gain R² vs fit mean")
    axes[0].legend(frameon=False)
    for key, offset, label in (
        ("candidate_minus_unchanged", -0.1, "State"),
        ("summary_minus_unchanged", 0.1, "Input summary"),
    ):
        delta = np.array([reports[name][key]["mean_diff"] for name in names])
        low = np.array([reports[name][key]["ci95"][0] for name in names])
        high = np.array([reports[name][key]["ci95"][1] for name in names])
        axes[1].errorbar(
            x + offset, delta, yerr=[delta - low, high - delta], fmt="o", capsize=3, label=label
        )
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_xticks(x, names, rotation=20, ha="right")
    axes[1].set_ylabel("Later Global WP: probe − unchanged")
    axes[1].legend(frameon=False)
    figure.suptitle("E428 synthetic sensitivity of the E422 gain assay")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    figure.clear()


def main() -> None:
    """Run the bounded E428 CPU calibration and save its figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap != 2000:
        parser.error("The preregistered E428 calibration requires 2000 bootstrap draws")
    reports = run_calibration(args.output, args.bootstrap)
    plot_calibration(reports, args.output / "e428_sensitivity.png")
    print(json.dumps({name: value["candidate_r2"] for name, value in reports.items()}))
    report = json.loads((args.output / "report.json").read_text())
    if not report["calibration_passed"]:
        raise SystemExit("E428 calibration failed; inspect the saved report and figure")


if __name__ == "__main__":
    main()
