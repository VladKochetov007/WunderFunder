from __future__ import annotations

import hashlib
import json
import logging
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from causal_features import history_features, sufficient_statistics, weighted_correlations

ROOT = Path(__file__).resolve().parent
MODELS = [
    "current",
    "current_instant_clock",
    "event_memory",
    "row_memory",
    "delayed_event_memory",
]
START = 611


def update_state(phase: str, detail: dict | None = None) -> None:
    """Persist phase boundaries and failures in the analyst-owned directory."""
    state = json.loads((ROOT / "STATE.json").read_text())
    state["phase"] = phase
    if detail:
        state.update(detail)
    (ROOT / "STATE.json").write_text(json.dumps(state, indent=2))


def read_sequence(pf: pq.ParquetFile, rg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one row group and only required columns."""
    names = pf.schema_arrow.names
    names = names[names.index("i0_p0") : names.index("a7") + 1]
    columns = names + ["t0", "t1"]
    if "is_scored" in pf.schema_arrow.names:
        columns += ["is_scored"]
    table = pf.read_row_group(rg, columns=columns, use_threads=False)
    features = np.column_stack([table[c].to_numpy() for c in names])
    targets = np.column_stack([table[c].to_numpy() for c in ["t0", "t1"]])
    selected = (
        table["is_scored"].to_numpy().astype(bool)
        if "is_scored" in columns
        else np.ones(len(features), dtype=bool)
    )
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError(f"Nonfinite data in row group {rg}")
    return features, targets, selected


def matched_hazard(features: np.ndarray, histories: dict[str, np.ndarray]) -> np.ndarray:
    """Compare age arms within exact price state and coarse recent-rate strata.

    Next event is an evaluation label, never a model input. Return numerator,
    support weight and young/old weighted hazard sums for real and stale age.
    """
    prices = np.ascontiguousarray(features[:, :22])
    packed = prices.view(np.dtype((np.void, prices.dtype.itemsize * 22))).ravel()
    _, state = np.unique(packed, return_inverse=True)
    rate_bucket = np.digitize(histories["rate"], [0.03, 0.1, 0.3])
    strata = state * 4 + rate_bucket
    outcome = histories["events"][START + 1 :].astype(float)
    strata = strata[START:-1]
    result = []
    for age in [histories["age"][START:-1], histories["age"][START - 512 : -513]]:
        young = (age >= 1) & (age <= 3)
        old = (age >= 8) & (age <= 31)
        size = int(strata.max()) + 1
        counts = np.array(
            [np.bincount(strata[arm], minlength=size) for arm in [young, old]], dtype=float
        )
        hits = np.array(
            [np.bincount(strata[arm], weights=outcome[arm], minlength=size) for arm in [young, old]]
        )
        eligible = np.all(counts >= 10, axis=0)
        counts, hits = counts[:, eligible], hits[:, eligible]
        weight = 2 / (1 / counts[0] + 1 / counts[1])
        hazards = hits / counts
        result.append(
            [
                float((weight * (hazards[1] - hazards[0])).sum()),
                float(weight.sum()),
                float((weight * hazards[0]).sum()),
                float((weight * hazards[1]).sum()),
                int(eligible.sum()),
            ]
        )
    return np.array(result)


def main() -> None:
    """Execute the frozen memory falsifier and a matched age-hazard probe."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.monotonic()
    hypotheses = json.loads((ROOT / "HYPOTHESES.json").read_text())
    provenance = json.loads((ROOT / "continuation_samples.json").read_text())
    update_state(
        "probe_training",
        {
            "pending_work": ["train", "score", "figures/report"],
            "commands": [
                "run_cpu.sh eda-statphys env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B prepare_continuation.py: exit 0",
                "run_cpu.sh eda-statphys env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B continuation_probe.py: admission exit 75; preserved ADMISSION_FAILURE.txt",
                "same continuation_probe.py bounded command retried under explicit controller authorization",
            ],
            "last_failure": "Historical admission exit75; controller authorized retry; ADMISSION_FAILURE.txt",
        },
    )
    pf = pq.ParquetFile("datasets/train.parquet")
    accumulators = {}
    for index, record in enumerate(provenance["samples"]["train"]):
        features, targets, _ = read_sequence(pf, record["row_group"])
        histories = history_features(features)
        y = np.clip(targets[START::8].astype(float), -2, 2)
        weights = np.abs(y)
        for model in MODELS:
            x = histories[model][START::8]
            if model not in accumulators:
                dimension = x.shape[1]
                accumulators[model] = {
                    "sum": np.zeros(dimension),
                    "square": np.zeros(dimension),
                    "count": 0,
                    "gram": np.zeros((2, dimension, dimension)),
                    "rhs": np.zeros((dimension, 2)),
                    "wx": np.zeros((dimension, 2)),
                    "wy": np.zeros(2),
                    "w": np.zeros(2),
                }
            acc = accumulators[model]
            acc["sum"] += x.sum(0)
            acc["square"] += (x * x).sum(0)
            acc["count"] += len(x)
            for target in range(2):
                wx = x * weights[:, target, None]
                acc["gram"][target] += x.T @ wx
                acc["rhs"][:, target] += wx.T @ y[:, target]
                acc["wx"][:, target] += wx.sum(0)
                acc["wy"][target] += (weights[:, target] * y[:, target]).sum()
                acc["w"][target] += weights[:, target].sum()
        if (index + 1) % 12 == 0:
            logging.info("train sequences %s/48", index + 1)
    coefficients = {}
    for model, acc in accumulators.items():
        variance = acc["square"] / acc["count"] - (acc["sum"] / acc["count"]) ** 2
        beta = np.empty((len(variance) + 1, 2))
        for target in range(2):
            weight = acc["w"][target]
            mean_x = acc["wx"][:, target] / weight
            mean_y = acc["wy"][target] / weight
            gram = acc["gram"][target] - weight * np.outer(mean_x, mean_x)
            rhs = acc["rhs"][:, target] - weight * mean_x * mean_y
            penalty = np.maximum(variance, 1e-8) * weight * 0.001
            beta[:-1, target] = np.linalg.solve(gram + np.diag(penalty), rhs)
            beta[-1, target] = mean_y - mean_x @ beta[:-1, target]
        coefficients[model] = beta
    np.savez(ROOT / "continuation_coefficients.npz", **coefficients)
    del accumulators
    update_state("probe_scoring")
    pf = pq.ParquetFile("datasets/valid.parquet")
    moments, hazards, row_counts = [], [], []
    causal_checks = []
    for index, record in enumerate(provenance["samples"]["valid"]):
        features, targets, selected = read_sequence(pf, record["row_group"])
        histories = history_features(features)
        selected[:START] = False
        sequence_moments = []
        for model in MODELS:
            beta = coefficients[model]
            predictions = histories[model][selected] @ beta[:-1] + beta[-1]
            sequence_moments.append(sufficient_statistics(targets[selected], predictions))
        moments.append(sequence_moments)
        hazards.append(matched_hazard(features, histories))
        row_counts.append(int(selected.sum()))
        if index == 0:
            prefix = history_features(features[:1000])
            for model in MODELS:
                if not np.array_equal(prefix[model], histories[model][:1000]):
                    raise AssertionError(f"Prefix invariance failed: {model}")
                causal_checks.append(f"prefix invariance passed: {model}")
            np.savez(
                ROOT / "continuation_example.npz",
                age=histories["age"],
                rate=histories["rate"],
                targets=targets,
                selected=selected,
            )
        if (index + 1) % 8 == 0:
            logging.info("validation sequences %s/32", index + 1)
    moments, hazards = np.array(moments), np.array(hazards)
    np.savez(ROOT / "continuation_moments.npz", moments=moments, hazards=hazards)
    correlations = weighted_correlations(moments.sum(0))
    scores = correlations.mean(-1)
    rng = np.random.default_rng(610104)
    bootstrap = rng.integers(0, len(moments), (400, len(moments)))
    bootstrap_scores = weighted_correlations(moments[bootstrap].sum(1)).mean(-1)
    deltas = {}
    event_index = MODELS.index("event_memory")
    for control in ["current_instant_clock", "delayed_event_memory", "row_memory"]:
        control_index = MODELS.index(control)
        draws = bootstrap_scores[:, event_index] - bootstrap_scores[:, control_index]
        deltas[control] = {
            "event_minus_control": float(scores[event_index] - scores[control_index]),
            "sequence_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    hazard_summary = {}
    for index, name in enumerate(["real_age", "delayed_age"]):
        summed = hazards[:, index].sum(0)
        draws = hazards[bootstrap, index].sum(1)
        diff = draws[:, 0] / np.maximum(draws[:, 1], 1e-12)
        hazard_summary[name] = {
            "old_minus_young": float(summed[0] / max(summed[1], 1e-12)),
            "young_hazard": float(summed[2] / max(summed[1], 1e-12)),
            "old_hazard": float(summed[3] / max(summed[1], 1e-12)),
            "support_weight": float(summed[1]),
            "eligible_strata": int(summed[4]),
            "sequence_bootstrap_95": np.quantile(diff, [0.025, 0.975]).tolist(),
        }
    success = all(
        deltas[name]["event_minus_control"] >= 0.005
        and deltas[name]["sequence_bootstrap_95"][0] > 0
        for name in ["current_instant_clock", "delayed_event_memory"]
    )
    rejection = any(
        deltas[name]["event_minus_control"] <= 0
        for name in ["current_instant_clock", "delayed_event_memory"]
    )
    results = {
        "models": MODELS,
        "scores": dict(zip(MODELS, scores.tolist(), strict=True)),
        "target_correlations": dict(zip(MODELS, correlations.tolist(), strict=True)),
        "deltas": deltas,
        "hazard": hazard_summary,
        "h1_verdict": "supported_local_proxy"
        if success
        else ("rejected_on_this_proxy" if rejection else "inconclusive"),
        "train_rows": 48 * len(range(START, 20000, 8)),
        "valid_scored_rows": sum(row_counts),
        "valid_sequence_row_counts": row_counts,
        "prefix_causality_checks": causal_checks,
        "seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "hypotheses_sha256": hashlib.sha256((ROOT / "HYPOTHESES.json").read_bytes()).hexdigest(),
        "limitations": hypotheses["no_claims"]
        + [
            "global WP only on sampled validation rows after common step-611 burn-in",
            "sequence bootstrap assumes sequence units; campaign components unknown",
            "current-state ridge is not the campaign baseline",
            "target census exposure means not an independent dataset confirmation",
            "512-row delay preserves dimensions/regime, not exact marginal distribution",
        ],
    }
    (ROOT / "continuation_results.json").write_text(json.dumps(results, indent=2))
    logging.info(json.dumps(results, indent=2))
    update_state(
        "probe_complete_pending_figures_report",
        {
            "measured": results,
            "last_failure": None,
            "artifacts": json.loads((ROOT / "STATE.json").read_text())["artifacts"]
            + [
                "continuation_results.json",
                "continuation_moments.npz",
                "continuation_coefficients.npz",
                "continuation_example.npz",
            ],
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        (ROOT / "continuation_failure.txt").write_text(failure)
        update_state("failed_no_automatic_retry", {"last_failure": failure})
        raise
