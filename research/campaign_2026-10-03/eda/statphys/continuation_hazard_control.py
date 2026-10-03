from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from causal_features import history_features
from continuation_probe import START, read_sequence, update_state

ROOT = Path(__file__).resolve().parent


def common_strata_hazard(
    features: np.ndarray, histories: dict[str, np.ndarray], auxiliary_columns: tuple[int, ...]
) -> np.ndarray:
    """Compare real and stale age on identical price/activity/clock strata."""
    prices = np.ascontiguousarray(features[:, :22])
    packed = prices.view(np.dtype((np.void, prices.dtype.itemsize * 22))).ravel()
    _, state = np.unique(packed, return_inverse=True)
    keys = [state, np.digitize(histories["rate"], [0.03, 0.1, 0.3])]
    keys += [np.floor(features[:, column] / 0.5).astype(int) for column in auxiliary_columns]
    _, groups = np.unique(np.column_stack(keys), axis=0, return_inverse=True)
    groups = groups[START:-1]
    outcome = histories["events"][START + 1 :].astype(float)
    size = int(groups.max()) + 1
    counts, hits = [], []
    for age in [histories["age"][START:-1], histories["age"][START - 512 : -513]]:
        arms = [(age >= 1) & (age <= 3), (age >= 8) & (age <= 31)]
        counts.append([np.bincount(groups[arm], minlength=size) for arm in arms])
        hits.append(
            [np.bincount(groups[arm], weights=outcome[arm], minlength=size) for arm in arms]
        )
    counts, hits = np.array(counts, dtype=float), np.array(hits)
    eligible = np.all(counts >= 10, axis=(0, 1))
    counts, hits = counts[:, :, eligible], hits[:, :, eligible]
    # Both clocks share the conservative harmonic support across all four arms.
    weight = 4 / (1 / counts).sum(axis=(0, 1))
    hazards = hits / counts
    deltas = hazards[:, 1] - hazards[:, 0]
    return np.array(
        [
            (weight * deltas[0]).sum(),
            (weight * deltas[1]).sum(),
            weight.sum(),
            eligible.sum(),
            (weight * hazards[0, 0]).sum(),
            (weight * hazards[0, 1]).sum(),
        ]
    )


def main() -> None:
    """Run a frozen stricter control for the earlier exploratory age effect."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    update_state("followup_common_support_hazard")
    provenance = json.loads((ROOT / "continuation_samples.json").read_text())
    pf = pq.ParquetFile("datasets/valid.parquet")
    measurements = []
    for record in provenance["samples"]["valid"]:
        features, _, _ = read_sequence(pf, record["row_group"])
        histories = history_features(features)
        measurements.append(
            [
                common_strata_hazard(features, histories, ()),
                common_strata_hazard(features, histories, (106, 107)),
            ]
        )
    values = np.array(measurements)
    np.savez(ROOT / "continuation_hazard_control_moments.npz", values=values)
    rng = np.random.default_rng(610105)
    indices = rng.integers(0, len(values), (400, len(values)))
    results = {}
    for index, name in enumerate(["price_rate_common_support", "plus_a2_a3"]):
        totals = values[:, index].sum(0)
        samples = values[indices, index].sum(1)
        differences = (samples[:, 0] - samples[:, 1]) / np.maximum(samples[:, 2], 1e-12)
        results[name] = {
            "real_old_minus_young": float(totals[0] / max(totals[2], 1e-12)),
            "stale_old_minus_young": float(totals[1] / max(totals[2], 1e-12)),
            "real_minus_stale": float((totals[0] - totals[1]) / max(totals[2], 1e-12)),
            "contrast_bootstrap_95": np.quantile(differences, [0.025, 0.975]).tolist(),
            "support_weight": float(totals[2]),
            "common_strata": int(totals[3]),
        }
    tested = results["plus_a2_a3"]
    if tested["common_strata"] < 50:
        verdict = "inconclusive_insufficient_common_support"
    elif tested["real_minus_stale"] <= -0.02 and tested["contrast_bootstrap_95"][1] < 0:
        verdict = "supported_conditional_age_association"
    else:
        verdict = "rejected_or_inconclusive_conditional_age_effect"
    output = {
        "hypothesis": "H3",
        "results": results,
        "verdict": verdict,
        "scope": "same exposed validation sample; next-event labels, not target prediction",
    }
    (ROOT / "continuation_hazard_control.json").write_text(json.dumps(output, indent=2))
    logging.info(json.dumps(output, indent=2))
    update_state("followup_complete_pending_figures_report", {"followup": output})


if __name__ == "__main__":
    main()
