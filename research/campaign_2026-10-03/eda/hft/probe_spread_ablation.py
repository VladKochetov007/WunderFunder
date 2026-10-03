from __future__ import annotations

import json
import logging
import pickle
import traceback

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from continuation_common import (
    OWN,
    PRICE0,
    PRICE1,
    QUANTA,
    ROOT,
    TRADE0,
    load_decoder,
    moments,
    prefix_stats,
    quantum_diagnostic,
    save_json,
    set_phase,
    sha256,
)
from probe_causal_sign_scale import scores_from_stats

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Distinguish current-book information from causal regime offsets."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    old = json.loads((OWN / "sampling_manifest.json").read_text())
    exposed = set(old["prior_valid_row_groups_conservative"])
    exposed.update(old["selected_valid_row_groups"])
    groups = [group for group in range(1400, 1873) if group not in exposed][:12]
    manifest = {
        "selected_valid_row_groups": groups,
        "prior_exposure_overlap": sorted(exposed & set(groups)),
        "scope": "Own-unseen second cohort; campaign-wide exposure unknown.",
        "hypothesis_sha256": sha256(OWN / "HYPOTHESES_H3.json"),
        "scale_model_sha256": sha256(OWN / "warmup_scale_model_v1.pkl"),
        "decoder_sha256": sha256(OWN / "decoder.pkl"),
    }
    save_json("sampling_manifest_h3.json", manifest)
    set_phase("H3_running")
    with (OWN / "warmup_scale_model_v1.pkl").open("rb") as stream:
        scale = pickle.load(stream)
    decoder = load_decoder()["sp"]
    input_columns = PRICE0 + PRICE1 + TRADE0 + ["a2", "a3", "a5", "a6", "a7"]
    columns = input_columns + ["seq_ix", "need_prediction", "is_scored", "t0", "t1"]
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    pooled = {}
    by_regime = {}
    records = []
    for group in groups:
        frame = parquet.read_row_group(group, columns=columns).to_pandas()
        # Predictions are constructed before accessing targets or scoring mask.
        inputs = frame[input_columns]
        prefix = pd.DataFrame([prefix_stats(inputs)])[scale["features"]]
        probabilities = scale["model"].predict_proba(prefix)[0]
        estimated_q = float(probabilities @ QUANTA[scale["model"].classes_])
        prices = inputs[PRICE0].to_numpy(dtype=np.float32)
        spread = decoder.predict_proba(prices) @ decoder.classes_
        predictions = {
            "current_spread": -estimated_q * spread / 2,
            "regime_only": np.full(len(frame), -estimated_q / 2),
            "warmup_spread": np.full(
                len(frame), -estimated_q * spread[:100].mean() / 2
            ),
        }
        targets = frame[["t0", "t1"]].to_numpy(dtype=np.float64)
        q, residual, regime = quantum_diagnostic(targets)
        mask = frame.need_prediction.to_numpy() & frame.is_scored.to_numpy()
        record = {
            "row_group": group,
            "seq_ix": int(frame.seq_ix.iloc[0]),
            "regime": regime,
            "diagnostic_q": q,
            "estimated_q": estimated_q,
            "scored_rows": int(mask.sum()),
            "quantization_max_residual": residual,
            "predicted_class": int(scale["model"].classes_[probabilities.argmax()]),
        }
        for name, prediction in predictions.items():
            stats = np.stack(
                [moments(targets[mask, j], prediction[mask]) for j in range(2)]
            )
            pooled.setdefault(name, np.zeros((2, 6)))
            pooled[name] += stats
            by_regime.setdefault((regime, name), np.zeros((2, 6)))
            by_regime[(regime, name)] += stats
        records.append(record)
        pd.DataFrame(records).to_csv(OWN / "spread_ablation_v1.csv", index=False)
        LOGGER.info("H3 %s", json.dumps(record))
    results = {
        "sequences": len(records),
        "scored_rows": sum(record["scored_rows"] for record in records),
        "pooled_wp": {name: scores_from_stats(stats) for name, stats in pooled.items()},
        "wp_by_diagnostic_regime": {
            str(regime): {
                name: scores_from_stats(stats)
                for (r, name), stats in by_regime.items()
                if r == regime
            }
            for regime in sorted({record["regime"] for record in records})
        },
        "pooled_sufficient_statistics": {
            name: stats.tolist() for name, stats in pooled.items()
        },
        "scale_class_accuracy": sum(
            record["regime"] == record["predicted_class"] for record in records
        )
        / len(records),
    }
    current = results["pooled_wp"]["current_spread"]["mean"]
    best_control = max(
        results["pooled_wp"][name]["mean"] for name in ["regime_only", "warmup_spread"]
    )
    improvements = []
    for scores in results["wp_by_diagnostic_regime"].values():
        controls = [
            scores[name]["mean"]
            for name in ["regime_only", "warmup_spread"]
            if scores[name]["mean"] is not None
        ]
        if controls and scores["current_spread"]["mean"] is not None:
            improvements.append(scores["current_spread"]["mean"] - max(controls))
    results["increment_over_stronger_control"] = current - best_control
    results["median_within_regime_increment"] = float(np.median(improvements))
    results["material_current_book_hypothesis_rejected"] = bool(
        current - best_control <= 0.02 or np.median(improvements) <= 0
    )
    save_json("spread_ablation_summary_v1.json", results)
    manifest["selected_valid_seq_ix"] = [record["seq_ix"] for record in records]
    save_json("sampling_manifest_h3.json", manifest)
    set_phase(
        "H3_complete",
        additional_inspected_sequence_ids=manifest["selected_valid_seq_ix"],
    )
    LOGGER.info("H3 SUMMARY %s", json.dumps(results["pooled_wp"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        (OWN / "spread_ablation_failure_v1.txt").write_text(failure)
        set_phase("H3_failed", last_failure=failure)
        raise
