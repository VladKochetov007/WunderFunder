from __future__ import annotations

import json
import logging
import pickle
import traceback

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from continuation_common import (
    OWN,
    PRICE0,
    PRICE1,
    QUANTA,
    ROOT,
    TRADE0,
    correlation,
    decode_prices,
    ema,
    load_decoder,
    moments,
    prefix_stats,
    quantum_diagnostic,
    save_json,
    set_phase,
    sha256,
)

LOGGER = logging.getLogger(__name__)


def fit_scale_model() -> tuple[HistGradientBoostingClassifier, list[str], float]:
    """Fit only the frozen saved TRAIN warm-up prefixes, never VALID data."""
    table = pd.read_csv(OWN / "regime_stats.csv")
    train = table[(table.split == "train") & (table["T"] == 100)]
    features = [name for name in train if name not in ["y", "g", "split"]]
    if len(train) != 300 or train.g.nunique() != 300:
        raise ValueError("The frozen TRAIN-prefix cohort differs from 300 sequences")
    model = HistGradientBoostingClassifier(
        max_iter=80, max_leaf_nodes=7, l2_regularization=1, random_state=19
    ).fit(train[features], train.y)
    prior_q = float(QUANTA[train.y.to_numpy(dtype=int)].mean())
    with (OWN / "warmup_scale_model_v1.pkl").open("wb") as stream:
        pickle.dump({"model": model, "features": features, "prior_q": prior_q}, stream)
    save_json(
        "warmup_scale_training_v1.json",
        {
            "train_row_groups": train.g.astype(int).tolist(),
            "feature_order": features,
            "rows_per_prefix": 100,
            "prior_q": prior_q,
            "parameters": model.get_params(),
            "training_accuracy_diagnostic_only": float(
                model.score(train[features], train.y)
            ),
            "source_hash": sha256(OWN / "regime_stats.csv"),
        },
    )
    return model, features, prior_q


def predict_inputs(
    inputs: pd.DataFrame,
    scale_model: HistGradientBoostingClassifier,
    features: list[str],
    prior_q: float,
    decoder: dict[str, object],
) -> tuple[dict[str, np.ndarray], float, int, float]:
    """Predict from inputs alone and verify full/prefix causality exactly."""
    prefix = pd.DataFrame([prefix_stats(inputs)])[features]
    probabilities = scale_model.predict_proba(prefix)[0]
    estimated_q = float(probabilities @ QUANTA[scale_model.classes_])
    predicted_class = int(scale_model.classes_[probabilities.argmax()])
    prices = inputs[PRICE0].to_numpy(dtype=np.float32)
    ask, bid, spread = decode_prices(prices, decoder)
    midpoint = (ask + bid) / 2
    prefix_length = 512
    short_ask, short_bid, short_spread = decode_prices(prices[:prefix_length], decoder)
    errors = [
        np.max(np.abs(ask[:prefix_length] - short_ask)),
        np.max(np.abs(bid[:prefix_length] - short_bid)),
        np.max(np.abs(spread[:prefix_length] - short_spread)),
    ]
    variants = {}
    for scale_name, scale in [("estimated", estimated_q), ("fixed", prior_q)]:
        variants[f"spread_{scale_name}"] = np.c_[-spread / 2, -spread / 2] * scale
        for span in [20, 100]:
            displacement = midpoint - ema(midpoint, span)
            short_displacement = (short_ask + short_bid) / 2
            short_displacement -= ema(short_displacement.copy(), span)
            errors.append(
                np.max(np.abs(displacement[:prefix_length] - short_displacement))
            )
            momentum = np.c_[displacement, -displacement] * scale
            variants[f"momentum{span}_{scale_name}"] = momentum
            variants[f"reversion{span}_{scale_name}"] = -momentum
    error = float(max(errors))
    if error > 1e-10:
        raise ValueError(f"Decoder/EMA prefix causality discrepancy {error}")
    if not all(np.isfinite(values).all() for values in variants.values()):
        raise ValueError("A causal prediction is not finite")
    return variants, estimated_q, predicted_class, error


def scores_from_stats(stats: np.ndarray) -> dict[str, float | None]:
    """Summarize the two target correlations without averaging sequence scores."""
    t0 = correlation(stats[0])
    t1 = correlation(stats[1])
    return {
        "t0": t0,
        "t1": t1,
        "mean": (t0 + t1) / 2 if t0 is not None and t1 is not None else None,
    }


def main() -> None:
    """Run frozen sign/scale variants on exactly the H1 own-unseen cohort."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = json.loads((OWN / "sampling_manifest.json").read_text())
    set_phase("H2_running")
    model, features, prior_q = fit_scale_model()
    decoder = load_decoder()
    input_columns = PRICE0 + PRICE1 + TRADE0 + ["a2", "a3", "a5", "a6", "a7"]
    columns = input_columns + [
        "seq_ix",
        "step_in_seq",
        "need_prediction",
        "is_scored",
        "t0",
        "t1",
    ]
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    pooled = {}
    by_regime = {}
    records = []
    for group in manifest["selected_valid_row_groups"]:
        frame = parquet.read_row_group(group, columns=columns).to_pandas()
        predictions, estimated_q, predicted_class, causal_error = predict_inputs(
            frame[input_columns], model, features, prior_q, decoder
        )
        targets = frame[["t0", "t1"]].to_numpy(dtype=np.float64)
        q, residual, regime = quantum_diagnostic(targets)
        mask = frame.need_prediction.to_numpy() & frame.is_scored.to_numpy()
        record = {
            "row_group": group,
            "seq_ix": int(frame.seq_ix.iloc[0]),
            "diagnostic_q": q,
            "regime": regime,
            "estimated_q": estimated_q,
            "predicted_class": predicted_class,
            "class_correct": predicted_class == regime,
            "causal_prefix_max_error": causal_error,
            "quantization_max_residual": residual,
            "scored_rows": int(mask.sum()),
        }
        for name, prediction in predictions.items():
            stats = np.stack(
                [moments(targets[mask, j], prediction[mask, j]) for j in range(2)]
            )
            pooled.setdefault(name, np.zeros((2, 6)))
            pooled[name] += stats
            key = (regime, name)
            by_regime.setdefault(key, np.zeros((2, 6)))
            by_regime[key] += stats
            for label, score in scores_from_stats(stats).items():
                record[f"{name}_{label}"] = score
        records.append(record)
        pd.DataFrame(records).to_csv(OWN / "causal_sign_scale_v1.csv", index=False)
        LOGGER.info(
            "H2 group=%d seq_ix=%d true_q=%.5f estimated_q=%.5f class=%d",
            group,
            record["seq_ix"],
            q,
            estimated_q,
            predicted_class,
        )
    table = pd.DataFrame(records)
    results = {
        "sequences": len(table),
        "scored_rows": int(table.scored_rows.sum()),
        "warmup_class_accuracy": float(table.class_correct.mean()),
        "prior_q": prior_q,
        "causal_prefix_max_error": float(table.causal_prefix_max_error.max()),
        "pooled_wp": {name: scores_from_stats(stats) for name, stats in pooled.items()},
        "wp_by_diagnostic_regime": {
            str(regime): {
                name: scores_from_stats(stats)
                for (r, name), stats in by_regime.items()
                if r == regime
            }
            for regime in sorted(table.regime.unique())
        },
        "quantum_confusion": pd.crosstab(table.regime, table.predicted_class).to_dict(),
        "pooled_sufficient_statistics": {
            name: stats.tolist() for name, stats in pooled.items()
        },
        "claim_limit": "Own-disjoint, predeclared exploratory score; no baseline comparison, no campaign-independence claim and no row-level uncertainty.",
    }
    results["quantum_confusion"] = {
        str(key): {str(k): int(v) for k, v in value.items()}
        for key, value in results["quantum_confusion"].items()
    }
    save_json("causal_sign_scale_summary_v1.json", results)
    set_phase("H2_complete", last_failure=None)
    LOGGER.info("H2 SUMMARY %s", json.dumps(results["pooled_wp"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        (OWN / "causal_sign_scale_failure_v1.txt").write_text(failure)
        set_phase("H2_failed", last_failure=failure)
        raise
