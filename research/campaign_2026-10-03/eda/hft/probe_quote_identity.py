from __future__ import annotations

import json
import logging
import traceback

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from continuation_common import (
    OWN,
    PRICE0,
    ROOT,
    TRADE0,
    change_mask,
    quantum_diagnostic,
    sampling_manifest,
    save_json,
    set_phase,
)

LOGGER = logging.getLogger(__name__)


def modal_purity(groups: np.ndarray, outcomes: np.ndarray) -> tuple[float | None, int]:
    """Measure modal outcome agreement only for groups observed >=5 times."""
    counts = np.bincount(groups)
    eligible = counts >= 5
    denominator = int(counts[eligible].sum())
    if not denominator:
        return None, 0
    pairs, frequencies = np.unique(np.c_[groups, outcomes], axis=0, return_counts=True)
    modes = np.zeros(len(counts), dtype=np.int64)
    np.maximum.at(modes, pairs[:, 0], frequencies)
    return float(modes[eligible].sum() / denominator), denominator


def permute_within(
    outcomes: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Keep sequence/event-mask marginals while removing state association."""
    shuffled = outcomes.copy()
    for value in np.unique(strata):
        rows = np.flatnonzero(strata == value)
        shuffled[rows] = rng.permutation(shuffled[rows])
    return shuffled


def analyze_sequence(frame: pd.DataFrame, group: int) -> tuple[dict, dict]:
    """Falsify physical identities and save small raw-data witnesses."""
    prices = frame[PRICE0].to_numpy(dtype=np.float32)
    targets = frame[["t0", "t1"]].to_numpy(dtype=np.float64)
    q, residual, regime = quantum_diagnostic(targets)
    ticks = np.rint(targets / q).astype(np.int64)
    implied_spread = -ticks.sum(axis=1)
    _, states = np.unique(prices, axis=0, return_inverse=True)
    states = states.ravel()
    _, spread_codes = np.unique(implied_spread, return_inverse=True)
    state_purity, state_coverage = modal_purity(states, spread_codes)
    bid = change_mask(prices[:, :11])
    ask = change_mask(prices[:, 11:])
    side = bid.astype(int) + 2 * ask.astype(int)
    events = np.flatnonzero(side)
    jumps = np.diff(ticks, axis=0, prepend=ticks[:1])
    _, transitions = np.unique(
        np.c_[states[events - 1], states[events]], axis=0, return_inverse=True
    )
    _, jump_codes = np.unique(jumps[events], axis=0, return_inverse=True)
    transitions = transitions.ravel()
    jump_codes = jump_codes.ravel()
    transition_purity, transition_coverage = modal_purity(transitions, jump_codes)
    trade = change_mask(frame[TRADE0].to_numpy())
    rng = np.random.default_rng(19000 + group)
    state_controls = []
    transition_controls = []
    for _ in range(20):
        state_control, _ = modal_purity(
            states, permute_within(spread_codes, trade, rng)
        )
        transition_control, _ = modal_purity(
            transitions, permute_within(jump_codes, side[events], rng)
        )
        state_controls.append(state_control)
        transition_controls.append(transition_control)
    scored = frame.is_scored.to_numpy() & frame.need_prediction.to_numpy()
    none = side == 0
    none[0] = False
    bid_only = side == 1
    ask_only = side == 2
    record = {
        "row_group": group,
        "seq_ix": int(frame.seq_ix.iloc[0]),
        "q": q,
        "regime": regime,
        "quantization_max_residual": residual,
        "rows": len(frame),
        "scored_rows": int(scored.sum()),
        "negative_spread_fraction": float((implied_spread < 0).mean()),
        "negative_spread_scored_fraction": float((implied_spread[scored] < 0).mean()),
        "state_count": len(np.unique(states)),
        "state_purity": state_purity,
        "state_coverage_rows": state_coverage,
        "state_control_mean": float(np.mean(state_controls)),
        "events": len(events),
        "transition_purity": transition_purity,
        "transition_coverage_events": transition_coverage,
        "transition_control_mean": float(np.mean(transition_controls)),
        "label_change_without_price": int(np.any(jumps[none] != 0, axis=1).sum()),
        "no_price_rows": int(none.sum()),
        "bid_only_opposite_target_change": float((jumps[bid_only, 0] != 0).mean())
        if bid_only.any()
        else None,
        "ask_only_opposite_target_change": float((jumps[ask_only, 1] != 0).mean())
        if ask_only.any()
        else None,
    }
    witness = {"row_group": group, "seq_ix": record["seq_ix"]}
    negative = np.flatnonzero(implied_spread < 0)
    if len(negative):
        row = int(negative[0])
        witness["negative_spread"] = {
            "step": int(frame.step_in_seq.iloc[row]),
            "targets": targets[row].tolist(),
            "price_vector": prices[row].tolist(),
            "implied_spread_ticks": int(implied_spread[row]),
        }
    for state in np.unique(states):
        rows = np.flatnonzero(states == state)
        values = implied_spread[rows]
        if len(rows) >= 5 and values.min() != values.max():
            low = int(rows[np.argmin(values)])
            high = int(rows[np.argmax(values)])
            witness["identical_state_different_sum"] = {
                "steps": [int(frame.step_in_seq.iloc[r]) for r in [low, high]],
                "targets": targets[[low, high]].tolist(),
                "price_vector": prices[low].tolist(),
                "implied_spread_ticks": implied_spread[[low, high]].tolist(),
            }
            break
    return record, witness


def main() -> None:
    """Stream the frozen 20-sequence cohort and preserve every probe output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = sampling_manifest()
    save_json("sampling_manifest.json", manifest)
    set_phase("H1_running", sampling_manifest="sampling_manifest.json")
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    columns = (
        PRICE0
        + TRADE0
        + ["t0", "t1", "seq_ix", "step_in_seq", "is_scored", "need_prediction"]
    )
    records = []
    witnesses = []
    for group in manifest["selected_valid_row_groups"]:
        frame = parquet.read_row_group(group, columns=columns).to_pandas()
        record, witness = analyze_sequence(frame, group)
        records.append(record)
        witnesses.append(witness)
        pd.DataFrame(records).to_csv(OWN / "quote_identity_v1.csv", index=False)
        save_json("quote_identity_witnesses_v1.json", witnesses)
        set_phase("H1_running", inspected_sequence_ids=[r["seq_ix"] for r in records])
        LOGGER.info("H1 %s", json.dumps(record))
    table = pd.DataFrame(records)
    state_rows = table.state_coverage_rows.to_numpy()
    event_rows = table.transition_coverage_events.to_numpy()
    summary = {
        "sequences": len(table),
        "rows": int(table.rows.sum()),
        "scored_rows": int(table.scored_rows.sum()),
        "negative_spread_fraction": float(table.negative_spread_fraction.mean()),
        "state_purity": float(np.average(table.state_purity, weights=state_rows)),
        "state_control": float(
            np.average(table.state_control_mean, weights=state_rows)
        ),
        "transition_purity": float(
            np.average(table.transition_purity, weights=event_rows)
        ),
        "transition_control": float(
            np.average(table.transition_control_mean, weights=event_rows)
        ),
        "transition_coverage_events": int(event_rows.sum()),
        "label_change_without_price": int(table.label_change_without_price.sum()),
        "no_price_rows": int(table.no_price_rows.sum()),
        "shared_reference_spread_rejected_sequences": int(
            (
                (table.negative_spread_fraction > 0.001) | (table.state_purity < 0.99)
            ).sum()
        ),
        "exact_transition_decoder_rejected_sequences": int(
            (table.transition_purity < 0.99).sum()
        ),
    }
    save_json("quote_identity_summary_v1.json", summary)
    manifest["selected_valid_seq_ix"] = table.seq_ix.tolist()
    manifest["dataset_metadata"] = {
        "path": "datasets/valid.parquet",
        "num_rows": parquet.metadata.num_rows,
        "num_row_groups": parquet.num_row_groups,
    }
    save_json("sampling_manifest.json", manifest)
    set_phase("H1_complete", last_failure=None)
    LOGGER.info("H1 SUMMARY %s", json.dumps(summary))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        (OWN / "quote_identity_failure_v1.txt").write_text(failure)
        set_phase("H1_failed", last_failure=failure)
        raise
