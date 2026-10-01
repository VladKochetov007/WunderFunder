from __future__ import annotations

import argparse
import json
import pickle
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from audit import ROOT, digest_file, event_rows, price_keys, read_sequence, target_quantum


def main() -> None:
    """Compare historical auxiliary event labels on the already frozen TRAIN sample."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start_wall, start_cpu = time.monotonic(), time.process_time()
    directory = Path(__file__).parent
    manifest = json.loads((directory / "preregistration_v2.json").read_text())
    cache = ROOT / "cache/class3_moves_full.pkl"
    with cache.open("rb") as handle:
        historical_moves, historical_spreads = pickle.load(handle)
    moves = {
        (
            np.asarray(left, dtype=np.float32).tobytes(),
            np.asarray(right, dtype=np.float32).tobytes(),
        ): move
        for (left, right), move in historical_moves.items()
    }
    spreads = {
        np.asarray(key, dtype=np.float32).tobytes(): value
        for key, value in historical_spreads.items()
    }
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    records = []
    totals = {horizon: Counter() for horizon in (8, 24)}
    for group in manifest["fit_train_groups"]:
        prices, targets, _ = read_sequence(source, group, has_mask=False)
        regime, _, _ = target_quantum(targets)
        if regime != 3:
            continue
        events = event_rows(prices)
        keys = price_keys(prices[events])
        delta = np.zeros((len(events), 2), dtype=np.int64)
        unknown = np.zeros(len(events), dtype=bool)
        for index in range(1, len(events)):
            move = moves.get((keys[index - 1], keys[index]))
            if move is None:
                unknown[index] = True
            else:
                delta[index] = move
        ask = np.cumsum(delta[:, 0])
        bid = np.zeros(len(events), dtype=np.int64)
        for index, key in enumerate(keys):
            spread = spreads.get(key)
            if spread is None:
                unknown[index] = True
                bid[index] = bid[index - 1] + delta[index, 1] if index else ask[index] - 1
            else:
                bid[index] = ask[index] - spread
        midpoint = (ask + bid) / 2
        quotes = np.flatnonzero((np.diff(ask) != 0) | (np.diff(bid) != 0)) + 1
        prefix = np.r_[0.0, np.cumsum(midpoint)]
        quote_prefix = np.r_[0.0, np.cumsum(midpoint[quotes])]
        unknown_prefix = np.cumsum(unknown)
        current = np.arange(len(events))
        end_rows = np.r_[events[1:], len(prices)]
        row_weight = np.maximum(0, end_rows - np.maximum(events, 99))
        comparisons = {}
        for horizon in (8, 24):
            next_quote = np.searchsorted(quotes, current, side="right")
            valid = (current + horizon < len(events)) & (next_quote + horizon <= len(quotes))
            selected = current[valid]
            if not len(selected):
                comparisons[str(horizon)] = {"eligible_required_rows": 0}
                continue
            q_start = next_quote[selected]
            q_end = quotes[q_start + horizon - 1]
            selected_known = unknown_prefix[q_end] == unknown_prefix[selected]
            selected = selected[selected_known]
            q_start, q_end = q_start[selected_known], q_end[selected_known]
            weight = row_weight[selected]
            raw_mean = (prefix[selected + horizon + 1] - prefix[selected + 1]) / horizon
            quote_mean = (quote_prefix[q_start + horizon] - quote_prefix[q_start]) / horizon
            raw_label = np.clip(0.65687 * (raw_mean - midpoint[selected]), -2, 2)
            quote_label = np.clip(0.65687 * (quote_mean - midpoint[selected]), -2, 2)
            absolute_error = np.abs(raw_label - quote_label)
            raw_span = events[selected + horizon] - events[selected]
            quote_span = events[q_end] - events[selected]
            result = {
                "eligible_required_rows": int(weight.sum()),
                "different_label_required_rows": int(weight[absolute_error > 1e-3].sum()),
                "absolute_difference_weighted_sum": float(weight @ absolute_error),
                "squared_difference_weighted_sum": float(weight @ (absolute_error**2)),
                "code_horizon_row_span_weighted_sum": float(weight @ raw_span),
                "quote_horizon_row_span_weighted_sum": float(weight @ quote_span),
            }
            comparisons[str(horizon)] = result
            totals[horizon].update(result)
        records.append(
            {
                "train_row_group": group,
                "representation_code_changes": len(events) - 1,
                "historically_decoded_quote_changes": len(quotes),
                "unknown_transition_or_spread_events": int(unknown.sum()),
                "comparisons": comparisons,
            }
        )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "frozen_train_groups_sha256": digest_file(directory / "preregistration_v2.json"),
        "historical_decoder_sha256": digest_file(cache),
        "historical_auxiliary_script_sha256": digest_file(ROOT / "scripts/cache_quote_path.py"),
        "selected_train_class3_sequences": len(records),
        "representation_code_changes": sum(r["representation_code_changes"] for r in records),
        "historically_decoded_quote_changes": sum(
            r["historically_decoded_quote_changes"] for r in records
        ),
        "records": records,
        "totals": {str(h): dict(value) for h, value in totals.items()},
        "resources": {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.monotonic() - start_wall,
            "peak_rss_kib": usage.ru_maxrss,
            "exit_code": 0,
        },
        "interpretation": [
            "TRAIN-only clock audit; no validation source was opened.",
            "Historical decoder uses label-derived TRAIN tables and is not asserted to identify physical quotes.",
            "Code-event and nonzero decoded-quote-event clocks differ even when all mappings are known.",
            "Differences diagnose auxiliary-label semantics, not a measured predictive gain or causal explanation for a prior run's score.",
            "Unknown intervals were excluded; no competition labels/cache/model/feature implementation was changed.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
