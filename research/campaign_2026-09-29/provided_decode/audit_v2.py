from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from audit import (
    PRICE_COLUMNS,
    ROOT,
    RULES,
    TARGET_QUANTA,
    common_law,
    digest_file,
    event_rows,
    invert_pair,
    price_keys,
    read_sequence,
    score_record,
    source_identity,
    target_quantum,
)

LOGGER = logging.getLogger(__name__)
QUOTE_HORIZONS = (2, 4, 8)
ROW_HORIZONS = (50, 200, 1000)
VERSION = "provided-constrained-transport-v2"


def frozen_sample(fit_count: int, evaluation_count: int) -> dict[str, object]:
    """Choose a new component panel before reading any of its labels."""
    chain_path = ROOT / "cache" / "chain_folds.npz"
    with np.load(chain_path) as data:
        split, group, chain, fold = (data[key] for key in ("split", "row_group", "chain", "fold"))
        excluded = set(chain[(split == "valid") & np.isin(fold, [0, 1])].tolist())
        train = [
            (int(g), int(c))
            for g, c in zip(group[split == "train"], chain[split == "train"], strict=True)
            if int(c) not in excluded
        ]
        candidates = [
            (int(g), int(c))
            for g, c in zip(
                group[(split == "valid") & (fold == 1)],
                chain[(split == "valid") & (fold == 1)],
                strict=True,
            )
        ]

    def order(pair: tuple[int, int]) -> bytes:
        return hashlib.sha256(f"provided-audit-v2:{pair[0]}:{pair[1]}".encode()).digest()

    train = sorted(train, key=order)[:fit_count]
    panel = []
    seen = set()
    for g, c in sorted(candidates, key=order):
        if c not in seen:
            panel.append((g, c))
            seen.add(c)
        if len(panel) == evaluation_count:
            break
    previous = json.loads((Path(__file__).parent / "sample_manifest.json").read_text())
    if set(previous["evaluation_components"]) & seen:
        raise ValueError("Second panel reuses first-panel evaluation components")
    if {c for _, c in train} & seen:
        raise ValueError("Train and evaluation share a component")
    return {
        "version": VERSION,
        "chain_metadata_sha256": digest_file(chain_path),
        "fit_train_groups": [g for g, _ in train],
        "evaluation_valid_groups": [g for g, _ in panel],
        "fit_components": [c for _, c in train],
        "evaluation_components": [c for _, c in panel],
        "selected_without_targets_or_masks": True,
        "quote_horizons": QUOTE_HORIZONS,
        "row_horizons": ROW_HORIZONS,
        "rules": RULES,
        "minimum_spread_support": 3,
        "minimum_spread_confidence": 0.9,
        "minimum_move_support": 3,
        "minimum_move_confidence": 0.8,
        "coverage_gate": {
            "minimum_union_quote4_eligible_required_rows": 1000,
            "minimum_union_quote4_required_row_fraction": 0.05,
            "evaluation_labels_read_only_after_gate_passes": True,
        },
        "quote_clock": "Nonzero known decoded ask/bid moves, excluding pure representation-code changes. Unknown intervening moves invalidate the entire interval.",
        "row_clock": "Mean/end over the next N publication rows; not identified physical time.",
        "rounding": "nearest even origin parity0; separately frozen parity1 and half-up",
        "spread_is_label_supervised": True,
    }


def fit_constrained(
    source: pq.ParquetFile, groups: list[int]
) -> tuple[
    dict[int, dict[tuple[bytes, bytes], tuple[int, int]]],
    dict[int, dict[bytes, int]],
    dict[str, object],
]:
    """Fit TRAIN-only transport with two non-clamped target identities."""
    spread_counts = defaultdict(lambda: defaultdict(Counter))
    regimes: Counter[int] = Counter()
    for group in groups:
        prices, targets, _ = read_sequence(source, group, has_mask=False)
        regime, _, ticks = target_quantum(targets)
        regimes[regime] += 1
        events = event_rows(prices)
        implied, _ = invert_pair(ticks)
        for key, start, stop in zip(
            price_keys(prices[events]), events, np.r_[events[1:], len(prices)], strict=True
        ):
            spread_counts[regime][key].update(implied[start:stop:7].tolist())
    spreads = {regime: {} for regime in range(4)}
    for regime, counters in spread_counts.items():
        for key, counter in counters.items():
            value, support = counter.most_common(1)[0]
            if value >= 1 and support >= 3 and support / counter.total() >= 0.9:
                spreads[regime][key] = value
    votes = defaultdict(lambda: defaultdict(Counter))
    side_votes = {"long": 0, "short": 0, "both_available": 0, "both_disagree": 0}
    observed_transitions: Counter[int] = Counter()
    for group in groups:
        prices, targets, _ = read_sequence(source, group, has_mask=False)
        regime, _, ticks = target_quantum(targets)
        events = event_rows(prices)
        keys = price_keys(prices[events])
        for index in range(1, len(events)):
            observed_transitions[regime] += 1
            previous, current = keys[index - 1], keys[index]
            before, after = spreads[regime].get(previous), spreads[regime].get(current)
            if before is None or after is None:
                continue
            row = events[index]
            old, new = ticks[row - 1], ticks[row]
            candidates = []
            if old[0] > -before - 2 and new[0] > -after - 2:
                candidates.append(-int(new[0] - old[0]))
                side_votes["long"] += 1
            if old[1] > -before - 2 and new[1] > -after - 2:
                candidates.append(int(new[1] - old[1]) + after - before)
                side_votes["short"] += 1
            if len(candidates) == 2:
                side_votes["both_available"] += 1
                side_votes["both_disagree"] += candidates[0] != candidates[1]
            votes[regime][(previous, current)].update(candidates)
    moves = {regime: {} for regime in range(4)}
    fit_stats = {}
    for regime in range(4):
        for key, counter in votes[regime].items():
            if not counter:
                continue
            ask_delta, support = counter.most_common(1)[0]
            if support >= 3 and support / counter.total() >= 0.8:
                spread_delta = spreads[regime][key[1]] - spreads[regime][key[0]]
                moves[regime][key] = (ask_delta, ask_delta - spread_delta)
        fit_stats[str(regime)] = {
            "fit_sequences": regimes[regime],
            "observed_transition_occurrences": observed_transitions[regime],
            "observed_spread_codes": len(spread_counts[regime]),
            "accepted_spread_codes": len(spreads[regime]),
            "voted_transition_codes": len(votes[regime]),
            "accepted_transition_codes": len(moves[regime]),
            "accepted_nonzero_quote_transition_codes": sum(
                a != 0 or b != 0 for a, b in moves[regime].values()
            ),
        }
    return (
        moves,
        spreads,
        {
            "by_regime": fit_stats,
            "target_identity_votes": side_votes,
            "identity_agreement_is_partly_algebraic": True,
            "not_a_physical_quote_identity": "TRAIN target inversion and common-F-stability assumptions supply the coordinate scale/spread; these are not recovered from features alone.",
        },
    )


def decoded_path(
    prices: np.ndarray, moves: dict[tuple[bytes, bytes], tuple[int, int]], spreads: dict[bytes, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Represent known increments while marking every missing transition."""
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
    spread = np.array([spreads.get(key, -1) for key in keys])
    quote_indices = np.flatnonzero(np.any(delta != 0, axis=1) & ~unknown)
    return events, delta, unknown, spread, quote_indices, np.cumsum(delta, axis=0)


def quote_coverage(
    prices: np.ndarray,
    moves: dict[tuple[bytes, bytes], tuple[int, int]],
    spreads: dict[bytes, int],
    horizon: int,
) -> np.ndarray:
    """Check feature-only horizon availability without touching evaluation labels."""
    events, _, unknown, spread, quote, _ = decoded_path(prices, moves, spreads)
    current = np.searchsorted(events, np.arange(len(prices)), side="right") - 1
    future_index = np.searchsorted(quote, current, side="right") + horizon - 1
    if not len(quote):
        return np.zeros(len(prices), dtype=bool)
    in_range = future_index < len(quote)
    end = quote[np.minimum(future_index, len(quote) - 1)]
    gaps = np.cumsum(unknown)
    return in_range & (spread[current] > 0) & (gaps[end] == gaps[current])


def predictions(
    prices: np.ndarray, moves: dict[tuple[bytes, bytes], tuple[int, int]], spreads: dict[bytes, int]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, dict[str, object]]:
    """Compare frozen quote-clock references and separate barrier/touch exits."""
    events, delta, unknown, spread, quote, cumulative = decoded_path(prices, moves, spreads)
    current = np.searchsorted(events, np.arange(len(prices)), side="right") - 1
    gaps = np.cumsum(unknown)
    rows = np.arange(len(prices))
    row_path = cumulative[current]
    row_prefix = np.vstack((np.zeros(2), np.cumsum(row_path, axis=0)))
    result = {}
    tie_counts = {}
    for clock, horizons in (("quote", QUOTE_HORIZONS), ("row", ROW_HORIZONS)):
        for horizon in horizons:
            if clock == "quote":
                future_index = np.searchsorted(quote, current, side="right") + horizon - 1
                in_range = future_index < len(quote)
                end = quote[np.minimum(future_index, len(quote) - 1)] if len(quote) else current
            else:
                in_range = rows + horizon < len(prices)
                end = np.searchsorted(events, rows + horizon, side="right") - 1
                end = np.minimum(end, len(events) - 1)
            eligible = in_range & (spread[current] > 0) & (gaps[end] == gaps[current])
            alternatives = {
                name: np.zeros((len(prices), 2), dtype=np.int64)
                for name in RULES + ("common_mean_parity1", "common_mean_half_up")
            }
            ties = 0
            for origin, (start, stop) in enumerate(
                zip(events, np.r_[events[1:], len(prices)], strict=True)
            ):
                for last in np.unique(end[start:stop]):
                    selected = np.arange(start, stop)[
                        (end[start:stop] == last) & eligible[start:stop]
                    ]
                    if not len(selected):
                        continue
                    width = int(spread[origin])
                    interval_ask = cumulative[origin + 1 : last + 1, 0] - cumulative[origin, 0]
                    interval_bid = (
                        cumulative[origin + 1 : last + 1, 1] - cumulative[origin, 1] - width
                    )
                    if not len(interval_ask):
                        interval_ask, interval_bid = np.array([0]), np.array([-width])
                    if np.any(interval_ask - interval_bid < 1):
                        eligible[selected] = False
                        continue
                    if clock == "quote":
                        future_quotes = quote[(quote > origin) & (quote <= last)]
                        mean_ask = np.full(
                            len(selected),
                            np.mean(cumulative[future_quotes, 0] - cumulative[origin, 0]),
                        )
                        mean_bid = np.full(
                            len(selected),
                            np.mean(cumulative[future_quotes, 1] - cumulative[origin, 1] - width),
                        )
                    else:
                        next_rows = selected + horizon + 1
                        row_mean = (
                            row_prefix[next_rows] - row_prefix[selected + 1]
                        ) / horizon - cumulative[origin]
                        mean_ask, mean_bid = row_mean[:, 0], row_mean[:, 1] - width
                    mean_mid = (mean_ask + mean_bid) / 2
                    ties += int(np.count_nonzero(mean_mid % 1 == 0.5))
                    for name, future in (
                        ("common_mean", np.rint(mean_mid)),
                        ("common_mean_parity1", np.rint(mean_mid + 1) - 1),
                        ("common_mean_half_up", np.floor(mean_mid + 0.5)),
                        (
                            "common_end",
                            np.full(
                                len(selected), np.rint((interval_ask[-1] + interval_bid[-1]) / 2)
                            ),
                        ),
                    ):
                        alternatives[name][selected] = common_law(
                            future, np.full(len(selected), -width), np.zeros(len(selected))
                        ).astype(np.int64)
                    stop_loss = -width - 2
                    stopped_long = np.any(interval_bid <= stop_loss)
                    stopped_short = np.any(-width - interval_ask <= stop_loss)
                    alternatives["independent_stop_end"][selected] = (
                        stop_loss if stopped_long else int(interval_bid[-1]),
                        stop_loss if stopped_short else int(-width - interval_ask[-1]),
                    )
                    alternatives["independent_stop_mean"][selected, 0] = (
                        stop_loss if stopped_long else np.rint(mean_bid).astype(np.int64)
                    )
                    alternatives["independent_stop_mean"][selected, 1] = (
                        stop_loss if stopped_short else np.rint(-width - mean_ask).astype(np.int64)
                    )
            for name, prediction in alternatives.items():
                result[f"{clock}{horizon}/{name}"] = (prediction, eligible.copy())
            tie_counts[f"{clock}{horizon}"] = ties
    is_quote_event = np.zeros(len(prices), dtype=bool)
    is_quote_event[events[quote]] = True
    return (
        result,
        is_quote_event,
        {
            "representation_code_events": len(events),
            "known_decoded_quote_changes": len(quote),
            "unknown_transition_occurrences": int(unknown.sum()),
            "known_transition_occurrences": int(len(events) - 1 - unknown.sum()),
            "events_without_confident_train_spread": int(np.sum(spread < 1)),
            "rounding_tie_rows": tie_counts,
        },
    )


def main() -> None:
    """Run the preregistered second supplied-only diagnostic and coverage gate."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--fit-sequences", type=int, default=96)
    parser.add_argument("--evaluation-sequences", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_wall, start_cpu = time.monotonic(), time.process_time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = frozen_sample(args.fit_sequences, args.evaluation_sequences)
    (args.output.parent / "preregistration_v2.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    train_path, valid_path = ROOT / "datasets/train.parquet", ROOT / "datasets/valid.parquet"
    train, valid = pq.ParquetFile(train_path), pq.ParquetFile(valid_path)
    LOGGER.info("Fitting constrained TRAIN-only transport")
    moves, spreads, fit = fit_constrained(train, manifest["fit_train_groups"])
    gate_records = []
    eligible_union = total_required = 0
    for group in manifest["evaluation_valid_groups"]:
        table = valid.read_row_group(group, columns=PRICE_COLUMNS, use_threads=False)
        prices = np.stack([table.column(name).to_numpy() for name in PRICE_COLUMNS], axis=1).astype(
            np.float32
        )
        per_regime = {
            regime: quote_coverage(prices, moves[regime], spreads[regime], 4) for regime in range(4)
        }
        union = np.logical_or.reduce(list(per_regime.values()))
        required = np.arange(len(prices)) >= 99
        eligible_union += int(np.sum(union & required))
        total_required += int(required.sum())
        gate_records.append(
            {
                "row_group": group,
                "per_regime_quote4_eligible_required_rows": {
                    str(k): int(np.sum(v & required)) for k, v in per_regime.items()
                },
                "union_eligible_required_rows": int(np.sum(union & required)),
            }
        )
    fraction = eligible_union / max(1, total_required)
    gate_passes = eligible_union >= 1000 and fraction >= 0.05
    LOGGER.info(
        "Feature-only coverage gate: eligible=%s fraction=%s passes=%s",
        eligible_union,
        fraction,
        gate_passes,
    )
    records = []
    totals = defaultdict(Counter)
    if gate_passes:
        for group in manifest["evaluation_valid_groups"]:
            LOGGER.info("Grading new component row group %s", group)
            prices, targets, scored = read_sequence(valid, group, has_mask=True)
            regime, error, ticks = target_quantum(targets)
            alternatives, is_quote_event, coverage = predictions(
                prices, moves[regime], spreads[regime]
            )
            measurements = {}
            for name, (prediction, eligible) in alternatives.items():
                value = score_record(
                    prediction, ticks, eligible, scored, is_quote_event, TARGET_QUANTA[regime]
                )
                measurements[name] = value
                totals[(regime, name)].update(value)
            records.append(
                {
                    "row_group": group,
                    "regime_from_targets_diagnostic_only": regime,
                    "target_lattice_max_error": error,
                    "coverage": coverage,
                    "rules": measurements,
                }
            )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    output = {
        "version": VERSION,
        "command": sys.argv,
        "preregistration": manifest,
        "fit": fit,
        "feature_only_gate": {
            "passes": gate_passes,
            "eligible_required_rows": eligible_union,
            "total_required_rows": total_required,
            "fraction": fraction,
            "records": gate_records,
            "evaluation_labels_read_before_gate": False,
        },
        "evaluation_records": records,
        "totals_by_regime_and_rule": {
            f"class{regime}/{name}": dict(value) for (regime, name), value in totals.items()
        },
        "resources": {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.monotonic() - start_wall,
            "peak_rss_kib": usage.ru_maxrss,
            "exit_code": 0,
        },
        "source_identities": {
            "train": source_identity(train_path),
            "valid": source_identity(valid_path),
            "script_sha256": digest_file(Path(__file__)),
            "shared_audit_module_sha256": digest_file(Path(__file__).with_name("audit.py")),
        },
        "boundaries": [
            "All mappings use supplied TRAIN only, excluding both v1 and v2 evaluation components.",
            "TRAIN target inversion does not establish physical quote identity or label-free scale.",
            "Evaluation regime is target-derived for retrospective grading only, not an inference feature.",
            "Unknown transitions/spreads invalidate downstream tests and retain unknown mass in denominators.",
            "Future books and targets are diagnostic only; no competition trainer, features or model was modified.",
            "No external numeric state or examples were read.",
            "Rounding and alternative exits are fixed controls; no winner is selected for a submission.",
        ],
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Wrote %s; resources=%s", args.output, output["resources"])


if __name__ == "__main__":
    main()
