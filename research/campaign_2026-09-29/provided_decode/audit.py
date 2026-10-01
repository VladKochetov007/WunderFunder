from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
PRICE_COLUMNS = [f"i0_p{j}" for j in range(22)]
TARGET_QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
EVENT_HORIZONS = (2, 4, 8)
ROW_HORIZONS = (50, 200, 1000)
RULES = ("common_mean", "common_end", "independent_stop_end", "independent_stop_mean")
PARSER_VERSION = "provided-law-audit-v1"


def digest_file(path: Path) -> str:
    """Hash small source/manifest artifacts without loading them all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: Path) -> dict[str, object]:
    """Identify immutable supplied sources by metadata and footer hash."""
    stat = path.stat()
    with path.open("rb") as handle:
        handle.seek(max(0, stat.st_size - (1 << 20)))
        footer_hash = hashlib.sha256(handle.read()).hexdigest()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "last_1mib_sha256": footer_hash,
    }


def freeze_sample(train_count: int, evaluation_count: int) -> dict[str, object]:
    """Select fixed components using only pre-existing supplied-data chain metadata."""
    path = ROOT / "cache" / "chain_folds.npz"
    with np.load(path) as data:
        split, groups, component, fold = (
            data[key] for key in ("split", "row_group", "chain", "fold")
        )
        evaluation_components = set(component[(split == "valid") & (fold == 0)].tolist())
        fit = [
            (int(group), int(chain))
            for group, chain in zip(
                groups[split == "train"], component[split == "train"], strict=True
            )
            if int(chain) not in evaluation_components
        ]
        evaluation = [
            (int(group), int(chain))
            for group, chain in zip(
                groups[(split == "valid") & (fold == 0)],
                component[(split == "valid") & (fold == 0)],
                strict=True,
            )
        ]

    def order(pair):
        return hashlib.sha256(f"provided-audit-v1:{pair[0]}:{pair[1]}".encode()).digest()

    fit = sorted(fit, key=order)[:train_count]
    selected = []
    seen = set()
    for group, chain in sorted(evaluation, key=order):
        if chain not in seen:
            selected.append((group, chain))
            seen.add(chain)
        if len(selected) == evaluation_count:
            break
    if set(chain for _, chain in fit) & set(chain for _, chain in selected):
        raise ValueError("Fit/evaluation component overlap")
    return {
        "chain_metadata_sha256": digest_file(path),
        "fit_train_groups": [group for group, _ in fit],
        "evaluation_valid_groups": [group for group, _ in selected],
        "fit_components": [chain for _, chain in fit],
        "evaluation_components": [chain for _, chain in selected],
        "selection_uses_targets_or_masks": False,
        "event_horizons": EVENT_HORIZONS,
        "row_horizons": ROW_HORIZONS,
        "rules": RULES,
        "minimum_mapping_support": 3,
        "minimum_mapping_confidence": 0.95,
        "fixed_rounding": "nearest even, origin parity 0; ties additionally audited with parity 1 and half-up",
        "physical_time_horizon": "not testable without an independently identified timestamp feature",
    }


def read_sequence(
    source: pq.ParquetFile, group: int, *, has_mask: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one supplied sequence, preserving the documented group order."""
    names = PRICE_COLUMNS + ["t0", "t1", "need_prediction"] + (["is_scored"] if has_mask else [])
    table = source.read_row_group(group, columns=names, use_threads=False)
    prices = np.stack([table.column(name).to_numpy() for name in PRICE_COLUMNS], axis=1).astype(
        np.float32
    )
    targets = np.stack([table.column(name).to_numpy() for name in ("t0", "t1")], axis=1).astype(
        np.float64
    )
    mask = table.column("need_prediction").to_numpy().astype(bool)
    if has_mask:
        mask &= table.column("is_scored").to_numpy().astype(bool)
    return prices, targets, mask


def target_quantum(targets: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Identify a supplied-label lattice; this is not a label-free price quantum."""
    errors = [np.max(np.abs(np.rint(targets / q) * q - targets)) for q in TARGET_QUANTA]
    regime = int(np.argmin(errors))
    if errors[regime] > 1e-3:
        raise ValueError("Targets do not match the declared supplied-label lattice")
    return regime, float(errors[regime]), np.rint(targets / TARGET_QUANTA[regime]).astype(np.int32)


def invert_pair(ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert the proposed law for diagnostics; never use this as an inference input."""
    first, second = ticks[:, 0], ticks[:, 1]
    positive_first, positive_second = first >= 1, second >= 1
    spread = np.where(
        positive_first, -second - 2, np.where(positive_second, -first - 2, -first - second)
    )
    future_relative = np.where(positive_first, first + spread, -second)
    return spread, future_relative


def common_law(future: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Implement the hypothesized common-reference payoff without importing old code."""
    spread = ask - bid
    return np.column_stack(
        (
            np.where(future < bid, -spread - 2, future - ask),
            np.where(future > ask, -spread - 2, bid - future),
        )
    )


def event_rows(prices: np.ndarray) -> np.ndarray:
    """Detect full-block changes without treating column indices as depth order."""
    return np.r_[0, np.flatnonzero(np.any(prices[1:] != prices[:-1], axis=1)) + 1]


def price_keys(prices: np.ndarray) -> list[bytes]:
    """Use the frozen code grammar resolution, explicitly separate from physical units."""
    return [row.tobytes() for row in np.round(prices, 4)]


def feature_only_probe(prices: np.ndarray) -> dict[str, object]:
    """Falsify literal/common monotone coordinate interpretations using no labels."""
    bid, ask = prices[:, :11], prices[:, 11:]
    direct = np.min(ask, axis=1) - np.max(bid, axis=1)
    reversed_direct = np.min(-ask, axis=1) - np.max(-bid, axis=1)
    events = event_rows(prices)[1:]
    delta = prices[events] - prices[events - 1]
    if len(delta):
        bid_delta, ask_delta = np.median(delta[:, :11], axis=1), np.median(delta[:, 11:], axis=1)
        residual = np.maximum(
            np.max(np.abs(delta[:, :11] - bid_delta[:, None]), axis=1),
            np.max(np.abs(delta[:, 11:] - ask_delta[:, None]), axis=1),
        )
        exact_translation = float(np.mean(residual <= 1e-4))
    else:
        exact_translation = math.nan
    column = prices[::31, 0].astype(np.float64)
    raw_unique, transformed_unique = np.unique(column), np.unique(ndtr(column))
    return {
        "rows": len(prices),
        "literal_positive_spread_fraction": float(np.mean(direct > 0)),
        "decreasing_common_transform_positive_spread_fraction": float(np.mean(reversed_direct > 0)),
        "uniform_bid_and_ask_translation_event_fraction": exact_translation,
        "p0_sample_raw_unique": len(raw_unique),
        "p0_sample_phi_unique": len(transformed_unique),
        "phi_preserves_sample_order_and_equality": bool(
            len(raw_unique) == len(transformed_unique) and np.all(np.diff(ndtr(raw_unique)) > 0)
        ),
        "identification_limit": "Unknown per-column transformations and unordered representation components prevent interpreting these values as quote levels. An invertible transform alone cannot identify physical tick scale or translation origin.",
    }


def fit_diagnostic(
    source: pq.ParquetFile, groups: list[int]
) -> tuple[
    dict[int, dict[tuple[bytes, bytes], tuple[int, int]]],
    dict[int, dict[bytes, int]],
    dict[str, object],
]:
    """Fit a separately labeled supervised diagnostic solely on supplied training components."""
    moves = defaultdict(lambda: defaultdict(Counter))
    spreads = defaultdict(lambda: defaultdict(Counter))
    probes = []
    regimes: Counter[int] = Counter()
    for group in groups:
        prices, targets, _ = read_sequence(source, group, has_mask=False)
        probes.append({"row_group": group, **feature_only_probe(prices)})
        regime, _, ticks = target_quantum(targets)
        regimes[regime] += 1
        event = event_rows(prices)
        keys = price_keys(prices[event])
        implied, _ = invert_pair(ticks)
        for index in range(1, len(event)):
            row = event[index]
            delta = ticks[row] - ticks[row - 1]
            moves[regime][(keys[index - 1], keys[index])][(-int(delta[0]), int(delta[1]))] += 1
        for key, start, stop in zip(keys, event, np.r_[event[1:], len(prices)], strict=True):
            spreads[regime][key].update(implied[start:stop:7].tolist())
    accepted_moves, accepted_spreads, stats = {}, {}, {}
    for regime in range(4):
        accepted_moves[regime], accepted_spreads[regime] = {}, {}
        for key, counter in moves[regime].items():
            value, support = counter.most_common(1)[0]
            if support >= 3 and support / counter.total() >= 0.95:
                accepted_moves[regime][key] = value
        for key, counter in spreads[regime].items():
            value, support = counter.most_common(1)[0]
            if value >= 1 and support >= 3 and support / counter.total() >= 0.95:
                accepted_spreads[regime][key] = value
        stats[str(regime)] = {
            "fit_sequences": regimes[regime],
            "observed_transition_codes": len(moves[regime]),
            "accepted_transition_codes": len(accepted_moves[regime]),
            "observed_price_codes": len(spreads[regime]),
            "accepted_spread_codes": len(accepted_spreads[regime]),
            "transition_modal_mass_fraction": sum(
                c.most_common(1)[0][1] for c in moves[regime].values()
            )
            / max(1, sum(c.total() for c in moves[regime].values())),
        }
    return (
        accepted_moves,
        accepted_spreads,
        {
            "supervised_fit_statistics": stats,
            "feature_only_fit_probes": probes,
            "spread_origin": "Provided TRAIN target-inverted modal code spreads, strict support/confidence gate. This diagnostic cannot establish label-free coordinates; no evaluation targets/masks used in fitting.",
        },
    )


def score_record(
    predicted: np.ndarray,
    truth: np.ndarray,
    eligible: np.ndarray,
    scored: np.ndarray,
    is_event: np.ndarray,
    quantum: float,
) -> dict[str, float | int]:
    """Report validity, target-weighted failure and event/quiet behavior over required rows."""
    exact = np.all(predicted == truth, axis=1)
    weight = np.abs(np.clip(truth * quantum, -2, 2)).sum(axis=1)
    result = {}
    for name, mask in (
        ("required", np.arange(len(truth)) >= 99),
        ("scored", scored),
        ("scored_event", scored & is_event),
        ("scored_quiet", scored & ~is_event),
    ):
        known = eligible & mask
        total_mass = weight[mask].sum()
        known_mass = weight[known].sum()
        result[f"{name}_rows"] = int(mask.sum())
        result[f"{name}_eligible_rows"] = int(known.sum())
        result[f"{name}_exact_rows"] = int((known & exact).sum())
        result[f"{name}_target_mass"] = float(total_mass)
        result[f"{name}_eligible_target_mass"] = float(known_mass)
        result[f"{name}_failure_target_mass"] = float(weight[known & ~exact].sum())
    return result


def horizon_predictions(
    prices: np.ndarray,
    regime: int,
    moves: dict[tuple[bytes, bytes], tuple[int, int]],
    spreads: dict[bytes, int],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Generate diagnostic future rules, invalidating unknown coordinate intervals."""
    event = event_rows(prices)
    keys = price_keys(prices[event])
    count = len(event)
    deltas = np.zeros((count, 2), dtype=np.int64)
    unknown = np.zeros(count, dtype=bool)
    for index in range(1, count):
        move = moves.get((keys[index - 1], keys[index]))
        if move is None:
            unknown[index] = True
        else:
            deltas[index] = move
    cumulative = np.cumsum(deltas, axis=0)
    unknown_prefix = np.cumsum(unknown)
    known_spread = np.array([spreads.get(key, -1) for key in keys])
    current = np.searchsorted(event, np.arange(len(prices)), side="right") - 1
    result = {}
    tie_counts = {}
    for clock, horizons in (("event", EVENT_HORIZONS), ("row", ROW_HORIZONS)):
        for horizon in horizons:
            end = (
                current + horizon
                if clock == "event"
                else np.searchsorted(event, np.arange(len(prices)) + horizon, side="right") - 1
            )
            is_in_range = (
                end < count if clock == "event" else np.arange(len(prices)) + horizon < len(prices)
            )
            end = np.clip(end, 0, count - 1)
            eligible = (
                is_in_range
                & (end > current)
                & (known_spread[current] > 0)
                & (unknown_prefix[end] == unknown_prefix[current])
            )
            predictions = {rule: np.zeros((len(prices), 2), dtype=np.int64) for rule in RULES}
            parity_predictions = {
                name: np.zeros((len(prices), 2), dtype=np.int64)
                for name in ("common_mean_parity1", "common_mean_half_up")
            }
            ties = 0
            for start, stop in zip(event, np.r_[event[1:], len(prices)], strict=True):
                origin = int(current[start])
                spread = int(known_spread[origin])
                local_ends = np.unique(end[start:stop])
                for last in local_ends:
                    mask = (
                        (np.arange(start, stop) >= start)
                        & (end[start:stop] == last)
                        & eligible[start:stop]
                    )
                    selected = np.arange(start, stop)[mask]
                    if not len(selected):
                        continue
                    future_ask = cumulative[origin + 1 : last + 1, 0] - cumulative[origin, 0]
                    future_bid = (
                        -spread + cumulative[origin + 1 : last + 1, 1] - cumulative[origin, 1]
                    )
                    path_spread = future_ask - future_bid
                    future_known = known_spread[origin + 1 : last + 1]
                    has_conflict = np.any(path_spread < 1) or np.any(
                        (future_known > 0) & (future_known != path_spread)
                    )
                    if has_conflict:
                        eligible[selected] = False
                        continue
                    mean_mid = float(np.mean((future_ask + future_bid) / 2))
                    mean_even = np.rint(mean_mid)
                    ties += int((mean_mid % 1 == 0.5) * len(selected))
                    for name, future in (
                        ("common_mean", mean_even),
                        ("common_end", np.rint((future_ask[-1] + future_bid[-1]) / 2)),
                        ("common_mean_parity1", np.rint(mean_mid + 1) - 1),
                        ("common_mean_half_up", np.floor(mean_mid + 0.5)),
                    ):
                        pair = common_law(np.array([future]), np.array([-spread]), np.array([0]))[
                            0
                        ].astype(np.int64)
                        (predictions if name in predictions else parity_predictions)[name][
                            selected
                        ] = pair
                    stop_loss = -spread - 2
                    stopped_long = bool(np.any(future_bid <= stop_loss))
                    stopped_short = bool(np.any(-spread - future_ask <= stop_loss))
                    predictions["independent_stop_end"][selected] = (
                        stop_loss if stopped_long else int(future_bid[-1]),
                        stop_loss if stopped_short else int(-spread - future_ask[-1]),
                    )
                    predictions["independent_stop_mean"][selected] = (
                        stop_loss if stopped_long else int(np.rint(np.mean(future_bid))),
                        stop_loss if stopped_short else int(np.rint(-spread - np.mean(future_ask))),
                    )
            for name, predicted in (predictions | parity_predictions).items():
                result[f"{clock}{horizon}/{name}"] = (predicted, eligible.copy())
            tie_counts[f"{clock}{horizon}"] = ties
    return result, {
        "book_events": count,
        "unknown_transition_occurrences": int(unknown.sum()),
        "known_transition_occurrences": int(count - 1 - unknown.sum()),
        "current_events_without_confident_train_spread": int(np.sum(known_spread < 1)),
        "common_mean_tie_rows": tie_counts,
        "regime": regime,
    }


def main() -> None:
    """Audit supplied-only feature identification and competing target mechanisms."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--fit-sequences", type=int, default=48)
    parser.add_argument("--evaluation-sequences", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_wall, start_cpu = time.monotonic(), time.process_time()
    manifest = freeze_sample(args.fit_sequences, args.evaluation_sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    train_path, valid_path = (
        ROOT / "datasets" / "train.parquet",
        ROOT / "datasets" / "valid.parquet",
    )
    train, valid = pq.ParquetFile(train_path), pq.ParquetFile(valid_path)
    LOGGER.info("Fitting independent supervised diagnostic on %s train groups", args.fit_sequences)
    moves, spreads, fit = fit_diagnostic(train, manifest["fit_train_groups"])
    records = []
    totals = defaultdict(Counter)
    for group in manifest["evaluation_valid_groups"]:
        LOGGER.info("Evaluating fixed held-out group %s", group)
        prices, targets, scored = read_sequence(valid, group, has_mask=True)
        regime, quantization_error, ticks = target_quantum(targets)
        implied_spread, future_relative = invert_pair(ticks)
        inverted = common_law(future_relative, np.zeros(len(prices)), implied_spread)
        events = event_rows(prices)
        is_event = np.zeros(len(prices), dtype=bool)
        is_event[events] = True
        codes = price_keys(prices)
        spread_variants = defaultdict(set)
        for key, spread in zip(codes, implied_spread, strict=True):
            spread_variants[key].add(int(spread))
        ambiguity = np.array([len(spread_variants[key]) > 1 for key in codes])
        rules, coverage = horizon_predictions(prices, regime, moves[regime], spreads[regime])
        scored_weight = np.abs(np.clip(targets, -2, 2)).sum(axis=1)
        measurements = {}
        for name, (predicted, eligible) in rules.items():
            measurement = score_record(
                predicted, ticks, eligible, scored, is_event, TARGET_QUANTA[regime]
            )
            measurements[name] = measurement
            totals[(regime, name)].update(measurement)
        records.append(
            {
                "row_group": group,
                "regime_from_targets_diagnostic_only": regime,
                "target_quantization_max_error": quantization_error,
                "feature_only": feature_only_probe(prices),
                "implied_spread_ambiguous_price_code_row_fraction": float(ambiguity[99:].mean()),
                "implied_spread_ambiguity_scored_mass_fraction": float(
                    scored_weight[scored & ambiguity].sum()
                    / max(1e-12, scored_weight[scored].sum())
                ),
                "inverse_law_identity_fraction_not_independent_evidence": float(
                    np.mean(np.all(inverted == ticks, axis=1))
                ),
                "target_change_off_book_events": int(
                    np.count_nonzero(np.any(ticks[1:] != ticks[:-1], axis=1) & ~is_event[1:])
                ),
                "coverage": coverage,
                "rules": measurements,
            }
        )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    output = {
        "parser_version": PARSER_VERSION,
        "command": sys.argv,
        "sample_manifest": manifest,
        "source_identities": {
            "train": source_identity(train_path),
            "valid": source_identity(valid_path),
            "script_sha256": digest_file(Path(__file__)),
        },
        "fit": fit,
        "evaluation_records": records,
        "totals_by_regime_and_rule": {
            f"class{regime}/{rule}": dict(counter) for (regime, rule), counter in totals.items()
        },
        "resources": {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.monotonic() - start_wall,
            "peak_rss_kib": usage.ru_maxrss,
            "exit_code": 0,
        },
        "interpretation": [
            "No external data, mappings, transformations or statistics were read by this audit.",
            "The strictly label-free probe cannot identify physical tick scale or translation origin from unknown encoded features.",
            "Supervised move/spread diagnostics are trained only on supplied train components excluded from the evaluation fold.",
            "Evaluation targets and future books are diagnostics only. No prediction model or competition feature pipeline was changed.",
            "Unknown transitions, unknown/ambiguous trained spreads and inconsistent reconstructed coordinates invalidate downstream horizon comparisons.",
            "Inverse-law identity follows from algebraic inversion and is not evidence for a common physical future price.",
            "Row horizons are publication-clock controls, not established physical-time horizons.",
            "No constrained readout gain is claimed; CE/rational readout repetitions require new causal information and a separate matched evaluation.",
        ],
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    LOGGER.info("Wrote %s; resources=%s", args.output, output["resources"])


if __name__ == "__main__":
    main()
