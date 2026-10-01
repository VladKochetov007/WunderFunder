"""Independent TRAIN-fitted reproduction of class-3 quote and code event clocks.

This is a retrospective target-law diagnostic. It never fits on validation labels or
masks and does not identify physical quotes from price codes without supervision.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRICE_COLUMNS = [f"i0_p{index}" for index in range(22)]
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
Q3 = 0.65687
LENGTH = 20_000


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def parquet_identity(path: Path) -> dict[str, int | str]:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - (1 << 20)))
        value.update(handle.read())
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "footer_1mib_sha256": value.hexdigest()}


def read_group(source: pq.ParquetFile, index: int, labels: bool) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    names = PRICE_COLUMNS + (["t0", "t1", "need_prediction"] if labels else [])
    if labels and source.schema_arrow.get_field_index("is_scored") >= 0:
        names.append("is_scored")
    table = source.read_row_group(index, columns=names, use_threads=False)
    prices = np.stack([table[name].to_numpy() for name in PRICE_COLUMNS], axis=1).astype(np.float32)
    if len(prices) != LENGTH:
        raise ValueError(f"group {index} length {len(prices)}")
    if not labels:
        return prices, None, None
    y = np.stack([table[name].to_numpy() for name in ("t0", "t1")], axis=1).astype(np.float64)
    scored = table["need_prediction"].to_numpy().astype(bool)
    if "is_scored" in names:
        scored &= table["is_scored"].to_numpy().astype(bool)
    return prices, y, scored


def class_and_ticks(y: np.ndarray) -> tuple[int, np.ndarray]:
    errors = [np.max(np.abs(np.rint(y / quantum) * quantum - y)) for quantum in QUANTA]
    regime = int(np.argmin(errors))
    if errors[regime] > 1e-3:
        raise ValueError(f"target lattice error {errors[regime]}")
    return regime, np.rint(y / QUANTA[regime]).astype(np.int32)


def event_index(prices: np.ndarray) -> tuple[np.ndarray, list[bytes]]:
    events = np.r_[0, 1 + np.flatnonzero(np.any(prices[1:] != prices[:-1], axis=1))]
    keys = [row.tobytes() for row in np.round(prices[events], 4)]
    return events, keys


def implied_spread(ticks: np.ndarray) -> np.ndarray:
    first, second = ticks[:, 0], ticks[:, 1]
    return np.where(first >= 1, -second - 2,
                    np.where(second >= 1, -first - 2, -first - second))


def fit_decoder(train: pq.ParquetFile, groups: list[int]) -> tuple[dict[bytes, int], dict[tuple[bytes, bytes], tuple[int, int]], dict]:
    spread_votes: dict[bytes, Counter[int]] = defaultdict(Counter)
    class_counts: Counter[int] = Counter()
    for group in groups:
        prices, y, _ = read_group(train, group, labels=True)
        assert y is not None
        regime, ticks = class_and_ticks(y)
        class_counts[regime] += 1
        if regime != 3:
            continue
        events, keys = event_index(prices)
        spread = implied_spread(ticks)
        for key, first, last in zip(keys, events, np.r_[events[1:], LENGTH], strict=True):
            spread_votes[key].update(int(value) for value in spread[first:last:7])
    spreads = {}
    for key, counter in spread_votes.items():
        value, count = counter.most_common(1)[0]
        if value >= 1 and count >= 3 and count / counter.total() >= 0.90:
            spreads[key] = value

    move_votes: dict[tuple[bytes, bytes], Counter[int]] = defaultdict(Counter)
    identity = Counter()
    transitions = 0
    for group in groups:
        prices, y, _ = read_group(train, group, labels=True)
        assert y is not None
        regime, ticks = class_and_ticks(y)
        if regime != 3:
            continue
        events, keys = event_index(prices)
        for index in range(1, len(events)):
            transitions += 1
            prev, curr = keys[index - 1], keys[index]
            before, after = spreads.get(prev), spreads.get(curr)
            if before is None or after is None:
                identity["unknown_spread"] += 1
                continue
            row = events[index]
            old, new = ticks[row - 1], ticks[row]
            votes: list[int] = []
            if old[0] > -before - 2 and new[0] > -after - 2:
                votes.append(-int(new[0] - old[0]))
                identity["t0_vote"] += 1
            if old[1] > -before - 2 and new[1] > -after - 2:
                votes.append(int(new[1] - old[1]) + after - before)
                identity["t1_vote"] += 1
            if len(votes) == 2:
                identity["two_votes"] += 1
                identity["disagree"] += votes[0] != votes[1]
            move_votes[(prev, curr)].update(votes)
    moves = {}
    for pair, counter in move_votes.items():
        if not counter:
            continue
        ask, count = counter.most_common(1)[0]
        if count >= 3 and count / counter.total() >= 0.80:
            difference = spreads[pair[1]] - spreads[pair[0]]
            moves[pair] = (ask, ask - difference)
    return spreads, moves, {
        "class_sequence_counts": dict(class_counts),
        "observed_class3_transitions": transitions,
        "observed_spread_codes": len(spread_votes),
        "accepted_spread_codes": len(spreads),
        "observed_transition_codes": len(move_votes),
        "accepted_transition_codes": len(moves),
        "known_nonzero_move_codes": sum(a != 0 or b != 0 for a, b in moves.values()),
        "target_identity_votes": dict(identity),
        "supervised_coordinate_warning": "TRAIN label identities determine spread and move scale; no label-free physical quote identity is established.",
    }


def path(prices: np.ndarray, spreads: dict[bytes, int], moves: dict[tuple[bytes, bytes], tuple[int, int]]) -> dict[str, np.ndarray]:
    events, keys = event_index(prices)
    move = np.zeros((len(events), 2), dtype=np.int32)
    unknown = np.zeros(len(events), dtype=bool)
    for index in range(1, len(events)):
        found = moves.get((keys[index - 1], keys[index]))
        if found is None:
            unknown[index] = True
        else:
            move[index] = found
    quotes = np.flatnonzero(np.any(move != 0, axis=1) & ~unknown)
    return {
        "events": events,
        "quotes": quotes,
        "unknown_prefix": np.cumsum(unknown),
        "spread": np.array([spreads.get(key, -1) for key in keys], dtype=np.int32),
        "cumulative": np.cumsum(move, axis=0),
    }


def event_eligibility(state: dict[str, np.ndarray], horizon: int, clock: str) -> tuple[np.ndarray, np.ndarray]:
    events, quotes = state["events"], state["quotes"]
    origins = np.arange(len(events))
    if clock == "code":
        end = origins + horizon
        in_range = end < len(events)
        end = np.minimum(end, len(events) - 1)
    else:
        position = np.searchsorted(quotes, origins, side="right")
        in_range = position + horizon <= len(quotes)
        end = quotes[np.minimum(position + horizon - 1, len(quotes) - 1)] if len(quotes) else origins
    known = state["unknown_prefix"][end] == state["unknown_prefix"][origins]
    eligible = in_range & known & (state["spread"] >= 1)
    return eligible, end


def midpoint_mean(state: dict[str, np.ndarray], clock: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    origins = np.arange(len(state["events"]))
    eligible, _ = event_eligibility(state, horizon, clock)
    cumulative = state["cumulative"].astype(np.float64)
    midpoint = np.mean(cumulative, axis=1)
    if clock == "code":
        prefix = np.r_[0.0, np.cumsum(midpoint)]
        stop = np.minimum(origins + horizon, len(midpoint) - 1)
        mean = (prefix[stop + 1] - prefix[origins + 1]) / horizon - midpoint[origins]
    else:
        quotes = state["quotes"]
        position = np.searchsorted(quotes, origins, side="right")
        if not len(quotes):
            return np.full(len(origins), np.nan), eligible
        prefix = np.r_[0.0, np.cumsum(midpoint[quotes])]
        start = np.minimum(position, len(quotes))
        stop = np.minimum(position + horizon, len(quotes))
        mean = (prefix[stop] - prefix[start]) / horizon - midpoint[origins]
    mean[~eligible] = np.nan
    return mean, eligible


def law_ticks(mean_displacement: np.ndarray, spread: np.ndarray) -> np.ndarray:
    reference = np.rint(mean_displacement - spread / 2)
    first = np.where(reference < -spread, -spread - 2, reference)
    second = np.where(reference > 0, -spread - 2, -spread - reference)
    return np.stack((first, second), axis=1)


def record_law(predicted: np.ndarray, truth: np.ndarray, eligible: np.ndarray, mask: np.ndarray,
               mass: np.ndarray, quote_rows: np.ndarray) -> dict[str, float | int]:
    selected = mask & eligible
    exact = np.all(predicted == truth, axis=1)
    answer: dict[str, float | int] = {}
    for name, partition in (("all", np.ones(len(mask), bool)),
                            ("event", quote_rows), ("quiet", ~quote_rows)):
        member = selected & partition
        entire = mask & partition
        answer[f"{name}_scored_rows"] = int(entire.sum())
        answer[f"{name}_eligible_rows"] = int(member.sum())
        answer[f"{name}_target_mass"] = float(mass[entire].sum())
        answer[f"{name}_eligible_mass"] = float(mass[member].sum())
        answer[f"{name}_exact_pairs"] = int((member & exact).sum())
        answer[f"{name}_failure_mass"] = float(mass[member & ~exact].sum())
    return answer


def evaluate_class3(prices: np.ndarray, y: np.ndarray, scored: np.ndarray,
                    spreads: dict[bytes, int], moves: dict[tuple[bytes, bytes], tuple[int, int]]) -> dict:
    state = path(prices, spreads, moves)
    current = np.searchsorted(state["events"], np.arange(LENGTH), side="right") - 1
    quote_rows = np.zeros(LENGTH, dtype=bool)
    quote_rows[state["events"][state["quotes"]]] = True
    mass = np.abs(np.clip(y, -2, 2)).sum(axis=1)
    truth = np.rint(y / Q3)
    means = {}
    for clock in ("code", "quote"):
        for horizon in (4, 8, 24):
            means[clock, horizon] = midpoint_mean(state, clock, horizon)
    code4, code_eligible = means["code", 4]
    quote4, quote_eligible = means["quote", 4]
    common = code_eligible[current] & quote_eligible[current]
    tick_code = law_ticks(code4, state["spread"])[current]
    tick_quote = law_ticks(quote4, state["spread"])[current]
    law = {
        "code4_all_available": record_law(tick_code, truth, code_eligible[current], scored, mass, quote_rows),
        "quote4_all_available": record_law(tick_quote, truth, quote_eligible[current], scored, mass, quote_rows),
        "code4_matched": record_law(tick_code, truth, common, scored, mass, quote_rows),
        "quote4_matched": record_law(tick_quote, truth, common, scored, mass, quote_rows),
    }
    auxiliary = {}
    for horizon in (8, 24):
        code_mean, code_known = means["code", horizon]
        quote_mean, quote_known = means["quote", horizon]
        eligible = scored & code_known[current] & quote_known[current]
        code_label = np.clip(Q3 * code_mean[current], -2, 2)
        quote_label = np.clip(Q3 * quote_mean[current], -2, 2)
        delta = np.abs(code_label - quote_label)
        auxiliary[str(horizon)] = {
            "full_scored_rows": int(scored.sum()),
            "full_scored_mass": float(mass[scored].sum()),
            "matched_eligible_rows": int(eligible.sum()),
            "matched_eligible_mass": float(mass[eligible].sum()),
            "different_rows_gt_0p001": int((eligible & (delta > 0.001)).sum()),
            "different_target_mass_gt_0p001": float(mass[eligible & (delta > 0.001)].sum()),
            "weighted_absolute_difference_sum": float(np.sum(mass[eligible] * delta[eligible])),
            "weighted_squared_difference_sum": float(np.sum(mass[eligible] * delta[eligible] ** 2)),
            "unweighted_absolute_difference_sum": float(np.sum(delta[eligible])),
        }
    return {
        "representation_code_changes": len(state["events"]) - 1,
        "known_decoded_quote_changes": len(state["quotes"]),
        "unknown_transition_occurrences": int(state["unknown_prefix"][-1]),
        "law": law,
        "auxiliary": auxiliary,
    }


def main() -> None:
    start_wall, start_cpu = time.monotonic(), time.process_time()
    manifest_path = HERE / "preregistration.json"
    manifest = json.loads(manifest_path.read_text())
    chain_path = ROOT / "cache/chain_folds.npz"
    if digest(chain_path) != manifest["chain_metadata_sha256"]:
        raise ValueError("Frozen component metadata changed")
    train_path = ROOT / "datasets/train.parquet"
    valid_path = ROOT / "datasets/valid.parquet"
    train, valid = pq.ParquetFile(train_path), pq.ParquetFile(valid_path)
    spreads, moves, fit = fit_decoder(train, manifest["fit_train_groups"])
    print(f"TRAIN fit: {fit}", flush=True)
    gate_rows = 0
    for group in manifest["evaluation_valid_groups"]:
        prices, _, _ = read_group(valid, group, labels=False)
        state = path(prices, spreads, moves)
        eligible, _ = event_eligibility(state, 4, "quote")
        current = np.searchsorted(state["events"], np.arange(LENGTH), side="right") - 1
        gate_rows += int(eligible[current][99:].sum())
    gate_fraction = gate_rows / (len(manifest["evaluation_valid_groups"]) * (LENGTH - 99))
    gate_passes = gate_rows >= 1000 and gate_fraction >= 0.05
    print(f"Feature-only gate: {gate_rows} rows, fraction {gate_fraction:.6f}, passes {gate_passes}", flush=True)
    records = []
    totals: dict[str, Counter] = defaultdict(Counter)
    evaluated_classes = Counter()
    if gate_passes:
        for group in manifest["evaluation_valid_groups"]:
            prices, y, scored = read_group(valid, group, labels=True)
            assert y is not None and scored is not None
            regime, _ = class_and_ticks(y)
            evaluated_classes[regime] += 1
            if regime != 3:
                continue
            result = evaluate_class3(prices, y, scored, spreads, moves)
            records.append({"valid_group": group, **result})
            for name, values in result["law"].items():
                totals[f"law/{name}"].update(values)
            for name, values in result["auxiliary"].items():
                totals[f"auxiliary/{name}"].update(values)
    output = {
        "version": "clock-reproduction-v1",
        "command": sys.argv,
        "preregistration_sha256": digest(manifest_path),
        "script_sha256": digest(Path(__file__)),
        "source_identities": {name: parquet_identity(p)
                              for name, p in (("train", train_path), ("valid", valid_path))},
        "fit": fit,
        "feature_only_gate": {"eligible_required_rows": gate_rows, "fraction": gate_fraction,
                              "passes": gate_passes, "evaluation_labels_read_before_gate": False},
        "evaluated_class_sequence_counts": dict(evaluated_classes),
        "records": records,
        "totals": {name: dict(counter) for name, counter in totals.items()},
        "resources": {"wall_seconds": time.monotonic() - start_wall,
                      "cpu_seconds": time.process_time() - start_cpu,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "exit_code": 0},
        "interpretation_boundary": [
            "Fold-2 evaluation components are disjoint from TRAIN fit and prior v1/v2 panels, but fold 2 is historically inspected.",
            "The decoder is supervised by TRAIN labels; decoded nonzero quote events are a diagnostic coordinate, not independently identified physical quotes.",
            "Evaluation target lattice determines retrospective class only; no evaluation label or mask enters the fit.",
            "Unknown transitions retain full scored target mass in denominators.",
            "Auxiliary-label differences establish semantics, not a causal model gain.",
        ],
    }
    destination = HERE / "results.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {destination} in {output['resources']['wall_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
