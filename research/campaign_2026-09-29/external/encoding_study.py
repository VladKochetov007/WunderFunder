from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from book_replay import (
    PARSER_VERSION,
    DepthReplay,
    geometry,
    keyed_edit,
    price_key,
    probe_perturbations,
    rank_edit,
    top_levels,
)

LOGGER = logging.getLogger(__name__)
METRICS = [
    "rank_l1",
    "interior_add",
    "interior_remove",
    "entered_mass",
    "exited_mass",
    "shared_prices",
    "bid_mass_q25",
    "bid_mass_q50",
    "bid_mass_q75",
    "bid_mean_distance",
    "bid_concentration",
    "ask_mass_q25",
    "ask_mass_q50",
    "ask_mass_q75",
    "ask_mean_distance",
    "ask_concentration",
]


def provenance(path: Path, digest: str, records: int) -> dict[str, object]:
    """Describe the exact consumed prefix rather than implying a full-file hash."""
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
        "hash_scope": "consumed original UTF-8 lines including header, excluding cap-stopping line",
        "processed_records": records,
    }


def summarize(values: list[list[float]]) -> dict[str, object]:
    """Keep chronological exploration/falsification slices explicit."""
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"count": 0, "status": "no_eligible_records"}
    split = max(1, len(array) * 2 // 3)
    result = {}
    for name, part in (("explore", array[:split]), ("falsify", array[split:])):
        if len(part) == 0:
            result[name] = {"count": 0}
            continue
        result[name] = {
            "count": len(part),
            "mean": dict(zip(METRICS, part.mean(axis=0).tolist(), strict=False)),
            "median": dict(zip(METRICS, np.median(part, axis=0).tolist(), strict=False)),
            "rank_edit_without_shared_price_edit_count": int(
                np.count_nonzero((part[:, 0] > 0) & (part[:, 1] == 0) & (part[:, 2] == 0))
            ),
        }
    return result


def snapshot_metrics(
    previous: tuple[list[tuple[int, float]], list[tuple[int, float]]] | None,
    current: tuple[list[tuple[int, float]], list[tuple[int, float]]],
) -> list[float]:
    """Compute shape and price-keyed edit summaries on occupied top-25 levels."""
    bid, ask = current
    anchor = (bid[0][0] + ask[0][0]) / 2
    scale = ask[0][0] - bid[0][0]
    bid_shape = geometry(bid, anchor, scale)
    ask_shape = geometry(ask, anchor, scale)
    if previous is None:
        edits = [0.0] * 6
    else:
        before_bid, before_ask = previous
        first = keyed_edit(dict(before_bid), dict(bid))
        second = keyed_edit(dict(before_ask), dict(ask))
        edits = [rank_edit(before_bid, bid) + rank_edit(before_ask, ask)]
        edits += [
            first[key] + second[key]
            for key in (
                "interior_add",
                "interior_remove",
                "entered_mass",
                "exited_mass",
                "shared_prices",
            )
        ]
    return edits + bid_shape + ask_shape


def observed_geometry_probe(
    current: tuple[list[tuple[int, float]], list[tuple[int, float]]],
) -> dict[str, float]:
    """Repeat analytical symmetry checks on measured, eligible books."""
    bid, ask = current
    anchor = (bid[0][0] + ask[0][0]) / 2
    scale = ask[0][0] - bid[0][0]
    result = {}
    for side, raw in (("bid", bid), ("ask", ask)):
        base = geometry(raw, anchor, scale)
        variants = {
            "translation": ([(p + 10**12, q) for p, q in raw], anchor + 10**12, scale),
            "price_scale": ([(p * 3, q) for p, q in raw], anchor * 3, scale * 3),
            "quantity_scale": ([(p, q * 13) for p, q in raw], anchor, scale),
            "mirror": ([(int(2 * anchor) - p, q) for p, q in raw], anchor, scale),
        }
        for name, (v, center, unit) in variants.items():
            result[f"{side}_{name}"] = max(
                abs(a - b) for a, b in zip(base, geometry(v, center, unit), strict=False)
            )
    return result


def publication_signature(
    events: list[tuple[int, tuple[float, ...]]],
    *,
    transient_trade: bool,
    stride: int = 1,
    delay_trade: int = 0,
) -> dict[str, object]:
    """Measure grouped co-change signatures under declared publication policies."""
    state = np.zeros(7)
    previous = state.copy()
    signatures: Counter[str] = Counter()
    pending: dict[int, list[tuple[float, ...]]] = {}
    rows = 0
    for index, (kind, payload) in enumerate(events):
        if transient_trade:
            state[4:] = 0
        if kind == 0:
            state[:4] = payload
        elif kind == 1:
            if delay_trade:
                pending.setdefault(index + delay_trade, []).append(payload)
            else:
                state[4:] = payload
        for delayed in pending.pop(index, []):
            state[4:] = delayed
        if (index + 1) % stride:
            continue
        changed = state != previous
        signature = (
            f"{int(changed[[0, 2]].any())}{int(changed[[1, 3]].any())}{int(changed[4:].any())}"
        )
        signatures[signature] += 1
        previous = state.copy()
        rows += 1
    return {
        "emitted_rows": rows,
        "signature_order": "price, quantity, trade",
        "signatures": dict(sorted(signatures.items())),
        "unpublished_tail_messages": len(events) % stride,
        "delayed_trades_pending_at_end": sum(map(len, pending.values())),
    }


def gzip_study(path: Path, cap: int, duration_ns: int) -> dict[str, object]:
    """Replay a bounded prefix with strict causal initialization and gap handling."""
    replay = DepthReplay()
    digest = hashlib.sha256()
    streams: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    rows = []
    geometry_errors: dict[str, float] = {}
    geometry_cases = 0
    events = []
    depth_quotes: dict[int, tuple[float, ...]] = {}
    ticker_quotes: dict[int, tuple[float, ...]] = {}
    previous_book = None
    first_ns = last_ns = None
    regressions = 0
    trade_gaps = 0
    previous_trade_id = None
    exchange_lags_ms = []
    snapshots = []
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if count >= cap:
                break
            receive, encoded = line.split(" ", 1)
            received_ns = int(receive)
            if first_ns is None:
                first_ns = received_ns
            if received_ns - first_ns > duration_ns:
                break
            if last_ns is not None and received_ns < last_ns:
                regressions += 1
            last_ns = received_ns
            digest.update(line.encode())
            message = json.loads(encoded)
            data = message.get("data", message)
            stream = message.get("stream", "snapshot" if "lastUpdateId" in data else "control")
            streams[stream] += 1
            count += 1
            if "E" in data:
                exchange_lags_ms.append(received_ns / 1e6 - int(data["E"]))
            if "lastUpdateId" in data:
                replay.snapshot(data, received_ns)
                snapshots.append(
                    {
                        "line": count,
                        "received_ns": received_ns,
                        "update_id": replay.update_id,
                        "bid_levels": len(replay.bids),
                        "ask_levels": len(replay.asks),
                    }
                )
                previous_book = None
                events.append((2, ()))
            elif data.get("e") == "depthUpdate":
                status = replay.update(data, received_ns)
                statuses[status] += 1
                if status == "applied" and replay.has_known_top():
                    current = replay.top()
                    rows.append(snapshot_metrics(previous_book, current))
                    previous_book = current
                    bid, ask = current
                    depth_quotes[int(data["u"])] = (
                        float(bid[0][0]),
                        bid[0][1],
                        float(ask[0][0]),
                        ask[0][1],
                    )
                    if len(rows) % 100 == 1:
                        geometry_cases += 1
                        for name, error in observed_geometry_probe(current).items():
                            geometry_errors[name] = max(geometry_errors.get(name, 0.0), error)
                else:
                    previous_book = None
                    if status == "applied":
                        statuses["unknown_top25_coverage"] += 1
                events.append((2, ()))
            elif data.get("e") == "trade":
                trade_id = int(data["t"])
                if previous_trade_id is not None and trade_id > previous_trade_id + 1:
                    trade_gaps += 1
                previous_trade_id = trade_id
                events.append(
                    (1, (float(price_key(data["p"])), float(data["q"]), -1.0 if data["m"] else 1.0))
                )
            elif stream.endswith("@bookTicker"):
                quote = (
                    float(price_key(data["b"])),
                    float(data["B"]),
                    float(price_key(data["a"])),
                    float(data["A"]),
                )
                ticker_quotes[int(data["u"])] = quote
                events.append((0, quote))
            else:
                events.append((2, ()))
    common = depth_quotes.keys() & ticker_quotes.keys()
    mismatches = [
        key
        for key in common
        if not np.allclose(depth_quotes[key], ticker_quotes[key], rtol=1e-12, atol=1e-8)
    ]
    split = len(events) * 2 // 3
    signatures = {}
    for part_name, part in (("explore", events[:split]), ("falsify", events[split:])):
        signatures[part_name] = {
            "hold": publication_signature(part, transient_trade=False),
            "transient": publication_signature(part, transient_trade=True),
            "coalesce5": publication_signature(part, transient_trade=False, stride=5),
            "delay_trade5": publication_signature(part, transient_trade=False, delay_trade=5),
        }
    lag = np.asarray(exchange_lags_ms)
    return {
        "provenance": provenance(path, digest.hexdigest(), count),
        "first_received_ns": first_ns,
        "last_received_ns": last_ns,
        "stream_counts": dict(streams),
        "receive_clock_regressions": regressions,
        "snapshots": snapshots,
        "depth_status_counts": dict(statuses),
        "gap_count": replay.gap_count,
        "zero_quantity_deletions": replay.deletion_count,
        "trade_id_gap_intervals": trade_gaps,
        "compatible_quote_comparisons": {
            "exact_shared_update_ids": len(common),
            "mismatches": len(mismatches),
            "first_mismatch_ids": mismatches[:10],
            "status": "unknown" if not common else "checked",
        },
        "exchange_event_receive_lag_ms": {}
        if not len(lag)
        else {"min": float(lag.min()), "median": float(np.median(lag)), "max": float(lag.max())},
        "geometry_symmetry": {"cases": geometry_cases, "max_errors": geometry_errors},
        "liquidity_edits_and_geometry": summarize(rows),
        "publication_policies": signatures,
        "limits": [
            "Depth update batches do not reveal gross within-batch arrivals/cancellations.",
            "Publication policies are constructed alternatives; their distinct signatures do not identify the competition generator.",
            "Top-25 entry and exit are censored separately from shared-price edits.",
        ],
    }


def md_study(directory: Path, book_cap: int, trade_cap: int, duration_ns: int) -> dict[str, object]:
    """Inspect explicit snapshots using a receive-order trade merge, with no hidden event claims."""
    book_path, trade_path = directory / "lob.csv", directory / "trades.csv"
    book_hash, trade_hash = hashlib.sha256(), hashlib.sha256()
    rows = []
    previous = None
    book_count = trade_count = 0
    first_ns = last_book_ns = last_trade_ns = None
    receive_regressions = trade_regressions = equal_timestamp_trades = 0
    sides: Counter[str] = Counter()
    cumulative_trade_mass = 0.0
    geometry_errors: dict[str, float] = {}
    geometry_cases = 0
    with book_path.open(encoding="utf-8") as books, trade_path.open(encoding="utf-8") as trades:
        book_header, trade_header = books.readline(), trades.readline()
        book_hash.update(book_header.encode())
        trade_hash.update(trade_header.encode())
        header = next(csv.reader([book_header]))
        if len(header) != 102 or next(csv.reader([trade_header])) != [
            "",
            "local_timestamp",
            "side",
            "price",
            "amount",
        ]:
            raise ValueError("Unexpected MD schema")
        pending_trade = trades.readline()
        pending_fields = next(csv.reader([pending_trade])) if pending_trade else None
        for line in books:
            if book_count >= book_cap:
                break
            fields = next(csv.reader([line]))
            ts = int(fields[1]) * 1000
            if first_ns is None:
                first_ns = ts
            if ts - first_ns > duration_ns:
                break
            if trade_count >= trade_cap and pending_fields and ts > int(pending_fields[1]) * 1000:
                break
            if last_book_ns is not None and ts < last_book_ns:
                receive_regressions += 1
                previous = None
            last_book_ns = ts
            while pending_fields and int(pending_fields[1]) * 1000 < ts and trade_count < trade_cap:
                trade_ts = int(pending_fields[1]) * 1000
                if last_trade_ns is not None and trade_ts < last_trade_ns:
                    trade_regressions += 1
                last_trade_ns = trade_ts
                trade_hash.update(pending_trade.encode())
                cumulative_trade_mass += float(pending_fields[4])
                sides[pending_fields[2]] += 1
                trade_count += 1
                pending_trade = trades.readline()
                pending_fields = next(csv.reader([pending_trade])) if pending_trade else None
            if pending_fields and int(pending_fields[1]) * 1000 == ts:
                equal_timestamp_trades += 1
            book_hash.update(line.encode())
            book_count += 1
            ask = [(price_key(fields[2 + k * 4]), float(fields[3 + k * 4])) for k in range(25)]
            bid = [(price_key(fields[4 + k * 4]), float(fields[5 + k * 4])) for k in range(25)]
            current = (bid, ask)
            if any(q <= 0 or not math.isfinite(q) for _, q in bid + ask) or bid[0][0] >= ask[0][0]:
                raise ValueError("Invalid explicit MD snapshot")
            if bid != top_levels(dict(bid), 25, "bid") or ask != top_levels(dict(ask), 25, "ask"):
                raise ValueError("MD levels are not ordered occupied prices")
            rows.append(snapshot_metrics(previous, current))
            previous = current
            if book_count % 100 == 1:
                geometry_cases += 1
                for name, error in observed_geometry_probe(current).items():
                    geometry_errors[name] = max(geometry_errors.get(name, 0.0), error)
    return {
        "book_provenance": provenance(book_path, book_hash.hexdigest(), book_count),
        "trade_provenance": provenance(trade_path, trade_hash.hexdigest(), trade_count),
        "first_book_ns": first_ns,
        "last_book_ns": last_book_ns,
        "receive_clock_regressions": receive_regressions,
        "trade_clock_regressions": trade_regressions,
        "book_rows_with_ambiguous_trade_timestamp_tie": equal_timestamp_trades,
        "received_trade_side_counts_unverified_semantics": dict(sides),
        "received_trade_mass_before_last_snapshot": cumulative_trade_mass,
        "geometry_symmetry": {"cases": geometry_cases, "max_errors": geometry_errors},
        "liquidity_edits_and_geometry": summarize(rows),
        "interpretation": "Net edits between observed snapshots are inferences. Intermediate updates, exchange time, gross cancellations and MD aggressor-side semantics are unobserved. Timestamp ties are excluded from same-row trade availability.",
    }


def main() -> None:
    """Run and serialize the isolated bounded external encoding study."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path("/home/vlad/development/ultimate/data")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-messages", type=int, default=200_000)
    parser.add_argument("--max-md-book-rows", type=int, default=100_000)
    parser.add_argument("--max-md-trades", type=int, default=200_000)
    parser.add_argument("--duration-seconds", type=int, default=600)
    args = parser.parse_args()
    if (
        min(args.max_messages, args.max_md_book_rows, args.max_md_trades, args.duration_seconds)
        <= 0
    ):
        parser.error("All prefix caps must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_wall, start_cpu = time.monotonic(), time.process_time()
    output = {
        "parser_version": PARSER_VERSION,
        "command": sys.argv,
        "limits": vars(args) | {"source_root": str(args.source_root), "output": str(args.output)},
        "environment": {
            key: os.environ.get(key)
            for key in (
                "WNN_SLICE",
                "WNN_CPU_PCT",
                "WNN_CPU_QUOTA",
                "WNN_MEM_MAX",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "ARROW_NUM_THREADS",
            )
        },
        "competition_transfer": "No external numeric parameters, examples, weights, symbol/time correspondences or learned mappings may enter the competition pipeline.",
        "toy_perturbations": probe_perturbations(),
        "gzip": {},
    }
    for name in ("btcusdt_20250702.gz", "ethusdt_20250702.gz", "usdcusdt_20250702.gz"):
        LOGGER.info("Starting %s", name)
        output["gzip"][name] = gzip_study(
            args.source_root / "binance_hft" / name,
            args.max_messages,
            args.duration_seconds * 10**9,
        )
        LOGGER.info("Completed %s", name)
    LOGGER.info("Starting MD receive-order snapshot study")
    output["md"] = md_study(
        args.source_root / "MD",
        args.max_md_book_rows,
        args.max_md_trades,
        args.duration_seconds * 10**9,
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    output["resources"] = {
        "wall_seconds": time.monotonic() - start_wall,
        "cpu_seconds": time.process_time() - start_cpu,
        "peak_rss_kib": usage.ru_maxrss,
        "exit_code": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Wrote %s; resources=%s", args.output, output["resources"])


if __name__ == "__main__":
    main()
