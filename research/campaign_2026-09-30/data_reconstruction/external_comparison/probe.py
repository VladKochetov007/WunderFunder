from __future__ import annotations

import csv
import gzip
import hashlib
import json
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]


def summary(x: np.ndarray, widths: tuple[int, int, int]) -> dict[str, object]:
    """Measure code equality and publication-row gaps without physical-time claims."""
    if widths[0] == 2:
        initialized = np.flatnonzero(np.all(x[:, :2] > 0, axis=1))
        if not len(initialized):
            raise ValueError("External prefix has no initialized positive quote")
        x = x[initialized[0] :]
    boundaries = np.cumsum((0,) + widths)
    d = x[1:] != x[:-1]
    flags = np.column_stack(
        [d[:, a:b].any(axis=1) for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)]
    )
    signatures = Counter("".join(str(int(v)) for v in row) for row in flags)
    loc = np.flatnonzero(flags[:, 0])
    gaps = np.diff(loc)
    price_jump = np.abs(np.diff(x[:, : boundaries[1]], axis=0)).sum(axis=1)
    price_jump = price_jump[price_jump > 0]
    quantity = x[:, boundaries[1] : boundaries[2]]
    q0, q1 = quantity[:-1], quantity[1:]
    valid = (q0 > 0) & (q1 > 0) & (q0 != q1)
    log_ratio = np.abs(np.log(q1[valid] / q0[valid]))

    def quantiles(values: np.ndarray) -> dict[str, object]:
        return (
            {
                "count": len(values),
                "q50_q90_q99_max": np.quantile(values, [0.5, 0.9, 0.99, 1]).tolist(),
            }
            if len(values)
            else {"count": 0}
        )

    return {
        "rows": len(x),
        "signature_order": "price,quantity,trade",
        "signatures": dict(signatures),
        "event_rates": flags.mean(axis=0).tolist(),
        "price_interevent_publication_rows": quantiles(gaps),
        "price_quiet_run_completed": quantiles(gaps - 1),
        "trade_refresh_with_price_fraction": float(flags[flags[:, 2], 0].mean())
        if flags[:, 2].any()
        else None,
        "encoded_quantity_positive_fraction": float((quantity > 0).mean()),
        "positive_quantity_change_abs_log_ratio": quantiles(log_ratio),
        "median_normalized_price_code_l1_jump": quantiles(price_jump / np.median(price_jump))
        if len(price_jump)
        else {"count": 0},
    }


def policies(events: list[tuple[int, tuple[float, ...]]]) -> dict[str, np.ndarray]:
    """Construct explicit publication alternatives using observed quote/trade messages."""
    result = {}
    for name, transient, stride, delay in (
        ("hold", False, 1, 0),
        ("transient", True, 1, 0),
        ("coalesce5", False, 5, 0),
        ("delay_trade5", False, 1, 5),
    ):
        state = np.zeros(7)
        rows = []
        pending = {}
        for i, (kind, payload) in enumerate(events):
            if transient:
                state[4:] = 0
            if kind == 0:
                state[:4] = payload
            elif kind == 1:
                if delay:
                    pending.setdefault(i + delay, []).append(payload)
                else:
                    state[4:] = payload
            for value in pending.pop(i, []):
                state[4:] = value
            if (i + 1) % stride == 0:
                rows.append(state[[0, 2, 1, 3, 4, 5, 6]].copy())
        result[name] = np.asarray(rows)
    return result


def gzip_events(path: Path) -> tuple[list[tuple[int, tuple[float, ...]]], dict[str, object]]:
    """Read a strict capped raw receive-order prefix without reconstructing depth."""
    digest = hashlib.sha256()
    events = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for _ in range(20000):
            line = handle.readline()
            if not line:
                break
            digest.update(line.encode())
            _, raw = line.split(" ", 1)
            msg = json.loads(raw)
            data = msg.get("data", msg)
            stream = msg.get("stream", "")
            if stream.endswith("@bookTicker"):
                events.append(
                    (0, (float(data["b"]), float(data["B"]), float(data["a"]), float(data["A"])))
                )
            elif data.get("e") == "trade":
                events.append((1, (float(data["p"]), float(data["q"]), -1.0 if data["m"] else 1.0)))
            else:
                events.append((2, ()))
    return events, {
        "path": str(path),
        "consumed_lines": len(events),
        "consumed_utf8_sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def md_events(directory: Path) -> tuple[list[tuple[int, tuple[float, ...]]], dict[str, object]]:
    """Merge MD top quotes and trades by strict receive order; ties remain unresolved."""
    books, trades = directory / "lob.csv", directory / "trades.csv"
    book_hash, trade_hash = hashlib.sha256(), hashlib.sha256()
    events = []
    consumed = ties = 0
    with books.open() as b, trades.open() as t:
        book_hash.update(b.readline().encode())
        trade_hash.update(t.readline().encode())
        pending = t.readline()
        fields = next(csv.reader([pending])) if pending else None
        for _ in range(1200):
            line = b.readline()
            if not line:
                break
            book_hash.update(line.encode())
            raw = next(csv.reader([line]))
            ts = int(raw[1])
            while fields and int(fields[1]) < ts and consumed < 20000:
                trade_hash.update(pending.encode())
                events.append(
                    (1, (float(fields[3]), float(fields[4]), 1.0 if fields[2] == "buy" else -1.0))
                )
                consumed += 1
                pending = t.readline()
                fields = next(csv.reader([pending])) if pending else None
            ties += bool(fields and int(fields[1]) == ts)
            events.append((0, (float(raw[4]), float(raw[5]), float(raw[2]), float(raw[3]))))
    return events, {
        "book_path": str(books),
        "trade_path": str(trades),
        "book_consumed_sha256": book_hash.hexdigest(),
        "trade_consumed_sha256": trade_hash.hexdigest(),
        "book_rows": 1200,
        "trade_rows": consumed,
        "ambiguous_tied_book_rows": ties,
        "side_semantics": "unverified MD side, equality only",
    }


def ratio_distortion(x: np.ndarray) -> dict[str, object]:
    """Falsify preservation of physical quantity ratios under nonlinear coding."""
    qty = x[:, 2:4]
    keep = np.all(qty > 0, axis=1)
    qty = qty[keep]
    raw = qty[:, 0] / qty[:, 1]
    transformed = np.log1p(qty[:, 0]) / np.log1p(qty[:, 1])
    error = np.abs(np.log(transformed / raw))
    return {
        "positive_quote_cases": len(qty),
        "log_ratio_error_q50_q90_q99": np.quantile(error, [0.5, 0.9, 0.99]).tolist(),
        "ratio_changed_gt1pct_fraction": float((error > np.log(1.01)).mean()),
        "equality_changes_preserved": bool(
            np.array_equal(qty[1:] != qty[:-1], np.log1p(qty[1:]) != np.log1p(qty[:-1]))
        ),
    }


def main() -> None:
    """Compare bounded external publication distributions with supplied code blocks."""
    start = time.monotonic()
    panel = json.loads((HERE.parent / "selection.json").read_text())
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    result = {
        "contract_sha256": hashlib.sha256((HERE / "preregistration.json").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "competition": [],
        "external": {},
        "information_boundary": "external mechanism description only; no inference export",
    }
    for group in panel["test_train_groups"]:
        names = [
            f"i{inst}_{kind}{j}"
            for inst in range(2)
            for kind, width in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
            for j in range(width)
        ]
        table = source.read_row_group(group, columns=names, use_threads=False)
        x = np.column_stack([table.column(name).to_numpy() for name in names])
        for inst in range(2):
            part = x[:, inst * 52 : (inst + 1) * 52]
            split = len(part) * 2 // 3
            for name, rows in (("first2thirds", part[:split]), ("lastthird", part[split:])):
                result["competition"].append(
                    {
                        "group": group,
                        "instrument": inst,
                        "partition": name,
                        "summary": summary(rows, (22, 22, 8)),
                    }
                )
    base = Path("/home/vlad/development/ultimate/data")
    for name in ("btcusdt_20250702.gz", "ethusdt_20250702.gz", "usdcusdt_20250702.gz", "MD"):
        events, provenance = (
            md_events(base / "MD") if name == "MD" else gzip_events(base / "binance_hft" / name)
        )
        split = len(events) * 2 // 3
        report = {"provenance": provenance, "partitions": {}}
        for partition, events_part in (
            ("first2thirds", events[:split]),
            ("lastthird", events[split:]),
        ):
            variants = policies(events_part)
            report["partitions"][partition] = {
                policy: summary(x, (2, 2, 3)) for policy, x in variants.items()
            }
            report["partitions"][partition]["log1p_ratio_falsifier"] = ratio_distortion(
                variants["hold"]
            )
        result["external"][name] = report
    result["resources"] = {
        "wall_seconds": time.monotonic() - start,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(HERE / "results.json"), "resources": result["resources"]}))


if __name__ == "__main__":
    main()
