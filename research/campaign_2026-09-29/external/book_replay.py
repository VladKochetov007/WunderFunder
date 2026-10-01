from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

from attrs import define, field

# JSON payloads have heterogeneous nested exchange fields; validate at ingestion.
Payload = dict[str, Any]
Book = dict[int, float]
PRICE_SCALE = 100_000_000
PARSER_VERSION = "external-encoding-v1"


def price_key(value: str) -> int:
    """Preserve Binance decimal price identity without floating point keys."""
    scaled = Decimal(value) * PRICE_SCALE
    if scaled != scaled.to_integral_value():
        raise ValueError(f"Price exceeds supported decimal precision: {value}")
    return int(scaled)


def levels(payload: list[list[str]]) -> Book:
    """Read quantity replacement values, retaining deletion markers."""
    result = {}
    for price, quantity in payload:
        amount = float(quantity)
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Invalid depth quantity")
        result[price_key(price)] = amount
    return result


def top_levels(book: Book, count: int, side: str) -> list[tuple[int, float]]:
    """Extract ordered occupied levels from a price-keyed book."""
    return [(p, book[p]) for p in sorted(book, reverse=side == "bid")[:count]]


def geometry(book_levels: list[tuple[int, float]], anchor: float, scale: float) -> list[float]:
    """Describe liquidity by normalized mass rather than rank-specific quantities."""
    total = sum(q for _, q in book_levels)
    if total <= 0 or scale <= 0:
        raise ValueError("Geometry needs positive mass and price scale")
    distances = [abs(p - anchor) / scale for p, _ in book_levels]
    masses = [q / total for _, q in book_levels]
    quantiles = []
    cumulative = 0.0
    index = 0
    for target in (0.25, 0.5, 0.75):
        while index < len(masses) - 1 and cumulative + masses[index] < target:
            cumulative += masses[index]
            index += 1
        quantiles.append(distances[index])
    return quantiles + [
        sum(d * q for d, q in zip(distances, masses, strict=False)),
        sum(q * q for q in masses),
    ]


def keyed_edit(old: Book, new: Book) -> dict[str, float]:
    """Measure edits on shared prices; edge entry/exit remains separately censored."""
    shared = old.keys() & new.keys()
    positive = sum(max(new[p] - old[p], 0.0) for p in shared)
    negative = sum(max(old[p] - new[p], 0.0) for p in shared)
    return {
        "interior_add": positive,
        "interior_remove": negative,
        "entered_mass": sum(new[p] for p in new.keys() - old.keys()),
        "exited_mass": sum(old[p] for p in old.keys() - new.keys()),
        "shared_prices": float(len(shared)),
    }


def rank_edit(old: list[tuple[int, float]], new: list[tuple[int, float]]) -> float:
    """Measure quantity movement when comparing occupied rank slots naively."""
    return sum(abs(q0 - q1) for (_, q0), (_, q1) in zip(old, new, strict=False))


@define
class DepthReplay:
    """Causal snapshot/diff replay with explicit invalid and unknown intervals."""

    bids: Book = field(factory=dict)
    asks: Book = field(factory=dict)
    update_id: int | None = None
    is_valid: bool = False
    snapshot_received_ns: int | None = None
    bid_boundary: int | None = None
    ask_boundary: int | None = None
    has_bridged: bool = False
    gap_count: int = 0
    ignored_updates: int = 0
    deletion_count: int = 0
    snapshot_count: int = 0
    invalid_reason: str = "missing_snapshot"

    def snapshot(self, data: Payload, received_ns: int) -> None:
        """Install a recorded snapshot only at its receive position."""
        bids = {p: q for p, q in levels(data["bids"]).items() if q > 0}
        asks = {p: q for p, q in levels(data["asks"]).items() if q > 0}
        if not bids or not asks or max(bids) >= min(asks):
            raise ValueError("Empty or crossed initial snapshot")
        self.bids, self.asks = bids, asks
        self.update_id = int(data["lastUpdateId"])
        self.snapshot_received_ns = received_ns
        self.bid_boundary, self.ask_boundary = min(bids), max(asks)
        self.has_bridged = False
        self.is_valid = True
        self.snapshot_count += 1
        self.invalid_reason = ""

    def update(self, data: Payload, received_ns: int) -> str:
        """Apply only a continuous update available after snapshot receipt."""
        first, last = int(data["U"]), int(data["u"])
        if first > last:
            raise ValueError("Reversed update interval")
        if not self.is_valid or self.update_id is None:
            return "invalid"
        if self.snapshot_received_ns is None or received_ns < self.snapshot_received_ns:
            raise ValueError("Attempt to backfill a snapshot into earlier receive time")
        if last <= self.update_id:
            self.ignored_updates += 1
            return "covered"
        if first > self.update_id + 1:
            self.gap_count += 1
            self.is_valid = False
            self.invalid_reason = "sequence_gap"
            return "gap"
        for book, updates in ((self.bids, data["b"]), (self.asks, data["a"])):
            for p, q in levels(updates).items():
                if q == 0:
                    book.pop(p, None)
                    self.deletion_count += 1
                else:
                    book[p] = q
        self.update_id = last
        self.has_bridged = True
        if not self.bids or not self.asks or max(self.bids) >= min(self.asks):
            self.is_valid = False
            self.invalid_reason = "empty_or_crossed_book"
            return "invalid_book"
        return "applied"

    def top(self, count: int = 25) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Return known occupied levels without filling unknown quantities."""
        return top_levels(self.bids, count, "bid"), top_levels(self.asks, count, "ask")

    def has_known_top(self, count: int = 25) -> bool:
        """Require requested levels to remain inside original snapshot coverage."""
        if not self.is_valid or not self.has_bridged:
            return False
        bid, ask = self.top(count)
        return (
            len(bid) == count
            and len(ask) == count
            and self.bid_boundary is not None
            and self.ask_boundary is not None
            and bid[-1][0] >= self.bid_boundary
            and ask[-1][0] <= self.ask_boundary
        )


def probe_perturbations() -> dict[str, object]:
    """Check analytical symmetries and distinguish co-change information losses."""
    raw = [(102, 2.0), (104, 7.0), (107, 3.0), (109, 4.0)]
    base = geometry(raw, 100.0, 1.0)
    variants = {
        "price_translate": ([(p + 800, q) for p, q in raw], 900.0, 1.0),
        "common_price_scale": ([(p * 3, q) for p, q in raw], 300.0, 3.0),
        "quantity_unit_scale": ([(p, q * 13) for p, q in raw], 100.0, 1.0),
        "mirror_side": ([(200 - p, q) for p, q in raw], 100.0, 1.0),
    }
    errors = {
        name: max(abs(a - b) for a, b in zip(base, geometry(v, anchor, scale), strict=False))
        for name, (v, anchor, scale) in variants.items()
    }
    old = {101: 2.0, 100: 7.0, 99: 3.0, 98: 4.0}
    new = {102: 5.0, 101: 2.0, 100: 7.0, 99: 3.0}
    rank_mass = rank_edit(top_levels(old, 4, "bid"), top_levels(new, 4, "bid"))
    edit = keyed_edit(old, new)
    # A non-injective normalization collapses different known quantity edits.
    clipped_first = [min(q, 1.0) for _, q in raw]
    clipped_second = [min(q * 2, 1.0) for _, q in raw]
    return {
        "geometry_max_errors": errors,
        "rank_shift": {"rank_l1": rank_mass, **edit},
        "quantity_clipping_collapses_distinct_books": clipped_first == clipped_second,
        "identification_limit": "Invertible transformations preserve information; clipping/coalescing can destroy it. Marginal resemblance cannot identify a generator.",
    }
