from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PRICE_COLUMNS = [f"i0_p{index}" for index in range(22)]
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
UNKNOWN = 255
ROWS = 20_000


def sha256(path: Path) -> str:
    """Hash files with bounded working memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def packed_codes(prices: np.ndarray) -> np.ndarray:
    """Encode exact provided float32 row values with the predeclared grammar."""
    rounded = np.ascontiguousarray(np.round(prices.astype(np.float32), 4), dtype="<f4")
    return rounded.view("V88").reshape(-1)


def lookup(prices: np.ndarray, table: dict[bytes, int]) -> np.ndarray:
    """Return spread bins/255unknown using current prices alone."""
    codes = packed_codes(prices)
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    stops = np.r_[starts[1:], len(codes)]
    values = np.fromiter(
        (table.get(bytes(codes[index]), UNKNOWN) for index in starts),
        dtype=np.uint8,
        count=len(starts),
    )
    return np.repeat(values, stops - starts)


def load_table(path: Path) -> dict[bytes, int]:
    """Load accepted pooled code labels; class is never an inference input."""
    return {
        bytes.fromhex(row["code_float32_round4_hex"]): min(row["spread_ticks"], 12) - 1
        for row in json.loads(path.read_text())
    }


def read_prices(source: pq.ParquetFile, group: int) -> np.ndarray:
    """Read only22 current price columns from a PyArrow parquet reader."""
    table = source.read_row_group(group, columns=PRICE_COLUMNS, use_threads=False)
    if table.num_rows != ROWS:
        raise ValueError("unexpected row-group length")
    return np.column_stack([table.column(name).to_numpy() for name in PRICE_COLUMNS]).astype(
        np.float32
    )
