"""Shared helpers for the cr_mature probes: feature-only class-3 quote-path decoder and event runs.

The decoder reuses the TRAIN-fit code tables of scripts/de_target_law.py (target_law_tables.npz): a code hash of the
22 i0 price fields, a (code_prev, code_next) -> (d_ask, d_bid) move table and a code -> spread table. Quote paths
are cumulative sums of tabled moves; an unknown transition starts a new segment. Bid is anchored per segment so
that ask - bid equals the tabled spread. Event marker E: a6 == float32 lower endpoint (feature only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
PRICE = [f"i0_p{i}" for i in range(22)]
ENDPOINT = np.float32(-5.199337482452393)
TABLES = ROOT / "research/campaign_2026-10-01/data_eval/target_law_tables.npz"


class Decoder:
    def __init__(self) -> None:
        t = np.load(TABLES)
        self.mult = t["mult"]
        self.spread = dict(zip(t["spread_code"].tolist(), zip(t["spread_class"].tolist(), t["spread_ticks"].tolist())))
        self.moves = {(a, b): (int(d[0]), int(d[1])) for a, b, d in zip(t["move_prev"].tolist(), t["move_next"].tolist(), t["move_d"])}

    def code_hash(self, x: np.ndarray) -> np.ndarray:
        q = np.rint(x.astype(np.float64) * 1e4).astype(np.int64)
        with np.errstate(over="ignore"):
            h = (q * self.mult[None]).sum(1)
            h ^= h >> np.int64(29)
        return h

    def decode(self, x: np.ndarray, a6: np.ndarray) -> dict:
        """Per-row ask, bid (ticks, anchored), segment id (-1 = unanchored), event flag E. Feature-only."""
        code = self.code_hash(x)
        T = len(code)
        ev = np.flatnonzero(code[1:] != code[:-1]) + 1
        da = np.zeros(T, np.int64)
        db = np.zeros(T, np.int64)
        unk = np.zeros(T, np.int64)
        for e, a, b in zip(ev.tolist(), code[ev - 1].tolist(), code[ev].tolist(), strict=True):
            m = self.moves.get((a, b))
            if m is None:
                unk[e] = 1
            else:
                da[e], db[e] = m
        ask, bid, seg = np.cumsum(da), np.cumsum(db), np.cumsum(unk)
        sp = np.array([self.spread.get(c, (-1, -1))[1] for c in code.tolist()])
        off = ask - bid - sp
        seg_ok = np.zeros(seg.max() + 1, bool)
        shift = np.zeros(seg.max() + 1, np.int64)
        for g in np.unique(seg):
            on = (seg == g) & (sp >= 1)
            if on.any():
                vals, cnt = np.unique(off[on], return_counts=True)
                shift[g], seg_ok[g] = vals[cnt.argmax()], True
        bid = bid + shift[seg]
        seg = np.where(seg_ok[seg], seg, -1)
        return {"ask": ask, "bid": bid, "seg": seg, "E": a6.astype(np.float32) == ENDPOINT, "spread": sp}


def read_group(source: pq.ParquetFile, group: int, labels: bool = True):
    cols = PRICE + ["a6"] + (["t0", "t1"] if labels else [])
    t = source.read_row_group(group, columns=cols, use_threads=False)
    x = np.column_stack([t.column(k).to_numpy() for k in PRICE]).astype(np.float32)
    a6 = t.column("a6").to_numpy().astype(np.float32)
    y = np.column_stack([t.column(k).to_numpy() for k in ("t0", "t1")]).astype(np.float64) if labels else None
    return x, a6, y
