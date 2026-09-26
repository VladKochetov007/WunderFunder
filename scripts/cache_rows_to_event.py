"""Per-row distance to the next / since the last i0 price-code event, all validation sequences.

    WNN_THREADS=4 scripts/limited .venv/bin/python scripts/cache_rows_to_event.py
Targets only change on i0 price-code rows (physicist F7), so rows-to-next-event is the natural
axis for where a model gains or loses. Writes cache/rte_valid.npz: to_next, since_last (int16,
capped at 30000), aligned with cache/valid_targets.npz.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SF = None


def _init():
    global _SF
    from wnn.data import SequenceFile
    _SF = SequenceFile(ROOT / "datasets" / "valid.parquet")


def _one(args):
    i, g = args
    x = _SF.read(int(g), targets=False, mask=False).features
    T = len(x)
    ev = np.flatnonzero(np.r_[False, (x[1:, 0:22] != x[:-1, 0:22]).any(1)])
    t = np.arange(T)
    k = np.searchsorted(ev, t, side="right")
    nxt = np.where(k < len(ev), ev[np.minimum(k, len(ev) - 1)], T + 30000)
    j = np.searchsorted(ev, t, side="right") - 1
    last = np.where(j >= 0, ev[np.maximum(j, 0)], -30000)
    return i, np.minimum(nxt - t, 30000).astype(np.int16), np.minimum(t - last, 30000).astype(np.int16)


def main():
    rg = np.load(ROOT / "cache" / "valid_targets.npz")["row_group"]
    to_next = np.zeros((len(rg), 20000), np.int16)
    since = np.zeros_like(to_next)
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "4")), initializer=_init) as ex:
        for i, a, b in ex.map(_one, [(i, int(g)) for i, g in enumerate(rg)], chunksize=8):
            to_next[i], since[i] = a, b
    np.savez(ROOT / "cache" / "rte_valid.npz", to_next=to_next, since_last=since)
    print("saved", to_next.shape, "median to_next", int(np.median(to_next)))


if __name__ == "__main__":
    main()
