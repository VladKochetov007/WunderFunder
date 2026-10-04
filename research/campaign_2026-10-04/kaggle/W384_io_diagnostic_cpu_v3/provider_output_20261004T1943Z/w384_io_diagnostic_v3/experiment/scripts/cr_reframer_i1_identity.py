"""R01/R03/R05 probe: is i1 a decodable, non-duplicate instrument stream on i0's encoder? Are volume codes censored?

Samples validation sequences per market class and reports
  * i0 / i1 price-code event rates per class,
  * vocabulary overlap: share of i1 22-column price codes (rounded 1e-4) present in the i0 code vocabulary of each class,
  * duplicate streams: share of i1 52-column states found verbatim among i0 52-column states of the sampled sequences,
  * censoring: share of i0/i1 volume values at each column's maximum / minimum code, per class.

    WNN_CPU_PCT=6 WNN_MEM_MAX=6G scripts/limited .venv/bin/python scripts/cr_reframer_i1_identity.py OUT.json [PER_CLASS]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.data import FEATURE_COLUMNS  # noqa: E402

I0, I1 = slice(0, 52), slice(52, 104)


def keyset(block: np.ndarray) -> np.ndarray:
    """Row hashes of a rounded block (void view of int32 codes)."""
    r = np.ascontiguousarray(np.rint(block * 1e4).astype(np.int32))
    return r.view(np.dtype((np.void, r.shape[1] * 4))).ravel()


def main() -> None:
    out = Path(sys.argv[1])
    per_class = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    with np.load(ROOT / "cache/valid_targets.npz") as z:
        rg, cls = z["row_group"].astype(int), z["market_class"].astype(int)
    rng = np.random.default_rng(7)
    picks = {c: rng.choice(rg[cls == c], per_class, replace=False) for c in range(4)}
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    vocab_i0 = {c: set() for c in range(4)}
    i1_codes = {c: [] for c in range(4)}
    i0_states, i1_states = [], []
    stats = {c: Counter() for c in range(4)}
    vol_hi = {c: {"i0": [], "i1": []} for c in range(4)}
    vol_lo = {c: {"i0": [], "i1": []} for c in range(4)}
    col_max = {}
    for c in range(4):
        for g in picks[c]:
            t = pf.read_row_group(int(g), columns=FEATURE_COLUMNS)
            x = np.column_stack([t.column(k).to_numpy() for k in FEATURE_COLUMNS]).astype(np.float32)
            p0, p1 = x[:, 0:22], x[:, 52:74]
            e0 = np.any(p0[1:] != p0[:-1], 1)
            e1 = np.any(p1[1:] != p1[:-1], 1)
            stats[c]["rows"] += len(e0)
            stats[c]["i0_px_events"] += int(e0.sum())
            stats[c]["i1_px_events"] += int(e1.sum())
            k0 = keyset(p0)
            k1 = keyset(p1)
            vocab_i0[c].update(np.unique(k0).tolist())
            i1_codes[c].append(np.unique(k1))
            stats[c]["i0_codes"] += len(np.unique(k0))
            stats[c]["i1_codes"] += len(np.unique(k1))
            i0_states.append((int(g), np.unique(keyset(x[:, I0]))))
            i1_states.append((int(g), c, np.unique(keyset(x[:, I1]))))
            for name, off in (("i0", 0), ("i1", 52)):
                v = x[:, off + 22:off + 44]
                vol_hi[c][name].append(v)
            print("read", c, g, flush=True)
    # censoring: per column global max/min over all sampled rows (any class)
    report = {"per_class": per_class, "classes": {}}
    for name in ("i0", "i1"):
        allv = np.concatenate([np.concatenate(vol_hi[c][name]) for c in range(4)])
        hi, lo = allv.max(0), allv.min(0)
        sentinel = np.isclose(allv, -5.19934, atol=1e-3)
        lo_ns = np.where(sentinel, np.inf, allv).min(0)
        col_max[name] = {"max": hi.round(4).tolist(), "min_non_sentinel": lo_ns.round(4).tolist()}
        for c in range(4):
            v = np.concatenate(vol_hi[c][name])
            s = np.isclose(v, -5.19934, atol=1e-3)
            at_hi = np.isclose(v, hi[None], atol=1e-4)
            at_lo = np.isclose(v, lo_ns[None], atol=1e-4)
            report["classes"].setdefault(str(c), {})[f"{name}_vol_at_max_share_by_col"] = at_hi.mean(0).round(4).tolist()
            report["classes"][str(c)][f"{name}_vol_at_min_share_by_col"] = at_lo.mean(0).round(4).tolist()
            report["classes"][str(c)][f"{name}_vol_sentinel_share"] = float(s.mean())
            report["classes"][str(c)][f"{name}_vol_rows_any_at_max"] = float(at_hi.any(1).mean())
            report["classes"][str(c)][f"{name}_vol_distinct_values_median_col"] = float(
                np.median([len(np.unique(v[:, j])) for j in range(v.shape[1])]))
    report["vol_code_range"] = col_max
    # vocabulary overlap of i1 codes with each class's i0 vocabulary
    for c in range(4):
        u = np.unique(np.concatenate(i1_codes[c]))
        overlap = {str(d): float(np.mean([k in vocab_i0[d] for k in u.tolist()])) for d in range(4)}
        report["classes"][str(c)].update({
            "i0_px_event_rate": stats[c]["i0_px_events"] / stats[c]["rows"],
            "i1_px_event_rate": stats[c]["i1_px_events"] / stats[c]["rows"],
            "i0_vocab_size": len(vocab_i0[c]),
            "i1_vocab_size": int(len(u)),
            "i1_code_share_in_i0_vocab_of_class": overlap,
        })
    # duplicate streams: i1 states of one sequence found among i0 states of other sampled sequences
    all_i0 = {}
    for g, keys in i0_states:
        for k in keys.tolist():
            all_i0.setdefault(k, set()).add(g)
    dup = {c: [] for c in range(4)}
    for g, c, keys in i1_states:
        hits = [k in all_i0 and (all_i0[k] - {g}) for k in keys.tolist()]
        dup[c].append(float(np.mean([bool(h) for h in hits])))
    for c in range(4):
        report["classes"][str(c)]["i1_state_share_found_as_other_i0_state"] = {
            "mean": float(np.mean(dup[c])), "max": float(np.max(dup[c]))}
    out.write_text(json.dumps(report, indent=1))
    for c in range(4):
        r = report["classes"][str(c)]
        print(c, {k: r[k] for k in ("i0_px_event_rate", "i1_px_event_rate", "i0_vocab_size", "i1_vocab_size",
                                   "i1_code_share_in_i0_vocab_of_class", "i1_state_share_found_as_other_i0_state",
                                   "i0_vol_rows_any_at_max", "i1_vol_rows_any_at_max", "i0_vol_sentinel_share")})


if __name__ == "__main__":
    main()
