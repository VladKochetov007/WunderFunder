"""Stream every released sequence to test the claimed target/event implication."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    """Count all need/scored target changes, with bounded examples of exceptions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    start = time.time()
    result = {"training_steps": 0, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "all row groups, four price-code columns, labels and official masks; no complete feature audit",
              "splits": {}}
    for split in ["train", "valid"]:
        cache = np.load(args.cache / f"{split}_targets.npz")
        classes = dict(zip(cache["row_group"].tolist(), cache["market_class"].tolist()))
        path = args.data / f"{split}.parquet"
        pf = pq.ParquetFile(path)
        rows = []
        examples = []
        columns = ["seq_ix", "need_prediction", "i0_p0", "i0_p11", "i1_p0", "i1_p11", "t0", "t1"]
        if split == "valid":
            columns.append("is_scored")
        for group in range(pf.num_row_groups):
            table = pf.read_row_group(group, columns=columns, use_threads=False)
            x = np.column_stack([table[c].to_numpy() for c in columns[2:6]])
            y = np.column_stack([table[c].to_numpy() for c in ["t0", "t1"]])
            need = table["need_prediction"].to_numpy().astype(bool)
            selected = need & table["is_scored"].to_numpy().astype(bool) if split == "valid" else need
            assert len(y) == 20_000 and not need[:99].any()
            assert np.isfinite(x).all() and np.isfinite(y).all()
            change = np.r_[False, np.any(y[1:] != y[:-1], axis=1)]
            i0and = np.r_[False, np.all(x[1:, :2] != x[:-1, :2], axis=1)]
            i0or = np.r_[False, np.any(x[1:, :2] != x[:-1, :2], axis=1)]
            i1or = np.r_[False, np.any(x[1:, 2:] != x[:-1, 2:], axis=1)]
            bad = change & ~i0and & need
            age = np.arange(len(y)) - np.maximum.accumulate(np.where(i0and, np.arange(len(y)), 0))
            mass = np.abs(np.clip(y.astype(np.float64), -2, 2))
            absdelta = np.r_[0., np.max(np.abs(y[1:].astype(np.float64) - y[:-1]), axis=1)]
            row = {"row_group": group, "seq_ix": int(table["seq_ix"][0].as_py()),
                   "market_class": classes.get(group, -1), "rows": len(y), "need": int(need.sum()),
                   "scored": int(selected.sum()), "changes": int((change & need).sum()),
                   "outside_and": int(bad.sum()), "outside_or": int((change & ~i0or & need).sum()),
                   "outside_both_instruments_or": int((change & ~i0or & ~i1or & need).sum()),
                   "outside_and_scored": int((bad & selected).sum()),
                   "outside_and_max_abs_delta": float(absdelta[bad].max(initial=0)),
                   "selected_mass": mass[selected].sum(0).tolist(),
                   "age64_selected_mass": mass[selected & (age >= 64)].sum(0).tolist(),
                   "age64_need_mass": mass[need & (age >= 64)].sum(0).tolist(),
                   "need_mass": mass[need].sum(0).tolist(),
                   "clipped_selected_mass": (mass * (np.abs(y) >= 2))[selected].sum(0).tolist()}
            rows.append(row)
            if len(examples) < 25:
                for step in np.flatnonzero(bad)[:25-len(examples)]:
                    examples.append({"row_group": group, "step": int(step), "class": classes.get(group, -1),
                                     "scored": bool(selected[step]), "delta_y": (y[step]-y[step-1]).tolist(),
                                     "delta_price_codes": (x[step]-x[step-1]).tolist()})
            if group % 1000 == 0:
                print(split, group, "of", pf.num_row_groups, flush=True)
        aggregate = {}
        for cls in [-1, 0, 1, 2, 3]:
            part = [r for r in rows if r["market_class"] == cls]
            if not part:
                continue
            keys = [k for k in part[0] if k not in ["row_group", "seq_ix", "market_class", "outside_and_max_abs_delta"]]
            aggregate[str(cls)] = {k: np.array([r[k] for r in part]).sum(0).tolist() for k in keys}
            aggregate[str(cls)]["sequences"] = len(part)
            aggregate[str(cls)]["max_exception_delta"] = max(r["outside_and_max_abs_delta"] for r in part)
        result["splits"][split] = {"path": str(path), "bytes": path.stat().st_size,
                                   "row_groups": len(rows), "aggregate": aggregate,
                                   "examples": examples, "sequence_statistics": rows}
        del pf
    result["elapsed_seconds"] = time.time() - start
    result["maxrss_linux_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("done", result["elapsed_seconds"], "seconds", flush=True)


if __name__ == "__main__":
    main()
