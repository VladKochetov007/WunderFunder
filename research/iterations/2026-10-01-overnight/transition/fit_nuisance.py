"""Fit legal-fold code centers/priors from four raw columns; root launches this once allocated."""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from features import SCHEMA, detect, encode


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_groups(cf, fold):
    allowed = (cf["split"] == "valid") & ((cf["fold"] != fold) if fold >= 0 else True)
    chosen = []
    for chain in np.unique(cf["chain"][allowed]):
        selected = allowed & (cf["chain"] == chain)
        chosen.append({"chain": int(chain), "row_group": int(cf["row_group"][selected].min())})
    if fold >= 0:
        held_chains = set(cf["chain"][(cf["split"] == "valid") & (cf["fold"] == fold)].tolist())
        if any(row["chain"] in held_chains for row in chosen):
            raise ValueError("nuisance training chain overlaps held-out fold")
    return sorted(chosen, key=lambda d: d["chain"])


def fit_values(sequences):
    values, weights = [], []
    for x in sequences:
        if len(x):
            ids = np.linspace(0, len(x)-1, min(512, len(x))).astype(int)
            values.append(x[ids])
            weights.append(np.full(len(ids), 1.0/len(ids)))
    if not values:
        raise ValueError("no detected training marks")
    x, weight = np.concatenate(values), np.concatenate(weights)
    centers = []
    for j in range(4):
        order = np.argsort(x[:, j], kind="stable")
        position = min(np.searchsorted(np.cumsum(weight[order]), .5*weight.sum()), len(order)-1)
        centers.append(float(x[order[position], j]))
    total = np.zeros(16)
    transitions = np.zeros((16, 16))
    nonempty = 0
    for values in sequences:
        m = encode(values, np.asarray(centers))
        if len(m):
            total += np.bincount(m, minlength=16)/len(m)
            nonempty += 1
        if len(m) > 1:
            transitions += np.bincount(m[:-1]*16+m[1:], minlength=256).reshape(16,16)/(len(m)-1)
    pi = (total+1/16)/(nonempty+1)
    R = (transitions+pi[None,:])/(transitions.sum(1)[:,None]+1)
    return {"centers": centers, "pi": pi.tolist(), "R": R.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=[-1, 0, 1, 2, 3], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    start = time.monotonic()
    chain_file = args.root/"cache/chain_folds.npz"
    source = args.root/"datasets/valid.parquet"
    stat = source.stat()
    with source.open("rb") as handle:
        handle.seek(-8, 2)
        footer_length, magic = struct.unpack("<I4s", handle.read(8))
        if magic != b"PAR1":
            raise ValueError("parquet footer")
        handle.seek(-footer_length-8, 2)
        footer_hash = hashlib.sha256(handle.read()).hexdigest()
    chosen = select_groups(np.load(chain_file), args.fold)
    pf = pq.ParquetFile(source)
    columns = [f"i0_dv{j}" for j in range(4)]
    sequences = []
    for row in chosen:
        table = pf.read_row_group(row["row_group"], columns=columns, use_threads=False)
        x = np.column_stack([table[c].to_numpy() for c in columns]).astype(np.float32)
        positions = detect(x)
        sequences.append(x[positions])
        row.update(raw4_sha256=hashlib.sha256(x.tobytes()).hexdigest(), rows=len(x), events=len(positions))
    result = {"schema": SCHEMA, "fold": args.fold, "selected": chosen, "fitted": fit_values(sequences),
              "source": {"path": str(source.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                         "footer_sha256": footer_hash, "columns": columns},
              "chain_file_sha256": digest(chain_file), "code_sha256": digest(__file__),
              "feature_code_sha256": digest(Path(__file__).with_name("features.py")),
              "targets_read": False, "classes_read": False, "mask_read": False,
              "runtime_seconds": time.monotonic()-start}
    if source.stat().st_mtime_ns != stat.st_mtime_ns or source.stat().st_size != stat.st_size:
        raise RuntimeError("source changed during fitting")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.out), "chains": len(chosen), "events": sum(row["events"] for row in chosen), "runtime_seconds": result["runtime_seconds"]}))

if __name__ == "__main__":
    main()
