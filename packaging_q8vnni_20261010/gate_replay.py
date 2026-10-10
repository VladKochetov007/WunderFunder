"""Delivered-ZIP gate run inside the offline Python 3.11 container (no GPU, no network).

Extracts the ZIP bytes, imports its solution.py, builds PredictionModel and replays fold-1 evaluation
sequences through the scorer callback (DataPoint per row). Checks: native library imported from the
requested level directory, reset between sequences (a replay of the first 120 rows after the whole
run is bitwise equal), determinism (a second fresh model gives bitwise identical predictions on
the first --det sequences), finite shape-(2,) outputs, no unexpected warmup predictions, no torch/CUDA
import. Records wall-clock us/row of the callback loop (feature construction included, parquet read
excluded) and stores predictions for the host-side metric comparison.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/workspace")
from utils import FEATURE_COLUMNS, DataPoint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--zip", type=Path, required=True)
ap.add_argument("--level", choices=("v2", "avx2", "v3"), required=True)
ap.add_argument("--split", type=Path, required=True)
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--sequences", type=int, default=30)
ap.add_argument("--det", type=int, default=2)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
assert sys.version_info[:2] == (3, 11)
assert not os.environ.get("LD_LIBRARY_PATH") and not os.environ.get("PYTHONPATH")
os.environ["WNN_NATIVE_LEVEL"] = a.level
groups = json.loads(a.split.read_text())["evaluation"]["valid"][: a.sequences]
pf = pq.ParquetFile(a.data)


def load(g):
    t = pf.read_row_group(g, columns=["seq_ix", "need_prediction", *FEATURE_COLUMNS], use_threads=False)
    f = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    return int(t["seq_ix"][0].as_py()), f, t["need_prediction"].to_numpy().astype(bool)


def replay(model, seq, feats, need):
    out = np.zeros((len(feats), 2), np.float32)
    for i in range(len(feats)):
        v = model.predict(DataPoint(seq, i, bool(need[i]), feats[i]))
        if not need[i]:
            assert v is None, f"warmup prediction at {i}"
            continue
        v = np.asarray(v, np.float32)
        assert v.shape == (2,) and np.isfinite(v).all(), f"bad output at {i}"
        out[i] = v
    return out


with tempfile.TemporaryDirectory(prefix="gate-") as tmp:
    pkg = Path(tmp)
    with zipfile.ZipFile(a.zip) as z:
        assert z.testzip() is None
        for m in z.namelist():
            assert not Path(m).is_absolute() and ".." not in Path(m).parts
        z.extractall(pkg)
    spec = importlib.util.spec_from_file_location("sol", pkg / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import wnn_native

    native = Path(wnn_native.__file__).resolve()
    assert native.parent == pkg / a.level, native
    model = mod.PredictionModel()
    prec = getattr(getattr(model, "model", None), "precision", None)
    preds, rows, elapsed, first = {}, 0, 0.0, None
    for k, g in enumerate(groups):
        seq, f, need = load(g)
        if k == 0:
            first = (seq, f, need)
        t0 = time.perf_counter()
        p = replay(model, seq, f, need)
        elapsed += time.perf_counter() - t0
        rows += len(f)
        preds[str(g)] = p
        print("seq", g, f"{elapsed / rows * 1e6:.1f} us/row", flush=True)
    seq, f, need = first
    reset = replay(model, seq, f[:120], need[:120])
    reset_ok = bool(np.array_equal(reset, preds[str(groups[0])][:120]))
    det_ok = True
    m2 = mod.PredictionModel()
    for g in groups[: a.det]:
        s, ff, nn = load(g)
        det_ok &= bool(np.array_equal(replay(m2, s, ff, nn), preds[str(g)]))
    cuda = [m for m in sys.modules if m.split(".")[0] in ("torch", "jax", "tensorflow")]
    report = {
        "zip_sha256": hashlib.sha256(a.zip.read_bytes()).hexdigest(),
        "zip_bytes": a.zip.stat().st_size,
        "python": sys.version,
        "numpy": np.__version__,
        "level": a.level,
        "resolved_precision": prec,
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "sequences": len(groups),
        "rows": rows,
        "us_per_row": elapsed / rows * 1e6,
        "reset_prefix_bitwise_equal": reset_ok,
        "determinism_bitwise_equal": det_ok,
        "ml_frameworks_imported": cuda,
        "passed": bool(reset_ok and det_ok and not cuda),
    }
np.savez(a.out.with_suffix(".npz"), groups=np.array(groups), **{f"pred_{g}": p for g, p in preds.items()})
a.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
