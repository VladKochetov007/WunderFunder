"""Organizer baseline predictions on a whole labelled split in sequence mode.

The ONNX graph is two GRU ops; making the time axis symbolic lets one session
call process a full 20,000-row sequence. Rows are independent across processes.

    scripts/limited .venv/bin/python scripts/predict_baseline.py valid
Writes cache/pred_baseline_<split>.npy (B, T, 2) float32 in cache row order.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        import onnx
        import onnxruntime as ort

        m = onnx.load(ROOT / "baseline" / "baseline.onnx")
        for v in list(m.graph.input) + list(m.graph.output):
            if v.name in ("features", "prediction"):
                v.type.tensor_type.shape.dim[1].dim_param = "T"
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 1
        opt.inter_op_num_threads = 1
        _SESSION = ort.InferenceSession(m.SerializeToString(), opt, providers=["CPUExecutionProvider"])
    return _SESSION


def _run(args):
    split, g = args
    from wnn.data import SequenceFile

    x = SequenceFile(ROOT / "datasets" / f"{split}.parquet").read(g, targets=False, mask=False).features
    h = np.zeros((1, 1, 128), np.float32)
    out = _session().run(None, {"features": x[None], "hidden_0": h, "hidden_1": h})[0]
    return out[0].astype(np.float32)


def main():
    split = sys.argv[1]
    groups = np.load(ROOT / "cache" / f"{split}_targets.npz")["row_group"]
    workers = int(os.environ.get("WNN_THREADS", "8"))
    pred = np.lib.format.open_memmap(ROOT / "cache" / f"pred_baseline_{split}.npy", mode="w+", dtype=np.float32,
                                     shape=(len(groups), 20_000, 2))
    with ProcessPoolExecutor(workers) as ex:
        for i, p in enumerate(ex.map(_run, [(split, int(g)) for g in groups], chunksize=4)):
            pred[i] = p
    pred.flush()


if __name__ == "__main__":
    main()
