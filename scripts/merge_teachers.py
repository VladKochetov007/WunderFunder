"""Merge two cross-fitted half teachers into one distillation target file per split.

    .venv/bin/python scripts/merge_teachers.py NAME FOLD

train: each sequence takes the teacher that did NOT train on it (half assignment as --train-half);
valid: mean of both (neither trained on validation). Writes cache/pred_NAME_fFOLD_{train,valid}.npy.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
name, fold = sys.argv[1], int(sys.argv[2])
a = {s: np.load(ROOT / "cache" / f"pred_{name}_h0_f{fold}_{s}.npy", mmap_mode="r") for s in ("train", "valid")}
b = {s: np.load(ROOT / "cache" / f"pred_{name}_h1_f{fold}_{s}.npy", mmap_mode="r") for s in ("train", "valid")}
rg = np.load(ROOT / "cache" / "train_targets.npz")["row_group"]
half = np.array([int(hashlib.sha256(f"half:{int(g)}".encode()).hexdigest(), 16) % 2 for g in rg])
out = np.lib.format.open_memmap(ROOT / "cache" / f"pred_{name}_f{fold}_train.npy", mode="w+", dtype=np.float32,
                                shape=a["train"].shape)
for i in range(len(rg)):  # teacher h0 trained on half 0 -> use it for half-1 sequences, and vice versa
    out[i] = a["train"][i] if half[i] == 1 else b["train"][i]
out.flush()
np.save(ROOT / "cache" / f"pred_{name}_f{fold}_valid.npy", ((np.asarray(a["valid"]) + np.asarray(b["valid"])) / 2).astype(np.float32))
print("merged", name, fold, "half sizes", np.bincount(half))
