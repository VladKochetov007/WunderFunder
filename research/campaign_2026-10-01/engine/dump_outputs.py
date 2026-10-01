"""Dump raw EnsemblePredictor outputs (2 incumbent GRU128 members, valid row group 0) for
cross-build / regression comparisons:  dump_outputs.py LIB PRECISION OUT.npy"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import engine_fp16_check as c

native = c.load_native(Path(sys.argv[1]))
members = [c.load_npz(p) for p in c.H128]
kw = {} if sys.argv[2] == "legacy" else {"precision": sys.argv[2]}
model = native.EnsemblePredictor(members, [np.float32(0.5)] * 2, fast_activations=True, **kw)
np.save(sys.argv[3], c.run_direct(model, list(c.read_sequences(c.VALID, [0]))))
