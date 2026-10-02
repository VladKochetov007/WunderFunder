"""Raw per-row outputs of one member under several precisions for one wnn_native build, so two
builds can be compared bit for bit:  dump_precisions.py LIB WEIGHTS OUT.npz PREC [PREC ...]"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import engine_fp16_check as e  # noqa: E402

lib, weights, out, *precs = sys.argv[1:]
native = e.load_native(Path(lib))
member = e.load_npz(Path(weights))
seqs = list(e.read_sequences(e.VALID, [0]))
res = {p: e.run_direct(e.ensemble(native, [member], p), seqs) for p in precs}
for p in precs:
    print(p, native.EnsemblePredictor([member], [np.float32(1)], precision=p).precision)
np.savez(out, **res)
