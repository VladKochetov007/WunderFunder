"""Raw per-row outputs of a blend (equal coefficients) under several precisions for one wnn_native
build, so builds can be compared bit for bit and against fp32:
    pair_dump.py LIB OUT.npz GROUPS PREC[,PREC..] WEIGHTS [WEIGHTS ...]
GROUPS is a comma list of datasets/valid.parquet row groups (one 20k-row sequence each)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "scripts"))
import engine_fp16_check as e  # noqa: E402

lib, out, groups, precs, *weights = sys.argv[1:]
native = e.load_native(Path(lib))
members = [e.load_npz(Path(w)) for w in weights]
seqs = list(e.read_sequences(e.VALID, [int(g) for g in groups.split(",")]))
res = {}
for p in precs.split(","):
    model = e.ensemble(native, members, p)
    res[p] = e.run_direct(model, seqs)
    print(p, "effective", model.precision, flush=True)
    if p.startswith("q16x"):
        res[p + "_repeat"] = e.run_direct(e.ensemble(native, members, p), seqs)
np.savez(out, groups=np.array([int(g) for g in groups.split(",")]), **res)
