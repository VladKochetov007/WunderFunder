"""Predict full sequences (causal features -> LightGBM) for a list of parquet row groups; saves (N,20000,2) float32."""
import sys; sys.path.append("/tmp/claude-1000/pylib/lgb")
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, lightgbm as lgb
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gbm_features import sequence_features, ROOT  # noqa
from wnn.data import SequenceFile  # noqa

MODELDIR = Path(sys.argv[1]); PARQUET = sys.argv[2]; GROUPS = json.loads(Path(sys.argv[3]).read_text()); OUT = Path(sys.argv[4])
_M = None


def work(g):
    global _M
    if _M is None:
        keep = np.load(MODELDIR / "kept_idx.npy")
        _M = (keep, [lgb.Booster(model_file=str(MODELDIR / f"lgb_t{k}.txt")) for k in range(2)])
    keep, models = _M
    s = SequenceFile(ROOT / "datasets" / PARQUET).read(g, targets=False, mask=False)
    _, F = sequence_features(s.features)
    F = F[:, keep]
    return np.stack([np.clip(m.predict(F, num_threads=1), -2, 2) for m in models], 1).astype(np.float32)


if __name__ == "__main__":
    with Pool(1, maxtasksperchild=10) as pool:
        res = list(pool.imap(work, GROUPS, chunksize=1))
    np.save(OUT, np.stack(res))
    print("saved", OUT, len(res))
