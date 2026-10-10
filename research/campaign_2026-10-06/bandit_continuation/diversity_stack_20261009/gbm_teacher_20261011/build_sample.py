"""Build the subsampled fold-1 training feature matrix (importance-sampled rows, stratified by |clip y|)."""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gbm_features import sequence_features, ROOT  # noqa
from wnn.data import SequenceFile  # noqa

OUT = Path(sys.argv[1]); NSEQ = int(sys.argv[2]); SEED = int(sys.argv[3]); KEEP = None if sys.argv[4] == "-" else np.load(sys.argv[4]); P_BASE, P_HEAVY = float(sys.argv[5]), float(sys.argv[6])


def work(args):
    g, comp = args
    f = SequenceFile(ROOT / "datasets/train.parquet")
    s = f.read(g)
    names, F = sequence_features(s.features)
    y = np.clip(s.targets.astype(np.float64), -2, 2)
    rng = np.random.default_rng(1000 + g)
    heavy = (np.abs(y) > 1.0).any(1)
    p = P_BASE + P_HEAVY * heavy
    p[:99] = 0
    keep = rng.random(len(p)) < p
    idx = np.flatnonzero(keep)
    Fk = F[idx] if KEEP is None else F[idx][:, KEEP]
    return g, comp, idx, Fk, y[idx], 1.0 / p[idx], names


if __name__ == "__main__":
    split = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json").read_text())
    comp = {r["row_group"]: r["component"] for r in split["sequences"]["train"]}
    fit = [g for g in split["fit"]["train"] if g not in (528,)]
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(fit, NSEQ, replace=False).tolist())
    t = time.time()
    with Pool(1, maxtasksperchild=10) as pool:
        res = list(pool.imap(work, [(g, comp[g]) for g in pick], chunksize=1))
    names = res[0][6]
    np.savez(OUT, X=np.concatenate([r[3] for r in res]), y=np.concatenate([r[4] for r in res]).astype(np.float32),
             isw=np.concatenate([r[5] for r in res]).astype(np.float32),
             group=np.concatenate([np.full(len(r[2]), r[0]) for r in res]), comp=np.concatenate([np.full(len(r[2]), r[1]) for r in res]),
             row=np.concatenate([r[2] for r in res]), names=np.array(names))
    print("done", time.time() - t, sum(len(r[2]) for r in res))
