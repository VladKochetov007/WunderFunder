"""Zero-fit adversarial validation: can sequence-level feature summaries tell train from valid?

Per sequence: mean, std and last-row value of every 4th raw feature column (rows 0..999). A
gradient-boosted classifier is cross-fitted over chain components (sequences linked in time stay
in one fold) and reports out-of-fold AUC. AUC ~0.5 means valid is distributed like train
(consistent with time-interleaving); AUC >> 0.5 means a systematic shift.
Usage: adversarial_validation.py OUT_JSON
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

NON = {"seq_ix", "step_in_seq", "need_prediction", "t0", "t1", "is_scored"}
CHAINS = "/home/vlad/development/wunder/research_tmp/chrono_root/cache/chronological_chains.npz"


def columns() -> list[str]:
    names = [n for n in pq.ParquetFile("datasets/train.parquet").schema_arrow.names if n not in NON]
    return names[::4]


def summarize(job: tuple[str, int, list[str]]) -> np.ndarray:
    split, group, cols = job
    t = pq.ParquetFile(f"datasets/{split}.parquet").read_row_group(group, columns=cols, use_threads=False)
    x = np.column_stack([t[c].to_numpy() for c in cols]).astype(np.float64)
    return np.concatenate([x.mean(0), x.std(0), x[-1]])


def components(split: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    parent = list(range(len(split)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in enumerate(nxt):
        if j >= 0:
            parent[find(i)] = find(int(j))
    return np.array([find(i) for i in range(len(split))])


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    z = np.load(CHAINS)
    split, groups = z["split"], z["row_group"]
    comp = components(split, z["next"])
    cols = columns()
    with ProcessPoolExecutor(4) as pool:
        feats = np.stack(list(pool.map(summarize, [(s, int(g), cols) for s, g in zip(split, groups)], chunksize=64)))
    np.savez(sys.argv[1].replace(".json", "_features.npz"), feats=feats, split=split, row_group=groups,
             component=comp, columns=np.array(cols))
    y = (split == "valid").astype(int)
    rng = np.random.default_rng(0)
    fold_of_comp = {c: f for c, f in zip(np.unique(comp), rng.integers(0, 5, len(np.unique(comp))))}
    folds = np.array([fold_of_comp[c] for c in comp])
    oof = np.zeros(len(y))
    for f in range(5):
        tr, te = folds != f, folds == f
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        clf.fit(feats[tr], y[tr])
        oof[te] = clf.predict_proba(feats[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    shuffled = float(np.mean([roc_auc_score(rng.permutation(y), oof) for _ in range(20)]))
    names = [f"{s}:{c}" for s in ("mean", "std", "last") for c in cols]
    uni = {n: float(max(roc_auc_score(y, feats[:, i]), 1 - roc_auc_score(y, feats[:, i]))) for i, n in enumerate(names)}
    top = dict(sorted(uni.items(), key=lambda kv: -kv[1])[:15])
    out = {"top_univariate_auc": top, "n_train": int((y == 0).sum()), "n_valid": int(y.sum()), "n_columns": len(cols),
           "features": int(feats.shape[1]), "oof_auc": auc, "label_shuffle_auc_mean": shuffled,
           "cv": "5-fold over chain components"}
    np.save(sys.argv[1].replace(".json", "_oof.npy"), oof)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
