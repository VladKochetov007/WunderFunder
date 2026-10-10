"""Can the market class of a sequence be read from its first rows (causal, label-free)?

For sequences of the validation set the class comes from the labels (wnn.data.market_class).  Features: per column, the fraction
of changing rows and the standard deviation of first differences over the first N rows.  Logistic regression with cross-validation
(grouped by sequence), balanced accuracy reported.  If class is predictable early, a per-class output map becomes possible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from utils import FEATURE_COLUMNS  # noqa: E402
from wnn.data import market_class  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--sequences", type=int, default=600)
    args = parser.parse_args()
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as cache:
        groups, y, mask = cache["row_group"], cache["y"], cache["mask"]
    pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    pick = np.linspace(0, len(groups) - 1, min(args.sequences, len(groups))).astype(int)
    feats, classes = [], []
    for i in pick:
        table = pf.read_row_group(int(groups[i]), columns=list(FEATURE_COLUMNS), use_threads=False)
        x = np.column_stack([table[c].to_numpy()[: args.rows] for c in FEATURE_COLUMNS]).astype(np.float64)
        d = np.diff(x, axis=0)
        feats.append(np.concatenate([(d != 0).mean(0), np.log1p(np.abs(d).std(0))]))
        classes.append(market_class(y[i][mask[i]]))
    X, c = np.array(feats), np.array(classes)
    print("class counts:", {int(k): int((c == k).sum()) for k in np.unique(c)})
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    pred = cross_val_predict(model, X, c, cv=5)
    print(f"balanced accuracy ({args.rows} warm-up rows, {len(c)} sequences, 5-fold): {balanced_accuracy_score(c, pred):.3f}  (chance 0.25)")
    print("per-class recall:", {int(k): round(float((pred[c == k] == k).mean()), 3) for k in np.unique(c)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
