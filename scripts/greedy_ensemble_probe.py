"""Greedy equal-weight ensemble selection over banked fold-1 predictions: how much does an N-member ensemble add (diagnostic, selection-biased)?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from crossfold_blend import load_fold  # noqa: E402
from grade_prediction_bank import score  # noqa: E402


def main() -> int:
    y, mask, comp, models = load_fold(ROOT / "research/campaign_2026-10-06/bandit_continuation/prediction_bank/BANK_FOLD1.jsonl", 1,
                                      ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json", ROOT / "cache/valid_targets.npz")
    sub = np.arange(0, len(y), 3)  # every third evaluation sequence for speed during selection
    ys, ms = y[sub], mask[sub]
    names, arrs = [], {}
    for st, (n, arr, o) in models.items():
        names.append(n)
    scores = {}
    for st, (n, arr, o) in models.items():
        if "lead" in n:  # lookahead probes are not causal
            continue
        try:
            p = np.asarray(arr[o][sub], dtype=np.float32)
            scores[n] = (score(ys, ms, p)["wp"], p)
        except Exception:
            continue
    ranked = sorted(scores, key=lambda n: -scores[n][0])[:14]
    print("top singles (subsample):", [(n, round(scores[n][0], 5)) for n in ranked[:8]])
    chosen = [ranked[0]]; cur = scores[ranked[0]][1].copy()
    print(f"1 member: {scores[ranked[0]][0]:.5f} {chosen}")
    for k in range(2, 6):
        best = None
        for n in ranked:
            if n in chosen:
                continue
            w = score(ys, ms, (cur * (k - 1) + scores[n][1]) / k)["wp"]
            if best is None or w > best[0]:
                best = (w, n)
        chosen.append(best[1]); cur = (cur * (k - 1) + scores[best[1]][1]) / k
        print(f"{k} members: {best[0]:.5f} +{best[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
