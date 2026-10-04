"""Score equal-weight K-member blends of fold models on pure held-out chains.

    scripts/limited .venv/bin/python scripts/ensemble_scan.py \
        --reference ab62_since+ab62_base_s4 \
        --sets "four=ab62_since+ab62_base_s4+ab68_ema6_since_ep5+ab73_ema6_since_s4_ep5" \
        --pool ab62_since,ab62_base_s4,ab68_ema6_since_ep5,... --k-max 4

Each member name is a fold prefix (``cache/pred_<name>_f<k>_valid.npy``); ``baseline`` is the
train-only organizer model scaled by 0.25 (``cache/pred_baseline_valid.npy``). Blends are clipped
to +-2 before scoring, exactly like the scorer. Predeclared ``--sets`` are the decision
evidence; the greedy forward search over ``--pool`` is exploratory (it selects on the same rows).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import decide
import numpy as np

LAMBDAS = (0.0, 0.02, 0.05)
BASELINE_SCALE = 0.25


def parse_blend(text: str) -> dict[str, float]:
    """Parse ``a+b+baseline*0.1`` into member weights; unweighted members share the rest equally."""
    fixed, free = {}, []
    for part in text.split("+"):
        name, _, weight = part.partition("*")
        if weight:
            fixed[name] = float(weight)
        else:
            free.append(name)
    rest = 1.0 - sum(fixed.values())
    if free and rest <= 0:
        raise ValueError(f"no weight left for equal members in {text!r}")
    return {**fixed, **{name: rest / len(free) for name in free}}


class FoldPreds:
    """Hold member predictions for the selected sequences of each fold in memory."""

    def __init__(self, folds: list[int], idx_by_fold: dict[int, np.ndarray]):
        self.folds = folds
        self.idx = idx_by_fold
        self.cache: dict[str, dict[int, np.ndarray]] = {}

    def get(self, name: str) -> dict[int, np.ndarray]:
        if name not in self.cache:
            out = {}
            if name == "baseline":
                full = np.load(decide.ROOT / "cache" / "pred_baseline_valid.npy", mmap_mode="r")
                for k in self.folds:
                    out[k] = np.asarray(full[self.idx[k]], np.float32) * BASELINE_SCALE
            else:
                for k in self.folds:
                    path = decide.ROOT / "cache" / f"pred_{name}_f{k}_valid.npy"
                    out[k] = np.asarray(np.load(path, mmap_mode="r")[self.idx[k]], np.float32)
            self.cache[name] = out
        return self.cache[name]


def blend_sums(preds: FoldPreds, blend: dict[str, float], y, mask) -> np.ndarray:
    """Return per-sequence (scored/unscored, target, six moments) sums in fold order."""
    parts = []
    need = np.arange(y.shape[1]) >= decide.WARMUP
    for k in preds.folds:
        idx = preds.idx[k]
        out = np.zeros((len(idx), 2, 2, 6), np.float64)
        for start in range(0, len(idx), 64):
            sl = slice(start, start + 64)
            rows = idx[sl]
            pred = sum(w * preds.get(name)[k][sl].astype(np.float64) for name, w in blend.items())
            pred = np.clip(pred, -2, 2)
            target = np.clip(y[rows].astype(np.float64), -2, 2)
            for t in range(2):
                out[sl, 0, t] = decide._sums6(target[..., t], pred[..., t], need & mask[rows])
                out[sl, 1, t] = decide._sums6(target[..., t], pred[..., t], need & ~mask[rows])
        parts.append(out)
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="blend spec, e.g. a+b")
    parser.add_argument("--sets", default="", help='predeclared "name=a+b+c;name2=..."')
    parser.add_argument("--pool", default="", help="comma list for exploratory greedy search")
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--all-chains", action="store_true", help="keep chains linked to train data")
    parser.add_argument("--boot", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    folds = [int(v) for v in args.folds.split(",")]
    with np.load(decide.ROOT / "cache" / "valid_targets.npz") as valid:
        y, mask, classes = valid["y"], valid["mask"], valid["market_class"]
    with np.load(decide.ROOT / "cache" / "train_targets.npz") as train:
        train_mix = np.bincount(train["market_class"], minlength=4) / len(train["market_class"])
    chain, fold = decide._chains()
    keep = np.isin(fold, folds)
    if not args.all_chains:
        chain_data = np.load(decide.ROOT / "cache" / "chain_folds.npz")
        keep &= ~np.isin(chain, chain_data["chain"][chain_data["split"] == "train"])
    idx_by_fold = {k: np.flatnonzero(keep & (fold == k)) for k in folds}
    order = np.concatenate([idx_by_fold[k] for k in folds])
    seq_fold, seq_chain, seq_class = fold[order], chain[order], classes[order]
    counts = np.bincount(seq_class, minlength=4)
    mixes = {"valid": counts / counts.sum(), "train": train_mix}
    mix_w = {name: mix[seq_class] * len(seq_class) / counts[seq_class] for name, mix in mixes.items()}
    rng = np.random.default_rng(41)
    boots = [decide._resample(np.arange(len(order)), rng, seq_chain) for _ in range(args.boot)]
    preds = FoldPreds(folds, idx_by_fold)
    ref_blend = parse_blend(args.reference)
    ref = blend_sums(preds, ref_blend, y, mask)

    def wp(s, lam, weights=None):
        return decide.wp(s if weights is None else s * weights[:, None, None, None], lam)

    def report(name: str, blend: dict[str, float]) -> dict:
        s = blend_sums(preds, blend, y, mask)
        row = {"blend": blend, "n": int(len(order))}
        for lam in LAMBDAS:
            d = wp(s, lam) - wp(ref, lam)
            sd = float(np.std([wp(s[b], lam) - wp(ref[b], lam) for b in boots])) if boots else float("nan")
            row[f"lam{lam:g}"] = {
                "wp": wp(s, lam),
                "delta": d,
                "sd": sd,
                "folds": {str(k): wp(s[seq_fold == k], lam) - wp(ref[seq_fold == k], lam) for k in folds},
            }
        row["train_mix_lam0_delta"] = wp(s, 0.0, mix_w["train"]) - wp(ref, 0.0, mix_w["train"])
        row["class_lam0_delta"] = {
            str(c): wp(s[seq_class == c], 0.0) - wp(ref[seq_class == c], 0.0) for c in range(4)
        }
        r0 = row["lam0"]
        print(
            f"{name:28s} lam0 {r0['delta']:+.5f}±{r0['sd']:.5f} folds "
            + " ".join(f"{v:+.5f}" for v in r0["folds"].values())
            + f" | lam.02 {row['lam0.02']['delta']:+.5f} lam.05 {row['lam0.05']['delta']:+.5f}"
            + f" | trainmix {row['train_mix_lam0_delta']:+.5f} | wp {r0['wp']:.5f}",
            flush=True,
        )
        return row

    print(f"reference {args.reference}: wp lam0 {wp(ref, 0.0):.5f} on {len(order)} sequences")
    results = {"reference": args.reference, "sets": {}, "greedy": []}
    for spec in filter(None, args.sets.split(";")):
        name, _, blend = spec.partition("=")
        results["sets"][name] = report(name, parse_blend(blend))

    pool = [p for p in args.pool.split(",") if p]
    if pool:
        print("-- greedy (exploratory, selects on these rows)")
        chosen: list[str] = []
        for _ in range(min(args.k_max, len(pool))):
            best = None
            for cand in pool:
                if cand in chosen:
                    continue
                blend = {m: 1 / (len(chosen) + 1) for m in [*chosen, cand]}
                score = wp(blend_sums(preds, blend, y, mask), 0.0)
                if best is None or score > best[0]:
                    best = (score, cand)
            chosen.append(best[1])
            results["greedy"].append(report("greedy:" + "+".join(chosen), {m: 1 / len(chosen) for m in chosen}))

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
