"""Chain-grouped validation folds (cache/chain_folds.npz) from cache/chains.npz.

    .venv/bin/python scripts/build_chain_folds.py [--folds 4]

Every sequence of one continuous stream (chain) lands in the same fold, so a held-out fold never has a
time-adjacent neighbour in the training pool among validation sequences. The fold of a chain is a salted
sha256 of the sorted (split, row_group) members, so it does not depend on chain numbering. Folds are
then rebalanced greedily by validation sequence count within each market class.

Fields: split, row_group, chain, fold (same fold for all members; train-only chains get a fold too, unused),
pure (True for a validation sequence whose chain has no train member: no train neighbour in time).
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()

    c = np.load(ROOT / "cache" / "chains.npz")
    split, rg, chain = c["split"], c["row_group"], c["chain"].copy()
    # sequences without a chain id become singleton chains
    lone = chain < 0
    chain[lone] = chain.max() + 1 + np.arange(lone.sum())

    members = defaultdict(list)
    for i, ch in enumerate(chain):
        members[int(ch)].append(i)
    classes = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["valid"]

    has_train = {ch: any(split[i] == "train" for i in idx) for ch, idx in members.items()}
    # chain class = majority market class of its validation members (-1 if none)
    chain_class, chain_nvalid = {}, {}
    for ch, idx in members.items():
        v = [classes[rg[i]] for i in idx if split[i] == "valid"]
        chain_nvalid[ch] = len(v)
        chain_class[ch] = int(np.bincount(np.asarray(v) + 1).argmax() - 1) if v else -1

    def key(ch: int) -> int:
        tag = ",".join(sorted(f"{split[i]}:{rg[i]}" for i in members[ch]))
        return int(hashlib.sha256(f"chainfold:{tag}".encode()).hexdigest(), 16)

    fold_of = {}
    load = defaultdict(lambda: np.zeros(args.folds))
    # largest chains first, each into the currently lightest fold of its class (ties by hash)
    for ch in sorted(members, key=lambda ch: (-chain_nvalid[ch], key(ch))):
        k = chain_class[ch]
        if chain_nvalid[ch] == 0:
            fold_of[ch] = key(ch) % args.folds
            continue
        order = np.lexsort(((np.arange(args.folds) + key(ch) % args.folds) % args.folds, load[k]))
        fold_of[ch] = int(order[0])
        load[k][fold_of[ch]] += chain_nvalid[ch]

    fold = np.array([fold_of[int(ch)] for ch in chain])
    pure = np.array([split[i] == "valid" and not has_train[int(chain[i])] for i in range(len(chain))])
    np.savez(ROOT / "cache" / "chain_folds.npz", split=split, row_group=rg, chain=chain, fold=fold, pure=pure)

    v = split == "valid"
    for f in range(args.folds):
        sel = v & (fold == f)
        print(f"fold {f}: valid {sel.sum()} pure {(sel & pure).sum()} classes {np.bincount(classes[rg[sel]] + 1, minlength=5)[1:]}")


if __name__ == "__main__":
    main()
