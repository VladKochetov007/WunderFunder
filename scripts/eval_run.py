"""Evaluate a saved run on the dev split and store per-sequence scores.

    scripts/limited .venv/bin/python scripts/eval_run.py ab2_C_mix_aux20_200 [--holdout]
Writes artifacts/runs/<name>/scores_<split>.npz with row groups, market class
and per-target WP (NaN for ineligible blocks). ``--holdout`` is for final
grading only (research/contract.yaml).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import train_rnn
from torch.utils.data import DataLoader
from train_rnn import Sequences, make_model, predict

from wnn.data import market_class
from wnn.metric import block_scores
from wnn.research_features import PRIOR_FEATURE_MODES, validate_research_feature_prior
from wnn.splits import valid_row_groups

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--devB", action="store_true", help="score only dev half B")
    args = ap.parse_args()
    run = ROOT / "artifacts" / "runs" / args.name
    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    a = ckpt["args"]
    n_out = 2 + 2 * len([v for v in a.get("aux", "").split(",") if v])
    device = torch.device("cuda")
    if a.get("prop_input"):
        from wnn.propensity import load_propensity

        train_rnn.INPUT_PROPENSITY = load_propensity(
            ROOT / "artifacts" / "runs" / a["prop_input"] / "model.pt", device
        )
    train_rnn.RESEARCH_FEATURES = a.get("research_features", "none")
    train_rnn.RESEARCH_FEATURE_CONTROL = a.get("research_feature_control", "none")
    train_rnn.RESEARCH_FEATURE_PRIOR = None
    if train_rnn.RESEARCH_FEATURES in PRIOR_FEATURE_MODES:
        prior = ckpt.get("research_feature_prior")
        if not isinstance(prior, dict):
            raise ValueError("prior-based checkpoint lacks its fitting-only feature prior")
        train_rnn.RESEARCH_FEATURE_PRIOR = validate_research_feature_prior(
            train_rnn.RESEARCH_FEATURES, prior
        )
    model = make_model(a, ckpt["n_in"], n_out).to(device)
    model.load_state_dict(ckpt["model"])
    dev, hold = valid_row_groups(ROOT / "datasets" / "valid.parquet")
    groups = hold if args.holdout else dev
    split = "holdout" if args.holdout else "dev"
    if args.devB:
        from wnn.splits import dev_halves

        groups, split = dev_halves(ROOT / "datasets" / "valid.parquet")[1], "devB"
    loader = DataLoader(Sequences("valid", groups), batch_size=16, num_workers=2, prefetch_factor=1)
    scores, classes = [], []
    for x, y, m in loader:
        p = predict(model, x, 4000, device)
        scores.append(block_scores(y.numpy(), p, m.numpy()))
        classes += [market_class(t) for t in y.numpy()]
    scores = np.concatenate(scores)
    classes = np.array(classes)
    np.savez(run / f"scores_{split}.npz", row_groups=groups, classes=classes, scores=scores)
    per = scores.mean(1)
    print(
        json.dumps(
            {
                "run": args.name,
                "split": split,
                "n": int(len(per)),
                "wp": float(np.nanmean(per)),
                "by_class": {
                    int(c): round(float(np.nanmean(per[classes == c])), 4)
                    for c in np.unique(classes)
                },
            }
        )
    )


if __name__ == "__main__":
    main()
