"""Causal propensity model P(is_scored | rows <= t), fit on validation devA only.

    scripts/limited .venv/bin/python scripts/train_propensity.py --name prop_v1
Scores AUC on devB. The frozen model supplies soft scoring weights when training
on train.parquet (which has no mask), keeping devB and holdout untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_rnn import Sequences, build_inputs

from wnn.data import N_FEATURES, WARMUP
from wnn.features import N_EVENT_FEATURES
from wnn.propensity import PropensityNet, propensity
from wnn.resources import configure_torch
from wnn.splits import dev_halves

ROOT = Path(__file__).resolve().parents[1]


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--heldout-fold", type=int, default=-1,
                    help="if >= 0: train on all validation sequences outside this chain fold (cache/chain_folds.npz), "
                         "evaluate on it; default: train devA, evaluate devB")
    ap.add_argument("--dev-all", action="store_true", help="train on all dev sequences (devA + devB), evaluate on the holdout")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda")
    configure_torch(gpu_fraction=0.3)
    out_dir = ROOT / "artifacts" / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_a, dev_b = dev_halves(ROOT / "datasets" / "valid.parquet")
    if args.dev_all:
        from wnn.splits import valid_row_groups
        dev_a, dev_b = valid_row_groups(ROOT / "datasets" / "valid.parquet")
    elif args.heldout_fold >= 0:
        rg = np.load(ROOT / "cache" / "valid_targets.npz")["row_group"]
        cf = np.load(ROOT / "cache" / "chain_folds.npz")
        v = cf["split"] == "valid"
        fold_of = dict(zip(cf["row_group"][v].tolist(), cf["fold"][v].tolist()))
        f = np.array([fold_of[int(g)] for g in rg])
        dev_a, dev_b = rg[f != args.heldout_fold], rg[f == args.heldout_fold]
    train = DataLoader(Sequences("valid", dev_a), batch_size=16, shuffle=True, num_workers=2, prefetch_factor=1)
    test = DataLoader(Sequences("valid", dev_b), batch_size=16, num_workers=2, prefetch_factor=1)
    model = PropensityNet(N_FEATURES + N_EVENT_FEATURES, args.hidden).to(device)
    with torch.no_grad():
        x, _, _ = next(iter(train))
        f = build_inputs(x.to(device))
        model.mean.copy_(f.reshape(-1, f.shape[-1]).mean(0))
        model.std.copy_(f.reshape(-1, f.shape[-1]).std(0).clamp_min(1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    history, best_auc = [], -1.0
    for epoch in range(args.epochs):
        model.train()
        for x, _, m in train:
            inputs = build_inputs(x.to(device))
            target = m.to(device).float()
            h = None
            for s in range(0, 20_000, 2000):
                with torch.autocast("cuda", dtype=torch.float16):
                    logit, h = model(inputs[:, s : s + 2000], h)
                h = h.detach()
                keep = (torch.arange(s, min(s + 2000, 20_000), device=device) >= WARMUP)[None].expand_as(logit)
                loss = F.binary_cross_entropy_with_logits(logit.float()[keep], target[:, s : s + 2000][keep])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        model.eval()
        ps, ms = [], []
        for x, _, m in test:
            p = propensity(model, build_inputs(x.to(device)))
            ps.append(p[:, WARMUP:].cpu().numpy().ravel())
            ms.append(m[:, WARMUP:].numpy().ravel())
        p, m = np.concatenate(ps), np.concatenate(ms)
        sub = np.random.default_rng(0).choice(len(p), 2_000_000, replace=False)
        res = {"epoch": epoch, "devB_auc": round(auc(p[sub], m[sub]), 4), "base_rate": round(float(m.mean()), 4),
               "mean_p": round(float(p.mean()), 4)}
        history.append(res)
        print("PROP", json.dumps(res), flush=True)
        if res["devB_auc"] > best_auc:  # keep the best epoch: the constant-lr last epoch can drift in calibration
            best_auc = res["devB_auc"]
            torch.save({"model": model.state_dict(), "hidden": args.hidden, "n_in": N_FEATURES + N_EVENT_FEATURES},
                       out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
