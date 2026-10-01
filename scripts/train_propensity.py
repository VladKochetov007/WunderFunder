"""Causal propensity model P(is_scored | rows <= t), fit on validation devA only.

    scripts/limited .venv/bin/python scripts/train_propensity.py --name prop_v1
Scores AUC on devB. The frozen model supplies soft scoring weights when training
on train.parquet (which has no mask), keeping devB and holdout untouched.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_rnn import Sequences, build_inputs, checkpoint_selected, configure_checkpoint_policy

from wnn.data import N_FEATURES, WARMUP
from wnn.features import N_EVENT_FEATURES
from wnn.propensity import PropensityNet, propensity
from wnn.resources import configure_torch
from wnn.splits import dev_halves, file_sha256, load_campaign_split

ROOT = Path(__file__).resolve().parents[1]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-policy", choices=["best", "fixed"], default="best")
    ap.add_argument("--checkpoint-epoch", type=int, default=None)
    ap.add_argument("--no-eval-during-fit", action="store_true")
    ap.add_argument("--split-manifest", type=Path, default=None)
    ap.add_argument(
        "--heldout-fold",
        type=int,
        default=-1,
        help="if >= 0: train on all validation sequences outside this chain fold (cache/chain_folds.npz), "
        "evaluate on it; default: train devA, evaluate devB",
    )
    ap.add_argument(
        "--dev-all",
        action="store_true",
        help="train on all dev sequences (devA + devB), evaluate on the holdout",
    )
    args = ap.parse_args()
    configure_checkpoint_policy(args, ap)
    campaign_split = load_campaign_split(args.split_manifest, ROOT) if args.split_manifest else None
    if campaign_split is not None:
        if args.dev_all or args.heldout_fold not in (-1, campaign_split.fold):
            ap.error("propensity fit arguments disagree with the campaign outer fold")
        args.heldout_fold = campaign_split.fold
        args.split_manifest = str(campaign_split.path)
        args.split_manifest_sha256 = campaign_split.sha256
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    configure_torch(gpu_fraction=0.3)
    out_dir = ROOT / "artifacts" / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if campaign_split is not None:
        dev_a = np.asarray(campaign_split.valid_groups)
        dev_b = np.asarray(campaign_split.evaluation_groups)
        campaign_split.assert_fit_groups((), dev_a)
    elif args.dev_all:
        from wnn.splits import valid_row_groups

        dev_a, dev_b = valid_row_groups(ROOT / "datasets" / "valid.parquet")
    elif args.heldout_fold >= 0:
        rg = np.load(ROOT / "cache" / "valid_targets.npz")["row_group"]
        cf = np.load(ROOT / "cache" / "chain_folds.npz")
        v = cf["split"] == "valid"
        fold_of = dict(zip(cf["row_group"][v].tolist(), cf["fold"][v].tolist(), strict=True))
        f = np.array([fold_of[int(g)] for g in rg])
        dev_a, dev_b = rg[f != args.heldout_fold], rg[f == args.heldout_fold]
    else:
        dev_a, dev_b = dev_halves(ROOT / "datasets" / "valid.parquet")
    train = DataLoader(
        Sequences("valid", dev_a), batch_size=16, shuffle=True, num_workers=2, prefetch_factor=1
    )
    test = (
        None
        if args.no_eval_during_fit
        else DataLoader(Sequences("valid", dev_b), batch_size=16, num_workers=2, prefetch_factor=1)
    )
    model = PropensityNet(N_FEATURES + N_EVENT_FEATURES, args.hidden).to(device)
    with torch.no_grad():
        x, _, _ = next(iter(train))
        f = build_inputs(x.to(device))
        model.mean.copy_(f.reshape(-1, f.shape[-1]).mean(0))
        model.std.copy_(f.reshape(-1, f.shape[-1]).std(0).clamp_min(1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    history, best_auc = [], -1.0
    provenance = {
        "seed": args.seed,
        "args": vars(args).copy(),
        "split_manifest_sha256": campaign_split.sha256 if campaign_split is not None else None,
        "fit_row_groups": {"valid": dev_a.tolist()},
        "evaluation_row_groups": dev_b.tolist(),
        "checkpoint_sha256": {},
    }
    fitting_started = time.time()
    print(json.dumps(vars(args)), flush=True)
    for epoch in range(args.epochs):
        epoch_started = time.time()
        model.train()
        for x, _, m in train:
            inputs = build_inputs(x.to(device))
            target = m.to(device).float()
            h = None
            for s in range(0, 20_000, 2000):
                with torch.autocast("cuda", dtype=torch.float16):
                    logit, h = model(inputs[:, s : s + 2000], h)
                h = h.detach()
                keep = (torch.arange(s, min(s + 2000, 20_000), device=device) >= WARMUP)[
                    None
                ].expand_as(logit)
                loss = F.binary_cross_entropy_with_logits(
                    logit.float()[keep], target[:, s : s + 2000][keep]
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        res = {"epoch": epoch, "train_seconds": round(time.time() - epoch_started, 1)}
        evaluation_score = None
        if args.no_eval_during_fit:
            res["evaluation"] = "withheld"
        else:
            model.eval()
            ps, ms = [], []
            for x, _, m in test:
                p = propensity(model, build_inputs(x.to(device)))
                ps.append(p[:, WARMUP:].cpu().numpy().ravel())
                ms.append(m[:, WARMUP:].numpy().ravel())
            p, m = np.concatenate(ps), np.concatenate(ms)
            sub = np.random.default_rng(0).choice(len(p), min(len(p), 2_000_000), replace=False)
            evaluation_score = round(auc(p[sub], m[sub]), 4)
            res.update(
                devB_auc=evaluation_score,
                base_rate=round(float(m.mean()), 4),
                mean_p=round(float(p.mean()), 4),
            )
        history.append(res)
        print("PROP", json.dumps(res), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "hidden": args.hidden,
            "n_in": N_FEATURES + N_EVENT_FEATURES,
            "epoch": epoch,
            "args": vars(args),
            "provenance": {
                key: value for key, value in provenance.items() if key != "checkpoint_sha256"
            },
        }
        snapshot = out_dir / f"model_ep{epoch}.pt"
        torch.save(checkpoint, snapshot)
        provenance["checkpoint_sha256"][snapshot.name] = file_sha256(snapshot)
        if checkpoint_selected(
            epoch, args.checkpoint_policy, args.checkpoint_epoch, evaluation_score, best_auc
        ):
            torch.save(checkpoint, out_dir / "model.pt")
            provenance["checkpoint_sha256"]["model.pt"] = file_sha256(out_dir / "model.pt")
            provenance["selected_epoch"] = epoch
        if evaluation_score is not None:
            best_auc = max(best_auc, evaluation_score)
        provenance["fitting_seconds"] = round(time.time() - fitting_started, 1)
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
