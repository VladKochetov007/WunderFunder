"""Train a stateful GRU on GPU with truncated BPTT and score it on validation.

Research only: multicore parquet decoding and CUDA. Example:
    python scripts/train_rnn.py --name gru256 --epochs 3 --train-seqs 10607
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wnn.data import N_FEATURES, WARMUP, SequenceFile, market_class
from wnn.features import N_EVENT_FEATURES, event_features_torch
from wnn.metric import block_scores, score
from wnn.nn import RNNModel, weighted_corr_loss, weighted_mse_loss
from wnn.resources import configure_torch, dataloader_budget

ROOT = Path(__file__).resolve().parents[1]


class Sequences(Dataset):
    def __init__(self, split: str, indices):
        self.path = ROOT / "datasets" / f"{split}.parquet"
        self.indices = list(indices)
        self._file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self._file is None:
            self._file = SequenceFile(self.path)
        seq = self._file.read(self.indices[i])
        scored = seq.scored if seq.scored is not None else np.zeros(len(seq.features), bool)
        return (
            torch.from_numpy(seq.features),
            torch.from_numpy(seq.targets),
            torch.from_numpy(scored),
        )


def build_inputs(x):
    return torch.cat([x, event_features_torch(x)], dim=-1)


@torch.no_grad()
def fit_normalization(model, loader, device, batches=2):
    feats = []
    for b, (x, _, _) in enumerate(loader):
        if b >= batches:
            break
        f = build_inputs(x.to(device))
        feats.append(f.reshape(-1, f.shape[-1])[::7])
    f = torch.cat(feats)
    model.mean.copy_(f.mean(0))
    model.std.copy_(f.std(0).clamp_min(1e-3))


@torch.no_grad()
def predict(model, x, chunk, device):
    model.eval()
    inputs = build_inputs(x.to(device, non_blocking=True))
    h, outs = None, []
    for s in range(0, inputs.shape[1], chunk):
        with torch.autocast("cuda", dtype=torch.float16):
            out, h = model(inputs[:, s : s + chunk], h)
        outs.append(out.float())
    return torch.cat(outs, dim=1).cpu().numpy()


def evaluate(model, loader, chunk, device):
    preds, ys, masks, classes = [], [], [], []
    for x, y, m in loader:
        preds.append(predict(model, x, chunk, device))
        ys.append(y.numpy())
        masks.append(m.numpy())
        classes += [market_class(t) for t in y.numpy()]
    p, y, m = np.concatenate(preds), np.concatenate(ys), np.concatenate(masks)
    res = score(y, p, m)
    per = block_scores(y, p, m).mean(axis=1)
    cls = np.array(classes)
    res["by_class"] = {int(c): float(np.nanmean(per[cls == c])) for c in np.unique(cls)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-seqs", type=int, default=10607)
    ap.add_argument("--valid-seqs", type=int, default=600)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--loss", choices=["corr", "wmse", "mix"], default="mix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-valid", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    duty = configure_torch()
    out_dir = ROOT / "artifacts" / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    train_idx = rng.permutation(10607)[: args.train_seqs]
    n_valid = 1873 if args.full_valid else args.valid_seqs
    valid_idx = np.random.default_rng(12345).permutation(1873)[:n_valid]

    seq_bytes = 20_000 * (N_FEATURES + 3) * 4
    workers, prefetch = dataloader_budget(args.batch * seq_bytes)
    train_loader = DataLoader(
        Sequences("train", train_idx),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        prefetch_factor=prefetch,
    )
    valid_loader = DataLoader(
        Sequences("valid", valid_idx), batch_size=16, num_workers=2, prefetch_factor=1
    )
    print(f"loader workers {workers} prefetch {prefetch}", flush=True)

    n_in = N_FEATURES + N_EVENT_FEATURES
    model = RNNModel(n_in, args.hidden, args.layers, args.dropout).to(device)
    fit_normalization(model, train_loader, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(train_loader) * math.ceil(20_000 / args.chunk)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total, pct_start=0.05, final_div_factor=100
    )
    scaler = torch.amp.GradScaler()
    print(json.dumps(vars(args)), flush=True)
    print(
        f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M steps {total}", flush=True
    )

    best, step, t0 = -1.0, 0, time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        for b, (x, y, _) in enumerate(train_loader):
            inputs = build_inputs(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            valid = (
                torch.arange(inputs.shape[1], device=device)[None].expand(inputs.shape[0], -1)
                >= WARMUP
            )
            h = None
            for s in range(0, inputs.shape[1], args.chunk):
                sl = slice(s, s + args.chunk)
                with torch.autocast("cuda", dtype=torch.float16):
                    out, h = model(inputs[:, sl], h)
                h = h.detach()
                if args.loss == "corr":
                    loss = weighted_corr_loss(out, y[:, sl], valid[:, sl])
                elif args.loss == "wmse":
                    loss = weighted_mse_loss(out, y[:, sl], valid[:, sl])
                else:
                    loss = weighted_corr_loss(
                        out, y[:, sl], valid[:, sl]
                    ) + 0.5 * weighted_mse_loss(out, y[:, sl], valid[:, sl])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                torch.cuda.synchronize()
                duty.tick()
            if b % 20 == 0:
                rate = (b + 1) * args.batch / (time.time() - t0)
                print(
                    f"ep {epoch} batch {b}/{len(train_loader)} loss {loss.item():.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} seq/s {rate:.1f}",
                    flush=True,
                )
        t0 = time.time()
        res = evaluate(model, valid_loader, 4000, device)
        res.update(epoch=epoch, eval_seconds=round(time.time() - t0, 1))
        history.append(res)
        print("VALID", json.dumps(res), flush=True)
        if res["weighted_pearson"] > best:
            best = res["weighted_pearson"]
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "n_in": n_in},
                out_dir / "model.pt",
            )
        t0 = time.time()
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    print("BEST", best)


if __name__ == "__main__":
    main()
