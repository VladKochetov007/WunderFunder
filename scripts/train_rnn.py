"""Train a stateful GRU on GPU with truncated BPTT; select on the validation dev split.

    scripts/limited .venv/bin/python scripts/train_rnn.py --name gru --epochs 2 --aux 20,200
The holdout split (wnn.splits) is never read here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, N_FEATURES, WARMUP, SequenceFile, market_class
from wnn.features import FEATURE_NAMES, N_EVENT_FEATURES, event_features_torch
from wnn.metric import block_scores, score
from wnn.nn import DualClockModel, RNNModel, weighted_corr_loss, weighted_mse_loss
from wnn.propensity import load_propensity, propensity
from wnn.resources import configure_torch, dataloader_budget
from wnn.splits import dev_halves, valid_row_groups

ROOT = Path(__file__).resolve().parents[1]
SEQ_BYTES = 20_000 * (N_FEATURES + 3) * 4
EV_I0PX = N_FEATURES + FEATURE_NAMES.index("ev_i0px")


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
        return torch.from_numpy(seq.features), torch.from_numpy(seq.targets), torch.from_numpy(scored)


INPUT_PROPENSITY = None  # frozen PropensityNet appended as an input column when set


def build_inputs(x):
    inputs = torch.cat([x, event_features_torch(x)], dim=-1)
    if INPUT_PROPENSITY is not None:
        inputs = torch.cat([inputs, propensity(INPUT_PROPENSITY, inputs)[..., None].to(inputs.dtype)], dim=-1)
    return inputs


def shifted_targets(y, valid, shifts):
    """Training-only labels: the target ``sh`` rows later. Never used as an input."""
    out = []
    for sh in shifts:
        ys = torch.cat([y[:, sh:], y[:, -1:].expand(-1, sh, -1)], dim=1)
        keep = valid.clone()
        keep[:, -sh:] = False
        out.append((ys, keep))
    return out


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


def forward_chunk(model, inputs, sl, state):
    if isinstance(model, DualClockModel):
        return model(inputs[:, sl], inputs[:, sl, EV_I0PX] > 0.5, state)
    return model(inputs[:, sl], state)


def detach_state(state):
    if isinstance(state, tuple):
        return tuple(v.detach() for v in state)
    return state.detach()


def make_model(args_dict, n_in, n_out):
    if args_dict.get("arch", "gru") == "dual":
        return DualClockModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["event_hidden"],
                              args_dict["dropout"], n_out=n_out)
    return RNNModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out=n_out)


@torch.no_grad()
def predict(model, x, chunk, device):
    model.eval()
    inputs = build_inputs(x.to(device, non_blocking=True))
    state, outs = None, []
    for s in range(0, inputs.shape[1], chunk):
        with torch.autocast("cuda", dtype=torch.float16):
            out, state = forward_chunk(model, inputs, slice(s, s + chunk), state)
        outs.append(out[..., :2].float())
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
    res["by_class"] = {int(c): round(float(np.nanmean(per[cls == c])), 4) for c in np.unique(cls)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-seqs", type=int, default=10607)
    ap.add_argument("--valid-seqs", type=int, default=600, help="dev sequences used for selection")
    ap.add_argument("--full-valid", action="store_true", help="all 1505 dev sequences")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--loss", choices=["corr", "wmse", "mix"], default="mix")
    ap.add_argument("--aux", default="", help="future row shifts as auxiliary targets, e.g. 20,200")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--arch", choices=["gru", "dual"], default="gru")
    ap.add_argument("--event-hidden", type=int, default=128)
    ap.add_argument("--class-balance", action="store_true",
                    help="sample train sequences so market-class shares match validation")
    ap.add_argument("--train-source", choices=["train", "valid_devA"], default="train")
    ap.add_argument("--eval-source", choices=["dev", "devB"], default="dev")
    ap.add_argument("--init", default="", help="run name whose model.pt initializes weights")
    ap.add_argument("--mask-weight", action="store_true",
                    help="main loss only on is_scored rows (needs a validation train source)")
    ap.add_argument("--propensity", default="", help="run name of a frozen PropensityNet; weights main loss by p(x)")
    ap.add_argument("--prop-power", type=float, default=1.0, help="weights p(x)**power")
    ap.add_argument("--prop-input", default="", help="run name of a frozen PropensityNet appended as an input feature")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mask_weight and args.train_source == "train":
        ap.error("--mask-weight needs is_scored, which train.parquet does not have")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    duty = configure_torch()
    out_dir = ROOT / "artifacts" / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_path = ROOT / "datasets" / "valid.parquet"
    dev_groups, _ = valid_row_groups(valid_path)
    dev_a, dev_b = dev_halves(valid_path)
    train_split = "train"
    if args.train_source == "valid_devA":
        train_split = "valid"
        train_idx = np.random.default_rng(args.seed).permutation(dev_a)[: args.train_seqs]
    else:
        usable = np.array([g for g in range(10607) if g not in CORRUPT_TRAIN_ROW_GROUPS])
        train_idx = np.random.default_rng(args.seed).permutation(usable)[: args.train_seqs]
    eval_groups = dev_b if args.eval_source == "devB" else dev_groups
    n_valid = len(eval_groups) if args.full_valid else min(args.valid_seqs, len(eval_groups))
    valid_idx = np.random.default_rng(12345).permutation(eval_groups)[:n_valid]

    workers, prefetch = dataloader_budget(args.batch * SEQ_BYTES)
    sampler = None
    if args.class_balance:
        classes = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")
        tr_cls = classes[train_split][train_idx]
        va_share = np.array([(classes["valid"] == c).mean() for c in range(4)])
        tr_share = np.array([max((tr_cls == c).mean(), 1e-9) for c in range(4)])
        weights = np.where(tr_cls >= 0, va_share[np.clip(tr_cls, 0, 3)] / tr_share[np.clip(tr_cls, 0, 3)], 1.0)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(train_idx),
                                        replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(Sequences(train_split, train_idx), batch_size=args.batch, shuffle=sampler is None,
                              sampler=sampler, drop_last=True, num_workers=workers, prefetch_factor=prefetch)
    valid_loader = DataLoader(Sequences("valid", valid_idx), batch_size=16, num_workers=2,
                              prefetch_factor=1)

    shifts = [int(v) for v in args.aux.split(",") if v]
    prop_model = load_propensity(ROOT / "artifacts" / "runs" / args.propensity / "model.pt", device) if args.propensity else None
    n_in = N_FEATURES + N_EVENT_FEATURES
    if args.prop_input:
        global INPUT_PROPENSITY
        INPUT_PROPENSITY = load_propensity(ROOT / "artifacts" / "runs" / args.prop_input / "model.pt", device)
        n_in += 1
    model = make_model(vars(args), n_in, 2 + 2 * len(shifts)).to(device)
    if args.init:
        init = torch.load(ROOT / "artifacts" / "runs" / args.init / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(init["model"])
    else:
        fit_normalization(model, train_loader, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = len(train_loader) * math.ceil(20_000 / args.chunk) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=0.05, final_div_factor=100)
    scaler = torch.amp.GradScaler()

    def objective(o, t, v):
        if args.loss == "corr":
            return weighted_corr_loss(o, t, v)
        if args.loss == "wmse":
            return weighted_mse_loss(o, t, v)
        return weighted_corr_loss(o, t, v) + 0.5 * weighted_mse_loss(o, t, v)

    print(json.dumps(vars(args)), flush=True)
    print(f"loader workers {workers} prefetch {prefetch}; params "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M; steps {total}", flush=True)

    best, t0, history = -1.0, time.time(), []
    for epoch in range(args.epochs):
        model.train()
        for b, (x, y, scored) in enumerate(train_loader):
            inputs = build_inputs(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            steps = torch.arange(inputs.shape[1], device=device)[None].expand(inputs.shape[0], -1)
            valid = steps >= WARMUP
            main_valid = valid & scored.to(device) if args.mask_weight else valid
            if prop_model is not None:
                main_valid = valid.float() * propensity(prop_model, inputs[..., : N_FEATURES + N_EVENT_FEATURES]) ** args.prop_power
            aux = shifted_targets(y, valid, shifts)
            h = None
            for s in range(0, inputs.shape[1], args.chunk):
                sl = slice(s, s + args.chunk)
                with torch.autocast("cuda", dtype=torch.float16):
                    out, h = forward_chunk(model, inputs, sl, h)
                h = detach_state(h)
                loss = objective(out[..., :2], y[:, sl], main_valid[:, sl])
                for j, (ya, va) in enumerate(aux):
                    loss = loss + args.aux_weight * objective(out[..., 2 + 2 * j : 4 + 2 * j], ya[:, sl], va[:, sl])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                torch.cuda.synchronize()
                duty.tick()
            if b % 20 == 0:
                rate = (b + 1) * args.batch / (time.time() - t0)
                print(f"ep {epoch} batch {b}/{len(train_loader)} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} seq/s {rate:.1f}", flush=True)
        t_eval = time.time()
        res = evaluate(model, valid_loader, 4000, device)
        res.update(epoch=epoch, eval_seconds=round(time.time() - t_eval, 1), split=f"{args.eval_source}{n_valid}")
        history.append(res)
        print("VALID", json.dumps(res), flush=True)
        if res["weighted_pearson"] > best:
            best = res["weighted_pearson"]
            torch.save({"model": model.state_dict(), "args": vars(args), "n_in": n_in}, out_dir / "model.pt")
        t0 = time.time()
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    print("BEST", best)


if __name__ == "__main__":
    main()
