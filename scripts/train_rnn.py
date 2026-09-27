"""Train a stateful GRU on GPU with truncated BPTT; select on the validation dev split.

    scripts/limited .venv/bin/python scripts/train_rnn.py --name gru --epochs 2 --aux 20,200
The holdout split (wnn.splits) is read only with --eval-source holdout, for grading final fits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, N_FEATURES, WARMUP, SequenceFile, market_class
from wnn.features import FEATURE_NAMES, N_EVENT_FEATURES, event_features_torch, online_zscore_torch
from wnn.gmetric import global_wp
from wnn.metric import block_scores, score
from wnn.nn import DualClockModel, RNNModel, weighted_corr_loss, weighted_huber_loss, weighted_mse_loss
from wnn.outcomes import N_CODES, N_SP_AUX, ReadoutMoments, outcome_codes, readout, sequence_class, structure_labels
from wnn.propensity import load_propensity, propensity
from wnn.resources import configure_torch, dataloader_budget
from wnn.splits import dev_halves, valid_row_groups

ROOT = Path(__file__).resolve().parents[1]
SEQ_BYTES = 20_000 * (N_FEATURES + 3) * 4
EV_I0PX = N_FEATURES + FEATURE_NAMES.index("ev_i0px")


class Sequences(Dataset):
    def __init__(self, split: str, indices, teachers=None, prop_cache=None):
        self.path = ROOT / "datasets" / f"{split}.parquet"
        self.indices = list(indices)
        self._file = None
        # prop_cache: path of cache/propcache_<name>_<split>.npy (row group x 20000 float32), or None
        self.prop_cache, self._prop = prop_cache, None
        # teachers: list of (name, scale); predictions are appended to targets as columns 2..3
        self.teachers = teachers
        self.split = split
        self._teach = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self._file is None:
            self._file = SequenceFile(self.path)
        seq = self._file.read(self.indices[i])
        scored = seq.scored if seq.scored is not None else np.zeros(len(seq.features), bool)
        x = seq.features
        if HEURISTICS:
            from wnn.heuristics import heuristic_features

            x = np.concatenate([x, heuristic_features(x)], axis=1)
        y = seq.targets
        if self.teachers:
            if self._teach is None:
                rg = np.load(ROOT / "cache" / f"{self.split}_targets.npz")["row_group"]
                self._pos = {int(g): i for i, g in enumerate(rg)}
                self._teach = [(np.load(ROOT / "cache" / f"pred_{n}_{self.split}.npy", mmap_mode="r"), sc) for n, sc in self.teachers]
            j = self._pos[int(self.indices[i])]
            t = sum(sc * np.asarray(arr[j], dtype=np.float32) for arr, sc in self._teach) / len(self._teach)
            y = np.concatenate([y, t.astype(np.float32)], axis=1)
        if self.prop_cache is not None:
            if self._prop is None:
                self._prop = np.load(self.prop_cache, mmap_mode="r")
            p = np.array(self._prop[self.indices[i]], dtype=np.float32)
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(scored), torch.from_numpy(p)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(scored)


def true_mask_weights(main_valid, valid, scored, eps, branchless):
    """Rows of sequences that carry a true scoring mask get c * valid * (is_scored + eps), with c chosen so
    those sequences keep the total weight their propensity weights had; other sequences keep main_valid.

    branchless=True computes the same rows and the same c without boolean indexing or a host-side branch
    (no GPU->CPU sync); tests/test_perf_parity.py checks both against each other."""
    true_w = scored.float() + eps
    if branchless:
        hm = scored.any(dim=1)[:, None]
        c = (main_valid * hm).sum() / (valid.float() * true_w * hm).sum().clamp_min(1e-8)
        return torch.where(hm, valid.float() * true_w * c, main_valid)
    has_mask = scored.any(dim=1)
    if has_mask.any():
        c = main_valid[has_mask].sum() / (valid.float()[has_mask] * true_w[has_mask]).sum().clamp_min(1e-8)
        return torch.where(has_mask[:, None], valid.float() * true_w * c, main_valid)
    return main_valid


def prop_cache_path(name, split):
    """Propensity cache written by scripts/cache_propensity.py; refuses a cache built from other weights."""
    import hashlib

    path = ROOT / "cache" / f"propcache_{name}_{split}.npy"
    meta = path.with_suffix(".json")
    if not path.exists() or not meta.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/cache_propensity.py {name}")
    want = hashlib.sha256((ROOT / "artifacts" / "runs" / name / "model.pt").read_bytes()).hexdigest()
    got = json.loads(meta.read_text())
    if got.get("model_sha256") != want:
        raise RuntimeError(f"{path}: propensity weights changed since the cache was built")
    return path


class StageTimer:
    """Per-stage GPU times from CUDA events plus CPU wait on the loader (research profiling only).

    ``mark(name)`` records an event closing the stage that started at the previous mark; nothing is
    synchronized until ``report()``, so profiling does not add host round trips to the step.
    """

    def __init__(self, enabled: bool, warmup: int = 0):
        self.enabled, self.warmup = enabled, warmup
        self.steps, self.cur, self.loader_wait = [], None, []
        self.syncs = 0

    def _count_sync(self, message, *a, **k):
        if "synchroniz" in str(message):
            self.syncs += 1

    def start(self):
        if self.enabled and len(self.steps) == self.warmup:
            import warnings

            warnings.simplefilter("always")
            warnings.showwarning = self._count_sync
            torch.cuda.set_sync_debug_mode(os.environ.get("WNN_SYNC_DEBUG", "warn"))
        if self.enabled:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.cur = {"_events": [("start", ev)], "_wall": time.perf_counter()}

    def mark(self, name):
        if self.enabled and self.cur is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.cur["_events"].append((name, ev))

    def stop(self):
        if self.enabled and self.cur is not None:
            self.steps.append(self.cur)
            self.cur = None

    def report(self, warmup, batch):
        torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        rows, walls = [], []
        for st in self.steps[warmup:]:
            ev = st["_events"]
            acc = {}
            for (_, a), (name, b) in zip(ev[:-1], ev[1:]):
                acc[name] = acc.get(name, 0.0) + a.elapsed_time(b)
            acc["gpu_total"] = ev[0][1].elapsed_time(ev[-1][1])
            rows.append(acc)
        ends = [st["_wall"] for st in self.steps[warmup:]]
        step_wall = np.median(np.diff(ends)) * 1000 if len(ends) > 2 else float("nan")
        keys = sorted({k for r in rows for k in r})
        med = {k: float(np.median([r.get(k, 0.0) for r in rows])) for k in keys}
        med["loader_wait"] = float(np.median(self.loader_wait[warmup:])) * 1000 if self.loader_wait else 0.0
        med["step_wall"] = float(step_wall)
        med["seq_per_s"] = batch / (step_wall / 1000) if step_wall == step_wall else float("nan")
        med["syncs_per_step"] = self.syncs / max(1, len(self.steps) - warmup)
        return med


class CudaPrefetcher:
    """Iterate a DataLoader of pinned CPU batches, copying batch b+1 to the GPU on a side stream while
    batch b is being consumed. Yields device tensors plus the CPU copy of ``scored``."""

    def __init__(self, loader, device, timer=None):
        self.loader, self.device, self.timer = loader, device, timer
        self.stream = torch.cuda.Stream()

    def _load(self, it):
        t = time.perf_counter()
        try:
            x, y, scored, *rest = next(it)
        except StopIteration:
            return None
        if self.timer is not None and self.timer.enabled:
            self.timer.loader_wait.append(time.perf_counter() - t)
        with torch.cuda.stream(self.stream):
            dev = [v.to(self.device, non_blocking=True) for v in (x, y, scored, *rest)]
        return dev, scored

    def __iter__(self):
        it = iter(self.loader)
        nxt = self._load(it)
        while nxt is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            dev, scored_cpu = nxt
            for v in dev:
                v.record_stream(torch.cuda.current_stream())
            nxt = self._load(it)
            yield dev, scored_cpu

    def __len__(self):
        return len(self.loader)


class FastEMA:
    """EMA of parameters without host syncs: same update as AveragedModel with get_ema_multi_avg_fn
    (copy on the first call, then ``_foreach_lerp_``), but the counter lives on the host."""

    def __init__(self, model, decay):
        import copy

        self.module = copy.deepcopy(model)
        self.module.eval()
        for p_ in self.module.parameters():
            p_.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update_parameters(self, model):
        ema = [p_ for p_ in self.module.parameters()]
        cur = [p_.detach() for p_ in model.parameters()]
        if self.n == 0:
            torch._foreach_copy_(ema, cur)
        else:
            torch._foreach_lerp_(ema, cur, 1.0 - self.decay)
        self.n += 1


INPUT_PROPENSITY = None  # frozen PropensityNet appended as an input column when set
ONLINE_NORM = False  # append causal per-sequence z-scores of the 112 raw features
READOUT = None  # (m1, m0) moment tables when the head is a softmax over outcome codes
READOUT_TILTED = False
HEURISTICS = False  # append wnn.heuristics features (computed in loader workers)
RAW_ONLY = False  # zero the event-feature block (organizer-style raw view; keeps n_in for native export)


def build_inputs(x):
    raw, extra = x[..., :N_FEATURES], x[..., N_FEATURES:]
    ev = event_features_torch(raw)
    if RAW_ONLY:
        ev = torch.zeros_like(ev)
    inputs = torch.cat([raw, ev], dim=-1)
    if ONLINE_NORM:
        inputs = torch.cat([inputs, online_zscore_torch(raw)], dim=-1)
    if extra.shape[-1]:
        inputs = torch.cat([inputs, extra], dim=-1)
    if INPUT_PROPENSITY is not None:
        inputs = torch.cat([inputs, propensity(INPUT_PROPENSITY, inputs)[..., None].to(inputs.dtype)], dim=-1)
    return inputs


def next_change_labels(y, valid):
    """Training-only labels from the future of the target stream: log1p(rows until the next change
    of D = t0 - t1) and the sign of that change. Rows without a later change are masked out."""
    d = y[..., 0] - y[..., 1]
    b, t = d.shape
    idx = torch.arange(t, device=y.device).expand(b, t)
    changed = torch.zeros_like(d, dtype=torch.bool)
    changed[:, 1:] = d[:, 1:] != d[:, :-1]
    marker = torch.where(changed, idx, torch.full_like(idx, t))
    nxt = torch.flip(torch.cummin(torch.flip(marker, [1]), 1).values, [1])  # first change at >= row
    nxt = torch.cat([nxt[:, 1:], torch.full_like(nxt[:, :1], t)], 1)  # first change strictly after row
    has = nxt < t
    j = nxt.clamp(max=t - 1)
    jump = torch.gather(d, 1, j) - torch.gather(d, 1, (j - 1).clamp(min=0))
    rows_to = torch.log1p((nxt - idx).float())
    return rows_to, (jump > 0).float(), valid & has


def kth_change_labels(y, valid, k):
    """log1p(rows until the k-th later change of D = t0 - t1) and sign of D(k-th change) - D(now)."""
    d = y[..., 0] - y[..., 1]
    b, t = d.shape
    changed = torch.zeros_like(d, dtype=torch.long)
    changed[:, 1:] = (d[:, 1:] != d[:, :-1]).long()
    cnt = torch.cumsum(changed, 1)  # changes up to and including each row, non-decreasing
    pos = torch.searchsorted(cnt.contiguous(), (cnt + k).contiguous())  # first row reaching cnt + k
    has = pos < t
    j = pos.clamp(max=t - 1)
    idx = torch.arange(t, device=y.device).expand(b, t)
    rows_to = torch.log1p((j - idx).float())
    move = torch.gather(d, 1, j) - d
    return rows_to, (move > 0).float(), valid & has


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
    for b, batch in enumerate(loader):
        if b >= batches:
            break
        f = build_inputs(batch[0].to(device))
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
    return RNNModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out=n_out,
                    cell=args_dict.get("cell", "gru"), enc=args_dict.get("enc", "linear"),
                    tok_dim=args_dict.get("tok_dim", 32), att_layers=args_dict.get("att_layers", 1),
                    conv_k=args_dict.get("conv_k", 16))


@torch.no_grad()
def predict(model, x, chunk, device):
    model.eval()
    inputs = build_inputs(x.to(device, non_blocking=True))
    state, outs = None, []
    for s in range(0, inputs.shape[1], chunk):
        with torch.autocast("cuda", dtype=torch.float16):
            out, state = forward_chunk(model, inputs, slice(s, s + chunk), state)
        outs.append(readout(out, *READOUT, tilted=READOUT_TILTED) if READOUT is not None else out[..., :2].float())
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
    res["per_seq_wp"] = res["weighted_pearson"]
    g = global_wp(y, p, m)
    res.update(global_t0=g["t0"], global_t1=g["t1"], weighted_pearson=g["wp"])
    res["frac_abs_pred_gt2"] = float((np.abs(p[m]) > 2).mean())
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
    ap.add_argument("--loss", choices=["corr", "wmse", "mix", "huber", "ce_joint", "mse"], default="mix")
    ap.add_argument("--huber-delta", type=float, default=0.5)
    ap.add_argument("--ce-tilt", action="store_true", help="weight code cross-entropy rows by mean |clip(y)|")
    ap.add_argument("--ce-mse", type=float, default=0.0, help="add this weight x wMSE on the differentiable readout")
    ap.add_argument("--distill", default="", help="teacher runs name[*scale],...; cached pred_<name>_<split>.npy")
    ap.add_argument("--distill-alpha", type=float, default=0.5, help="target = (1-a) y + a mean(teachers)")
    ap.add_argument("--aux-next", type=float, default=0.0,
                    help="weight of auxiliary heads: log rows to next target change (MSE) and its sign (BCE)")
    ap.add_argument("--aux-next-k", default="", help="extra event horizons for the next-change aux heads, e.g. 2,4")
    ap.add_argument("--aux-struct", type=float, default=0.0,
                    help="weight of auxiliary cross-entropy heads for spread (ticks), branch and class")
    ap.add_argument("--aux", default="", help="future row shifts as auxiliary targets, e.g. 20,200")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--arch", choices=["gru", "dual"], default="gru")
    ap.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    ap.add_argument("--enc", choices=["linear", "mlp", "tokens", "conv"], default="linear",
                    help="per-row encoder before the recurrent cell (residual on the linear projection)")
    ap.add_argument("--tok-dim", type=int, default=32)
    ap.add_argument("--att-layers", type=int, default=1)
    ap.add_argument("--conv-k", type=int, default=16)
    ap.add_argument("--event-hidden", type=int, default=128)
    ap.add_argument("--class-balance", action="store_true",
                    help="sample train sequences so market-class shares match validation")
    ap.add_argument("--train-source", choices=["train", "valid_devA", "train+valid", "valid"], default="train",
                    help="train+valid: all train sequences plus validation devA and devB (holdout never)")
    ap.add_argument("--eval-source", choices=["dev", "devB", "holdout", "vfold"], default="dev")
    ap.add_argument("--valid-fold", type=int, default=-1,
                    help="K-fold over ALL validation sequences: train on train + other folds, evaluate on this fold")
    ap.add_argument("--n-valid-folds", type=int, default=4)
    ap.add_argument("--valid-repeat", type=int, default=1,
                    help="with train+valid: each validation training sequence appears this many times per epoch")
    ap.add_argument("--init", default="", help="run name whose model.pt initializes weights")
    ap.add_argument("--freeze", default="", help="comma-separated parameter-name prefixes kept fixed, e.g. inp,gru")
    ap.add_argument("--ema-teacher", type=float, default=0.0,
                    help="mean-teacher consistency: weight of (out - EMA-model out)^2 on the two target outputs")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--eval-ema", action="store_true", help="evaluate and save the EMA weights instead of the raw ones")
    ap.add_argument("--mask-weight", action="store_true",
                    help="main loss only on is_scored rows (needs a validation train source)")
    ap.add_argument("--propensity", default="", help="run name of a frozen PropensityNet; weights main loss by p(x)")
    ap.add_argument("--prop-power", type=float, default=1.0, help="weights p(x)**power")
    ap.add_argument("--prop-eps", type=float, default=0.0, help="floor added to propensity weights (mask robustness)")
    ap.add_argument("--class-weight", default="", help="per-market-class multipliers on the main loss weights, e.g. 1.19,1.32,2.02,0.79")
    ap.add_argument("--valid-true-mask", type=float, default=-1.0,
                    help="if >= 0: rows of sequences that carry a true scoring mask (validation) are weighted "
                         "c*(is_scored + this eps) instead of by the propensity net; c matches the mean propensity weight")
    ap.add_argument("--prop-input", default="", help="run name of a frozen PropensityNet appended as an input feature")
    ap.add_argument("--swa-frac", type=float, default=0.0,
                    help="average weights over the last fraction of optimizer steps; evaluate the average")
    ap.add_argument("--train-classes", default="", help="comma-separated market classes to train on (oracle expert test)")
    ap.add_argument("--online-norm", action="store_true")
    ap.add_argument("--heuristics", action="store_true", help="append the causal heuristic library as inputs")
    ap.add_argument("--raw-only", action="store_true", help="zero the event-feature inputs (raw 112-column view)")
    ap.add_argument("--seed", type=int, default=0)
    perf = ap.add_argument_group("throughput (see research/perf-2026-09-27.md)")
    perf.add_argument("--fast", action="store_true",
                      help="all verified throughput options: --sync-free --fused-adam --pin-memory --prop-cache")
    perf.add_argument("--sync-free", action="store_true", help="no per-step host<->GPU syncs in the train step")
    perf.add_argument("--fused-adam", action="store_true", help="fused AdamW (GradScaler skips inf steps on device)")
    perf.add_argument("--pin-memory", action="store_true", help="pinned persistent loaders + CUDA-stream prefetch")
    perf.add_argument("--prop-cache", action="store_true",
                      help="read propensity from cache/propcache_<name>_<split>.npy (scripts/cache_propensity.py)")
    perf.add_argument("--precision", choices=["fp16", "bf16"], default="fp16",
                      help="autocast dtype; bf16 trains without GradScaler")
    perf.add_argument("--max-batches", type=int, default=0, help="stop each epoch after this many batches (harness)")
    perf.add_argument("--profile-steps", type=int, default=0,
                      help="time this many batches per stage (after --profile-warmup), print a table and exit")
    perf.add_argument("--profile-warmup", type=int, default=5)
    perf.add_argument("--parity-dump", default="", help="save per-step losses and final weights here, then exit")
    perf.add_argument("--deterministic", action="store_true", help="cudnn deterministic, no benchmark (parity runs)")
    args = ap.parse_args()
    if args.fast:
        args.sync_free = args.fused_adam = args.pin_memory = args.prop_cache = True
    if args.profile_steps:
        args.max_batches = args.profile_warmup + args.profile_steps
    if args.mask_weight and args.train_source == "train":
        ap.error("--mask-weight needs is_scored, which train.parquet does not have")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    duty = configure_torch()
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    out_dir = ROOT / "artifacts" / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_path = ROOT / "datasets" / "valid.parquet"
    dev_groups, hold_groups = valid_row_groups(valid_path)
    dev_a, dev_b = dev_halves(valid_path)
    train_split = "train"
    extra_valid = None
    fold_groups = None
    if args.valid_fold >= 0:
        all_valid = np.r_[np.asarray(dev_groups), np.asarray(hold_groups)]
        # chain-grouped folds (cache/chain_folds.npz): sequences of one continuous stream share a fold
        cf = np.load(ROOT / "cache" / "chain_folds.npz")
        vmask = cf["split"] == "valid"
        fold_map = dict(zip(cf["row_group"][vmask].tolist(), cf["fold"][vmask].tolist()))
        fold_of = np.array([fold_map[int(g)] for g in all_valid]) % args.n_valid_folds
        fold_groups = all_valid[fold_of == args.valid_fold]
        if args.train_source == "train+valid":
            extra_valid = all_valid[fold_of != args.valid_fold]
    elif args.train_source == "train+valid":
        extra_valid = np.r_[np.asarray(dev_a), np.asarray(dev_b)]
    if args.train_source == "valid":
        # with --valid-fold: all valid minus the held-out chain fold; without: devA + devB (holdout never)
        train_split = "valid"
        pool = np.setdiff1d(all_valid, fold_groups) if fold_groups is not None else np.asarray(dev_groups)
        train_idx = np.random.default_rng(args.seed).permutation(pool)
    elif args.train_source == "valid_devA":
        train_split = "valid"
        train_idx = np.random.default_rng(args.seed).permutation(dev_a)[: args.train_seqs]
    else:
        usable = np.array([g for g in range(10607) if g not in CORRUPT_TRAIN_ROW_GROUPS])
        if args.train_classes:
            keep = {int(c) for c in args.train_classes.split(",")}
            cls = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["train"]
            usable = usable[np.isin(cls[usable], list(keep))]
        train_idx = np.random.default_rng(args.seed).permutation(usable)[: args.train_seqs]
    eval_groups = {"devB": dev_b, "holdout": hold_groups, "vfold": fold_groups}.get(args.eval_source, dev_groups)
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
    teachers = None
    if args.distill:
        teachers = [(t.partition("*")[0], float(t.partition("*")[2] or 1.0)) for t in args.distill.split(",")]
    prop_cache = {}
    if args.prop_cache and args.propensity:
        prop_cache = {sp: prop_cache_path(args.propensity, sp) for sp in ("train", "valid")}
    train_ds = Sequences(train_split, train_idx, teachers, prop_cache.get(train_split))
    if extra_valid is not None:
        valid_ds = Sequences("valid", extra_valid, teachers, prop_cache.get("valid"))
        train_ds = torch.utils.data.ConcatDataset([train_ds] + [valid_ds] * args.valid_repeat)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=sampler is None,
                              sampler=sampler, drop_last=True, num_workers=workers, prefetch_factor=prefetch,
                              pin_memory=args.pin_memory)  # no persistent workers: they change the shuffle order (parity E-169)
    valid_loader = DataLoader(Sequences("valid", valid_idx), batch_size=16, num_workers=2,
                              prefetch_factor=1)

    shifts = [int(v) for v in args.aux.split(",") if v]
    prop_model = load_propensity(ROOT / "artifacts" / "runs" / args.propensity / "model.pt", device) if args.propensity else None
    n_in = N_FEATURES + N_EVENT_FEATURES
    if args.raw_only:
        global RAW_ONLY
        RAW_ONLY = True
    if args.heuristics:
        global HEURISTICS
        from wnn.heuristics import K as N_HEUR

        HEURISTICS = True
        n_in += N_HEUR
    if args.online_norm:
        global ONLINE_NORM
        ONLINE_NORM = True
        n_in += N_FEATURES
    if args.prop_input:
        global INPUT_PROPENSITY
        INPUT_PROPENSITY = load_propensity(ROOT / "artifacts" / "runs" / args.prop_input / "model.pt", device)
        n_in += 1
    if args.loss == "ce_joint" and shifts:
        ap.error("--aux is not supported with the outcome-code head")
    n_out = N_CODES if args.loss == "ce_joint" else 2 + 2 * len(shifts)
    struct_at = n_out
    if args.aux_struct > 0:
        n_out += N_SP_AUX + 3 + 4
    next_at = n_out
    ks = [int(v) for v in args.aux_next_k.split(",") if v]
    if args.aux_next > 0:
        n_out += 2 + 2 * len(ks)
    model = make_model(vars(args), n_in, n_out).to(device)
    moments = ReadoutMoments(device) if args.loss == "ce_joint" else None
    if args.init:
        init = torch.load(ROOT / "artifacts" / "runs" / args.init / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(init["model"])
    else:
        fit_normalization(model, train_loader, device)
    frozen = tuple(v for v in args.freeze.split(",") if v)
    for pname, prm in model.named_parameters():
        if frozen and pname.startswith(frozen):
            prm.requires_grad_(False)
    opt = torch.optim.AdamW([prm for prm in model.parameters() if prm.requires_grad], lr=args.lr, weight_decay=args.wd,
                            fused=bool(args.fused_adam))
    total = len(train_loader) * math.ceil(20_000 / args.chunk) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=0.05, final_div_factor=100)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=args.precision == "fp16")
    swa_model = torch.optim.swa_utils.AveragedModel(model) if args.swa_frac > 0 else None
    ema_model = None
    if args.ema_teacher > 0 or args.eval_ema:
        if args.sync_free:
            ema_model = FastEMA(model, args.ema_decay)
        else:
            ema_model = torch.optim.swa_utils.AveragedModel(
                model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay))
            ema_model.eval()
    swa_start = int(total * (1.0 - args.swa_frac))
    step_count = 0

    def objective(o, t, v):
        if args.loss == "corr":
            return weighted_corr_loss(o, t, v)
        if args.loss == "wmse":
            return weighted_mse_loss(o, t, v)
        if args.loss == "mse":  # unweighted MSE on clipped targets (organizer-baseline style)
            yv = t.float().clamp(-2, 2)
            vv = v[..., None].float()
            return (vv * (o.float() - yv) ** 2).sum() / (vv.sum() * yv.shape[-1]).clamp_min(1e-8)
        if args.loss == "huber":
            return weighted_huber_loss(o, t, v, args.huber_delta)
        return weighted_corr_loss(o, t, v) + 0.5 * weighted_mse_loss(o, t, v)

    print(json.dumps(vars(args)), flush=True)
    print(f"loader workers {workers} prefetch {prefetch}; params "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M; steps {total}", flush=True)

    timer = StageTimer(args.profile_steps > 0, args.profile_warmup)
    losses_trace = []
    best, t0, history = -1.0, time.time(), []
    for epoch in range(args.epochs):
        model.train()
        batches = CudaPrefetcher(train_loader, device, timer) if args.pin_memory else train_loader
        for b, batch in enumerate(batches):
            if args.max_batches and b >= args.max_batches:
                break
            timer.start()
            prop_cached = None
            if args.pin_memory:
                (x, y, scored_dev, *rest), scored = batch
                if rest:
                    prop_cached = rest[0]
            else:
                x, y, scored, *rest = batch
                if rest:
                    prop_cached = rest[0].to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                scored_dev = scored.to(device, non_blocking=True)
            timer.mark("h2d")
            inputs = build_inputs(x)
            timer.mark("inputs")
            teach = None
            if y.shape[-1] > 2:
                teach, y = y[..., 2:4], y[..., :2]
            steps = torch.arange(inputs.shape[1], device=device)[None].expand(inputs.shape[0], -1)
            valid = steps >= WARMUP
            main_valid = valid & scored_dev if args.mask_weight else valid
            if prop_model is not None:
                if prop_cached is not None:
                    prob = prop_cached
                elif args.prop_cache:
                    raise RuntimeError("--prop-cache set but the batch carries no cached propensity")
                else:
                    prob = propensity(prop_model, inputs[..., : N_FEATURES + N_EVENT_FEATURES])
                main_valid = valid.float() * (prob ** args.prop_power + args.prop_eps)
                if args.class_weight:
                    cw = torch.tensor([float(v) for v in args.class_weight.split(",")], device=device)
                    main_valid = main_valid * cw[sequence_class(y)][:, None]
                if args.valid_true_mask >= 0:
                    main_valid = true_mask_weights(main_valid, valid, scored_dev, args.valid_true_mask, args.sync_free)
            timer.mark("weights")
            aux = shifted_targets(y, valid, shifts)
            if args.aux_struct > 0:
                sp_lab, br_lab, cl_lab = structure_labels(y, sequence_class(y))
                sw = main_valid.float()
            if args.aux_next > 0:
                nx_rows, nx_up, nx_ok = next_change_labels(y, valid)
                nw = main_valid.float() * nx_ok.float() if main_valid.dtype != torch.bool else (main_valid & nx_ok).float()
                kth = [kth_change_labels(y, valid, kk) for kk in ks]
            if moments is not None:
                codes = outcome_codes(y, sequence_class(y))
                row_w = main_valid.float()
                ce_w = row_w * y.float().clamp(-2, 2).abs().mean(-1) if args.ce_tilt else row_w
                if epoch == 0:
                    moments.update(codes, y, row_w)
            timer.mark("labels")
            h = h_ema = None
            for s in range(0, inputs.shape[1], args.chunk):
                sl = slice(s, s + args.chunk)
                with torch.autocast("cuda", dtype=amp_dtype):
                    out, h = forward_chunk(model, inputs, sl, h)
                h = detach_state(h)
                timer.mark("forward")
                if args.ema_teacher > 0:
                    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
                        out_ema, h_ema = forward_chunk(ema_model.module, inputs, sl, h_ema)
                    h_ema = detach_state(h_ema)
                if moments is not None:
                    ce = F.cross_entropy(out.float().reshape(-1, N_CODES), codes[:, sl].reshape(-1), reduction="none")
                    wv = ce_w[:, sl].reshape(-1)
                    loss = (ce * wv).sum() / wv.sum().clamp_min(1e-8)
                    if args.ce_mse > 0:
                        pred = readout(out, *moments.tables(), tilted=args.ce_tilt)
                        loss = loss + args.ce_mse * weighted_mse_loss(pred, y[:, sl], main_valid[:, sl])
                elif teach is not None:
                    yc = y[:, sl].float().clamp(-2, 2)
                    tgt = (1 - args.distill_alpha) * yc + args.distill_alpha * teach[:, sl].float().clamp(-2, 2)
                    w = yc.abs() * main_valid[:, sl][..., None]
                    loss = (w * (out[..., :2].float() - tgt) ** 2).sum() / w.sum().clamp_min(1e-8)
                else:
                    loss = objective(out[..., :2], y[:, sl], main_valid[:, sl])
                if args.aux_struct > 0:
                    o = out.float()
                    wv = sw[:, sl].reshape(-1)
                    ce = 0.0
                    for a0, a1, lab in ((struct_at, struct_at + N_SP_AUX, sp_lab), (struct_at + N_SP_AUX, struct_at + N_SP_AUX + 3, br_lab),
                                        (struct_at + N_SP_AUX + 3, struct_at + N_SP_AUX + 7, cl_lab)):
                        ce = ce + F.cross_entropy(o[..., a0:a1].reshape(-1, a1 - a0), lab[:, sl].reshape(-1), reduction="none")
                    loss = loss + args.aux_struct * (ce * wv).sum() / wv.sum().clamp_min(1e-8)
                if args.aux_next > 0:
                    o = out.float()
                    wv = nw[:, sl]
                    mse = (o[..., next_at] - (nx_rows[:, sl] - 3.0) / 2.0) ** 2
                    bce = F.binary_cross_entropy_with_logits(o[..., next_at + 1], nx_up[:, sl], reduction="none")
                    loss = loss + args.aux_next * ((mse + bce) * wv).sum() / wv.sum().clamp_min(1e-8)
                    for j, (kr, ku, kok) in enumerate(kth):
                        kw = main_valid[:, sl].float() * kok[:, sl].float()
                        col = next_at + 2 + 2 * j
                        kmse = (o[..., col] - (kr[:, sl] - 4.0) / 2.0) ** 2
                        kbce = F.binary_cross_entropy_with_logits(o[..., col + 1], ku[:, sl], reduction="none")
                        loss = loss + args.aux_next * ((kmse + kbce) * kw).sum() / kw.sum().clamp_min(1e-8)
                for j, (ya, va) in enumerate(aux):
                    loss = loss + args.aux_weight * objective(out[..., 2 + 2 * j : 4 + 2 * j], ya[:, sl], va[:, sl])
                if args.ema_teacher > 0:
                    cw = main_valid[:, sl].float()[..., None]
                    loss = loss + args.ema_teacher * (cw * (out[..., :2].float() - out_ema[..., :2].float()) ** 2).sum() / cw.sum().clamp_min(1e-8)
                timer.mark("loss")
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                timer.mark("backward")
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                timer.mark("optimizer")
                if args.parity_dump:
                    losses_trace.append(loss.detach().float())

                step_count += 1
                if swa_model is not None and step_count >= swa_start:
                    swa_model.update_parameters(model)
                if ema_model is not None:
                    ema_model.update_parameters(model)
                if duty is not None:  # explicit rate limit (WNN_GPU_DUTY < 1): sync, then sleep
                    torch.cuda.synchronize()
                    duty.tick()
                elif not args.sync_free:
                    torch.cuda.synchronize()
            timer.stop()
            if args.profile_steps and b + 1 == args.max_batches:
                break
            if b % 20 == 0:
                rate = (b + 1) * args.batch / (time.time() - t0)
                print(f"ep {epoch} batch {b}/{len(train_loader)} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} seq/s {rate:.1f}", flush=True)
        if args.profile_steps:
            med = timer.report(args.profile_warmup, args.batch)
            print("PROFILE", json.dumps({k: round(v, 2) for k, v in med.items()}), flush=True)
            return
        if args.parity_dump:
            torch.save({"losses": torch.stack(losses_trace).cpu(), "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "ema": None if ema_model is None else {k: v.detach().cpu() for k, v in ema_model.module.state_dict().items()}},
                       args.parity_dump)
            print("PARITY_DUMP", args.parity_dump, len(losses_trace), flush=True)
            return
        t_eval = time.time()
        eval_model = swa_model.module if swa_model is not None and step_count >= swa_start else model
        if args.eval_ema:
            eval_model = ema_model.module
        if moments is not None:
            global READOUT, READOUT_TILTED
            READOUT, READOUT_TILTED = moments.tables(), args.ce_tilt
        res = evaluate(eval_model, valid_loader, 4000, device)
        res.update(epoch=epoch, eval_seconds=round(time.time() - t_eval, 1), split=f"{args.eval_source}{n_valid}")
        history.append(res)
        print("VALID", json.dumps(res), flush=True)
        if res["weighted_pearson"] > best:
            best = res["weighted_pearson"]
            torch.save({"model": eval_model.state_dict(), "args": vars(args), "n_in": n_in,
                        "readout": None if READOUT is None else [t.cpu() for t in READOUT],
                        "readout_tilted": READOUT_TILTED}, out_dir / "model.pt")
        t0 = time.time()
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    print("BEST", best)


if __name__ == "__main__":
    main()
