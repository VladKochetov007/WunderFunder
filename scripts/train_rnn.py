"""Train a stateful GRU on GPU with truncated BPTT; select on the validation dev split.

    scripts/limited .venv/bin/python scripts/train_rnn.py --name gru --epochs 2 --aux 20,200
The holdout split (wnn.splits) is read only with --eval-source holdout, for grading final fits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from wnn.data import (
    CORRUPT_TRAIN_ROW_GROUPS,
    N_FEATURES,
    TARGET_QUANTA,
    WARMUP,
    SequenceFile,
    market_class,
)
from wnn.features import FEATURE_NAMES, N_EVENT_FEATURES, event_features_torch, online_zscore_torch
from wnn.metric import block_aggregate_score, block_scores, score
from wnn.nn import (
    DualClockModel,
    GatedRNNModel,
    PooledCorrMemory,
    RNNModel,
    weighted_corr_loss,
    weighted_huber_loss,
    weighted_mse_loss,
    weighted_pooled_corr_loss,
)
from wnn.outcomes import (
    N_CODES,
    N_SP_AUX,
    ReadoutMoments,
    outcome_codes,
    readout,
    sequence_class,
    structure_labels,
)
from wnn.propensity import load_propensity, propensity
from wnn.structured_state import StructuredStateModel
from wnn.event_content import EventContentModel
from wnn.resources import configure_torch, dataloader_budget
from wnn.run_identity import (EpochUpdateCounter, group_identity, normalization_identity,
                              sample_order_identity, scaler_source_identity)
from wnn.splits import dev_halves, file_sha256, load_campaign_split, valid_row_groups
from wnn.spread_aux import PackedSpreadCache, structural_loss, validate_spread_cache

ROOT = Path(__file__).resolve().parents[1]
SEQ_BYTES = 20_000 * (N_FEATURES + 3) * 4
EV_I0PX = N_FEATURES + FEATURE_NAMES.index("ev_i0px")


def bind_fold_teachers(teachers, campaign_split, splits):
    """Distillation targets under a clean fold must come from teachers fitted on that same fold.

    Each cached prediction needs a sidecar naming the teacher run, split and the student's split-manifest hash,
    and its bytes must match the recorded prediction hash; the evaluation fold never reached the teacher's fit.
    """
    bound = []
    for name, scale in teachers:
        for split in splits:
            pred = ROOT / "cache" / f"pred_{name}_{split}.npy"
            side = json.loads(pred.with_suffix(".json").read_text())
            if side.get("run") != name or side.get("split") != split:
                raise ValueError(f"teacher sidecar does not describe {name}/{split}")
            if side.get("split_manifest_sha256") != campaign_split.sha256:
                raise ValueError(f"teacher {name} was not fitted on the student's clean fold")
            order = np.load(ROOT / "cache" / f"{split}_targets.npz")["row_group"]
            if list(map(int, side.get("row_groups", []))) != order.tolist():
                raise ValueError(f"teacher {name}/{split} rows are not in cache target order")
            digest = file_sha256(pred)
            if digest != side.get("prediction_sha256"):
                raise ValueError(f"teacher prediction bytes changed since export: {pred}")
            bound.append({"run": name, "split": split, "scale": scale, "epoch": side.get("epoch"),
                          "checkpoint_sha256": side.get("checkpoint_sha256"), "prediction_sha256": digest})
    return bound


class Sequences(Dataset):
    def __init__(self, split: str, indices, teachers=None, prop_cache=None, quote_path=False,
                 quote_cache_root=None, spread_cache_root=None, thin=None):
        self.path = ROOT / "datasets" / f"{split}.parquet"
        # thin = (probability, low, high): event-preserving row thinning for training views (see thin_rows)
        self.thin = thin
        self.indices = list(indices)
        self._file = None
        # prop_cache: path of cache/propcache_<name>_<split>.npy (row group x 20000 float32), or None
        self.prop_cache, self._prop = prop_cache, None
        # teachers: list of (name, scale); predictions are appended to targets as columns 2..3
        self.teachers = teachers
        self.split = split
        self._teach = None
        self.quote_path = quote_path
        self.quote_cache_root = Path(quote_cache_root) if quote_cache_root else ROOT / "cache"
        self._quote = None
        self.spread_cache_root = Path(spread_cache_root) if spread_cache_root else None
        self._spread = None
        if teachers and quote_path:
            raise ValueError("distillation and quote-path auxiliaries share target columns")

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
        if self.quote_path:
            if self._quote is None:
                self._quote = np.load(self.quote_cache_root / f"quote_path_{self.split}.npy", mmap_mode="r")
            y = np.concatenate([y, np.asarray(self._quote[self.indices[i]], dtype=np.float32)], axis=1)
        if self.prop_cache is not None:
            if self._prop is None:
                self._prop = np.load(self.prop_cache, mmap_mode="r")
            p = np.array(self._prop[self.indices[i]], dtype=np.float32)
            result = (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(scored), torch.from_numpy(p))
        else:
            result = (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(scored))
        if self.spread_cache_root is not None:
            if self._spread is None:
                self._spread = PackedSpreadCache(self.spread_cache_root, self.split)
            result += (torch.from_numpy(self._spread.read(self.indices[i])),)
        if self.thin is not None:
            result = thin_rows(result, self.thin)
        return result


def thin_rows(result, thin):
    """Training view of one sequence without some rows whose i0 price codes did not change.

    Targets are constant between quote moves and each target depends only on the future quote path, so deleting a
    row with unchanged i0 price codes (columns 0..21) after the warm-up leaves every kept row's label exact; the
    stream simply looks as if fewer book updates were published. With probability ``thin[0]`` a rate q ~ U(thin[1],
    thin[2]) of those rows is dropped; the sequence is padded back to its length by repeating the last kept row, and
    the kept length is appended so the loop can exclude padding from every loss.
    """
    prob, low, high = thin
    x = result[0]
    n = x.shape[0]
    if torch.rand(()) >= prob:
        return result + (torch.tensor(n),)
    q = low + (high - low) * torch.rand(())
    quiet = torch.zeros(n, dtype=torch.bool)
    quiet[WARMUP:] = (x[WARMUP:, :22] == x[WARMUP - 1:-1, :22]).all(dim=1)
    keep = ~(quiet & (torch.rand(n) < q))
    kept = int(keep.sum())
    index = torch.cat([torch.nonzero(keep).squeeze(1), torch.full((n - kept,), int(torch.nonzero(keep)[-1]))])
    thinned = []
    for j, part in enumerate(result):
        part = part[index]
        if j == 2:  # is_scored: padded rows are never scored
            part = part.clone()
            part[kept:] = False
        thinned.append(part)
    return tuple(thinned) + (torch.tensor(kept),)


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
        rows = []
        for st in self.steps[warmup:]:
            ev = st["_events"]
            acc = {}
            for (_, a), (name, b) in zip(ev[:-1], ev[1:], strict=True):
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


def next_label_change(y):
    """(B, T) index of the first row strictly after t where the target pair changes (T if none): the label is
    identical on (t, next - 1], so a causal estimate made at row next - 1 is a valid target for row t."""
    b, t, _ = y.shape
    idx = torch.arange(t, device=y.device).expand(b, t)
    changed = torch.zeros((b, t), dtype=torch.bool, device=y.device)
    changed[:, 1:] = (y[:, 1:] != y[:, :-1]).any(-1)
    marker = torch.where(changed, idx, torch.full_like(idx, t))
    at_or_after = torch.flip(torch.cummin(torch.flip(marker, [1]), 1).values, [1])
    return torch.cat([at_or_after[:, 1:], torch.full_like(at_or_after[:, :1], t)], 1)


def episode_redundancy_weight(y: torch.Tensor, scale: float, floor: float) -> torch.Tensor:
    """Reduce duplicate class-3 targets in long offline label episodes.

    Episode length uses future labels only to set training weights. It is never
    added to model inputs or computed during inference.
    """
    batch, steps, _ = y.shape
    row = torch.arange(steps, device=y.device)[None].expand(batch, -1)
    changed = torch.ones((batch, steps), dtype=torch.bool, device=y.device)
    changed[:, 1:] = (y[:, 1:] != y[:, :-1]).any(-1)
    start = torch.cummax(torch.where(changed, row, 0), 1).values
    marker = torch.where(changed, row, steps)
    next_change = torch.flip(torch.cummin(torch.flip(marker, (1,)), 1).values, (1,))
    end = torch.cat((next_change[:, 1:], torch.full_like(next_change[:, :1], steps)), 1)
    length = (end - start).float()
    weight = torch.rsqrt(1.0 + length / scale).clamp_min(floor)
    return torch.where(sequence_class(y)[:, None] == 3, weight, 1.0)


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
    if args_dict.get("arch") == "structured":
        if (args_dict["hidden"] != 96 or args_dict["layers"] != 2
                or args_dict["dropout"] != 0
                or args_dict.get("enc", "linear") != "linear"):
            raise ValueError("structured card B requires hidden96, layers2, dropout0, enc linear")
        model = StructuredStateModel(n_in, n_out, args_dict.get("decay_init", "multi"))
        for prefix in filter(None, args_dict.get("freeze", "").split(",")):
            if not any(name.startswith(prefix) for name, _ in model.named_parameters()):
                raise ValueError(f"unknown structured freeze prefix {prefix!r}; backbone is inp,blocks")
        return model
    if args_dict.get("arch") == "event-content":
        if (args_dict["hidden"] != 96 or args_dict["layers"] != 1
                or args_dict["dropout"] != 0 or args_dict.get("enc", "linear") != "linear"):
            raise ValueError("D1 requires hidden96,layers1,dropout0,enc linear")
        return EventContentModel(n_in, n_out, args_dict.get("memory_arm", "endpoint-content"))
    if args_dict.get("arch") == "gated":
        return GatedRNNModel(
            n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out=n_out
        )
    if args_dict.get("arch", "gru") == "dual":
        return DualClockModel(
            n_in,
            args_dict["hidden"],
            args_dict["layers"],
            args_dict["event_hidden"],
            args_dict["dropout"],
            n_out=n_out,
        )
    return RNNModel(
        n_in,
        args_dict["hidden"],
        args_dict["layers"],
        args_dict["dropout"],
        n_out=n_out,
        cell=args_dict.get("cell", "gru"),
        enc=args_dict.get("enc", "linear"),
        tok_dim=args_dict.get("tok_dim", 32),
        att_layers=args_dict.get("att_layers", 1),
        conv_k=args_dict.get("conv_k", 16),
    )


@torch.no_grad()
def predict(model, x, chunk, device):
    model.eval()
    inputs = build_inputs(x.to(device, non_blocking=True))
    state, outs = None, []
    for s in range(0, inputs.shape[1], chunk):
        with torch.autocast("cuda", dtype=torch.float16):
            out, state = forward_chunk(model, inputs, slice(s, s + chunk), state)
        outs.append(
            readout(out, *READOUT, tilted=READOUT_TILTED)
            if READOUT is not None
            else out[..., :2].float()
        )
    return torch.cat(outs, dim=1).cpu().numpy()


def evaluate(model, loader, chunk, device):
    preds, ys, masks, classes = [], [], [], []
    for x, y, m in loader:
        preds.append(predict(model, x, chunk, device))
        ys.append(y.numpy())
        selected = m.numpy().copy()
        selected[:, :WARMUP] = False
        masks.append(selected)
        classes += [market_class(t) for t in y.numpy()]
    p, y, m = np.concatenate(preds), np.concatenate(ys), np.concatenate(masks)
    res = score(y, p, m)
    legacy = (
        block_aggregate_score(y, p, m)
        if res["eligible_blocks"]
        else {"weighted_pearson": None, "t0": None, "t1": None}
    )
    res.update(
        global_t0=res["t0"],
        global_t1=res["t1"],
        per_seq_wp=legacy["weighted_pearson"],
        block_t0=legacy["t0"],
        block_t1=legacy["t1"],
    )
    res["frac_abs_pred_gt2"] = float((np.abs(p[m]) > 2).mean())
    per = block_scores(y, p, m).mean(axis=1)
    cls = np.array(classes)
    res["by_class"] = {
        int(c): score(y[cls == c], p[cls == c], m[cls == c])["weighted_pearson"]
        for c in np.unique(cls)
    }
    res["block_by_class"] = {
        int(c): round(float(np.nanmean(per[cls == c])), 4)
        if np.isfinite(per[cls == c]).any()
        else None
        for c in np.unique(cls)
    }
    return res


def configure_checkpoint_policy(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Require predeclared checkpoint selection when evaluation is withheld."""
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.checkpoint_policy == "fixed":
        if args.checkpoint_epoch is None:
            args.checkpoint_epoch = args.epochs - 1
        if not 0 <= args.checkpoint_epoch < args.epochs:
            parser.error("--checkpoint-epoch must be within the fitted epochs")
    elif args.checkpoint_epoch is not None:
        parser.error("--checkpoint-epoch requires --checkpoint-policy fixed")
    if args.no_eval_during_fit and args.checkpoint_policy != "fixed":
        parser.error("--no-eval-during-fit requires --checkpoint-policy fixed")
    if args.split_manifest and not (args.no_eval_during_fit and args.checkpoint_policy == "fixed"):
        parser.error("campaign manifests require fixed checkpoints and --no-eval-during-fit")


def checkpoint_selected(
    epoch: int,
    policy: str,
    fixed_epoch: int | None,
    evaluation_score: float | None,
    best_score: float,
) -> bool:
    """Select a fixed epoch without consulting evaluation, or retain legacy best selection."""
    if policy == "fixed":
        return epoch == fixed_epoch
    if evaluation_score is None:
        raise ValueError("best-checkpoint selection requires evaluation")
    return evaluation_score > best_score


def checkpoint_model(
    model: torch.nn.Module,
    ema_model: FastEMA | torch.optim.swa_utils.AveragedModel | None,
    swa_model: torch.optim.swa_utils.AveragedModel | None,
    use_ema: bool,
) -> torch.nn.Module:
    """Choose fitted snapshot weights independently of whether evaluation runs."""
    if use_ema:
        if ema_model is None:
            raise ValueError("EMA checkpoint requested without an EMA model")
        return ema_model.module
    return swa_model.module if swa_model is not None else model


def inherit_init_arguments(args: argparse.Namespace, checkpoint: dict, options: list[str]) -> None:
    """Preserve the parent's feature construction and model/output dimensions during FT."""
    fields = (
        "hidden",
        "layers",
        "dropout",
        "arch",
        "decay_init",
        "memory_arm",
        "cell",
        "enc",
        "tok_dim",
        "att_layers",
        "conv_k",
        "event_hidden",
        "online_norm",
        "raw_only",
        "heuristics",
        "prop_input",
        "aux",
        "aux_struct",
        "aux_next",
        "aux_next_k",
        "aux_tier",
        "aux_mask",
        "aux_sign",
        "aux_quote",
        "quote_cache_root",
    )
    source = checkpoint["args"]
    for field in fields:
        if field in source:
            option = "--" + field.replace("_", "-")
            if (
                any(value == option or value.startswith(option + "=") for value in options)
                and getattr(args, field) != source[field]
            ):
                raise ValueError(f"{option} disagrees with the parent checkpoint")
            setattr(args, field, source[field])


def validate_campaign_checkpoint(checkpoint: dict, args: argparse.Namespace) -> None:
    """Reject initialization or propensity weights fitted outside this clean outer fold."""
    if args.split_manifest:
        source = checkpoint.get("args", {})
        if (source.get("split_manifest_sha256") != args.split_manifest_sha256
                or not source.get("no_eval_during_fit")
                or source.get("checkpoint_policy") != "fixed"
                or checkpoint.get("epoch") != source.get("checkpoint_epoch")):
            raise ValueError("source checkpoint lacks matching clean-fold fitting provenance")


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
    ap.add_argument("--checkpoint-policy", choices=["best", "fixed"], default="best")
    ap.add_argument("--checkpoint-epoch", type=int, default=None, help="zero-based fixed checkpoint epoch")
    ap.add_argument("--no-eval-during-fit", action="store_true", help="never construct or read an evaluation loader")
    ap.add_argument("--split-manifest", type=Path, default=None, help="frozen component-excluded outer-fold manifest")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--loss", choices=["corr", "wmse", "mix", "huber", "ce_joint", "mse", "gcorr", "wmse_gcorr", "gcorr_pool"],
                    default="mix")
    ap.add_argument("--thin-prob", type=float, default=0.0,
                    help="training-view row thinning: probability per sequence (0 disables)")
    ap.add_argument("--thin-range", default="0.1,0.4", help="low,high of the dropped share of quiet rows")
    ap.add_argument("--pool-decay", type=float, default=0.98,
                    help="--loss gcorr_pool: per-chunk decay of the pooled weighted sums kept from earlier chunks")
    ap.add_argument("--huber-delta", type=float, default=0.5)
    ap.add_argument("--ce-tilt", action="store_true", help="weight code cross-entropy rows by mean |clip(y)|")
    ap.add_argument("--ce-mse", type=float, default=0.0, help="add this weight x wMSE on the differentiable readout")
    ap.add_argument("--distill", default="", help="teacher runs name[*scale],...; cached pred_<name>_<split>.npy")
    ap.add_argument("--distill-alpha", type=float, default=0.5, help="target = (1-a) y + a mean(teachers)")
    ap.add_argument("--distill-alpha-far", type=float, default=None,
                    help="noise-adaptive distillation: a(u) = a + (far - a)(1 - exp(-u/tau)), u = rows until the label "
                         "changes (training-time only; inputs stay causal)")
    ap.add_argument("--distill-alpha-tau", type=float, default=32.0, help="rows scale of the noise-adaptive weight")
    ap.add_argument("--aux-next", type=float, default=0.0,
                    help="weight of auxiliary heads: log rows to next target change (MSE) and its sign (BCE)")
    ap.add_argument("--aux-tier", type=float, default=0.0,
                    help="class-3 winning-target tick-tier auxiliary CE; head discarded for prediction")
    ap.add_argument("--aux-mask", type=float, default=0.0,
                    help="validation-source scoring-mask auxiliary BCE; head discarded for prediction")
    ap.add_argument("--aux-sign", type=float, default=0.0,
                    help="current target-contrast direction BCE; head discarded for prediction")
    ap.add_argument("--aux-quote", type=float, default=0.0,
                    help="training-only class-3 future 8/24-book-event mean quote displacement MSE")
    ap.add_argument("--quote-cache-root", default="",
                    help="optional paired auxiliary cache directory; historical cache remains the default")
    ap.add_argument("--aux-next-k", default="", help="extra event horizons for the next-change aux heads, e.g. 2,4")
    ap.add_argument("--spread-aux-mode", choices=["original", "proxy_known", "current_known"], default="original")
    ap.add_argument("--spread-cache-root", default="", help="supervision-only fold0 known-code cache")
    ap.add_argument("--aux-struct", type=float, default=0.0,
                    help="weight of auxiliary cross-entropy heads for spread (ticks), branch and class")
    ap.add_argument("--aux", default="", help="future row shifts as auxiliary targets, e.g. 20,200")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--arch", choices=["gru", "dual", "gated", "structured", "event-content"], default="gru")
    ap.add_argument("--decay-init", choices=["multi", "single"], default="multi",
                    help="structured-state initial half-lives; identical learned bounds and phases")
    ap.add_argument("--memory-arm", choices=["endpoint-content", "endpoint-position", "raw-code-content"], default="endpoint-content")
    ap.add_argument("--gate-ce", type=float, default=0.0,
                    help="BCE weight for the gated model's target-difference sign head")
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
    ap.add_argument("--include-heldout-valid", action="store_true",
                    help="final fit only: include the reserved validation block in training")
    ap.add_argument("--valid-train-fraction", type=float, default=1.0,
                    help="fixed chain subset of eligible validation training sequences for a data-volume comparison")
    ap.add_argument("--epoch-samples", type=int, default=0,
                    help="draw this many training sequences with replacement each epoch to match optimizer steps")
    ap.add_argument("--eval-source", choices=["dev", "devB", "holdout", "vfold"], default="dev")
    ap.add_argument("--valid-fold", type=int, default=-1,
                    help="K-fold over ALL validation sequences: train on train + other folds, evaluate on this fold")
    ap.add_argument("--n-valid-folds", type=int, default=4)
    ap.add_argument("--valid-repeat", type=int, default=1,
                    help="with train+valid: each validation training sequence appears this many times per epoch")
    ap.add_argument("--init", default="", help="run name whose model.pt initializes weights; NAME@epN loads model_epN.pt")
    ap.add_argument("--inherit-init-preprocessing", action="store_true",
                    help="inherit parent feature construction and architecture; reject conflicting explicit flags")
    ap.add_argument("--freeze", default="", help="comma-separated parameter-name prefixes kept fixed, e.g. inp,gru")
    ap.add_argument("--ema-teacher", type=float, default=0.0,
                    help="mean-teacher consistency: weight of (out - EMA-model out)^2 on the two target outputs")
    ap.add_argument("--ema-metric-weight", action="store_true",
                    help="weight teacher consistency by |clipped target|, matching the scored loss")
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
    ap.add_argument("--loss-norm", choices=["chunk", "running"], default="chunk",
                    help="wmse main loss divided by each chunk's weight sum (chunk) or by a running mean of it "
                         "(running: rows keep the pooled metric's relative weights across chunks; brainteaser card B)")
    ap.add_argument("--since-weight", type=float, default=0.0,
                    help="if > 0: multiply main-loss row weights by max(floor, 1/(1+since/tau)), since = rows since the "
                         "last i0 price event (causal input only; ideas_round4 B2)")
    ap.add_argument("--since-floor", type=float, default=0.25)
    ap.add_argument("--episode-weight", type=float, default=0.0,
                    help="scale for training-only inverse-sqrt class-3 target episode length")
    ap.add_argument("--episode-floor", type=float, default=0.5)
    ap.add_argument("--td-alpha", type=float, default=0.0,
                    help="TD targets (ideas_round4 B1b): mix the label with the EMA teacher's output at the last row "
                         "before the next label change (same |y| weights); needs the EMA model")
    ap.add_argument("--td-ramp", type=float, default=0.5, help="epochs over which td-alpha ramps up from 0")
    ap.add_argument("--gcorr-weight", type=float, default=1.0, help="weight of the pooled-Pearson term in --loss wmse_gcorr")
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
    init_checkpoint = None
    if args.inherit_init_preprocessing:
        if not args.init:
            ap.error("--inherit-init-preprocessing requires --init")
        parent_run, _, parent_epoch = args.init.partition("@ep")
        parent_file = f"model_ep{int(parent_epoch)}.pt" if parent_epoch else "model.pt"
        init_checkpoint = torch.load(ROOT / "artifacts" / "runs" / parent_run / parent_file,
                                     map_location="cpu", weights_only=False)
        inherit_init_arguments(args, init_checkpoint, sys.argv[1:])
    configure_checkpoint_policy(args, ap)
    campaign_split = load_campaign_split(args.split_manifest, ROOT) if args.split_manifest else None
    if campaign_split is not None:
        if args.valid_fold not in (-1, campaign_split.fold):
            ap.error("--valid-fold disagrees with --split-manifest")
        if args.include_heldout_valid or args.train_source == "valid_devA":
            ap.error("campaign splits require train, valid, or train+valid sources without heldout inclusion")
        if args.prop_input:
            ap.error("campaign splits require explicit provenance for propensity input")
        args.valid_fold, args.n_valid_folds = campaign_split.fold, campaign_split.n_folds
        args.eval_source = "vfold"
        args.split_manifest = str(campaign_split.path)
        args.split_manifest_sha256 = campaign_split.sha256
    if args.loss == "gcorr_pool" and args.aux:
        ap.error("--loss gcorr_pool keeps one pooled memory for the main head; --aux shift heads would share it")
    if args.distill and args.loss != "wmse":
        ap.error("--distill trains wMSE toward the blended target; pass --loss wmse")
    if args.aux_quote > 0 and args.distill:
        ap.error("--aux-quote and --distill both use target columns 2:4")
    if args.quote_cache_root and args.aux_quote <= 0:
        ap.error("--quote-cache-root requires --aux-quote")
    if args.fast:
        args.sync_free = args.fused_adam = args.pin_memory = args.prop_cache = True
    if args.profile_steps:
        args.max_batches = args.profile_warmup + args.profile_steps
    if args.mask_weight and args.train_source == "train":
        ap.error("--mask-weight needs is_scored, which train.parquet does not have")
    if args.include_heldout_valid and (args.valid_fold >= 0 or args.train_source not in ("train+valid", "valid")):
        ap.error("--include-heldout-valid is only for a final train+valid or valid fit")
    if not 0 < args.valid_train_fraction <= 1:
        ap.error("--valid-train-fraction must be in (0, 1]")
    if args.valid_train_fraction < 1 and args.train_source not in ("train+valid", "valid"):
        ap.error("--valid-train-fraction requires a validation training source")
    if args.epoch_samples < 0 or (args.epoch_samples and args.valid_repeat != 1):
        ap.error("--epoch-samples must be nonnegative and requires --valid-repeat 1")
    if args.epoch_samples and args.class_balance:
        ap.error("--epoch-samples cannot be combined with --class-balance")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    duty = configure_torch()
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    out_dir = ROOT / "artifacts" / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_path = ROOT / "datasets" / "valid.parquet"
    if campaign_split is None:
        dev_groups, hold_groups = valid_row_groups(valid_path)
        dev_a, dev_b = dev_halves(valid_path)
    else:
        dev_groups = dev_a = np.asarray(campaign_split.valid_groups)
        hold_groups = dev_b = np.asarray(campaign_split.evaluation_groups)
    train_split = "train"
    extra_valid = None
    fold_groups = None
    all_valid = np.r_[np.asarray(dev_groups), np.asarray(hold_groups)]
    if args.valid_fold >= 0:
        if campaign_split is not None:
            fold_groups = np.asarray(campaign_split.evaluation_groups)
            if args.train_source == "train+valid":
                extra_valid = np.asarray(campaign_split.valid_groups)
        else:
            # Legacy folds exclude validation groups only; campaign manifests also exclude linked train groups.
            cf = np.load(ROOT / "cache" / "chain_folds.npz")
            vmask = cf["split"] == "valid"
            fold_map = dict(zip(cf["row_group"][vmask].tolist(), cf["fold"][vmask].tolist(), strict=True))
            fold_of = np.array([fold_map[int(g)] for g in all_valid]) % args.n_valid_folds
            fold_groups = all_valid[fold_of == args.valid_fold]
            if args.train_source == "train+valid":
                extra_valid = all_valid[fold_of != args.valid_fold]
    elif args.train_source == "train+valid":
        extra_valid = all_valid if args.include_heldout_valid else np.r_[np.asarray(dev_a), np.asarray(dev_b)]
    valid_pool = None
    if args.train_source == "valid":
        valid_pool = (np.setdiff1d(all_valid, fold_groups) if fold_groups is not None
                      else all_valid if args.include_heldout_valid else np.asarray(dev_groups))
    elif extra_valid is not None:
        valid_pool = np.asarray(extra_valid)
    eligible_valid_count = 0 if valid_pool is None else len(valid_pool)
    if valid_pool is not None and args.valid_train_fraction < 1:
        if campaign_split is not None:
            chain_map = {group: component for split, group, component in campaign_split.components if split == "valid"}
        else:
            cf = np.load(ROOT / "cache" / "chain_folds.npz")
            vmask = cf["split"] == "valid"
            chain_map = dict(zip(cf["row_group"][vmask].tolist(), cf["chain"][vmask].tolist(), strict=True))
        pool_chains = np.array([chain_map[int(group)] for group in valid_pool])
        valid_classes = np.load(ROOT / "research/artifacts/market-classes-v1.npz")["valid"][valid_pool]
        rng = np.random.default_rng(20260928)
        selected_chains = []
        for market_id in np.unique(valid_classes):
            in_class = valid_classes == market_id
            class_chains, class_sizes = np.unique(pool_chains[in_class], return_counts=True)
            order = rng.permutation(len(class_chains))
            target = max(1, round(in_class.sum() * args.valid_train_fraction))
            stop = np.searchsorted(np.cumsum(class_sizes[order]), target, side="left") + 1
            selected_chains.extend(class_chains[order[:stop]])
        valid_pool = valid_pool[np.isin(pool_chains, selected_chains)]
        if args.train_source == "train+valid":
            extra_valid = valid_pool
    if args.train_source == "valid":
        # with --valid-fold: all valid minus the held-out chain fold; without: devA + devB (holdout never)
        train_split = "valid"
        train_idx = np.random.default_rng(args.seed).permutation(valid_pool)
    elif args.train_source == "valid_devA":
        train_split = "valid"
        train_idx = np.random.default_rng(args.seed).permutation(dev_a)[: args.train_seqs]
    else:
        usable = (np.asarray(campaign_split.train_groups) if campaign_split is not None else
                  np.array([g for g in range(10607) if g not in CORRUPT_TRAIN_ROW_GROUPS]))
        if args.train_classes:
            keep = {int(c) for c in args.train_classes.split(",")}
            cls = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["train"]
            usable = usable[np.isin(cls[usable], list(keep))]
        train_idx = np.random.default_rng(args.seed).permutation(usable)[: args.train_seqs]
    if fold_groups is not None and extra_valid is not None and np.intersect1d(extra_valid, fold_groups).size:
        raise RuntimeError("held-out validation fold entered the training pool")
    if campaign_split is not None:
        campaign_split.assert_fit_groups(train_idx if train_split == "train" else (),
                                        train_idx if train_split == "valid" else extra_valid if extra_valid is not None else ())
    eval_groups = {"devB": dev_b, "holdout": hold_groups, "vfold": fold_groups}.get(args.eval_source, dev_groups)
    n_valid = len(eval_groups) if args.full_valid else min(args.valid_seqs, len(eval_groups))
    valid_idx = np.random.default_rng(12345).permutation(eval_groups)[:n_valid]

    workers, prefetch = dataloader_budget(args.batch * SEQ_BYTES)
    sampler = None
    if args.class_balance:
        classes = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")
        tr_cls = classes[train_split][train_idx]
        valid_classes = classes["valid"][list(campaign_split.valid_groups)] if campaign_split is not None else classes["valid"]
        va_share = np.array([(valid_classes == c).mean() for c in range(4)])
        tr_share = np.array([max((tr_cls == c).mean(), 1e-9) for c in range(4)])
        weights = np.where(tr_cls >= 0, va_share[np.clip(tr_cls, 0, 3)] / tr_share[np.clip(tr_cls, 0, 3)], 1.0)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(train_idx),
                                        replacement=True, generator=torch.Generator().manual_seed(args.seed))
    teachers = None
    if args.distill:
        teachers = [(t.partition("*")[0], float(t.partition("*")[2] or 1.0)) for t in args.distill.split(",")]
        if campaign_split is not None:
            used = {"train": ("train",), "valid": ("valid",)}.get(args.train_source, ("train", "valid"))
            args.distill_provenance = bind_fold_teachers(teachers, campaign_split, used)
    prop_cache = {}
    if args.prop_cache and args.propensity:
        prop_cache = {sp: prop_cache_path(args.propensity, sp) for sp in ("train", "valid")}
    quote_cache_provenance = None
    if args.quote_cache_root:
        if campaign_split is None:
            ap.error("--quote-cache-root requires a clean --split-manifest")
        cache_root = Path(args.quote_cache_root)
        if not cache_root.is_absolute():
            cache_root = ROOT / cache_root
        cache_root = cache_root.resolve()
        metadata_path = cache_root.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("split_manifest_sha256") != campaign_split.sha256
                or set(metadata.get("fit_train_groups", [])) != set(campaign_split.train_groups)
                or set(metadata.get("fit_valid_groups", [])) != set(campaign_split.valid_groups)
                or set(metadata.get("evaluation_valid_groups", [])) != set(campaign_split.evaluation_groups)):
            raise ValueError("quote-cache provenance disagrees with clean fold")
        if (not set(train_idx).issubset(metadata["fit_train_groups"] if train_split == "train"
                                            else metadata["fit_valid_groups"])
                or extra_valid is not None and not set(extra_valid).issubset(metadata["fit_valid_groups"])):
            raise ValueError("training row groups are outside the quote cache fit pool")
        cache_files = {}
        for split in ("train", "valid"):
            path = cache_root / f"quote_path_{split}.npy"
            entry = metadata["files"][f"{cache_root.name}/{split}"]
            if not path.is_file() or len(entry["sha256"]) != 64:
                raise ValueError(f"missing or unproven quote cache: {path}")
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"quote cache content hash disagrees with metadata: {path}")
            cache_files[split] = entry["sha256"]
        args.quote_cache_root = str(cache_root)
        quote_cache_provenance = {"metadata_sha256": file_sha256(metadata_path),
                                  "array_sha256_expected": cache_files,
                                  "variant": cache_root.name}
    spread_provenance = None
    if args.spread_aux_mode != "original":
        if (campaign_split is None or campaign_split.fold != 0 or args.aux_struct <= 0
                or not args.spread_cache_root or args.train_source != "train+valid"
                or args.init != "c20260929_since_s3_f0@ep3" or args.seed != 3
                or args.epochs != 1 or args.checkpoint_policy != "fixed"
                or args.checkpoint_epoch != 0 or not args.no_eval_during_fit):
            raise ValueError("known-code spread screen requires fixed one-epoch clean fold0 continuation from seed3 ep3")
        spread_root = Path(args.spread_cache_root)
        if not spread_root.is_absolute():
            spread_root = ROOT / spread_root
        spread_root = spread_root.resolve()
        spread_provenance = validate_spread_cache(spread_root, campaign_split.sha256, train_idx,
                                                   extra_valid, campaign_split.evaluation_groups)
        if json.loads((spread_root / "metadata.json").read_text())["N_SP_AUX"] != N_SP_AUX:
            raise ValueError("spread cache head dimension differs")
        spread_provenance["mode"] = args.spread_aux_mode
        spread_provenance["source_sha256"] = file_sha256(ROOT / "src/wnn/spread_aux.py")
        args.spread_cache_root = str(spread_root)
    elif args.spread_cache_root:
        raise ValueError("spread cache supplied without a matched known-code mode")
    thin = None
    if args.thin_prob > 0:
        low, high = (float(v) for v in args.thin_range.split(","))
        thin = (args.thin_prob, low, high)
    train_ds = Sequences(train_split, train_idx, teachers, prop_cache.get(train_split),
                         quote_path=args.aux_quote > 0, quote_cache_root=args.quote_cache_root,
                         spread_cache_root=args.spread_cache_root, thin=thin)
    if extra_valid is not None:
        valid_ds = Sequences("valid", extra_valid, teachers, prop_cache.get("valid"),
                             quote_path=args.aux_quote > 0, quote_cache_root=args.quote_cache_root,
                             spread_cache_root=args.spread_cache_root, thin=thin)
        train_ds = torch.utils.data.ConcatDataset([train_ds] + [valid_ds] * args.valid_repeat)
    if args.epoch_samples:
        sample_weights = np.ones(len(train_ds), dtype=np.float64)
        if extra_valid is not None:
            sample_weights[len(train_idx):] = eligible_valid_count / len(extra_valid)
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), args.epoch_samples,
                                        replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=sampler is None,
                              sampler=sampler, drop_last=True, num_workers=workers, prefetch_factor=prefetch,
                              pin_memory=args.pin_memory)  # no persistent workers: they change the shuffle order (parity E-169)
    valid_loader = None if args.no_eval_during_fit else DataLoader(
        Sequences("valid", valid_idx), batch_size=16, num_workers=2, prefetch_factor=1)

    shifts = [int(v) for v in args.aux.split(",") if v]
    provenance = {"seed": args.seed, "args": vars(args).copy(),
                  "split_manifest_sha256": campaign_split.sha256 if campaign_split is not None else None,
                  "fit_row_groups": {"train": train_idx.tolist() if train_split == "train" else [],
                                     "valid": train_idx.tolist() if train_split == "valid" else
                                     extra_valid.tolist() if extra_valid is not None else []},
                  "evaluation_row_groups": valid_idx.tolist(), "checkpoint_sha256": {}}
    if args.arch == "structured":
        provenance["model_source_sha256"] = file_sha256(ROOT / "src/wnn/structured_state.py")
    if spread_provenance is not None:
        provenance["spread_aux"] = spread_provenance
    if args.arch == "event-content":
        provenance["model_source_sha256"] = file_sha256(ROOT / "src/wnn/event_content.py")
    if quote_cache_provenance is not None:
        provenance["quote_cache"] = quote_cache_provenance
    if args.propensity:
        propensity_path = ROOT / "artifacts" / "runs" / args.propensity / "model.pt"
        if campaign_split is not None:
            validate_campaign_checkpoint(torch.load(propensity_path, map_location="cpu", weights_only=False), args)
        provenance["propensity_checkpoint_sha256"] = file_sha256(propensity_path)
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
    tier_at = n_out
    if args.aux_tier > 0:
        n_out += 4
    mask_at = n_out
    if args.aux_mask > 0:
        n_out += 1
    sign_at = n_out
    if args.aux_sign > 0:
        n_out += 1
    quote_at = n_out
    if args.aux_quote > 0:
        n_out += 2
    model = make_model(vars(args), n_in, n_out).to(device)
    moments = ReadoutMoments(device) if args.loss == "ce_joint" else None
    if args.init:
        init_run, _, init_epoch = args.init.partition("@ep")
        init_file = f"model_ep{int(init_epoch)}.pt" if init_epoch else "model.pt"
        init_path = ROOT / "artifacts" / "runs" / init_run / init_file
        init = init_checkpoint if init_checkpoint is not None else torch.load(init_path, map_location="cpu", weights_only=False)
        validate_campaign_checkpoint(init, args)
        provenance["initial_checkpoint_sha256"] = file_sha256(init_path)
        model.load_state_dict(init["model"])
    else:
        fit_normalization(model, train_loader, device)
    provenance["normalization_identity"] = normalization_identity(model)
    provenance["update_counter_implementation"] = scaler_source_identity()
    provenance["run_identity_source_sha256"] = file_sha256(ROOT / "src/wnn/run_identity.py")
    provenance["fit_membership_identity"] = {
        "train": group_identity(provenance["fit_row_groups"]["train"]),
        "valid": group_identity(provenance["fit_row_groups"]["valid"]),
        "valid_repeat": args.valid_repeat, "dataset_length": len(train_ds),
    }
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
    if args.ema_teacher > 0 or args.eval_ema or args.td_alpha > 0:
        if args.sync_free:
            ema_model = FastEMA(model, args.ema_decay)
        else:
            ema_model = torch.optim.swa_utils.AveragedModel(
                model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay))
            ema_model.eval()
    swa_start = int(total * (1.0 - args.swa_frac))
    step_count = 0

    pool_memory = PooledCorrMemory(args.pool_decay) if args.loss == "gcorr_pool" else None

    def objective(o, t, v):
        if pool_memory is not None:
            return pool_memory(o, t, v)
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
        if args.loss == "gcorr":
            return weighted_pooled_corr_loss(o, t, v)
        if args.loss == "wmse_gcorr":
            return weighted_mse_loss(o, t, v) + args.gcorr_weight * weighted_pooled_corr_loss(o, t, v)
        return weighted_corr_loss(o, t, v) + 0.5 * weighted_mse_loss(o, t, v)

    print(json.dumps(vars(args)), flush=True)
    print(json.dumps({"source_train_sequences": len(train_idx),
                      "eligible_valid_train_sequences": eligible_valid_count,
                      "used_valid_train_sequences": len(extra_valid) if extra_valid is not None else
                      len(train_idx) if train_split == "valid" else 0,
                      "eval_sequences": n_valid}), flush=True)
    print(f"loader workers {workers} prefetch {prefetch}; params "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M; steps {total}", flush=True)

    timer = StageTimer(args.profile_steps > 0, args.profile_warmup)
    norm_ema = None  # running weight-sum normalizer for --loss-norm running
    losses_trace = []
    best, t0, history = -1.0, time.time(), []
    fitting_started = t0
    cumulative_updates = {}
    provenance["update_diagnostics"] = []
    for epoch in range(args.epochs):
        model.train()
        epoch_counter = EpochUpdateCounter(device)
        epoch_order_identity = sample_order_identity(train_loader)
        batches = CudaPrefetcher(train_loader, device, timer) if args.pin_memory else train_loader
        for b, batch in enumerate(batches):
            if args.max_batches and b >= args.max_batches:
                break
            timer.start()
            prop_cached = spread_encoded = kept_len = None
            if args.pin_memory:
                (x, y, scored_dev, *rest), scored = batch
            else:
                x, y, scored, *rest = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                scored_dev = scored.to(device, non_blocking=True)
            if thin is not None:
                kept_len = rest.pop()
                if not args.pin_memory:
                    kept_len = kept_len.to(device, non_blocking=True)
            if args.spread_aux_mode != "original":
                spread_encoded = rest.pop()
                if not args.pin_memory:
                    spread_encoded = spread_encoded.to(device, non_blocking=True)
            if rest:
                prop_cached = rest[0] if args.pin_memory else rest[0].to(device, non_blocking=True)
            timer.mark("h2d")
            inputs = build_inputs(x)
            timer.mark("inputs")
            teach = None
            quote_label = None
            if y.shape[-1] > 2:
                if args.aux_quote > 0:
                    quote_label, y = y[..., 2:4], y[..., :2]
                else:
                    teach, y = y[..., 2:4], y[..., :2]
            steps = torch.arange(inputs.shape[1], device=device)[None].expand(inputs.shape[0], -1)
            valid = steps >= WARMUP
            if kept_len is not None:
                valid = valid & (steps < kept_len[:, None])
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
            if args.since_weight > 0:
                since = torch.expm1(inputs[..., N_FEATURES + FEATURE_NAMES.index("since_i0px")].float())
                sw_rows = torch.clamp(1.0 / (1.0 + since / args.since_weight), min=args.since_floor)
                main_valid = main_valid.float() * sw_rows
            if args.episode_weight > 0:
                main_valid = main_valid.float() * episode_redundancy_weight(
                    y, args.episode_weight, args.episode_floor
                )
            timer.mark("weights")
            aux = shifted_targets(y, valid, shifts)
            if args.aux_struct > 0:
                sp_lab, br_lab, cl_lab = structure_labels(y, sequence_class(y))
                if args.spread_aux_mode == "current_known":
                    sp_lab = spread_encoded.clamp_max(N_SP_AUX - 1).long()
                sw = main_valid.float()
            if args.aux_next > 0:
                nx_rows, nx_up, nx_ok = next_change_labels(y, valid)
                nw = main_valid.float() * nx_ok.float() if main_valid.dtype != torch.bool else (main_valid & nx_ok).float()
                kth = [kth_change_labels(y, valid, kk) for kk in ks]
            if args.aux_tier > 0:
                tier_label = (y.float().max(dim=-1).values / float(TARGET_QUANTA[3])).round().long().clamp(0, 3)
                tier_weight = (main_valid.float() * (sequence_class(y) == 3)[:, None]
                               * y.float().clamp(-2, 2).abs().mean(dim=-1))
            if args.aux_mask > 0:
                mask_rows = (valid & scored_dev.any(dim=1)[:, None]).float()
            if moments is not None:
                codes = outcome_codes(y, sequence_class(y))
                row_w = main_valid.float()
                ce_w = row_w * y.float().clamp(-2, 2).abs().mean(-1) if args.ce_tilt else row_w
                if epoch == 0:
                    moments.update(codes, y, row_w)
            nxt_label = next_label_change(y) if args.distill_alpha_far is not None else None
            if args.td_alpha > 0:
                nxt_all = next_label_change(y)
                ramp_steps = max(1, int(args.td_ramp * len(train_loader) * math.ceil(20_000 / args.chunk)))
                td_a = args.td_alpha * min(1.0, step_count / ramp_steps)
            timer.mark("labels")
            h = h_ema = None
            for s in range(0, inputs.shape[1], args.chunk):
                sl = slice(s, s + args.chunk)
                with torch.autocast("cuda", dtype=amp_dtype):
                    out, h = forward_chunk(model, inputs, sl, h)
                h = detach_state(h)
                timer.mark("forward")
                if args.ema_teacher > 0 or args.td_alpha > 0:
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
                    if args.distill_alpha_far is None:
                        a = args.distill_alpha
                    else:
                        # far from the label's resolution the realized target is mostly unknowable future path:
                        # lean on the teacher there, on the label near the next change
                        rows = torch.arange(sl.start, sl.start + yc.shape[1], device=yc.device)[None]
                        u = (nxt_label[:, sl] - rows).float().clamp_min(0)
                        a = (args.distill_alpha + (args.distill_alpha_far - args.distill_alpha)
                             * (1 - torch.exp(-u / args.distill_alpha_tau)))[..., None]
                    tgt = (1 - a) * yc + a * teach[:, sl].float().clamp(-2, 2)
                    w = yc.abs() * main_valid[:, sl][..., None]
                    loss = (w * (out[..., :2].float() - tgt) ** 2).sum() / w.sum().clamp_min(1e-8)
                elif args.td_alpha > 0:
                    yc = y[:, sl].float().clamp(-2, 2)
                    s_end = min(s + args.chunk, inputs.shape[1])
                    jg = torch.clamp(nxt_all[:, sl], max=s_end) - 1  # last row of the constant-label run inside the chunk
                    rows = torch.arange(s, s_end, device=device)[None]
                    teach_t = torch.gather(out_ema[..., :2].float(), 1, (jg - s).clamp(min=0)[..., None].expand(-1, -1, 2))
                    tgt = torch.where((jg > rows)[..., None], (1 - td_a) * yc + td_a * teach_t.clamp(-2, 2), yc)
                    w = yc.abs() * main_valid[:, sl][..., None]
                    loss = (w * (out[..., :2].float() - tgt) ** 2).sum() / w.sum().clamp_min(1e-8)
                elif args.loss_norm == "running" and args.loss == "wmse":
                    yc = y[:, sl].float().clamp(-2, 2)
                    w = yc.abs() * main_valid[:, sl][..., None]
                    wsum = w.sum().detach()
                    norm_ema = wsum if norm_ema is None else 0.99 * norm_ema + 0.01 * wsum
                    loss = (w * (out[..., :2].float() - yc) ** 2).sum() / norm_ema.clamp_min(1e-8)
                else:
                    loss = objective(out[..., :2], y[:, sl], main_valid[:, sl])
                if args.arch == "gated" and args.gate_ce > 0:
                    difference = y[:, sl, 0].float() - y[:, sl, 1].float()
                    gate_weight = main_valid[:, sl].float() * difference.abs().clamp_max(2.0)
                    gate_bce = F.binary_cross_entropy_with_logits(
                        out[..., n_out].float(), (difference > 0).float(), reduction="none"
                    )
                    loss = loss + args.gate_ce * (gate_bce * gate_weight).sum() / gate_weight.sum().clamp_min(1e-8)
                if args.aux_struct > 0:
                    loss = loss + args.aux_struct * structural_loss(
                        out, sp_lab[:, sl], br_lab[:, sl], cl_lab[:, sl], sw[:, sl], struct_at, N_SP_AUX,
                        None if spread_encoded is None else spread_encoded[:, sl])
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
                if args.aux_tier > 0:
                    ce = F.cross_entropy(
                        out[..., tier_at:tier_at + 4].float().reshape(-1, 4),
                        tier_label[:, sl].reshape(-1), reduction="none",
                    )
                    tw = tier_weight[:, sl].reshape(-1)
                    loss = loss + args.aux_tier * (ce * tw).sum() / tw.sum().clamp_min(1e-8)
                if args.aux_mask > 0:
                    ce = F.binary_cross_entropy_with_logits(
                        out[..., mask_at].float(), scored_dev[:, sl].float(),
                        pos_weight=out.new_tensor(8.0), reduction="none",
                    )
                    mw = mask_rows[:, sl]
                    loss = loss + args.aux_mask * (ce * mw).sum() / mw.sum().clamp_min(1.0)
                if args.aux_sign > 0:
                    clipped = y[:, sl].float().clamp(-2, 2)
                    difference = clipped[..., 0] - clipped[..., 1]
                    sign_weight = main_valid[:, sl].float() * difference.abs().clamp_max(2.0)
                    sign_bce = F.binary_cross_entropy_with_logits(
                        out[..., sign_at].float(), (difference > 0).float(), reduction="none"
                    )
                    loss = loss + args.aux_sign * (sign_bce * sign_weight).sum() / sign_weight.sum().clamp_min(1e-8)
                if args.aux_quote > 0:
                    label = quote_label[:, sl].float()
                    quote_weight = (
                        main_valid[:, sl].float()[..., None]
                        * (sequence_class(y) == 3)[:, None, None]
                        * torch.isfinite(label)
                        * y[:, sl].float().clamp(-2, 2).abs().mean(-1, keepdim=True)
                    )
                    quote_error = (out[..., quote_at:quote_at + 2].float() - torch.nan_to_num(label)) ** 2
                    loss = loss + args.aux_quote * (quote_error * quote_weight).sum() / quote_weight.sum().clamp_min(1e-8)
                for j, (ya, va) in enumerate(aux):
                    loss = loss + args.aux_weight * objective(out[..., 2 + 2 * j : 4 + 2 * j], ya[:, sl], va[:, sl])
                if args.ema_teacher > 0:
                    cw = main_valid[:, sl].float()[..., None]
                    consistency_scale = 1.0
                    if args.ema_metric_weight:
                        cw = cw * y[:, sl].float().clamp(-2, 2).abs()
                        consistency_scale = 2.0  # preserve the original two-target loss scale
                    loss = loss + args.ema_teacher * consistency_scale * (cw * (out[..., :2].float() - out_ema[..., :2].float()) ** 2).sum() / cw.sum().clamp_min(1e-8)
                timer.mark("loss")
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                timer.mark("backward")
                scaler.unscale_(opt)
                epoch_counter.observe_unscaled(scaler, opt)
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
                    epoch_counter.ema_updates += 1
                if duty is not None:  # explicit rate limit (WNN_GPU_DUTY < 1): sync, then sleep
                    torch.cuda.synchronize()
                    duty.tick()
                elif not args.sync_free:
                    torch.cuda.synchronize()
            timer.stop()
            epoch_counter.completed_batches += 1
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
        update_counts = epoch_counter.finish(cumulative_updates)
        update_counts["sample_order_identity"] = epoch_order_identity
        update_counts["normalization_identity"] = provenance["normalization_identity"]
        provenance["update_diagnostics"].append({"epoch_index": epoch, **update_counts})
        provenance["cumulative_updates"] = cumulative_updates.copy()
        train_seconds = round(time.time() - t0, 1)
        eval_model = checkpoint_model(model, ema_model,
                                      swa_model if step_count >= swa_start else None, args.eval_ema)
        if moments is not None:
            global READOUT, READOUT_TILTED
            READOUT, READOUT_TILTED = moments.tables(), args.ce_tilt
        evaluation_score = None
        if args.no_eval_during_fit:
            res = {"epoch": epoch, "train_seconds": train_seconds, "evaluation": "withheld"}
        else:
            t_eval = time.time()
            res = evaluate(eval_model, valid_loader, 4000, device)
            res.update(epoch=epoch, train_seconds=train_seconds, eval_seconds=round(time.time() - t_eval, 1),
                       split=f"{args.eval_source}{n_valid}")
            evaluation_score = res["weighted_pearson"]
        res["update_diagnostics"] = update_counts
        history.append(res)
        print("FIT" if args.no_eval_during_fit else "VALID", json.dumps(res), flush=True)
        # snapshot every epoch (~1 MB): lets any epoch be re-read out later without selection on the eval set
        checkpoint = {"model": eval_model.state_dict(), "args": vars(args), "n_in": n_in, "epoch": epoch,
                      "readout": None if READOUT is None else [t.cpu() for t in READOUT],
                      "readout_tilted": READOUT_TILTED,
                      "provenance": {key: value for key, value in provenance.items() if key != "checkpoint_sha256"}}
        snapshot = out_dir / f"model_ep{epoch}.pt"
        torch.save(checkpoint, snapshot)
        provenance["checkpoint_sha256"][snapshot.name] = file_sha256(snapshot)
        is_selected = checkpoint_selected(epoch, args.checkpoint_policy, args.checkpoint_epoch, evaluation_score, best)
        if evaluation_score is not None:
            best = max(best, evaluation_score)
        if is_selected or (args.checkpoint_policy == "best" and args.include_heldout_valid and epoch == args.epochs - 1):
            torch.save(checkpoint, out_dir / "model.pt")
            provenance["checkpoint_sha256"]["model.pt"] = file_sha256(out_dir / "model.pt")
            provenance["selected_epoch"] = epoch
        provenance["fitting_seconds"] = round(time.time() - fitting_started, 1)
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        t0 = time.time()
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    if args.checkpoint_policy == "fixed":
        print("SELECTED_FIXED", args.checkpoint_epoch)
    else:
        print("BEST", best)
    if args.include_heldout_valid and not args.no_eval_during_fit:
        print("SELECTED_FINAL", history[-1]["weighted_pearson"])


if __name__ == "__main__":
    main()
