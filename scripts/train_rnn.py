"""Train a stateful GRU on GPU with truncated BPTT; select on the validation dev split.

    scripts/limited .venv/bin/python scripts/train_rnn.py --name gru --epochs 2 --aux 20,200
The holdout split (wnn.splits) is read only with --eval-source holdout, for grading final fits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
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
    SEQUENCE_LENGTH,
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
    TCNModel,
    TwoClockModel,
    weighted_corr_loss,
    weighted_huber_loss,
    weighted_mse_loss,
    weighted_pooled_corr_loss,
)
from wnn.optim import GradStats, rnn_init_
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
from wnn.privileged import (PRIVILEGED_MODES, load_inner_manifest, privileged_inputs,
                            validate_crossfit_cache)
from wnn.structured_state import StructuredStateModel
from wnn.event_content import EventContentModel
from wnn.resources import configure_torch, cpu_threads, dataloader_budget
from wnn.research_features import (RESEARCH_FEATURE_MODES, research_feature_names,
                                   research_features_torch)
from wnn.run_identity import (EpochUpdateCounter, group_identity, normalization_identity,
                              sample_order_identity, scaler_source_identity)
from wnn.splits import dev_halves, file_sha256, load_campaign_split, valid_row_groups
from wnn.spread_aux import (PackedSpreadCache, structural_loss, validate_spread_cache,
                            validate_spread_parent)
from wnn.training_controls import (DistilledLoss, distilled_macro_losses,
                                   detach_boundary_offset, ema_decay_per_update, shuffled_masses)
from wnn.training_credit import credit_edge_scales
from wnn.training_dynamics import (Dynamics, add_arguments as add_dynamics_arguments,
                                   build_optimizer, build_scheduler, switch_optimizer)
from wnn.training_sampling import EpochOrderSampler, init_qmc_
from wnn.training_factorized import (configure_factorized, factorized_forward,
                                      factorized_inference_state)
from wnn.sentinel import (INDICATOR_MODES, VALUE_MODES, SentinelInputProjection,
                          SentinelPreprocessor)
from wnn.sentinel_calibration import validate_fit as validate_sentinel_fit
from wnn.training_qat import (QAT_MODES, configure_qat, inference_state,
                              invalidate_qat_cache, qat_forward)
from wnn.training_state import (atomic_save_training_state, capture_training_state,
                                load_training_state, loader_state, restore_rng,
                                restore_training_state)
from wnn.training_views import origin_windows, update_ema_for_rows

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
            if "crossfit_format" in side:
                validate_crossfit_cache(side, campaign_split, ROOT, split)
            bound.append({"run": name, "split": split, "scale": scale, "epoch": side.get("epoch"),
                          "checkpoint_sha256": side.get("checkpoint_sha256"), "prediction_sha256": digest})
    return bound


class Sequences(Dataset):
    def __init__(self, split: str, indices, teachers=None, prop_cache=None, quote_path=False,
                 quote_cache_root=None, spread_cache_root=None, thin=None):
        self.path = ROOT / "datasets" / f"{split}.parquet"
        # thin = (probability, low, high[, mode]): paired event-preserving training views.
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

    def batch_tensor_bytes(self, batch_size: int, *, heuristics: bool | None = None) -> int:
        """Size returned tensors from their schema without reading a sample or RNG.

        Cache headers supply auxiliary target widths; teacher predictions are
        averaged, so their column count is included once, not per teacher.
        Event/normalization features computed on device are not loader tensors.
        """
        if batch_size <= 0:
            raise ValueError("loader batch size must be positive")
        features = N_FEATURES
        use_heuristics = HEURISTICS if heuristics is None else heuristics
        if use_heuristics:
            from wnn.heuristics import K

            features += K
        targets = 2
        if self.teachers:
            widths = set()
            for name, _ in self.teachers:
                values = np.load(
                    ROOT / "cache" / f"pred_{name}_{self.split}.npy", mmap_mode="r"
                )
                if values.ndim != 3 or values.shape[1] != SEQUENCE_LENGTH:
                    raise ValueError("teacher cache does not contain full sequences")
                widths.add(values.shape[2])
            if len(widths) != 1:
                raise ValueError("teacher target widths differ")
            targets += widths.pop()
        elif self.quote_path:
            values = np.load(
                self.quote_cache_root / f"quote_path_{self.split}.npy", mmap_mode="r"
            )
            if values.ndim != 3 or values.shape[1] != SEQUENCE_LENGTH:
                raise ValueError("quote cache does not contain full sequences")
            targets += values.shape[2]
        row_bytes = (features + targets) * np.dtype(np.float32).itemsize
        row_bytes += np.dtype(np.bool_).itemsize
        if self.prop_cache is not None:
            row_bytes += np.dtype(np.float32).itemsize
        if self.spread_cache_root is not None:
            row_bytes += np.dtype(np.uint8).itemsize
        sequence_bytes = SEQUENCE_LENGTH * row_bytes
        if self.thin is not None:
            sequence_bytes += (SEQUENCE_LENGTH if len(self.thin) == 4 and self.thin[3] == "retained-mass"
                               else np.dtype(np.int64).itemsize)
        return batch_size * sequence_bytes

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
    The retained-mass control uses the identical random keep map but leaves all observations,
    targets and cached values on their original time axis, appending the supervised-row mask.
    """
    prob, low, high = thin[:3]
    mode = thin[3] if len(thin) == 4 else "compress"
    if mode not in ("compress", "retained-mass"):
        raise ValueError("unknown row-thinning mode")
    x = result[0]
    n = x.shape[0]
    if torch.rand(()) >= prob:
        return result + (torch.ones(n, dtype=torch.bool) if mode == "retained-mass" else torch.tensor(n),)
    q = low + (high - low) * torch.rand(())
    quiet = torch.zeros(n, dtype=torch.bool)
    quiet[WARMUP:] = (x[WARMUP:, :22] == x[WARMUP - 1:-1, :22]).all(dim=1)
    keep = ~(quiet & (torch.rand(n) < q))
    if mode == "retained-mass":
        return result + (keep,)
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
        # The first pinned iterator can lazily initialize CPU thread pools.
        # Reapply the main-thread budget after its pinning thread is active.
        torch.set_num_threads(cpu_threads())
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
RESEARCH_FEATURES = "none"  # causal full-sequence features, before TBPTT slicing
PRIVILEGED_MODE = "none"  # teacher-only current/future controls, never a deployed input
PRIVILEGED_HORIZON = 8
SENTINEL_PREPROCESSOR = None  # frozen, input-only raw-level intervention


def build_inputs(x):
    raw, extra = x[..., :N_FEATURES], x[..., N_FEATURES:]
    ev = event_features_torch(raw)
    if RAW_ONLY:
        ev = torch.zeros_like(ev)
    inputs = torch.cat([raw, ev], dim=-1)
    if ONLINE_NORM:
        inputs = torch.cat([inputs, online_zscore_torch(raw)], dim=-1)
    prop_inputs = inputs
    if RESEARCH_FEATURES != "none":
        inputs = torch.cat([inputs, research_features_torch(raw, RESEARCH_FEATURES)], dim=-1)
    if PRIVILEGED_MODE != "none":
        inputs = torch.cat([inputs, privileged_inputs(raw, PRIVILEGED_MODE, PRIVILEGED_HORIZON)], dim=-1)
    if extra.shape[-1]:
        inputs = torch.cat([inputs, extra], dim=-1)
        if INPUT_PROPENSITY is not None:
            prop_inputs = torch.cat([prop_inputs, extra], dim=-1)
    if INPUT_PROPENSITY is not None:
        inputs = torch.cat([inputs, propensity(INPUT_PROPENSITY, prop_inputs)[..., None].to(inputs.dtype)], dim=-1)
    if SENTINEL_PREPROCESSOR is not None:
        inputs = SENTINEL_PREPROCESSOR.torch(inputs)
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


def main_row_weights(inputs, y, valid, scored, prop_model, prop_cached, args, raw=None):
    """Shared fit/calibration weights, including the batch-level true-mask scale."""
    main_valid = valid & scored if args.mask_weight else valid
    if prop_model is not None:
        if prop_cached is not None:
            prob = prop_cached
        elif args.prop_cache:
            raise RuntimeError("--prop-cache set but the batch carries no cached propensity")
        else:
            prop_inputs = inputs[..., : N_FEATURES + N_EVENT_FEATURES]
            if SENTINEL_PREPROCESSOR is not None:
                if raw is None:
                    raise ValueError("sentinel weighting requires original raw propensity inputs")
                prop_inputs = torch.cat((raw[..., :N_FEATURES], prop_inputs[..., N_FEATURES:]), -1)
            prob = propensity(prop_model, prop_inputs)
        main_valid = valid.float() * (prob ** args.prop_power + args.prop_eps)
        if args.class_weight:
            cw = torch.tensor([float(v) for v in args.class_weight.split(",")], device=y.device)
            main_valid = main_valid * cw[sequence_class(y)][:, None]
        if args.valid_true_mask >= 0:
            main_valid = true_mask_weights(main_valid, valid, scored, args.valid_true_mask, args.sync_free)
    if args.since_weight > 0:
        since = torch.expm1(inputs[..., N_FEATURES + FEATURE_NAMES.index("since_i0px")].float())
        sw_rows = torch.clamp(1.0 / (1.0 + since / args.since_weight), min=args.since_floor)
        main_valid = main_valid.float() * sw_rows
    if args.episode_weight > 0:
        main_valid = main_valid.float() * episode_redundancy_weight(
            y, args.episode_weight, args.episode_floor)
    return main_valid


@torch.no_grad()
def calibrate_distill_reference(args, train_loader, prop_model, provenance, device, total):
    """Measure MAIN denominator on actual fit batches; do not fit or inspect eval rows."""
    from wnn.training_calibration import RecordedOrder, summarize_masses

    update_rows = args.update_rows or args.chunk
    masses, orders = [], []
    order = RecordedOrder(train_loader.sampler)
    original_sampler = train_loader.batch_sampler.sampler
    train_loader.batch_sampler.sampler = order
    for epoch in range(args.calibration_epochs):
        if isinstance(train_loader.sampler, EpochOrderSampler):
            train_loader.sampler.set_epoch(epoch)
        orders.append(sample_order_identity(train_loader))
        for batch, (x, target, scored, *rest) in enumerate(train_loader):
            x, target, scored = x.to(device), target.to(device), scored.to(device)
            y = target[..., :2]
            valid = torch.arange(x.shape[1], device=device)[None].expand(x.shape[:2]) >= WARMUP
            cached = rest[0].to(device) if rest else None
            weights = main_row_weights(build_inputs(x), y, valid, scored, prop_model, cached, args, x)
            mass = y.float().clamp(-2, 2).abs() * weights[..., None]
            windows = origin_windows(x.shape[1], update_rows, args.chunk, args.origin_view,
                                     args.origin_seed, epoch, batch)
            masses.extend(float(mass[:, window].sum()) for window in windows)
        orders[-1].update(order.identity(args.batch))
    train_loader.batch_sampler.sampler = original_sampler
    report = summarize_masses(masses)
    report.update({"format": "wnn.distill-reference-mass", "version": 1,
                   "estimator": "mean over actual sampled fit batches and update windows",
                   "calibration_epochs": args.calibration_epochs,
                   "sample_order": orders,
                   "contract": resume_contract(args, provenance, train_loader, total)})
    path = Path(args.calibrate_distill_reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"reference_mass": report["reference_mass"],
                      "calibration": str(path), "updates": report["updates"]}), flush=True)


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
        if b == 0:
            torch.set_num_threads(cpu_threads())
        if b >= batches:
            break
        f = build_inputs(batch[0].to(device))
        feats.append(f.reshape(-1, f.shape[-1])[::7])
    f = torch.cat(feats)
    model.mean.copy_(f.mean(0))
    model.std.copy_(f.std(0).clamp_min(1e-3))


def forward_chunk(model, inputs, sl, state):
    if getattr(model, "_recurrent_rank", 0):
        return factorized_forward(model, inputs[:, sl], state)
    if getattr(model, "_qat_preset", "none") != "none":
        return qat_forward(model, inputs[:, sl], state)
    if isinstance(model, DualClockModel):
        return model(inputs[:, sl], inputs[:, sl, EV_I0PX] > 0.5, state)
    return model(inputs[:, sl], state)


def inference_checkpoint_fields(model: torch.nn.Module, source: str) -> dict:
    """Export effective weights and representation metadata without changing raw resume state."""
    if getattr(model, "_recurrent_rank", 0):
        exported, metadata = factorized_inference_state(model, source)
        return {"model": exported, "factorization": metadata}
    exported, metadata = inference_state(model, source)
    result = {"model": exported, **({"quantization": metadata} if metadata is not None else {})}
    if hasattr(model, "_sentinel_preprocessing"):
        result["sentinel_preprocessing"] = model._sentinel_preprocessing
    return result


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
    if args_dict.get("arch") == "tcn":
        if args_dict["dropout"] != 0 or args_dict.get("enc", "linear") != "linear":
            raise ValueError("tcn requires dropout0, enc linear")
        model = TCNModel(n_in, args_dict["hidden"], args_dict["layers"], n_out=n_out,
                         mix=bool(args_dict.get("tcn_mix", 0)))
        for prefix in filter(None, args_dict.get("freeze", "").split(",")):
            if not any(name.startswith(prefix) for name, _ in model.named_parameters()):
                raise ValueError(f"unknown tcn freeze prefix {prefix!r}; trunk is inp,tcn")
        return model
    if args_dict.get("arch") == "twoclock":
        if args_dict["dropout"] != 0 or args_dict.get("enc", "linear") != "linear":
            raise ValueError("twoclock requires dropout0, enc linear")
        model = TwoClockModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["event_hidden"], n_out=n_out)
        frozen = tuple(filter(None, args_dict.get("freeze", "").split(",")))
        loose = [name for name, _ in model.named_parameters()
                 if frozen and not name.startswith(frozen) and not name.startswith("head.")]
        if loose:  # head-only FT must not leave part of the trunk trainable (the dual-clock event GRU confound)
            raise ValueError(f"twoclock --freeze must cover the whole trunk (inp,gru); trainable: {loose}")
        return model
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
    model = RNNModel(
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
    if args_dict.get("sentinel_indicators", "none") != "none":
        model.inp = SentinelInputProjection(model.inp, args_dict["sentinel_base_width"])
    return model


@torch.no_grad()
def predict(model, x, chunk, device):
    model.eval()
    inputs = build_inputs(x.to(device, non_blocking=True))
    state, outs = None, []
    for s in range(0, inputs.shape[1], chunk):
        with torch.autocast(device.type, dtype=torch.float16,
                            enabled=device.type == "cuda" and not getattr(model, "_inference_fp32", False)):
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
        "tcn_mix",
        "event_hidden",
        "online_norm",
        "raw_only",
        "heuristics",
        "research_features",
        "privileged_mode",
        "privileged_horizon",
        "sentinel_values",
        "sentinel_indicators",
        "sentinel_base_width",
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


def resume_contract(args: argparse.Namespace, provenance: dict, loader: DataLoader,
                    total_steps: int) -> dict:
    """Bind an epoch continuation to its original data, code and fixed schedule."""
    excluded = {"name", "resume_state", "save_training_state", "stop_after_epoch", "split_manifest",
                "fit_groups_manifest"}
    options = {key: value for key, value in vars(args).items() if key not in excluded}
    package = Path(__file__).resolve().parents[1] / "src/wnn"
    sources = {str(path.relative_to(package)): file_sha256(path) for path in sorted(package.rglob("*.py"))}
    sources["trainer"] = file_sha256(Path(__file__))
    data = {}
    used_splits = {split for split, groups in provenance["fit_row_groups"].items() if groups}
    for split in used_splits | {"valid"}:
        path = ROOT / "datasets" / f"{split}.parquet"
        registry_path = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
        if registry_path:
            from wnn.portability import verify_portable_source

            registry = Path(registry_path)
            logical_name = f"datasets/{split}.parquet"
            entry = json.loads(registry.read_text())["sources"][logical_name]
            verify_portable_source(path, logical_name, entry["original_identity"], registry)
            data[split] = {"bytes": path.stat().st_size, "sha256": entry["sha256"]}
        else:
            data[split] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    teachers = {}
    for teacher in filter(None, args.distill.split(",")):
        name = teacher.partition("*")[0]
        for split, groups in provenance["fit_row_groups"].items():
            if groups:
                filename = f"pred_{name}_{split}.npy"
                teachers[filename] = file_sha256(ROOT / "cache" / filename)
    caches = {}
    for split in used_splits:
        if args.distill:
            filename = f"{split}_targets.npz"
            caches[filename] = file_sha256(ROOT / "cache" / filename)
        if args.propensity and args.prop_cache:
            path = prop_cache_path(args.propensity, split)
            caches[path.name] = file_sha256(path)
            caches[path.with_suffix(".json").name] = file_sha256(path.with_suffix(".json"))
    if args.prop_input:
        caches["input_propensity"] = file_sha256(ROOT / "artifacts/runs" / args.prop_input / "model.pt")
    sampling = loader_state(loader)
    for key in ("loader_generator", "sampler_generator", "sampler_state", "sampler_epoch"):
        sampling.pop(key)
    return {
        "epochs": args.epochs, "total_steps": total_steps, "options": options,
        "sources": sources, "datasets": data, "teacher_predictions": teachers, "consumed_caches": caches,
        "fit_membership": provenance["fit_membership_identity"],
        "normalization": provenance["normalization_identity"],
        "ancestors": {key: provenance.get(key) for key in
                      ("initial_checkpoint_sha256", "propensity_checkpoint_sha256", "quote_cache", "spread_aux")},
        "loader": sampling,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
                    "python": platform.python_version(), "machine": platform.machine(),
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "devices": [] if args.device == "cpu" else
                    [{"name": torch.cuda.get_device_name(index),
                      "capability": list(torch.cuda.get_device_capability(index))}
                     for index in range(torch.cuda.device_count())],
                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "threads": torch.get_num_threads(),
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="CPU is available for bounded implementation checks; production default remains CUDA")
    ap.add_argument("--train-seqs", type=int, default=10607)
    ap.add_argument("--valid-seqs", type=int, default=600, help="dev sequences used for selection")
    ap.add_argument("--full-valid", action="store_true", help="all 1505 dev sequences")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--update-rows", type=int, default=0,
                    help="opt-in distilled macrostep rows; 0 preserves one update per --chunk")
    ap.add_argument("--detach-rows", type=int, default=0,
                    help="opt-in gradient detach interval; 0 uses --chunk, with state values carried")
    ap.add_argument("--detach-boundary-jitter", action="store_true",
                    help="randomize detach phase inside unchanged macro updates; requires chunk < detach rows")
    ap.add_argument("--detach-jitter-seed", type=int, default=271828)
    ap.add_argument("--origin-view", choices=["off", "prefix-fixed", "prefix-random", "prefix-discard"], default="off",
                    help="replay causal prefixes before updates; discard matches fixed replay work while retaining carried state")
    ap.add_argument("--origin-seed", type=int, default=79079)
    ap.add_argument("--credit-mode", choices=["deterministic", "random-uncompensated", "artbp"],
                    default="deterministic", help="random recurrent credit within fixed update windows only")
    ap.add_argument("--credit-keep-probability", type=float, default=.8)
    ap.add_argument("--credit-seed", type=int, default=100100)
    ap.add_argument("--qat", choices=QAT_MODES, default="none",
                    help="weight-only nearest-even q8 with identity STE; paired float control must use the same fp32 recipe")
    ap.add_argument("--recurrent-rank", type=int, default=0,
                    help="SVD-initialized rank per stacked GRU matrix; 0 preserves dense recurrence")
    ap.add_argument("--sentinel-values", choices=VALUE_MODES, default="unchanged")
    ap.add_argument("--sentinel-indicators", choices=INDICATOR_MODES, default="none")
    ap.add_argument("--sentinel-base-width", type=int, default=0,
                    help="inherited only from a frozen sentinel_prepare initializer")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--stop-after-epoch", type=int, default=None,
                    help="end training after this epoch while the lr schedule still spans --epochs (cheap screens)")
    ap.add_argument("--checkpoint-policy", choices=["best", "fixed"], default="best")
    ap.add_argument("--checkpoint-epoch", type=int, default=None, help="zero-based fixed checkpoint epoch")
    ap.add_argument("--no-eval-during-fit", action="store_true", help="never construct or read an evaluation loader")
    ap.add_argument("--split-manifest", type=Path, default=None, help="frozen component-excluded outer-fold manifest")
    ap.add_argument("--save-training-state", action="store_true",
                    help="atomically save full training_state.pt at each epoch boundary; fixed final checkpoint only")
    ap.add_argument("--resume-state", type=Path, default=None,
                    help="resume full epoch training state with the identical recipe and schedule horizon")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    research = ap.add_argument_group("optimizer research (src/wnn/optim.py; defaults reproduce AdamW + OneCycle)")
    research.add_argument("--optim", choices=["adamw", "muon", "sgd"], default="adamw",
                          help="muon: orthogonalised per-gate updates on the recurrent matrices, AdamW elsewhere")
    research.add_argument("--muon-scope", choices=["rnn", "hidden"], default="rnn",
                          help="hidden: Muon also on head.0.weight")
    research.add_argument("--beta2", type=float, default=0.999, help="Adam second-moment decay")
    research.add_argument("--sched", choices=["onecycle", "wsd", "sgdr"], default="onecycle",
                          help="wsd: 5%% linear warmup, constant lr, (1 - sqrt) cooldown over the last --wsd-decay")
    research.add_argument("--wsd-decay", type=float, default=0.2, help="cooldown share of all optimizer steps")
    research.add_argument("--wsd-final", type=float, default=0.01, help="final lr as a fraction of --lr")
    research.add_argument("--rnn-init", choices=["default", "chrono", "orth", "chrono_orth"], default="default",
                          help="chrono gate biases (memory timescales U[2, --chrono-tmax] rows) and/or orthogonal W_hh")
    research.add_argument("--chrono-tmax", type=float, default=2000.0)
    research.add_argument("--clip", type=float, default=1.0, help="global gradient-norm clip")
    research.add_argument("--grad-stats", action="store_true",
                          help="per-epoch pre-clip gradient-norm quantiles, clip rate and family shares")
    ap.add_argument("--loss", choices=["corr", "wmse", "mix", "huber", "ce_joint", "mse", "gcorr", "wmse_gcorr", "gcorr_pool"],
                    default="mix")
    ap.add_argument("--thin-prob", type=float, default=0.0,
                    help="training-view row thinning: probability per sequence (0 disables)")
    ap.add_argument("--thin-range", default="0.1,0.4", help="low,high of the dropped share of quiet rows")
    ap.add_argument("--thin-mode", choices=["compress", "retained-mass"], default="compress",
                    help="retained-mass preserves the observed stream and masks the same sampled quiet rows only in losses")
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
    ap.add_argument("--distill-boot-h", type=int, default=0,
                    help="stopped bootstrap: in the distillation blend replace y by the teacher at row t+h while the label "
                         "run outlives t+h, else keep y (training-time only; min(t+h, run end) is a stopping time)")
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
    ap.add_argument("--spread-cache-root", default="", help="supervision-only cache bound to the clean fitting population")
    ap.add_argument("--aux-struct", type=float, default=0.0,
                    help="weight of auxiliary cross-entropy heads for spread (ticks), branch and class")
    ap.add_argument("--aux", default="", help="future row shifts as auxiliary targets, e.g. 20,200")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--arch", choices=["gru", "dual", "gated", "structured", "event-content", "tcn", "twoclock"],
                    default="gru",
                    help="tcn: causal dilated conv stack, --hidden channels, --layers blocks (receptive field 2^layers); "
                         "twoclock: --layers x GRU(--hidden) on rows + GRU(--event-hidden) stepping on a6 endpoint rows")
    ap.add_argument("--tcn-mix", type=int, default=0, help="--arch tcn: GELU + 1x1 conv after each block's conv")
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
    ap.add_argument("--freeze", default="",
                    help="comma-separated parameter-name prefixes kept fixed, e.g. inp,gru (inp,tcn for --arch tcn)")
    ap.add_argument("--ema-teacher", type=float, default=0.0,
                    help="mean-teacher consistency: weight of (out - EMA-model out)^2 on the two target outputs")
    ap.add_argument("--ema-metric-weight", action="store_true",
                    help="weight teacher consistency by |clipped target|, matching the scored loss")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--ema-reference-rows", type=int, default=0,
                    help="0 keeps per-attempt decay; otherwise decay**(update_rows/reference_rows)")
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
    ap.add_argument("--research-features", choices=RESEARCH_FEATURE_MODES, default="none",
                    help="append causal volume features computed on full float32 sequences")
    ap.add_argument("--privileged-mode", choices=PRIVILEGED_MODES, default="none",
                    help="teacher-only equal-width current/future raw inputs; requires inner component crossfit")
    ap.add_argument("--privileged-horizon", type=int, default=8)
    ap.add_argument("--fit-groups-manifest", type=Path,
                    help="strict inner component split within --split-manifest, for fresh teacher fitting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss-norm", choices=["chunk", "running"], default="chunk",
                    help="wmse main loss divided by each chunk's weight sum (chunk) or by a running mean of it "
                         "(running: rows keep the pooled metric's relative weights across chunks; brainteaser card B)")
    ap.add_argument("--distill-denominator", choices=["chunk", "fixed", "shuffled"], default="chunk",
                    help="cached-distill MAIN loss: update mass, frozen reference mass, or shuffled mass-factor sham")
    ap.add_argument("--distill-reference-mass", type=float, default=0.0,
                    help="positive frozen fit-pool main mass for fixed/shuffled controls")
    ap.add_argument("--calibrate-distill-reference", default="",
                    help="write a fit-pool mass report and exit before optimizer construction")
    ap.add_argument("--calibration-epochs", type=int, default=1,
                    help="complete training-loader passes used only for reference calibration")
    ap.add_argument("--denominator-shuffle-seed", type=int, default=314159)
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
    perf.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16",
                      help="autocast dtype; bf16 trains without GradScaler")
    perf.add_argument("--max-batches", type=int, default=0, help="stop each epoch after this many batches (harness)")
    perf.add_argument("--profile-steps", type=int, default=0,
                      help="time this many batches per stage (after --profile-warmup), print a table and exit")
    perf.add_argument("--profile-warmup", type=int, default=5)
    perf.add_argument("--parity-dump", default="", help="save per-step losses and final weights here, then exit")
    perf.add_argument("--deterministic", action="store_true", help="cudnn deterministic, no benchmark (parity runs)")
    add_dynamics_arguments(ap)
    args = ap.parse_args()
    if args.recurrent_rank < 0:
        ap.error("--recurrent-rank cannot be negative")
    if args.qat != "none" and (args.precision != "fp32" or args.arch != "gru"
                               or args.cell != "gru" or args.enc != "linear"):
        ap.error("QAT requires explicit --precision fp32 with plain GRU and linear input encoder")
    if args.calibrate_distill_reference:
        if (args.device != "cpu" or args.resume_state or args.thin_prob or args.spread_aux_mode != "original"
                or args.calibration_epochs < 1 or args.profile_steps or args.max_batches
                or args.distill_denominator != "chunk" or args.loss != "wmse"):
            ap.error("reference calibration requires CPU wmse, chunk normalization, complete unthinned fit batches")
    if args.zoneout != 0:
        ap.error("exact timestep zoneout is unsupported for fused nn.GRU")
    if args.batch_order == "qmc":
        ap.error("QMC sampling requires a bound input-only strata cache; trainer integration is not implemented")
    if args.antithetic and args.batch_order != "seeded":
        ap.error("--antithetic requires --batch-order seeded and a same-seed control")
    if args.init_qmc and (args.init or args.rnn_init != "default" or args.enc != "linear" or args.arch != "gru"):
        ap.error("--init-qmc supports a fresh linear-encoder GRU/LSTM only")
    init_checkpoint = None
    if args.inherit_init_preprocessing:
        if not args.init:
            ap.error("--inherit-init-preprocessing requires --init")
        parent_run, _, parent_epoch = args.init.partition("@ep")
        parent_file = f"model_ep{int(parent_epoch)}.pt" if parent_epoch else "model.pt"
        init_checkpoint = torch.load(ROOT / "artifacts" / "runs" / parent_run / parent_file,
                                     map_location="cpu", weights_only=False)
        inherit_init_arguments(args, init_checkpoint, sys.argv[1:])
    if args.recurrent_rank and (args.precision != "fp32" or args.optim != "adamw"
            or args.optim_switch or args.qat != "none" or not args.init
            or args.arch != "gru" or args.cell != "gru" or args.enc != "linear"
            or args.recurrent_rank > args.hidden):
        ap.error("factorized recurrence needs dense --init, plain GRU, linear encoder, rank <= hidden, "
                 "AdamW, explicit FP32, no QAT or optimizer switch")
    if (args.sentinel_base_width or args.sentinel_values != "unchanged"
            or args.sentinel_indicators != "none"):
        if (not args.init or not args.inherit_init_preprocessing or args.sentinel_base_width < N_FEATURES
                or args.arch != "gru" or args.cell != "gru" or args.enc != "linear"
                or args.qat != "none" or args.recurrent_rank):
            ap.error("sentinel arms require a prepared plain dense GRU --init with --inherit-init-preprocessing")
        if args.sentinel_indicators != "none" and any("inp".startswith(prefix)
                for prefix in filter(None, args.freeze.split(","))):
            ap.error("sentinel indicator arms require trainable input-projection columns")
    configure_checkpoint_policy(args, ap)
    continuation = bool(args.save_training_state or args.resume_state)
    if continuation and (args.checkpoint_policy != "fixed" or args.checkpoint_epoch != args.epochs - 1):
        ap.error("training-state continuation requires a fixed last-epoch checkpoint")
    if continuation and (args.profile_steps or args.parity_dump):
        ap.error("profiling/parity early exits cannot save epoch continuation state")
    saved_training = load_training_state(args.resume_state) if args.resume_state else None
    if args.resume_state:
        args.resume_state = str(args.resume_state)
    campaign_split = load_campaign_split(args.split_manifest, ROOT) if args.split_manifest else None
    inner_crossfit = None
    if args.fit_groups_manifest is not None:
        if (campaign_split is None or args.train_source != "train+valid"
                or args.init or args.distill or args.propensity or args.prop_input
                or args.train_seqs != 10607 or args.valid_train_fraction != 1 or args.train_classes
                or args.privileged_mode == "none"):
            ap.error("inner crossfit requires a fresh train+valid teacher, no learned external ancestors, and full inner membership")
        inner_crossfit = load_inner_manifest(args.fit_groups_manifest, campaign_split)
        args.fit_groups_manifest_sha256 = file_sha256(args.fit_groups_manifest)
        args.fit_groups_manifest = str(args.fit_groups_manifest)
    elif args.privileged_mode != "none":
        ap.error("teacher-only current/future inputs require --fit-groups-manifest")
    if args.privileged_horizon < 1:
        ap.error("--privileged-horizon must be positive")
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
    if args.distill and args.loss_norm != "chunk":
        ap.error("--loss-norm running does not apply to cached distillation; use --distill-denominator")
    if args.distill_boot_h and not args.distill:
        ap.error("--distill-boot-h bootstraps from the distillation teacher; pass --distill")
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
    update_rows = args.update_rows or args.chunk
    detach_rows = args.detach_rows or args.chunk
    if not 0 <= args.thin_prob <= 1:
        ap.error("--thin-prob must lie in [0, 1]")
    if args.thin_prob:
        try:
            thin_low, thin_high = map(float, args.thin_range.split(","))
        except ValueError:
            ap.error("--thin-range requires two drop shares low,high")
        if not 0 <= thin_low <= thin_high <= 1:
            ap.error("--thin-range drop shares must satisfy 0 <= low <= high <= 1")
    if args.thin_mode == "retained-mass" and (
            not args.thin_prob or not args.init or not args.inherit_init_preprocessing):
        ap.error("retained-mass control requires positive thinning and inherited parent preprocessing")
    if args.credit_mode != "deterministic":
        if (args.optim != "sgd" or args.precision != "fp32" or not args.no_gradient_clip
                or args.grad_noise or args.optim_switch or args.detach_boundary_jitter
                or args.origin_view != "off" or args.chunk >= update_rows
                or args.detach_rows not in (0, update_rows)):
            ap.error("random credit requires unclipped fp32 SGD, chunk < update rows, full-window detach bound, and no other credit/noise/switch arms")
        try:
            credit_edge_scales(0, args.credit_keep_probability, args.credit_seed, 0, True)
        except ValueError as error:
            ap.error(str(error))
        detach_rows = update_rows
    controlled_loss = bool(args.update_rows or args.detach_rows or args.distill_denominator != "chunk"
                           or args.detach_boundary_jitter or args.origin_view != "off"
                           or args.credit_mode != "deterministic")
    if args.origin_view != "off":
        try:
            origin_windows(SEQUENCE_LENGTH, update_rows, args.chunk, args.origin_view, args.origin_seed, 0, 0)
        except ValueError as error:
            ap.error(str(error))
    if args.detach_boundary_jitter and (detach_rows <= args.chunk or args.detach_jitter_seed < 0):
        ap.error("detach-boundary jitter needs --detach-rows larger than --chunk and a nonnegative seed")
    if controlled_loss:
        if (args.chunk <= 0 or update_rows <= 0 or detach_rows <= 0
                or detach_rows % args.chunk or update_rows % detach_rows
                or SEQUENCE_LENGTH % update_rows):
            ap.error("controlled micro/detach/update rows must divide each other and the sequence length")
        if (not args.distill or args.loss != "wmse" or args.arch != "gru" or args.enc != "linear" or args.dropout
                or args.distill_alpha_far is not None or args.td_alpha
                or args.aux or args.aux_next_k or args.aux_tier or args.aux_mask
                or args.aux_sign or args.aux_quote or args.thin_prob
                or args.spread_aux_mode != "original" or args.loss_norm != "chunk"):
            ap.error("controlled macro losses support the cached-distilled GRU wmse/EMA/struct/next recipe only")
        if args.distill_denominator != "chunk" and (
                args.distill_reference_mass <= 0 or not math.isfinite(args.distill_reference_mass)):
            ap.error("fixed/shuffled distilled normalization needs a positive finite frozen reference mass")
    if args.ema_reference_rows and SEQUENCE_LENGTH % update_rows:
        ap.error("EMA row clock requires update rows dividing sequence length")
    ema_update_decay = ema_decay_per_update(args.ema_decay, update_rows, args.ema_reference_rows)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cpu" and (args.pin_memory or args.fused_adam):
        ap.error("CPU implementation checks require unfused, synchronous loaders (omit --fast)")
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
    if inner_crossfit is not None:
        extra_valid = valid_pool = np.asarray(inner_crossfit["fit"]["valid"], dtype=np.int64)
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
        if inner_crossfit is not None:
            usable = np.asarray(inner_crossfit["fit"]["train"], dtype=np.int64)
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
        else:
            for name, _ in teachers:
                for split in ("train", "valid"):
                    sidecar = ROOT / "cache" / f"pred_{name}_{split}.json"
                    if sidecar.exists() and "crossfit_format" in json.loads(sidecar.read_text()):
                        ap.error("component-crossfit teachers require their matching --split-manifest")
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
        if (campaign_split is None or args.aux_struct <= 0
                or not args.spread_cache_root or args.train_source != "train+valid"
                or not args.init or not args.inherit_init_preprocessing
                or args.checkpoint_policy != "fixed" or args.checkpoint_epoch != args.epochs - 1
                or not args.no_eval_during_fit):
            raise ValueError("known-code spread screen requires a fixed final-epoch clean continuation with inherited parent preprocessing")
        spread_root = Path(args.spread_cache_root)
        if not spread_root.is_absolute():
            spread_root = ROOT / spread_root
        spread_root = spread_root.resolve()
        spread_provenance = validate_spread_cache(spread_root, campaign_split.sha256, train_idx,
                                                   extra_valid, campaign_split.evaluation_groups,
                                                   sequence_rows=SEQUENCE_LENGTH)
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
        thin = (args.thin_prob, low, high, args.thin_mode)
    train_ds = Sequences(train_split, train_idx, teachers, prop_cache.get(train_split),
                         quote_path=args.aux_quote > 0, quote_cache_root=args.quote_cache_root,
                         spread_cache_root=args.spread_cache_root, thin=thin)
    train_batch_bytes = train_ds.batch_tensor_bytes(args.batch, heuristics=args.heuristics)
    if extra_valid is not None:
        valid_ds = Sequences("valid", extra_valid, teachers, prop_cache.get("valid"),
                             quote_path=args.aux_quote > 0, quote_cache_root=args.quote_cache_root,
                             spread_cache_root=args.spread_cache_root, thin=thin)
        train_batch_bytes = max(
            train_batch_bytes, valid_ds.batch_tensor_bytes(args.batch, heuristics=args.heuristics)
        )
        train_ds = torch.utils.data.ConcatDataset([train_ds] + [valid_ds] * args.valid_repeat)
    if args.epoch_samples:
        sample_weights = np.ones(len(train_ds), dtype=np.float64)
        if extra_valid is not None:
            sample_weights[len(train_idx):] = eligible_valid_count / len(extra_valid)
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), args.epoch_samples,
                                        replacement=True, generator=torch.Generator().manual_seed(args.seed))
    valid_ds = None if args.no_eval_during_fit else Sequences("valid", valid_idx)
    valid_batch_bytes = (
        0 if valid_ds is None else valid_ds.batch_tensor_bytes(16, heuristics=args.heuristics)
    )
    loader_ram = int(float(os.environ.get("WNN_LOADER_RAM_GB", "6")) * (1 << 30))
    # Validation is synchronous; reserve its consumed/in-flight copies even if
    # training's iterator still holds batches when evaluation begins.
    workers, prefetch = dataloader_budget(
        train_batch_bytes, ram_budget_bytes=loader_ram - 2 * valid_batch_bytes
    )
    if args.batch_order == "seeded":
        if sampler is not None:
            ap.error("seeded/antithetic order cannot replace weighted or replacement sampling")
        sampler = EpochOrderSampler(len(train_ds), args.seed, args.antithetic,
                                    batch_size=args.batch, drop_last=True)
        sampler.set_epoch(-1)  # identical normalization order in both paired arms
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=sampler is None,
                              sampler=sampler, drop_last=True, num_workers=workers,
                              prefetch_factor=prefetch if workers else None,
                              pin_memory=args.pin_memory)  # no persistent workers: they change the shuffle order (parity E-169)
    valid_loader = None if valid_ds is None else DataLoader(
        valid_ds, batch_size=16, num_workers=0, prefetch_factor=None)

    shifts = [int(v) for v in args.aux.split(",") if v]
    provenance = {"seed": args.seed, "args": vars(args).copy(),
                  "split_manifest_sha256": campaign_split.sha256 if campaign_split is not None else None,
                  "fit_row_groups": {"train": train_idx.tolist() if train_split == "train" else [],
                                     "valid": train_idx.tolist() if train_split == "valid" else
                                     extra_valid.tolist() if extra_valid is not None else []},
                  "evaluation_row_groups": valid_idx.tolist(), "checkpoint_sha256": {}}
    if inner_crossfit is not None:
        provenance["inner_crossfit"] = inner_crossfit
        provenance["deployment_safe"] = False
    if args.arch == "structured":
        provenance["model_source_sha256"] = file_sha256(ROOT / "src/wnn/structured_state.py")
    if spread_provenance is not None:
        provenance["spread_aux"] = spread_provenance
    if args.arch == "event-content":
        provenance["model_source_sha256"] = file_sha256(ROOT / "src/wnn/event_content.py")
    if args.arch in ("tcn", "twoclock"):
        provenance["model_source_sha256"] = file_sha256(ROOT / "src/wnn/nn.py")
    if quote_cache_provenance is not None:
        provenance["quote_cache"] = quote_cache_provenance
    if args.propensity:
        propensity_path = ROOT / "artifacts" / "runs" / args.propensity / "model.pt"
        if campaign_split is not None:
            validate_campaign_checkpoint(torch.load(propensity_path, map_location="cpu", weights_only=False), args)
        provenance["propensity_checkpoint_sha256"] = file_sha256(propensity_path)
    prop_model = load_propensity(ROOT / "artifacts" / "runs" / args.propensity / "model.pt", device) if args.propensity else None
    global RESEARCH_FEATURES
    RESEARCH_FEATURES = args.research_features
    global PRIVILEGED_MODE, PRIVILEGED_HORIZON
    PRIVILEGED_MODE, PRIVILEGED_HORIZON = args.privileged_mode, args.privileged_horizon
    global SENTINEL_PREPROCESSOR
    SENTINEL_PREPROCESSOR = None
    n_in = (N_FEATURES + N_EVENT_FEATURES + len(research_feature_names(RESEARCH_FEATURES))
            + (N_FEATURES if PRIVILEGED_MODE != "none" else 0))
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
    if args.sentinel_base_width:
        if n_in != args.sentinel_base_width:
            ap.error("sentinel parent base width differs from inherited preprocessing")
        if args.sentinel_indicators != "none":
            n_in += N_FEATURES
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
    if args.init_qmc and saved_training is None:
        init_qmc_(model, args.seed)
    if args.rnn_init != "default" and not args.init:
        rnn_init_(model, args.rnn_init, args.chrono_tmax, args.seed)
    moments = ReadoutMoments(device) if args.loss == "ce_joint" else None
    if args.init:
        init_run, _, init_epoch = args.init.partition("@ep")
        init_file = f"model_ep{int(init_epoch)}.pt" if init_epoch else "model.pt"
        init_path = ROOT / "artifacts" / "runs" / init_run / init_file
        init = init_checkpoint if init_checkpoint is not None else torch.load(init_path, map_location="cpu", weights_only=False)
        if args.recurrent_rank and (init.get("quantization") or init.get("factorization")):
            ap.error("factorized initialization must be a dense float checkpoint; use raw state for exact resume")
        if args.qat != "none" and init.get("factorization"):
            ap.error("combined QAT and factorized initialization is not implemented")
        validate_campaign_checkpoint(init, args)
        if args.spread_aux_mode != "original":
            validate_spread_parent(init, campaign_split.sha256, provenance["fit_row_groups"])
        provenance["initial_checkpoint_sha256"] = file_sha256(init_path)
        model.load_state_dict(init["model"])
        sentinel_metadata = init.get("sentinel_preprocessing")
        if bool(sentinel_metadata) != bool(args.sentinel_base_width):
            ap.error("sentinel metadata requires matching inherited preprocessing options")
        if sentinel_metadata:
            if (sentinel_metadata["value_mode"] != args.sentinel_values
                    or sentinel_metadata["indicators"] != args.sentinel_indicators
                    or sentinel_metadata["base_width"] != args.sentinel_base_width):
                ap.error("sentinel initializer modes differ from training options")
            SENTINEL_PREPROCESSOR = SentinelPreprocessor(sentinel_metadata)
            validate_sentinel_fit(sentinel_metadata["calibration"], ROOT,
                                  provenance["fit_row_groups"], provenance["split_manifest_sha256"])
            model._sentinel_preprocessing = sentinel_metadata
            provenance["sentinel_preprocessing"] = sentinel_metadata
    elif saved_training is None:
        fit_normalization(model, train_loader, device)
    configure_factorized(model, args.recurrent_rank)
    if saved_training is not None:
        model.load_state_dict(saved_training["model"])
    if args.precision == "fp32":
        model._inference_fp32 = True
    configure_qat(model, args.qat)
    if args.qat != "none":
        provenance["qat"] = {"preset": args.qat, "scheme": "symmetric_per_row_int8_nearest_even_v1",
                             "gradient": "identity STE; scales detached; no scale/round/clamp derivative",
                             "averaging": "latent float parameters, quantized for each effective forward/export",
                             "training_state": "raw latent float weights; derived quantization cache not serialized"}
    if args.recurrent_rank:
        provenance["factorization"] = {
            "rank": args.recurrent_rank, "initialization": model._factorization_initialization,
            "averaging": "latent factors; products formed after averaging",
            "training_compute": "materialized dense FP32 GRU; native runtime not validated"}
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
    total = len(train_loader) * math.ceil(SEQUENCE_LENGTH / update_rows) * args.epochs
    if args.calibrate_distill_reference:
        calibrate_distill_reference(args, train_loader, prop_model, provenance, device, total)
        return
    opt = build_optimizer(model, args)
    sched = build_scheduler(opt, args, total)
    dynamics = Dynamics(model, args, total)
    provenance["optimizer_switch_events"] = dynamics.events
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and args.precision != "fp32"
    scaler = torch.amp.GradScaler(device.type, enabled=args.precision == "fp16")
    grad_stats = GradStats(model, args.clip) if args.grad_stats else None
    swa_model = torch.optim.swa_utils.AveragedModel(model) if args.swa_frac > 0 else None
    ema_model = None
    if args.ema_teacher > 0 or args.eval_ema or args.td_alpha > 0:
        if args.sync_free:
            ema_model = FastEMA(model, ema_update_decay)
        else:
            ema_model = torch.optim.swa_utils.AveragedModel(
                model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_update_decay))
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
    provenance["training_clocks"] = {"micro_rows": args.chunk, "detach_rows": detach_rows,
                                     "update_rows": update_rows, "ema_decay_per_update": ema_update_decay,
                                     "ema_clock": "attempted updates including AMP skips"}
    if thin is not None:
        provenance["row_thinning"] = {
            "mode": args.thin_mode, "probability": args.thin_prob, "drop_range": [low, high],
            "observed_time_axis": "original" if args.thin_mode == "retained-mass" else "compressed then padded",
            "rows_per_sequence": SEQUENCE_LENGTH, "update_schedule": "unchanged padded row count",
            "normalization": "frozen parent" if args.init else "fit on training views"}
    if args.credit_mode != "deterministic":
        provenance["training_clocks"]["credit_estimator"] = {
            "mode": args.credit_mode, "keep_probability": args.credit_keep_probability,
            "target": "fixed update-window gradient conditional on detached incoming state",
            "optimizer_claim": "affine SGD/momentum step conditional on fixed incoming optimizer state",
            "maximum_graph_rows": update_rows, "random_edge_spacing_rows": args.chunk}
    if args.origin_view != "off":
        initial_windows = origin_windows(SEQUENCE_LENGTH, update_rows, args.chunk, args.origin_view,
                                         args.origin_seed, 0, 0)
        provenance["training_clocks"].update({
            "origin_view": args.origin_view,
            "prefix_rows_per_sequence": sum(window.start for window in initial_windows),
            "supervised_rows_per_sequence": SEQUENCE_LENGTH,
            "variable_update_width": args.origin_view == "prefix-random",
            "uses_replayed_state": args.origin_view != "prefix-discard",
            "ema_clock": "actual update rows" if args.ema_reference_rows else "attempted updates including AMP skips"})
    # Bind the runtime only after lazy loaders/optimizers have initialized, and
    # use that same explicit thread budget for uninterrupted and resumed fits.
    torch.set_num_threads(cpu_threads())
    training_contract = resume_contract(args, provenance, train_loader, total) if continuation else None
    start_epoch = 0
    prior_fit_seconds = 0.0
    if saved_training is not None:
        # Revalidate after constructing all objects/lineage; RNG restoration is last.
        if saved_training["contract"] != training_contract:
            ap.error("resume contract differs: recipe, schedule, data, code, lineage or runtime changed")
        saved_dynamics = saved_training["auxiliary"]["dynamics"]
        if saved_dynamics["is_switched"]:
            opt, _ = switch_optimizer(opt, sched, saved_dynamics["switch"][1])
        restore_training_state(saved_training, model=model, optimizer=opt, scheduler=sched,
                               scaler=scaler, loader=train_loader, ema=ema_model, swa=swa_model)
        dynamics.load_state_dict(saved_dynamics)
        provenance["optimizer_switch_events"] = dynamics.events
        progress = saved_training["progress"]
        start_epoch, step_count = saved_training["next_epoch"], progress["step_count"]
        best, history = progress["best"], progress["history"]
        cumulative_updates = progress["cumulative_updates"]
        provenance["update_diagnostics"] = progress["update_diagnostics"]
        provenance["resumed_from"] = {"path": str(args.resume_state),
                                       "sha256": file_sha256(Path(args.resume_state)), "next_epoch": start_epoch}
        prior_fit_seconds = progress["fitting_seconds"]
        auxiliary_state = saved_training["auxiliary"]
        norm_ema = auxiliary_state["norm_ema"]
        if norm_ema is not None:
            norm_ema = norm_ema.to(device)
        if pool_memory is not None:
            value = auxiliary_state["pool_memory"]
            pool_memory.memory = None if value is None else value.to(device)
        if moments is not None:
            for field, value in auxiliary_state["readout_moments"].items():
                getattr(moments, field).copy_(value.to(device))
        restore_rng(saved_training["rng"])
    for epoch in range(start_epoch, args.epochs):
        if args.stop_after_epoch is not None and epoch > args.stop_after_epoch:
            break  # cheaper screens: the lr schedule still spans --epochs, so earlier checkpoints are unchanged
        model.train()
        epoch_counter = EpochUpdateCounter(device)
        if isinstance(train_loader.sampler, EpochOrderSampler):
            train_loader.sampler.set_epoch(epoch)
        epoch_order_identity = sample_order_identity(train_loader)
        if isinstance(train_loader.sampler, EpochOrderSampler):
            epoch_order_identity["seeded_order"] = {"seed": args.seed, "epoch": epoch,
                                                    "antithetic": args.antithetic, "shared_dropped_tail": True}
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
                valid = valid & (kept_len if args.thin_mode == "retained-mass" else steps < kept_len[:, None])
            main_valid = main_row_weights(inputs, y, valid, scored_dev, prop_model, prop_cached, args, x)
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
            nxt_label = next_label_change(y) if args.distill_alpha_far is not None or args.distill_boot_h else None
            if args.td_alpha > 0:
                nxt_all = next_label_change(y)
                ramp_steps = max(1, int(args.td_ramp * len(train_loader) * math.ceil(20_000 / args.chunk)))
                td_a = args.td_alpha * min(1.0, step_count / ramp_steps)
            timer.mark("labels")
            h = h_ema = None
            donor_masses = None
            windows = origin_windows(inputs.shape[1], update_rows, args.chunk, args.origin_view,
                                     args.origin_seed, epoch, b)
            if args.distill_denominator == "shuffled":
                if args.origin_view == "off":
                    donor_masses = shuffled_masses(
                        y.float().clamp(-2, 2).abs() * main_valid[..., None], update_rows,
                        args.denominator_shuffle_seed + 104729 * epoch + b)
                else:
                    masses = torch.stack([(y[:, window].float().clamp(-2, 2).abs()
                                           * main_valid[:, window, None]).sum().detach() for window in windows])
                    order = np.random.default_rng(args.denominator_shuffle_seed + 104729 * epoch + b).permutation(len(windows))
                    donor_masses = masses[torch.as_tensor(order, device=device)]
            for window_index, sl in enumerate(windows):
                s = sl.start
                opt = dynamics.maybe_switch(step_count, opt, sched)
                if controlled_loss:
                    def raw_forward(window, state):
                        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_autocast):
                            return forward_chunk(model, inputs, window, state)

                    def teacher_forward(window, state):
                        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_autocast):
                            return forward_chunk(ema_model.module, inputs, window, state)

                    if args.origin_view != "off":
                        carried_h, carried_h_ema = h, h_ema
                        h = h_ema = None
                        with torch.no_grad():
                            for prefix_start in range(0, sl.start, args.chunk):
                                prefix = slice(prefix_start, prefix_start + args.chunk)
                                _, h = raw_forward(prefix, h)
                                if args.ema_teacher:
                                    _, h_ema = teacher_forward(prefix, h_ema)
                        if args.origin_view == "prefix-discard":
                            h, h_ema = carried_h, carried_h_ema
                    opt.zero_grad(set_to_none=True)
                    loss = torch.zeros((), device=device)
                    configuration = DistilledLoss(
                        args.distill_alpha, args.distill_boot_h, args.ema_teacher,
                        args.ema_metric_weight, args.aux_struct, struct_at, N_SP_AUX,
                        args.aux_next, next_at)
                    for contribution, h, h_ema in distilled_macro_losses(
                            forward=raw_forward, ema_forward=teacher_forward,
                            state=h, ema_state=h_ema, labels=y, teacher=teach,
                            row_weights=main_valid, next_change=nxt_label,
                            structural_labels=(sp_lab, br_lab, cl_lab) if args.aux_struct else None,
                            next_labels=(nx_rows, nx_up, nw) if args.aux_next else None,
                            rows=sl, micro_rows=args.chunk, detach_rows=detach_rows,
                            config=configuration, denominator=args.distill_denominator,
                            reference_mass=args.distill_reference_mass,
                            detach_offset_rows=detach_boundary_offset(
                                args.chunk, detach_rows, args.detach_jitter_seed, step_count)
                                if args.detach_boundary_jitter else 0,
                            credit_scales=credit_edge_scales(
                                (sl.stop - sl.start) // args.chunk - 1, args.credit_keep_probability,
                                args.credit_seed, step_count, args.credit_mode == "artbp")
                                if args.credit_mode != "deterministic" else None,
                            donor_mass=None if donor_masses is None else donor_masses[window_index]):
                        scaler.scale(contribution).backward()
                        loss = loss + contribution.detach()
                    timer.mark("loss")
                else:
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_autocast):
                        out, h = forward_chunk(model, inputs, sl, h)
                    h = detach_state(h)
                    timer.mark("forward")
                    if args.ema_teacher > 0 or args.td_alpha > 0:
                        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=use_autocast):
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
                        label = yc
                        if args.distill_boot_h:
                            # Training-only target substitution while the label run outlives t+h.
                            # This is an empirical denoising hypothesis, not an unbiasedness guarantee.
                            rows = torch.arange(sl.start, sl.start + yc.shape[1], device=yc.device)[None]
                            ahead = rows + args.distill_boot_h
                            alive = (nxt_label[:, sl] > ahead) & (ahead < teach.shape[1])
                            idx = ahead.clamp(max=teach.shape[1] - 1).expand(yc.shape[0], -1)[..., None].expand(-1, -1, 2)
                            label = torch.where(alive[..., None], teach.gather(1, idx).float().clamp(-2, 2), yc)
                        tgt = (1 - a) * label + a * teach[:, sl].float().clamp(-2, 2)
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
                if grad_stats is not None:
                    grad_stats.observe()
                dynamics.perturb_and_clip(step_count)
                scaler.step(opt)
                invalidate_qat_cache(model)
                scaler.update()
                sched.step()
                timer.mark("optimizer")
                if args.parity_dump:
                    losses_trace.append(loss.detach().float())

                step_count += 1
                if swa_model is not None and step_count >= swa_start:
                    swa_model.update_parameters(model)
                    invalidate_qat_cache(swa_model.module)
                if ema_model is not None:
                    if args.origin_view != "off":
                        update_ema_for_rows(ema_model, model, args.ema_decay, sl.stop - sl.start,
                                            args.ema_reference_rows)
                    else:
                        ema_model.update_parameters(model)
                    invalidate_qat_cache(ema_model.module)
                    epoch_counter.ema_updates += 1
                if duty is not None:  # explicit rate limit (WNN_GPU_DUTY < 1): sync, then sleep
                    torch.cuda.synchronize()
                    duty.tick()
                elif not args.sync_free and device.type == "cuda":
                    torch.cuda.synchronize()
            timer.stop()
            epoch_counter.completed_batches += 1
            if args.profile_steps and b + 1 == args.max_batches:
                break
            if b % 20 == 0:
                rate = (b + 1) * args.batch / (time.time() - t0)
                print(f"ep {epoch} batch {b}/{len(train_loader)} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} seq/s {rate:.1f}", flush=True)
        epoch_grad_stats = None if grad_stats is None else {"epoch": epoch, **grad_stats.finish()}
        if epoch_grad_stats is not None:
            print("GRAD_STATS", json.dumps(epoch_grad_stats), flush=True)
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
        if epoch_grad_stats is not None:
            update_counts["grad_stats"] = epoch_grad_stats
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
        representation = inference_checkpoint_fields(eval_model, "ema" if args.eval_ema else
                                                      "raw" if eval_model is model else "swa")
        checkpoint = {**representation, "args": vars(args), "n_in": n_in, "epoch": epoch,
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
        provenance["fitting_seconds"] = round(prior_fit_seconds + time.time() - fitting_started, 1)
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if continuation:
            training_state = capture_training_state(
                model=model, optimizer=opt, scheduler=sched, scaler=scaler, loader=train_loader,
                contract=training_contract, next_epoch=epoch + 1, ema=ema_model, swa=swa_model,
                progress={"step_count": step_count, "best": best, "history": history,
                          "cumulative_updates": cumulative_updates,
                          "update_diagnostics": provenance["update_diagnostics"],
                          "fitting_seconds": provenance["fitting_seconds"]},
                auxiliary={"norm_ema": norm_ema, "dynamics": dynamics.state_dict(),
                           "pool_memory": None if pool_memory is None else pool_memory.memory,
                           "readout_moments": None if moments is None else
                           {field: getattr(moments, field) for field in ("m1", "m0", "n")}},
            )
            atomic_save_training_state(training_state, out_dir / "training_state.pt")
        t0 = time.time()
    if saved_training is not None and start_epoch == args.epochs:
        # A crash after the final atomic state but before export needs no more updates.
        eval_model = checkpoint_model(model, ema_model, swa_model if step_count >= swa_start else None,
                                      args.eval_ema)
        provenance["selected_epoch"] = args.epochs - 1
        provenance["cumulative_updates"] = cumulative_updates.copy()
        provenance["fitting_seconds"] = prior_fit_seconds
        representation = inference_checkpoint_fields(eval_model, "ema" if args.eval_ema else
                                                      "raw" if eval_model is model else "swa")
        checkpoint = {**representation, "args": vars(args), "n_in": n_in,
                      "epoch": args.epochs - 1,
                      "readout": None if moments is None else [value.cpu() for value in moments.tables()],
                      "readout_tilted": bool(args.ce_tilt),
                      "provenance": {key: value for key, value in provenance.items() if key != "checkpoint_sha256"}}
        for filename in (f"model_ep{args.epochs - 1}.pt", "model.pt"):
            torch.save(checkpoint, out_dir / filename)
            provenance["checkpoint_sha256"][filename] = file_sha256(out_dir / filename)
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        atomic_save_training_state(saved_training, out_dir / "training_state.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    if args.checkpoint_policy == "fixed":
        print("SELECTED_FIXED", args.checkpoint_epoch)
    else:
        print("BEST", best)
    if args.include_heldout_valid and not args.no_eval_during_fit:
        print("SELECTED_FINAL", history[-1]["weighted_pearson"])


if __name__ == "__main__":
    main()
