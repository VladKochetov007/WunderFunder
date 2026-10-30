from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from wnn import student_frontier
from wnn.resources import dataloader_budget
from wnn.training.datasets import Sequences
from wnn.training.state import TrainingRun
from wnn.training_sampling import EpochOrderSampler


def select_train_eval_row_groups(train_idx: np.ndarray, sequence_count: int) -> list[int]:
    """Select a stable source-order prefix shared across fit seeds and configs."""
    groups = np.asarray(train_idx)
    if (
        groups.ndim != 1
        or groups.dtype.kind not in "iu"
        or not 1 <= sequence_count <= 32
        or len(set(map(int, groups))) != len(groups)
    ):
        raise ValueError("train-eval selection needs unique integer groups and a count in [1, 32]")
    ordered = np.sort(groups.astype(np.int64, copy=False))
    if len(ordered) < sequence_count:
        raise ValueError("train-eval prefix exceeds the fitting membership")
    return list(map(int, ordered[:sequence_count]))


def build_loaders(run: TrainingRun) -> None:
    """Construct lazy datasets and preserve sampler, pinning and drop-tail semantics."""
    environment = run.environment
    ap = run.options.ap
    args = run.options.args
    campaign_split = run.data.campaign_split
    distill_cache_validation = run.data.distill_cache_validation
    eligible_valid_count = run.data.eligible_valid_count
    extra_valid = run.data.extra_valid
    frontier_cache_binding = run.data.frontier_cache_binding
    frontier_contract = run.data.frontier_contract
    frontier_order = run.data.frontier_order
    frontier_stage = run.options.frontier_stage
    high = run.data.high
    low = run.data.low
    prefetch = run.data.prefetch
    prop_cache = run.data.prop_cache
    sampler = run.data.sampler
    teachers = run.data.teachers
    thin = run.data.thin
    train_eval_loader = None
    train_eval_members_sha256 = None
    train_eval_row_groups = None
    train_ds = run.data.train_ds
    train_idx = run.data.train_idx
    train_loader = run.data.train_loader
    train_split = run.data.train_split
    valid_idx = run.data.valid_idx
    valid_loader = run.data.valid_loader
    workers = run.data.workers
    if args.thin_prob > 0:
        low, high = (float(v) for v in args.thin_range.split(","))
        thin = (args.thin_prob, low, high, args.thin_mode)
    train_ds = Sequences(
        train_split,
        train_idx,
        teachers,
        prop_cache.get(train_split),
        quote_path=args.aux_quote > 0,
        quote_cache_root=args.quote_cache_root,
        spread_cache_root=args.spread_cache_root,
        thin=thin,
        distill_cache_root=args.distill_cache_root,
        distill_cache_variant=args.distill_cache_variant,
        distill_cache_metadata=distill_cache_validation,
        include_family_targets=args.distill_family_weight > 0,
        environment=environment,
    )
    train_batch_bytes = train_ds.batch_tensor_bytes(args.batch, heuristics=args.heuristics)
    if extra_valid is not None:
        valid_ds = Sequences(
            "valid",
            extra_valid,
            teachers,
            prop_cache.get("valid"),
            quote_path=args.aux_quote > 0,
            quote_cache_root=args.quote_cache_root,
            spread_cache_root=args.spread_cache_root,
            thin=thin,
            distill_cache_root=args.distill_cache_root,
            distill_cache_variant=args.distill_cache_variant,
            distill_cache_metadata=distill_cache_validation,
            include_family_targets=args.distill_family_weight > 0,
            environment=environment,
        )
        train_batch_bytes = max(
            train_batch_bytes, valid_ds.batch_tensor_bytes(args.batch, heuristics=args.heuristics)
        )
        train_ds = torch.utils.data.ConcatDataset([train_ds] + [valid_ds] * args.valid_repeat)
    if args.epoch_samples:
        sample_weights = np.ones(len(train_ds), dtype=np.float64)
        if extra_valid is not None:
            sample_weights[len(train_idx) :] = eligible_valid_count / len(extra_valid)
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights),
            args.epoch_samples,
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    valid_ds = (
        None if args.no_eval_during_fit else Sequences("valid", valid_idx, environment=environment)
    )
    valid_batch_bytes = (
        0 if valid_ds is None else valid_ds.batch_tensor_bytes(16, heuristics=args.heuristics)
    )
    loader_ram = int(float(os.environ.get("WNN_LOADER_RAM_GB", "6")) * (1 << 30))
    workers, prefetch = dataloader_budget(
        train_batch_bytes, ram_budget_bytes=loader_ram - 2 * valid_batch_bytes
    )
    if args.batch_order == "seeded":
        if sampler is not None:
            ap.error("seeded/antithetic order cannot replace weighted or replacement sampling")
        sampler = EpochOrderSampler(
            len(train_ds), args.seed, args.antithetic, batch_size=args.batch, drop_last=True
        )
        sampler.set_epoch(-1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=workers,
        prefetch_factor=prefetch if workers else None,
        pin_memory=args.pin_memory,
    )
    valid_loader = (
        None
        if valid_ds is None
        else DataLoader(valid_ds, batch_size=16, num_workers=0, prefetch_factor=None)
    )
    if args.train_eval_seqs:
        if train_split != "train":
            ap.error("--train-eval-seqs requires a fitting source named train")
        if campaign_split is None or not np.array_equal(
            np.sort(train_idx), np.asarray(campaign_split.train_groups, dtype=np.int64)
        ):
            ap.error("--train-eval-seqs requires the complete official train fit membership")
        train_eval_row_groups = select_train_eval_row_groups(train_idx, args.train_eval_seqs)
        members = [["train", group] for group in train_eval_row_groups]
        member_bytes = json.dumps(members, separators=(",", ":")).encode("ascii")
        train_eval_members_sha256 = hashlib.sha256(member_bytes).hexdigest()
        train_eval_ds = Sequences("train", train_eval_row_groups, environment=environment)
        train_eval_loader = DataLoader(
            train_eval_ds,
            batch_size=args.train_eval_seqs,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
    frontier_order = None
    if frontier_contract is not None:
        if frontier_cache_binding is None or extra_valid is None or train_split != "train":
            raise ValueError(
                "student-frontier fit requires validated clean train+valid cache membership"
            )
        frontier_order = student_frontier.validate_training_order(
            args,
            frontier_contract,
            frontier_stage,
            campaign_split,
            train_idx,
            extra_valid,
            train_loader,
        )
    run.data.frontier_order = frontier_order
    run.data.high = high
    run.data.low = low
    run.data.prefetch = prefetch
    run.data.sampler = sampler
    run.data.thin = thin
    run.data.train_ds = train_ds
    run.data.train_eval_loader = train_eval_loader
    run.data.train_eval_members_sha256 = train_eval_members_sha256
    run.data.train_eval_row_groups = train_eval_row_groups
    run.data.train_loader = train_loader
    run.data.valid_loader = valid_loader
    run.data.workers = workers
