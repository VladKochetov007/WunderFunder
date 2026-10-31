from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
from wnn.training.state import TrainingRun


def select_membership(run: TrainingRun) -> None:
    """Select complete fitting/evaluation groups with the original deterministic ordering."""
    environment = run.environment
    args = run.options.args
    campaign_split = run.data.campaign_split
    eligible_valid_count = run.data.eligible_valid_count
    extra_valid = run.data.extra_valid
    fit_subset = run.data.fit_subset
    inner_crossfit = run.data.inner_crossfit
    n_valid = run.data.n_valid
    sampler = run.data.sampler
    train_idx = run.data.train_idx
    train_split = run.data.train_split
    valid_idx = run.data.valid_idx
    valid_path = environment.root / "datasets" / "valid.parquet"
    if campaign_split is None:
        dev_groups, hold_groups = environment.valid_row_groups(valid_path)
        dev_a, dev_b = environment.dev_halves(valid_path)
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
            cf = np.load(environment.root / "cache" / "chain_folds.npz")
            vmask = cf["split"] == "valid"
            fold_map = dict(
                zip(cf["row_group"][vmask].tolist(), cf["fold"][vmask].tolist(), strict=True)
            )
            fold_of = np.array([fold_map[int(g)] for g in all_valid]) % args.n_valid_folds
            fold_groups = all_valid[fold_of == args.valid_fold]
            if args.train_source == "train+valid":
                extra_valid = all_valid[fold_of != args.valid_fold]
    elif args.train_source == "train+valid":
        extra_valid = (
            all_valid if args.include_heldout_valid else np.r_[np.asarray(dev_a), np.asarray(dev_b)]
        )
    valid_pool = None
    if args.train_source == "valid":
        valid_pool = (
            np.setdiff1d(all_valid, fold_groups)
            if fold_groups is not None
            else all_valid
            if args.include_heldout_valid
            else np.asarray(dev_groups)
        )
    elif extra_valid is not None:
        valid_pool = np.asarray(extra_valid)
    if inner_crossfit is not None:
        extra_valid = valid_pool = np.asarray(inner_crossfit["fit"]["valid"], dtype=np.int64)
    eligible_valid_count = 0 if valid_pool is None else len(valid_pool)
    if valid_pool is not None and args.valid_train_fraction < 1:
        if campaign_split is not None:
            chain_map = {
                group: component
                for split, group, component in campaign_split.components
                if split == "valid"
            }
        else:
            cf = np.load(environment.root / "cache" / "chain_folds.npz")
            vmask = cf["split"] == "valid"
            chain_map = dict(
                zip(cf["row_group"][vmask].tolist(), cf["chain"][vmask].tolist(), strict=True)
            )
        pool_chains = np.array([chain_map[int(group)] for group in valid_pool])
        valid_classes = np.load(environment.root / "research/artifacts/market-classes-v1.npz")[
            "valid"
        ][valid_pool]
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
    if fit_subset is not None:
        train_split = "train"
        train_idx = np.asarray(fit_subset["train_groups"], dtype=np.int64)
        extra_valid = np.asarray(fit_subset["valid_groups"], dtype=np.int64)
        valid_pool = extra_valid
        eligible_valid_count = len(extra_valid)
    elif args.train_source == "valid":
        train_split = "valid"
        train_idx = np.random.default_rng(args.seed).permutation(valid_pool)
    elif args.train_source == "valid_devA":
        train_split = "valid"
        train_idx = np.random.default_rng(args.seed).permutation(dev_a)[: args.train_seqs]
    else:
        usable = (
            np.asarray(campaign_split.train_groups)
            if campaign_split is not None
            else np.array([g for g in range(10607) if g not in environment.corrupt_train_groups])
        )
        if inner_crossfit is not None:
            usable = np.asarray(inner_crossfit["fit"]["train"], dtype=np.int64)
        if args.train_classes:
            keep = {int(c) for c in args.train_classes.split(",")}
            cls = np.load(environment.root / "research" / "artifacts" / "market-classes-v1.npz")[
                "train"
            ]
            usable = usable[np.isin(cls[usable], list(keep))]
        train_idx = np.random.default_rng(args.seed).permutation(usable)[: args.train_seqs]
    if (
        fold_groups is not None
        and extra_valid is not None
        and np.intersect1d(extra_valid, fold_groups).size
    ):
        raise RuntimeError("held-out validation fold entered the training pool")
    if campaign_split is not None:
        campaign_split.assert_fit_groups(
            train_idx if train_split == "train" else (),
            train_idx if train_split == "valid" else extra_valid if extra_valid is not None else (),
        )
    eval_groups = {"devB": dev_b, "holdout": hold_groups, "vfold": fold_groups}.get(
        args.eval_source, dev_groups
    )
    n_valid = (
        len(eval_groups)
        if args.full_valid
        or (
            campaign_split is not None
            and campaign_split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
        )
        else min(args.valid_seqs, len(eval_groups))
    )
    valid_idx = np.random.default_rng(12345).permutation(eval_groups)[:n_valid]
    sampler = None
    if args.class_balance:
        classes = np.load(environment.root / "research" / "artifacts" / "market-classes-v1.npz")
        tr_cls = classes[train_split][train_idx]
        valid_classes = (
            classes["valid"][list(campaign_split.valid_groups)]
            if campaign_split is not None
            else classes["valid"]
        )
        va_share = np.array([(valid_classes == c).mean() for c in range(4)])
        tr_share = np.array([max((tr_cls == c).mean(), 1e-09) for c in range(4)])
        weights = np.where(
            tr_cls >= 0, va_share[np.clip(tr_cls, 0, 3)] / tr_share[np.clip(tr_cls, 0, 3)], 1.0
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            len(train_idx),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    run.data.eligible_valid_count = eligible_valid_count
    run.data.extra_valid = extra_valid
    run.data.n_valid = n_valid
    run.data.sampler = sampler
    run.data.train_idx = train_idx
    run.data.train_split = train_split
    run.data.valid_idx = valid_idx
