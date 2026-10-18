"""Causal propensity model P(is_scored | rows <= t), fit on validation devA only.

    scripts/limited .venv/bin/python scripts/train_propensity.py --name prop_v1
Legacy mode scores AUC on devB. The opt-in strict_fp32_v1 profile requires a
campaign split, fits normalization on all ordered fit-valid groups, and withholds
evaluation. The frozen model supplies soft scoring weights when training on
train.parquet (which has no mask).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_rnn import Sequences, build_inputs, checkpoint_selected, configure_checkpoint_policy

from wnn.data import N_FEATURES, WARMUP
from wnn.features import N_EVENT_FEATURES
from wnn.propensity import (
    PROPENSITY_FIT_SOURCE_FILES,
    STRICT_FP32_PROFILE,
    PropensityNet,
    fit_normalization,
    ordered_groups_sha256,
    propensity,
)
from wnn.resources import configure_strict_fp32, configure_torch
from wnn.splits import dev_halves, file_sha256, load_campaign_split

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", SOURCE_ROOT)).resolve()


def _source_sha256(profile: str) -> dict[str, str]:
    """Bind relevant training, data, feature and precision code for strict runs."""
    if profile == STRICT_FP32_PROFILE:
        paths = PROPENSITY_FIT_SOURCE_FILES
    else:
        paths = (
            "scripts/train_propensity.py",
            "scripts/train_rnn.py",
            "src/wnn/propensity.py",
            "src/wnn/propensity_cli.py",
            "src/wnn/training/inputs.py",
        )
    return {path: file_sha256(SOURCE_ROOT / path) for path in paths}


def _prepare_run_directory(path: Path, *, strict: bool, resume: bool = False) -> None:
    """Create a strict run directory exclusively; legacy runs retain reuse behavior."""
    if strict:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise FileExistsError(f"strict propensity run path is a symlink: {path}")
        if resume:
            if not path.is_dir():
                raise FileNotFoundError(f"strict propensity resume directory is missing: {path}")
        else:
            path.mkdir(exist_ok=False)
    else:
        path.mkdir(parents=True, exist_ok=True)


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"strict propensity {name} contain non-finite values")


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    from wnn.propensity_cli import build_propensity_parser

    ap = build_propensity_parser()
    args = ap.parse_args()
    configure_checkpoint_policy(args, ap)
    strict_fp32 = args.precision_profile == STRICT_FP32_PROFILE
    if strict_fp32 and (
        Path(args.name).name != args.name or args.name in {".", ".."} or "\\" in args.name
    ):
        ap.error("strict_fp32_v1 --name must be a single run-directory name")
    if args.resume and not strict_fp32:
        ap.error("--resume is supported only by strict_fp32_v1")
    if args.loader_workers is None:
        args.loader_workers = 0 if strict_fp32 else 2
    if args.loader_workers < 0:
        ap.error("--loader-workers must be nonnegative")
    if strict_fp32 and args.split_manifest is None:
        ap.error("strict_fp32_v1 requires --split-manifest")
    if strict_fp32 and not (args.no_eval_during_fit and args.checkpoint_policy == "fixed"):
        ap.error("strict_fp32_v1 requires fixed checkpoint selection and --no-eval-during-fit")
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
    if strict_fp32:
        configure_strict_fp32()
    out_dir = ROOT / "artifacts" / "runs" / args.name
    _prepare_run_directory(out_dir, strict=strict_fp32, resume=args.resume)
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
    train_dataset = Sequences("valid", dev_a)
    loader_options = {"num_workers": args.loader_workers}
    if args.loader_workers:
        loader_options["prefetch_factor"] = 1
    train = DataLoader(train_dataset, batch_size=16, shuffle=True, **loader_options)
    test = (
        None
        if args.no_eval_during_fit
        else DataLoader(Sequences("valid", dev_b), batch_size=16, **loader_options)
    )
    model = PropensityNet(N_FEATURES + N_EVENT_FEATURES, args.hidden).to(device)
    model._propensity_profile = args.precision_profile
    normalization = None
    if strict_fp32 and not args.resume:
        stats_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=False,
            **loader_options,
        )
        normalization = fit_normalization(model, (build_inputs(x) for x, _, _ in stats_loader))
    elif not strict_fp32:
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
        "precision_profile": args.precision_profile,
        "split_manifest_sha256": campaign_split.sha256 if campaign_split is not None else None,
        "fit_row_groups": {"valid": dev_a.tolist()},
        "fit_row_groups_sha256": {"valid": ordered_groups_sha256(dev_a)},
        "evaluation_row_groups": dev_b.tolist(),
        "normalization": normalization,
        "n_in": N_FEATURES + N_EVENT_FEATURES,
        "source_sha256": _source_sha256(args.precision_profile),
        "checkpoint_sha256": {},
    }
    if strict_fp32:
        assert campaign_split is not None
        from wnn.propensity_fit import fit_strict_propensity

        print(json.dumps(vars(args)), flush=True)
        fit_strict_propensity(
            model=model,
            loader=train,
            optimizer=opt,
            build_inputs=build_inputs,
            args=vars(args).copy(),
            provenance=provenance,
            normalization=normalization,
            split=campaign_split,
            data_root=ROOT,
            source_root=SOURCE_ROOT,
            device=device,
            warmup=WARMUP,
            chunk_rows=2000,
            resume=args.resume,
            run_dir=out_dir,
        )
        return
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
                if strict_fp32:
                    with torch.autocast("cuda", enabled=False):
                        logit, h = model(inputs[:, s : s + 2000], h)
                    _require_finite(logit, "logits")
                else:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logit, h = model(inputs[:, s : s + 2000], h)
                h = h.detach()
                keep = (torch.arange(s, min(s + 2000, 20_000), device=device) >= WARMUP)[
                    None
                ].expand_as(logit)
                loss = F.binary_cross_entropy_with_logits(
                    logit.float()[keep], target[:, s : s + 2000][keep]
                )
                if strict_fp32:
                    _require_finite(loss, "loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if strict_fp32:
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            _require_finite(parameter.grad, f"gradient {name}")
                opt.step()
                if strict_fp32:
                    for name, parameter in model.named_parameters():
                        _require_finite(parameter, f"weight {name}")
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
