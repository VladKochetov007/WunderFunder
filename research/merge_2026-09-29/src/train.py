"""Truncated-BPTT training over streamed sequences.

Each batch is B whole sequences. The recurrent state is carried across
consecutive chunks of T rows with gradients cut at chunk boundaries, exactly
mirroring the row-by-row stateful inference the scorer performs. One optimiser
step per chunk.

    python src/train.py --tag dev --seqs 512 --epochs 2          # smoke run
    python src/train.py --tag gru256 --hidden 256 --epochs 8      # full data

Outputs go to runs/<tag>/: best.pt (weights + config + feature stats),
log.jsonl, and stdout progress.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (
    SEQUENCE_LENGTH,
    STEP_MASK,
    TRAIN_PATH,
    VALID_PATH,
    BatchStream,  # noqa: E402
    SequenceReader,
    feature_stats,
    load_subset,
)
from loss import CombinedLoss, GlobalPearsonLoss, SequencePearsonLoss, WeightedMSELoss  # noqa: E402
from metric import global_weighted_pearson, score_batch  # noqa: E402
from model import ModelConfig, Predictor, count_params  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wnn_connectome_starterpack" / "src"))
from wnn.splits import valid_row_groups  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def recency_weights(features: torch.Tensor, tau: float, floor: float) -> torch.Tensor:
    """Weight rows by causal age since the previous i0 book-price change."""
    batch, length, _ = features.shape
    event = torch.zeros((batch, length), dtype=torch.bool, device=features.device)
    event[:, 1:] = (features[:, 1:, :22] != features[:, :-1, :22]).any(dim=-1)
    row = torch.arange(length, device=features.device).expand(batch, -1)
    last_event = torch.cummax(torch.where(event, row, 0), dim=1).values
    age = row - last_event
    return torch.clamp(1.0 / (1.0 + age.float() / tau), min=floor)


class EMA:
    """Exponential moving average of weights, evaluated instead of the raw weights."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers(), strict=True):
            s.copy_(b)


@torch.no_grad()
def evaluate(
    model: Predictor,
    features: np.ndarray,
    targets: np.ndarray,
    scored: np.ndarray,
    device: torch.device,
    batch: int = 64,
    chunk: int = 4000,
    state_reset: int = 0,
) -> dict:
    """Score whole sequences with the exact per-sequence metric."""
    model.eval()
    preds = np.zeros(targets.shape, dtype=np.float32)
    step = math.gcd(chunk, state_reset) if state_reset else chunk
    for i in range(0, len(features), batch):
        x = torch.from_numpy(features[i : i + batch]).to(device)
        state = model.initial_state(x.shape[0], device)
        for t in range(0, SEQUENCE_LENGTH, step):
            if state_reset and t and t % state_reset == 0:
                state = model.initial_state(x.shape[0], device)
            out, state = model(x[:, t : t + step], state)
            preds[i : i + batch, t : t + step] = out[..., :2].float().cpu().numpy()
    model.train()
    result = score_batch(targets, preds, scored)
    result["global"] = global_weighted_pearson(targets, preds, scored)["weighted_pearson"]
    q = SEQUENCE_LENGTH // 4
    result["quarters"] = [
        round(
            score_batch(
                targets[:, i * q : (i + 1) * q],
                preds[:, i * q : (i + 1) * q],
                scored[:, i * q : (i + 1) * q],
            )["weighted_pearson"],
            4,
        )
        for i in range(4)
    ]
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rnn", default="gru", choices=["gru", "lstm"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--proj", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--diff", action="store_true", help="feed first differences alongside the raw row"
    )
    ap.add_argument("--batch", type=int, default=128, help="sequences per batch")
    ap.add_argument("--chunk", type=int, default=1000, help="TBPTT length in rows")
    ap.add_argument(
        "--global-loss",
        action="store_true",
        help="pool the Pearson statistics over the batch as well as over time, "
        "matching the global metric adopted 2026-09-15. Implies --seq-loss.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue from runs/<tag>/state.pt, written after every epoch. "
        "Restores model, EMA, optimiser, SWA accumulator, step and best. "
        "The batch order restarts at the epoch boundary rather than replaying "
        "the interrupted epoch, so a resumed run is not bit-identical to an "
        "uninterrupted one.",
    )
    ap.add_argument(
        "--state-reset",
        type=int,
        default=0,
        help="reset the recurrent state to its initial value every N rows (0 = never). "
        "Must divide evenly into --chunk multiples; inference must use the same N.",
    )
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--pearson-weight", type=float, default=0.9)
    ap.add_argument(
        "--teacher-train", default=None, help="teacher_label.py memmap for train.parquet"
    )
    ap.add_argument(
        "--teacher-valid", default=None, help="teacher_label.py memmap for valid.parquet"
    )
    ap.add_argument(
        "--distill",
        type=float,
        default=0.0,
        help="weight of MSE(student, teacher) on required rows, added to the loss",
    )
    ap.add_argument(
        "--swa",
        action="store_true",
        help="average raw weights over the last epoch and keep it if it scores best",
    )
    ap.add_argument(
        "--aux-mask",
        type=float,
        default=0.0,
        help="weight of an auxiliary BCE head predicting is_scored (third output channel, dropped at export)",
    )
    ap.add_argument(
        "--seq-loss",
        action="store_true",
        help="sequence-level Pearson with running statistics instead of per-chunk Pearson",
    )
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seqs", type=int, default=0, help="limit training sequences (0 = all)")
    ap.add_argument(
        "--train-path",
        default=str(TRAIN_PATH),
        help="parquet to train on; pass valid.parquet for mask experiments",
    )
    ap.add_argument(
        "--holdout",
        type=int,
        default=0,
        help="when training on valid.parquet: number of its sequences held out for evaluation",
    )
    ap.add_argument(
        "--loss-mask",
        default="all",
        choices=["all", "scored", "soft"],
        help="rows the loss sees: all required rows, the is_scored rows, or the predicted "
        "P(is_scored) weights from --soft-mask",
    )
    ap.add_argument("--soft-mask", default=None, help="float16 memmap from maskmodel.py label")
    ap.add_argument(
        "--add-valid",
        action="store_true",
        help="also train on the validation sequences not held out (their real is_scored mask)",
    )
    ap.add_argument(
        "--valid-fold",
        type=int,
        default=-1,
        help="chain fold kept for evaluation; other development folds may join training",
    )
    ap.add_argument(
        "--final-fit",
        action="store_true",
        help="train on all validation development groups and save fixed-epoch EMA without validation selection",
    )
    ap.add_argument("--n-valid-folds", type=int, default=4)
    ap.add_argument(
        "--since-weight",
        type=float,
        default=0.0,
        help="recency scale in rows for loss weights; 0 disables recency weighting",
    )
    ap.add_argument("--since-floor", type=float, default=0.25)
    ap.add_argument(
        "--val-seqs", type=int, default=192, help="validation subset for the per-epoch check"
    )
    ap.add_argument("--stats-seqs", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=20, help="chunks")
    args = ap.parse_args()
    if args.global_loss:
        args.seq_loss = True
    if args.since_weight < 0 or not 0.0 <= args.since_floor <= 1.0:
        ap.error("--since-weight must be nonnegative and --since-floor must lie in [0, 1]")
    if args.state_reset and args.state_reset % args.chunk:
        ap.error("--state-reset must be divisible by --chunk")
    if args.final_fit and (not args.add_valid or args.holdout or args.valid_fold >= 0 or args.swa):
        ap.error("--final-fit requires --add-valid and cannot use holdout, valid-fold, or SWA")

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.90, torch.cuda.current_device())
    run_dir = RUNS / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    def log(**kv):
        kv["time"] = time.time()
        with open(log_path, "a") as f:
            f.write(json.dumps(kv) + "\n")

    train_reader = SequenceReader(
        args.train_path,
        soft_mask=args.soft_mask if args.loss_mask == "soft" else None,
        teacher=args.teacher_train if args.distill > 0 else None,
    )
    valid_reader = SequenceReader(
        VALID_PATH, teacher=args.teacher_valid if args.distill > 0 else None
    )
    if args.distill > 0 and (
        args.teacher_train is None or (args.add_valid and args.teacher_valid is None)
    ):
        ap.error("--distill needs --teacher-train (and --teacher-valid with --add-valid)")
    rng = np.random.default_rng(args.seed)
    extra = []
    if args.final_fit:
        dev_groups, _ = valid_row_groups(VALID_PATH)
        extra = [(valid_reader, dev_groups)]
        train_idx = np.setdiff1d(np.arange(len(train_reader)), np.array([528]))
        val_idx = np.empty(0, dtype=np.int64)
    elif args.valid_fold >= 0:
        if args.valid_fold >= args.n_valid_folds:
            ap.error("--valid-fold must be smaller than --n-valid-folds")
        if not args.add_valid:
            ap.error("--valid-fold requires --add-valid so training excludes the evaluated fold")
        if args.holdout:
            ap.error("--valid-fold cannot be combined with --holdout")
        dev_groups, _ = valid_row_groups(VALID_PATH)
        fold_data = np.load(
            Path(__file__).resolve().parents[1]
            / "wnn_connectome_starterpack"
            / "cache"
            / "chain_folds.npz"
        )
        valid_rows = fold_data["split"] == "valid"
        fold_map = dict(
            zip(
                fold_data["row_group"][valid_rows].tolist(),
                fold_data["fold"][valid_rows].tolist(),
                strict=True,
            )
        )
        fold_of = np.array([fold_map[int(group)] % args.n_valid_folds for group in dev_groups])
        val_idx = dev_groups[fold_of == args.valid_fold]
        train_valid_idx = dev_groups[fold_of != args.valid_fold]
        extra = [(valid_reader, train_valid_idx)]
        train_idx = np.setdiff1d(np.arange(len(train_reader)), np.array([528]))
        if not len(val_idx) or not len(train_valid_idx):
            ap.error("the requested chain fold has an empty train or evaluation split")
    elif args.holdout:
        if args.add_valid:
            ap.error("--add-valid cannot be combined with --holdout")
        # training on the masked validation file: split it into train / held-out
        perm = rng.permutation(len(train_reader))
        val_idx, train_idx = perm[: args.holdout], perm[args.holdout :]
        val_idx = val_idx[: args.val_seqs]
        valid_reader = train_reader
    else:
        train_idx = np.setdiff1d(np.arange(len(train_reader)), np.array([528]))
        val_idx = rng.choice(
            len(valid_reader), size=min(args.val_seqs, len(valid_reader)), replace=False
        )
        if args.add_valid:
            dev_groups, _ = valid_row_groups(VALID_PATH)
            rest = np.setdiff1d(dev_groups, val_idx)
            extra = [(valid_reader, rest)]
    if args.seqs:
        train_idx = rng.choice(train_idx, size=min(args.seqs, len(train_idx)), replace=False)
    if args.loss_mask != "all" and not train_reader.has_mask:
        ap.error(f"--loss-mask {args.loss_mask} needs a masked training file or --soft-mask")
    # a fixed slice of training sequences scored with the exact metric: the
    # train/val gap is the overfitting diagnostic
    fit_idx = train_idx[: min(64, len(train_idx))]

    print(
        f"device={device} train_seqs={len(train_idx)}{' + valid ' + str(len(extra[0][1])) if extra else ''} "
        f"val_seqs={len(val_idx)}",
        flush=True,
    )
    t0 = time.time()
    val = None if args.final_fit else load_subset(valid_reader, val_idx, args.workers)
    fit = load_subset(train_reader, fit_idx, args.workers)
    fit_scored = np.broadcast_to(STEP_MASK, fit.targets.shape[:2])
    mean, std = feature_stats(train_reader, args.stats_seqs, args.seed)
    loaded = (
        "training subset + feature stats" if args.final_fit else "validation subset + feature stats"
    )
    print(f"loaded {loaded} in {time.time() - t0:.0f}s", flush=True)

    cfg = ModelConfig(
        rnn=args.rnn,
        hidden=args.hidden,
        layers=args.layers,
        proj=args.proj,
        dropout=args.dropout,
        diff=args.diff,
        out=3 if args.aux_mask > 0 else 2,
    )
    model = Predictor(cfg, mean, std).to(device)
    ema = EMA(model, args.ema)
    print(f"model {cfg.to_dict()} params={count_params(model):,}", flush=True)

    stream = BatchStream(
        train_reader, args.batch, train_idx, seed=args.seed, workers=args.workers, extra=extra
    )
    chunks_per_seq = math.ceil(SEQUENCE_LENGTH / args.chunk)
    steps_per_epoch = stream.batches_per_epoch() * chunks_per_seq
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup))

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (
            no_decay if p.ndim < 2 or "norm" in name.lower() or name.endswith("bias") else decay
        ).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.98),
    )

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress)))

    criterion = CombinedLoss(args.pearson_weight)
    seq_pearson = GlobalPearsonLoss() if args.global_loss else SequencePearsonLoss()
    mse = WeightedMSELoss()
    swa_sum, swa_n = None, 0
    step_mask = torch.from_numpy(STEP_MASK).to(device)
    best = -1.0
    step = 0
    start_epoch = 1
    state_path = run_dir / "state.pt"
    if args.resume:
        if not state_path.exists():
            print(f"--resume: no {state_path}, starting from scratch", flush=True)
        else:
            st = torch.load(state_path, map_location="cpu", weights_only=False)
            if st["config"] != cfg.to_dict():
                raise SystemExit(
                    f"--resume: config mismatch\n  saved {st['config']}\n  now   {cfg.to_dict()}"
                )
            if st["total_steps"] != total_steps:
                raise SystemExit(
                    f"--resume: LR schedule changed (total steps {st['total_steps']} -> "
                    f"{total_steps}); resuming would distort it. Match --epochs/--batch/--chunk."
                )
            model.load_state_dict(st["model"])
            ema.shadow.load_state_dict(st["ema"])
            opt.load_state_dict(st["opt"])
            swa_n = st["swa_n"]
            swa_sum = [t.to(device) for t in st["swa_sum"]] if st["swa_sum"] is not None else None
            step, best, start_epoch = st["step"], st["best"], st["epoch"] + 1
            torch.set_rng_state(st["rng"])
            print(
                f"resumed from epoch {st['epoch']} (step {step}, best {best:.4f}) -> starting epoch {start_epoch}",
                flush=True,
            )
            if start_epoch > args.epochs:
                print("nothing to do: saved epoch is already the last one", flush=True)
                return 0

    print(f"steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup_steps}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        run_loss, run_n = 0.0, 0
        for b, batch in enumerate(stream):
            x_all = torch.from_numpy(batch.features).to(device, non_blocking=True)
            y_all = torch.from_numpy(batch.targets).to(device, non_blocking=True)
            m_all = (
                torch.from_numpy(batch.scored).to(device)
                if args.loss_mask != "all"
                else step_mask.expand(x_all.shape[0], -1)
            )
            t_all = torch.from_numpy(batch.teacher).to(device) if args.distill > 0 else None
            state = model.initial_state(x_all.shape[0], device)
            seq_pearson.reset()
            recency_all = (
                recency_weights(x_all, args.since_weight, args.since_floor)
                if args.since_weight > 0
                else None
            )
            for t in range(0, SEQUENCE_LENGTH, args.chunk):
                if args.state_reset and t and t % args.state_reset == 0:
                    state = model.initial_state(x_all.shape[0], device)
                x = x_all[:, t : t + args.chunk]
                y = y_all[:, t : t + args.chunk]
                m = m_all[:, t : t + args.chunk]
                m_loss = m * recency_all[:, t : t + args.chunk] if recency_all is not None else m
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                out, state = model(x, state)
                pred = out[..., :2]
                if args.seq_loss:
                    loss = args.pearson_weight * seq_pearson.step(pred, y, m_loss) + (
                        1 - args.pearson_weight
                    ) * mse(pred, y, m_loss)
                else:
                    loss = criterion(pred, y, m_loss)
                if args.distill > 0:
                    req = (
                        step_mask[t : t + args.chunk]
                        .to(out.dtype)
                        .expand(x.shape[0], -1)
                        .unsqueeze(-1)
                    )
                    loss = loss + args.distill * (
                        (pred - t_all[:, t : t + args.chunk]) ** 2 * req
                    ).sum() / (req.sum() * 2)
                if args.aux_mask > 0:
                    # target: real mask on valid rows, predicted probability on train rows;
                    # only required (post warm-up) rows count
                    req = step_mask[t : t + args.chunk].to(out.dtype).expand(x.shape[0], -1)
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        out[..., 2], m.to(out.dtype), reduction="none"
                    )
                    loss = loss + args.aux_mask * (bce * req).sum() / req.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                ema.update(model)
                state = state.detach()
                step += 1
                if args.swa and epoch == args.epochs and step % 25 == 0:
                    with torch.no_grad():
                        cur = [p.detach().float().clone() for p in model.parameters()]
                        swa_sum = (
                            cur
                            if swa_sum is None
                            else [a + b for a, b in zip(swa_sum, cur, strict=True)]
                        )
                        swa_n += 1
                run_loss += loss.item()
                run_n += 1
                if step % args.log_every == 0:
                    elapsed = time.time() - t_epoch
                    print(
                        f"ep {epoch} batch {b + 1}/{stream.batches_per_epoch()} step {step} "
                        f"loss {run_loss / run_n:.4f} lr {lr_at(step):.2e} {elapsed:.0f}s",
                        flush=True,
                    )
                    log(epoch=epoch, step=step, loss=run_loss / run_n, lr=lr_at(step))
                    run_loss, run_n = 0.0, 0

        fit_raw = evaluate(
            model, fit.features, fit.targets, fit_scored, device, state_reset=args.state_reset
        )
        if args.final_fit:
            scores = raw = None
            current = None
            use_ema = True
            source = ema.shadow
            print(
                f"== epoch {epoch} done in {(time.time() - t_epoch) / 60:.1f} min: "
                f"fixed-epoch EMA (no validation selection) | "
                f"train-subset {fit_raw['weighted_pearson']:.4f}",
                flush=True,
            )
            log(epoch=epoch, step=step, fit_raw=fit_raw, checkpoint="fixed_epoch_ema")
        else:
            scores = evaluate(
                ema.shadow,
                val.features,
                val.targets,
                val.scored,
                device,
                state_reset=args.state_reset,
            )
            raw = evaluate(
                model, val.features, val.targets, val.scored, device, state_reset=args.state_reset
            )
            print(
                f"== epoch {epoch} done in {(time.time() - t_epoch) / 60:.1f} min: "
                f"GLOBAL ema {scores['global']:.4f} raw {raw['global']:.4f} | "
                f"per-seq ema {scores['weighted_pearson']:.4f} raw {raw['weighted_pearson']:.4f} "
                f"quarters {scores['quarters']} | train-subset {fit_raw['weighted_pearson']:.4f}",
                flush=True,
            )
            log(epoch=epoch, step=step, val_ema=scores, val_raw=raw, fit_raw=fit_raw)

            # select on the global metric, not the legacy per-sequence metric
            key = "global" if args.global_loss else "weighted_pearson"
            use_ema = scores[key] >= raw[key]
            current = max(scores[key], raw[key])
            source = ema.shadow if use_ema else model
        if args.swa and epoch == args.epochs and swa_n > 0:
            swa_model = copy.deepcopy(model)
            with torch.no_grad():
                for p, a in zip(swa_model.parameters(), swa_sum, strict=True):
                    p.copy_((a / swa_n).to(p.dtype))
            swa_scores = evaluate(
                swa_model,
                val.features,
                val.targets,
                val.scored,
                device,
                state_reset=args.state_reset,
            )
            print(
                f"   SWA over {swa_n} snapshots: val WP {swa_scores['weighted_pearson']:.4f} quarters {swa_scores['quarters']}",
                flush=True,
            )
            log(epoch=epoch, step=step, val_swa=swa_scores)
            if swa_scores[key] > current:
                current, source, use_ema = swa_scores[key], swa_model, False
        ckpt = {
            "config": cfg.to_dict(),
            "mean": mean,
            "std": std,
            "state_dict": {k: v.detach().cpu() for k, v in source.state_dict().items()},
            "state_reset": args.state_reset,
            "val_wp": current,
            "epoch": epoch,
            "ema": use_ema,
            "args": vars(args),
        }
        torch.save(ckpt, run_dir / "last.pt")
        for label, selected_model in (("ema", ema.shadow), ("raw", model)):
            fixed_ckpt = dict(ckpt)
            fixed_ckpt["state_dict"] = {
                k: v.detach().cpu() for k, v in selected_model.state_dict().items()
            }
            fixed_ckpt["ema"] = label == "ema"
            torch.save(fixed_ckpt, run_dir / f"{label}_ep{epoch}.pt")
        # best.pt/last.pt hold the *selected* weights (ema or swa), which cannot
        # continue training -- keep the raw optimiser state separately.
        torch.save(
            {
                "config": cfg.to_dict(),
                "total_steps": total_steps,
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "ema": {k: v.detach().cpu() for k, v in ema.shadow.state_dict().items()},
                "opt": opt.state_dict(),
                "swa_sum": [t.detach().cpu() for t in swa_sum] if swa_sum is not None else None,
                "swa_n": swa_n,
                "step": step,
                "epoch": epoch,
                "best": max(best, current) if current is not None else best,
                "rng": torch.get_rng_state(),
            },
            run_dir / "state.pt",
        )
        if args.final_fit or current > best:
            if current is not None:
                best = current
            torch.save(ckpt, run_dir / "best.pt")
            selected = (
                "fixed-epoch EMA"
                if args.final_fit
                else f"{'ema' if use_ema else 'raw'} WP {best:.4f}"
            )
            print(f"   saved best.pt ({selected})", flush=True)

    final_status = "fixed-epoch EMA" if args.final_fit else f"best val WP {best:.4f}"
    print(f"{final_status} -> {run_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
