"""Bounded synthetic optimizer screen: AdamW vs Muon variants on a tiny GRU.

Exploratory mechanism probe only: a synthetic causal sequence task that mimics
the competition's structure (chunked per-batch-normalized weighted MSE, targets
clipped to +-2, magnitude weights, metric-style per-channel weighted Pearson).
It is NOT competition-quality evidence; it only justifies (or not) a follow-up
on real cached data / Kaggle GPU.

Self-contained: imports only torch/numpy/stdlib. Strict FP32, single thread.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_float32_matmul_precision("highest")

# Task constants
N_SEQ = 48
SEQ_LEN = 1500
N_GROUPS = 4
FIELDS_PER_GROUP = 6
N_AR = N_GROUPS * FIELDS_PER_GROUP  # 24 smooth AR(1) fields
N_TICK = 4  # event tick channels (decaying one-hot bursts)
N_CLOCK = 1  # normalized row position
N_INTERACT = 3  # slow cross-group interaction features
INPUT_DIM = N_AR + N_TICK + N_CLOCK + N_INTERACT  # 32
N_CHAN = 4
TRAIN_FRAC = 0.7
DATA_SEED = 1234
CHAN_LAGS = (1, 4, 8, 16)

# Model/training constants
HIDDEN = 64
CHUNK = 192
BATCH_SEQS = 8
WARMUP = 50
BASE_LR = 3e-3
ADAM_BETAS = (0.9, 0.95)
ADAM_WD = 1e-2
MUON_MOMENTUM = 0.95
MUON_WD = 1e-2
NS_AGGRESSIVE = (3.4445, -4.7750, 2.0315)  # standard Muon quintic
NS_STABILIZING = (1.5, 0.0, -0.5)  # p(s)=1.5s-0.5s^3: p(1)=1, p'(1)=0
MUON_RMS = 0.2  # Kimi/Moonlight RMS-matching scale factor


def make_data() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic causal task; one fixed data seed for all arms."""
    rng = np.random.default_rng(DATA_SEED)
    x = np.zeros((N_SEQ, SEQ_LEN, INPUT_DIM), dtype=np.float32)
    rhos = rng.uniform(0.90, 0.99, size=(N_GROUPS, FIELDS_PER_GROUP))
    sigmas = rng.uniform(0.03, 0.08, size=(N_GROUPS, FIELDS_PER_GROUP))
    ar = np.zeros((N_SEQ, SEQ_LEN, N_AR), dtype=np.float32)
    for g in range(N_GROUPS):
        for f in range(FIELDS_PER_GROUP):
            eps = rng.normal(0.0, sigmas[g, f], size=(N_SEQ, SEQ_LEN))
            idx = g * FIELDS_PER_GROUP + f
            for t in range(1, SEQ_LEN):
                ar[:, t, idx] = rhos[g, f] * ar[:, t - 1, idx] + eps[:, t]
    x[:, :, :N_AR] = ar
    # Event ticks: rare bursts that decay; tick activity feeds target channels.
    ticks = np.zeros((N_SEQ, SEQ_LEN, N_TICK), dtype=np.float32)
    for t in range(1, SEQ_LEN):
        ticks[:, t] = 0.5 * ticks[:, t - 1]
        fire = rng.random((N_SEQ, N_TICK)) < 0.01
        ticks[:, t] += fire.astype(np.float32)
    x[:, :, N_AR : N_AR + N_TICK] = ticks
    # Normalized row clock (position within sequence).
    x[:, :, N_AR + N_TICK] = np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float32)[None, :]
    # Slow interaction features: products of group means at lag 5.
    gmeans = ar.reshape(N_SEQ, SEQ_LEN, N_GROUPS, FIELDS_PER_GROUP).mean(axis=3)
    for k in range(N_INTERACT):
        a, b = k, (k + 1) % N_GROUPS
        x[:, 5:, N_AR + N_TICK + 1 + k] = (gmeans[:, 5:, a] * gmeans[:, 5:, b]).astype(
            np.float32
        )
    # Targets: bounded nonlinear functions with different lags per channel.
    y = np.zeros((N_SEQ, SEQ_LEN, N_CHAN), dtype=np.float32)

    def lagged(v: np.ndarray, lag: int) -> np.ndarray:
        head = np.repeat(v[:, :1, :], lag, axis=1)
        return np.concatenate([head, v[:, : SEQ_LEN - lag, :]], axis=1)

    for c in range(N_CHAN):
        g_lag = lagged(gmeans[:, :, c : c + 1], CHAN_LAGS[c])[:, :, 0]
        other = lagged(gmeans[:, :, ((c + 1) % N_GROUPS) : ((c + 1) % N_GROUPS) + 1], 8)[
            :, :, 0
        ]
        noise = rng.normal(0.0, 0.25, size=(N_SEQ, SEQ_LEN))
        raw = (
            1.5 * np.tanh(2.2 * g_lag + 0.9 * ticks[:, :, c])
            + 0.3 * np.tanh(other) * g_lag
            + noise
        )
        y[:, :, c] = np.clip(raw, -2.0, 2.0).astype(np.float32)
    return x, y


def standardize(x: np.ndarray, train_end: int) -> np.ndarray:
    flat = x[:, :train_end].reshape(-1, x.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def weighted_mse(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Per-batch-normalized weighted MSE (batch weights sum to 1)."""
    return (w * (pred - y) ** 2).sum() / w.sum().clamp_min(1e-8)


def weighted_pearson(pred: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Metric-style per-channel weighted Pearson, then mean over channels."""
    scores = []
    pc = np.clip(pred, -2.0, 2.0)
    zy = np.clip(y, -2.0, 2.0)
    for c in range(N_CHAN):
        wc = w[:, :, c]
        if wc.sum() < 1e-8:
            continue
        ym = np.average(zy[:, :, c], weights=wc)
        pm = np.average(pc[:, :, c], weights=wc)
        cov = np.average((pc[:, :, c] - pm) * (zy[:, :, c] - ym), weights=wc)
        vp = np.average((pc[:, :, c] - pm) ** 2, weights=wc)
        vy = np.average((zy[:, :, c] - ym) ** 2, weights=wc)
        if vp < 1e-12 or vy < 1e-12:
            continue
        scores.append(cov / math.sqrt(vp * vy))
    return float(np.mean(scores)) if scores else float("nan")


def zeropower(g: torch.Tensor, two_stage: bool = False) -> torch.Tensor:
    """Newton-Schulz orthogonalization in pure FP32 (5 iterations).

    Standard Muon: 5 aggressive quintic iterations. Two-stage variant: 3
    aggressive, then 2 stabilizing (p(s)=1.5s-0.5s^3 keeps singular values at
    1 with zero derivative there). Operates on the wide side of the matrix.
    """
    transposed = False
    x = g.clone()
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    x = x / (x.norm() + 1e-7)
    for i in range(5):
        a, b, c = NS_STABILIZING if (two_stage and i >= 3) else NS_AGGRESSIVE
        t1 = x
        t2 = x @ x.T @ x
        t3 = t2 @ x.T @ x
        x = a * t1 + b * t2 + c * t3
    return x.T if transposed else x


class MuonState:
    """Minimal Muon for 2D parameters: heavy-ball momentum + Nesterov view,
    Newton-Schulz orthogonalization, 0.2*sqrt(max(m,n)) RMS-matching scale,
    decoupled weight decay (documented design choice)."""

    def __init__(self, params: list[torch.nn.Parameter], two_stage: bool):
        self.params = params
        self.buf = [torch.zeros_like(p) for p in params]
        self.two_stage = two_stage

    def step(self, lr: float) -> list[torch.Tensor]:
        updates = []
        for p, buf in zip(self.params, self.buf):
            if p.grad is None:
                updates.append(torch.zeros_like(p))
                continue
            buf.mul_(MUON_MOMENTUM).add_(p.grad)
            v = p.grad + MUON_MOMENTUM * buf
            scale = MUON_RMS * math.sqrt(max(p.size(0), p.size(1)))
            u = scale * zeropower(v, two_stage=self.two_stage)
            with torch.no_grad():
                p.add_(-lr * (u + MUON_WD * p))
            updates.append(u.detach())
        return updates


class AdamWState:
    """Minimal AdamW (bias-corrected) used standalone or on Muon leftovers."""

    def __init__(self, params: list[torch.nn.Parameter]):
        self.params = params
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    def step(self, lr: float) -> list[torch.Tensor]:
        self.t += 1
        updates = []
        b1, b2 = ADAM_BETAS
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                updates.append(torch.zeros_like(p))
                continue
            m.mul_(b1).add_(p.grad, alpha=1 - b1)
            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            u = (m / (1 - b1**self.t)) / ((v / (1 - b2**self.t)).sqrt() + 1e-8)
            with torch.no_grad():
                p.add_(-lr * (u + ADAM_WD * p))
            updates.append(u.detach())
        return updates


def lr_at(step: int, total: int) -> float:
    if step < WARMUP:
        return BASE_LR * (step + 1) / WARMUP
    prog = (step - WARMUP) / max(1, total - WARMUP)
    return BASE_LR * (0.1 + 0.45 * (1 + math.cos(math.pi * prog)))


def effective_rank(m: torch.Tensor) -> float:
    s = torch.linalg.svdvals(m.detach().float())
    p = s / s.sum().clamp_min(1e-12)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum().item()
    return math.exp(entropy) / min(m.size(0), m.size(1))


def evaluate(model, head, x_t, y_np, w_np, train_end):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(N_SEQ):
            h = torch.zeros(2, 1, HIDDEN)
            seq_p = []
            for start in range(0, SEQ_LEN, CHUNK):
                out, h = model(x_t[s : s + 1, start : start + CHUNK], h)
                seq_p.append(head(out)[0])
            preds.append(torch.cat(seq_p))
    pred = torch.stack(preds).numpy()
    val = (slice(None), slice(train_end, None))
    vp, vy, vw = pred[val], y_np[val], w_np[val]
    wmse = float((vw * (vp - vy) ** 2).sum() / vw.sum())
    return weighted_pearson(vp, vy, vw), wmse


def run(optim: str, seed: int, steps: int, out_path: Path) -> dict:
    t0 = time.time()
    torch.manual_seed(seed)
    x, y = make_data()
    train_end = int(SEQ_LEN * TRAIN_FRAC)
    x = standardize(x, train_end)
    w = np.abs(y)
    x_t = torch.from_numpy(x)

    model = torch.nn.GRU(INPUT_DIM, HIDDEN, num_layers=2, batch_first=True)
    head = torch.nn.Linear(HIDDEN, N_CHAN)
    named = list(model.named_parameters()) + [
        ("head.weight", head.weight),
        ("head.bias", head.bias),
    ]
    all_params = [p for _, p in named]

    muon_params, adamw_params, muon_names = [], [], []
    for name, p in named:
        is_2d = p.ndim == 2
        take_muon = (optim == "muon" and is_2d) or (
            optim == "muon_hybrid" and "weight_hh" in name
        ) or (optim == "muon_ns2stage" and is_2d)
        if take_muon:
            muon_params.append(p)
            muon_names.append(name)
        else:
            adamw_params.append(p)
    muon = MuonState(muon_params, two_stage=(optim == "muon_ns2stage"))
    adamw = AdamWState(adamw_params)

    seq_ids = list(range(BATCH_SEQS))
    seq_pos = [0] * BATCH_SEQS
    cursor = BATCH_SEQS
    states = [torch.zeros(2, 1, HIDDEN) for _ in range(BATCH_SEQS)]
    hh_param = dict(model.named_parameters())["weight_hh_l0"]
    last_hh_update = None
    last_hh_momentum = None
    curve = []

    model.train()
    for step in range(steps):
        lr = lr_at(step, steps)
        batch = []
        for slot, sid in enumerate(seq_ids):
            start = seq_pos[slot]
            if start == 0:
                states[slot] = torch.zeros(2, 1, HIDDEN)
            end = min(start + CHUNK, train_end)
            batch.append((slot, sid, start, end))
            if end >= train_end:
                seq_ids[slot] = cursor % N_SEQ
                cursor += 1
                seq_pos[slot] = 0
            else:
                seq_pos[slot] = end
        model.zero_grad(set_to_none=True)
        total_loss = 0.0
        carried = []
        for slot, sid, start, end in batch:
            xc = x_t[sid : sid + 1, start:end]
            yc = torch.from_numpy(y[sid : sid + 1, start:end])
            wc = torch.abs(yc)
            out, hn = model(xc, states[slot].detach())
            loss = weighted_mse(head(out), yc, wc) / len(batch)
            loss.backward()
            total_loss += loss.item()
            carried.append((slot, hn.detach()))
        for slot, hn in carried:
            states[slot] = hn
        if muon.params:
            updates = muon.step(lr)
            idx = next((i for i, p in enumerate(muon.params) if p is hh_param), None)
            if idx is not None:
                last_hh_update = updates[idx].clone()
                last_hh_momentum = muon.buf[idx].clone()
        if adamw.params:
            ups = adamw.step(lr)
            idx = next((i for i, p in enumerate(adamw.params) if p is hh_param), None)
            if idx is not None:
                last_hh_update = ups[idx].clone()
                last_hh_momentum = adamw.m[idx].clone()
        if (step + 1) % 100 == 0 or step == steps - 1:
            wp, wv = evaluate(model, head, x_t, y, w, train_end)
            curve.append(
                {
                    "step": step + 1,
                    "train_loss": total_loss,
                    "val_wpearson": wp,
                    "val_wmse": wv,
                }
            )

    wall = time.time() - t0
    wp, wv = evaluate(model, head, x_t, y, w, train_end)
    result = {
        "optim": optim,
        "seed": seed,
        "steps": steps,
        "final_val_wpearson": wp,
        "final_val_wmse": wv,
        "best_val_wpearson": max(c["val_wpearson"] for c in curve),
        "curve": curve,
        "wall_seconds": round(wall, 2),
        "erank_weight_hh_l0": effective_rank(hh_param),
        "erank_last_update_hh": (
            effective_rank(last_hh_update) if last_hh_update is not None else None
        ),
        "erank_last_momentum_hh": (
            effective_rank(last_hh_momentum) if last_hh_momentum is not None else None
        ),
        "muon_param_names": muon_names,
        "defaults": {
            "base_lr": BASE_LR,
            "adam_betas": list(ADAM_BETAS),
            "adam_wd": ADAM_WD,
            "muon_momentum": MUON_MOMENTUM,
            "muon_wd": MUON_WD,
            "ns_aggressive": list(NS_AGGRESSIVE),
            "ns_stabilizing": list(NS_STABILIZING),
            "muon_rms": MUON_RMS,
            "chunk": CHUNK,
            "batch_seqs": BATCH_SEQS,
            "warmup": WARMUP,
            "data_seed": DATA_SEED,
        },
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(
        f"{optim} seed{seed}: wp={wp:.4f} wmse={wv:.5f} wall={wall:.1f}s "
        f"erank[W]={result['erank_weight_hh_l0']:.3f} "
        f"erank[upd]={result['erank_last_update_hh']}"
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--optim", required=True, choices=["adamw", "muon", "muon_hybrid", "muon_ns2stage"]
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(args.optim, args.seed, args.steps, out_path)


if __name__ == "__main__":
    main()
