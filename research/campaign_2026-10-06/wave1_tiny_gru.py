"""Tiny CPU GRU: row WMSE vs episode WMSE on far no-trade rows.

Full-sequence BPTT, one sequence in memory. Alpha of the comparison is the
loss, not the architecture. Kill bar: episode minus row >= +0.005 WP on the
eval far-notrade cell.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from wave1_anatomy import COLS, _split_bit
from wave1_episode import _far_notrade
from wave1_stretch import ROOT, _add, _wp
from wnn.gmetric import GlobalAccumulator
from wnn.splits import is_holdout

OUT = Path(__file__).resolve().parent
HIDDEN = 32
EPOCHS = 2
LR = 1e-3
MAX_FIT = 250
MAX_EVAL = 250
CLIP = 2.0
SEED = int(os.environ.get("TINY_GRU_SEED", "0"))
LOSS_MASK = os.environ.get("TINY_LOSS_MASK", "far")


class TinyGRU(nn.Module):
    def __init__(self, d_in: int, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 2)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(x)
        return self.readout(hidden)


def _inputs(cols: dict[str, np.ndarray]) -> np.ndarray:
    levels = np.column_stack([np.arcsinh(cols[name].astype(np.float32)) for name in COLS])
    dp0 = np.zeros(levels.shape[0], dtype=np.float32)
    dp11 = np.zeros(levels.shape[0], dtype=np.float32)
    dtrade = np.zeros(levels.shape[0], dtype=np.float32)
    dp0[1:] = (cols["i0_p0"][1:] != cols["i0_p0"][:-1]).astype(np.float32)
    dp11[1:] = (cols["i0_p11"][1:] != cols["i0_p11"][:-1]).astype(np.float32)
    dtrade[1:] = (
        (cols["i0_dp0"][1:] != cols["i0_dp0"][:-1]) | (cols["i0_dv0"][1:] != cols["i0_dv0"][:-1])
    ).astype(np.float32)
    return np.column_stack([levels, dp0 * dp11, dtrade]).astype(np.float32)


def _row_loss(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    clipped_y = y.clamp(-CLIP, CLIP)
    clipped_p = pred.clamp(-CLIP, CLIP)
    weight = clipped_y.abs() * mask.unsqueeze(-1).to(pred.dtype)
    return (weight * (clipped_p - clipped_y).square()).sum() / weight.sum().clamp_min(1e-8)


def _episode_loss(
    pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, last: np.ndarray
) -> torch.Tensor:
    clipped_y = y.clamp(-CLIP, CLIP)
    clipped_p = pred.clamp(-CLIP, CLIP)
    scored = mask.cpu().numpy().astype(bool)
    if not scored.any():
        return pred.sum() * 0.0
    numer = pred.new_zeros(())
    denom = pred.new_zeros(())
    ids = last[scored]
    for sid in np.unique(ids):
        take = torch.from_numpy((last == sid) & scored)
        mass = clipped_y[take].abs().sum()
        if float(mass) <= 0:
            continue
        y_bar = clipped_y[take].mean(dim=0)
        p_bar = clipped_p[take].mean(dim=0)
        numer = numer + mass * (p_bar - y_bar).square().sum()
        denom = denom + mass
    return numer / denom.clamp_min(1e-8)


def _score_model(
    model: TinyGRU, seqs: list[int], y_all: np.ndarray, mask_all: np.ndarray, row_group: np.ndarray, pf: pq.ParquetFile
) -> dict[str, dict[str, float]]:
    model.eval()
    far_acc = GlobalAccumulator()
    all_acc = GlobalAccumulator()
    with torch.no_grad():
        for group in seqs:
            table = pf.read_row_group(int(row_group[group]), columns=list(COLS), use_threads=False)
            cols = {name: table.column(name).to_numpy() for name in COLS}
            x = torch.from_numpy(_inputs(cols)).unsqueeze(0)
            pred = model(x).squeeze(0).numpy().astype(np.float32)
            y = y_all[group]
            mask = mask_all[group]
            far, _last = _far_notrade(cols, y)
            _add(all_acc, y, pred, mask)
            _add(far_acc, y, pred, mask & far)
    return {"all": _wp(all_acc), "far_notrade": _wp(far_acc)}


def main() -> None:
    started = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cache = ROOT / "cache"
    with np.load(cache / "valid_targets.npz") as blob:
        y_all = np.asarray(blob["y"])
        mask_all = np.asarray(blob["mask"], dtype=bool)
        row_group = np.asarray(blob["row_group"])
        seq_ix = np.asarray(blob["seq_ix"])
    pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    fit, eval_ids = [], []
    for group in range(y_all.shape[0]):
        seq = int(seq_ix[group])
        if is_holdout(seq):
            continue
        if _split_bit(seq, "wave1b") == 0:
            if len(fit) < MAX_FIT:
                fit.append(group)
        elif len(eval_ids) < MAX_EVAL:
            eval_ids.append(group)

    d_in = len(COLS) + 2
    payload: dict[str, object] = {
        "hidden": HIDDEN,
        "epochs": EPOCHS,
        "fit_sequences": len(fit),
        "eval_sequences": len(eval_ids),
        "seed": SEED,
        "loss_mask": LOSS_MASK,
    }
    scores = {}
    for name, loss_fn in (("row", "row"), ("episode", "episode")):
        torch.manual_seed(SEED)
        model = TinyGRU(d_in, HIDDEN)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        model.train()
        for epoch in range(EPOCHS):
            total = 0.0
            n_steps = 0
            for group in fit:
                table = pf.read_row_group(int(row_group[group]), columns=list(COLS), use_threads=False)
                cols = {name: table.column(name).to_numpy() for name in COLS}
                x = torch.from_numpy(_inputs(cols)).unsqueeze(0)
                y = torch.from_numpy(y_all[group].astype(np.float32))
                mask = torch.from_numpy(mask_all[group])
                pred = model(x).squeeze(0)
                far, last = _far_notrade(cols, y_all[group])
                far_mask = mask & torch.from_numpy(far)
                if LOSS_MASK == "mixed" and loss_fn == "episode":
                    rest = mask & ~torch.from_numpy(far)
                    loss = _row_loss(pred, y, rest) + _episode_loss(pred, y, far_mask, last)
                elif LOSS_MASK == "all" or LOSS_MASK == "mixed":
                    loss = (
                        _row_loss(pred, y, mask)
                        if loss_fn == "row"
                        else _episode_loss(pred, y, mask, last)
                    )
                else:
                    loss = (
                        _row_loss(pred, y, far_mask)
                        if loss_fn == "row"
                        else _episode_loss(pred, y, far_mask, last)
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach())
                n_steps += 1
            print(f"{name} epoch {epoch} loss {total / max(n_steps, 1):.6f}", flush=True)
        scores[name] = _score_model(model, eval_ids, y_all, mask_all, row_group, pf)
        del model, opt
    payload["eval"] = scores
    payload["delta_episode_vs_row_far"] = (
        scores["episode"]["far_notrade"]["wp"] - scores["row"]["far_notrade"]["wp"]
    )
    payload["seconds"] = round(time.time() - started, 3)
    text = json.dumps(payload, indent=2) + "\n"
    tag = f"s{SEED}" if LOSS_MASK == "far" else f"{LOSS_MASK}_s{SEED}"
    (OUT / f"wave1_tiny_gru_{tag}.json").write_text(text, encoding="utf-8")
    if LOSS_MASK == "far" and SEED == 0:
        (OUT / "wave1_tiny_gru.json").write_text(text, encoding="utf-8")
    print(
        "TINY_GRU_DONE",
        f"row_far={scores['row']['far_notrade']['wp']:.6f}",
        f"ep_far={scores['episode']['far_notrade']['wp']:.6f}",
        f"delta={payload['delta_episode_vs_row_far']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
