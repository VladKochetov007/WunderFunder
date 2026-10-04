"""Measure historical batch-32 training memory before clean parent fitting."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import torch

from wnn.features import N_EVENT_FEATURES
from wnn.nn import RNNModel, weighted_mse_loss
from wnn.resources import configure_torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run one complete synthetic sequence batch with historical chunking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/campaign_2026-09-29/artifacts/gpu_batch32_smoke.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"smoke evidence already exists: {args.output}")
    started = time.monotonic()
    torch.manual_seed(3)
    configure_torch()
    device = torch.device("cuda")
    parent = torch.load(
        ROOT / "artifacts/runs/ab62_since_f0/model_ep3.pt", map_location="cpu", weights_only=False
    )
    n_out = len(parent["model"]["head.2.bias"])
    n_in = 112 + N_EVENT_FEATURES
    model = RNNModel(n_in, hidden=128, layers=2, n_out=n_out).to(device)
    teacher = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(32, 20_000, n_in, device=device)
    y = torch.randn(32, 20_000, 2, device=device).clamp(-2, 2)
    mask = torch.ones(32, 20_000, dtype=torch.bool, device=device)
    mask[:, :99] = False
    hidden = teacher_hidden = None
    for start in range(0, 20_000, 2000):
        section = slice(start, start + 2000)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out, hidden = model(x[:, section], hidden)
            with torch.no_grad():
                reference, teacher_hidden = teacher(x[:, section], teacher_hidden)
        hidden = hidden.detach()
        teacher_hidden = teacher_hidden.detach()
        loss = weighted_mse_loss(out[..., :2], y[:, section], mask[:, section])
        loss = loss + (out[..., :2].float() - reference[..., :2].float()).square().mean()
        loss = loss + 0.3 * out[..., 2:].float().square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    result = {
        "batch": 32,
        "chunk": 2000,
        "sequence_length": 20_000,
        "n_in": n_in,
        "n_out": n_out,
        "seed": 3,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
        "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
        "wall_seconds": time.monotonic() - started,
        "loss": float(loss.detach().cpu()),
        "limitation": "Synthetic feature batch; real trainer total-device cap remains mandatory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
