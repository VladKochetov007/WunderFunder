from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from wnn.optim import MuonAdamW  # noqa: E402


def _run(mode: str) -> dict[str, object]:
    """Replay identical synthetic gradients through the actual CPU optimizer."""
    generator = torch.Generator(device="cpu").manual_seed(1701)
    recurrent = torch.nn.Parameter(torch.randn(24, 8, generator=generator) * 0.01)
    head = torch.nn.Parameter(torch.randn(2, 8, generator=generator) * 0.01)
    optimizer = MuonAdamW(
        [{"params": [recurrent], "muon": True, "blocks": 3}, {"params": [head]}],
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        strict_fp32=True,
    )
    recurrent_updates, head_updates, factors = [], [], []
    for step in range(24):
        recurrent.grad = torch.randn(recurrent.shape, generator=generator) * 0.15
        head.grad = torch.randn(head.shape, generator=generator) * (4.0 if step % 3 == 0 else 0.15)
        if mode == "constant_scale_0.1":
            recurrent.grad.mul_(0.1)
            head.grad.mul_(0.1)
            factor = 0.1
        elif mode.startswith("global_clip_"):
            threshold = float(mode.rsplit("_", 1)[1])
            norm = torch.nn.utils.clip_grad_norm_([recurrent, head], threshold)
            factor = min(1.0, threshold / (float(norm) + 1e-6))
        else:
            factor = 1.0
        old_recurrent, old_head = recurrent.detach().clone(), head.detach().clone()
        optimizer.step()
        recurrent_updates.append((recurrent.detach() - old_recurrent).flatten().clone())
        head_updates.append((head.detach() - old_head).flatten().clone())
        factors.append(factor)
    return {
        "recurrent_updates": torch.stack(recurrent_updates),
        "head_updates": torch.stack(head_updates),
        "factors": factors,
    }


def _compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Summarize update lengths and directions, without interpreting model quality."""
    ratios = candidate.norm(dim=1) / reference.norm(dim=1)
    cosine = torch.nn.functional.cosine_similarity(reference, candidate, dim=1).clamp(-1, 1)
    angles = torch.acos(cosine) * (180.0 / math.pi)
    return {
        "median_update_norm_ratio": float(ratios.median()),
        "mean_update_angle_degrees": float(angles.mean()),
        "maximum_update_angle_degrees": float(angles.max()),
        "relative_trajectory_difference": float((candidate - reference).norm() / reference.norm()),
    }


def main() -> None:
    """Save an exact-code mechanism illustration; never fit or evaluate competition data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    baseline = _run("unclipped")
    comparisons: dict[str, object] = {}
    for mode in ("constant_scale_0.1", "global_clip_1", "global_clip_5"):
        candidate = _run(mode)
        comparisons[mode] = {
            "mean_raw_gradient_multiplier": sum(candidate["factors"]) / 24,
            "muon_recurrent": _compare(
                baseline["recurrent_updates"], candidate["recurrent_updates"]
            ),
            "adamw_head": _compare(baseline["head_updates"], candidate["head_updates"]),
        }
    result = {
        "schema": "wnn.synthetic-clipping-mechanism.v1",
        "evidence_scope": "CPU synthetic gradient replay, not a training or quality result",
        "device": "cpu",
        "torch": torch.__version__,
        "steps": 24,
        "seed": 1701,
        "weight_decay": 0,
        "source_sha256": hashlib.sha256(
            (SOURCE_ROOT / "src/wnn/optim.py").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparisons": comparisons,
        "sgd_bias_counterexample": {"raw_batch_gradients": [8, -2], "raw_mean": 3, "clip1_mean": 0},
        "limitations": [
            "Tiny matrices and prescribed gradients do not reproduce GRU training or data noise.",
            "Fixed momentum and LR omit OneCycle momentum/LR variation.",
            "Near scale-invariance depends on numerical epsilon, finite precision and optimizer state.",
            "Repeated gradients are shared inputs, not independent training replications.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
