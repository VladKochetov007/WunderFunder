from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from model import StructuredStateModel

OWN_DIR = Path(__file__).resolve().parent
REPO_ROOT = OWN_DIR.parents[2]
LOG = logging.getLogger(__name__)
VRAM_CAP = 5_000_000_000


def sha256(path: Path) -> str:
    """Hash exact source/checkpoint bytes for the controller's durable ledger."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_vram(samples: list[dict[str, object]], stage: str) -> None:
    """Sample total device bytes; root's continuous monitor remains mandatory."""
    torch.cuda.synchronize()
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    device_bytes = [int(value.strip()) * 2**20 for value in output.splitlines()]
    samples.append(
        {
            "stage": stage,
            "device_bytes": device_bytes,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
    )
    if max(device_bytes) > VRAM_CAP:
        raise RuntimeError(f"total device VRAM exceeds {VRAM_CAP} bytes at {stage}")


def main() -> None:
    """Profile batch32/chunk2000 synthetic student+EMA; root alone launches GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decay-init", choices=["multi", "single"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_path = args.output.with_suffix(".pt")
    if args.output.exists() or checkpoint_path.exists():
        raise FileExistsError("refusing to overwrite synthetic GPU profile evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from train_rnn import FastEMA, detach_state, forward_chunk

    from wnn.nn import weighted_mse_loss
    from wnn.resources import configure_torch

    started = time.monotonic()
    samples: list[dict[str, object]] = []
    result: dict[str, object] = {
        "synthetic_only": True,
        "competition_data_read": False,
        "batch": 32,
        "sequence_length": 20_000,
        "chunk": 2000,
        "n_in": 211,
        "n_out": 23,
        "seed": 3,
        "decay_init": args.decay_init,
        "precision": "outer fp16 autocast + GradScaler; model FP32 throughout",
        "teacher": "actual trainer FastEMA, consistency1.0, decay0.999",
        "aux_struct": 0.3,
        "aux_next": 0.3,
        "source_sha256": {
            path.name: sha256(path)
            for path in [
                OWN_DIR / "model.py",
                Path(__file__),
                OWN_DIR / "gpu_smoke_preregistration.md",
                REPO_ROOT / "scripts/train_rnn.py",
            ]
        },
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda_version": torch.version.cuda,
        "scope": os.environ.get("WNN_SCOPE_UNIT"),
        "total_device_cap_bytes": VRAM_CAP,
    }
    try:
        torch.manual_seed(3)
        configure_torch(gpu_fraction=0.70)
        device = torch.device("cuda")
        model = StructuredStateModel(211, 23, args.decay_init).to(device)
        teacher = FastEMA(model, 0.999)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2, fused=True)
        scaler = torch.amp.GradScaler("cuda")
        raw = torch.randn(32, 20_000, 112, device=device)
        ev = torch.randn(32, 20_000, 99, device=device)
        inputs = torch.cat([raw, ev], dim=-1)
        del ev
        y = torch.randn(32, 20_000, 2, device=device).clamp(-2, 2)
        weight = torch.rand(32, 20_000, device=device)
        weight[:, :99] = 0
        labels = torch.randint(0, 12, (32, 20_000, 3), device=device)
        labels[..., 1] %= 3
        labels[..., 2] %= 4
        next_rows = torch.log1p(torch.rand(32, 20_000, device=device) * 1000)
        next_up = torch.rand(32, 20_000, device=device)
        student_state = teacher_state = None
        result["parameters"] = sum(p.numel() for p in model.parameters())
        result["raw_resident_shape"] = list(raw.shape)
        sample_vram(samples, "resident_inputs_student_teacher_optimizer")
        forward_times: list[float] = []
        backward_times: list[float] = []
        for start in range(0, 20_000, 2000):
            section = slice(start, start + 2000)
            optimizer.zero_grad(set_to_none=True)
            mark = time.monotonic()
            with torch.autocast("cuda", dtype=torch.float16):
                out, student_state = forward_chunk(model, inputs, section, student_state)
                student_state = detach_state(student_state)
                with torch.no_grad():
                    target, teacher_state = forward_chunk(
                        teacher.module, inputs, section, teacher_state
                    )
                teacher_state = detach_state(teacher_state)
            torch.cuda.synchronize()
            forward_times.append(time.monotonic() - mark)
            if out.dtype != torch.float32 or any(s.dtype != torch.float32 for s in student_state):
                raise AssertionError("FP32 output/state policy violated")
            sample_vram(samples, f"chunk{start // 2000}:student_and_teacher_forward")
            w = weight[:, section]
            loss = weighted_mse_loss(out[..., :2], y[:, section], w)
            loss = loss + (
                (out[..., :2] - target[..., :2]).square() * w[..., None]
            ).sum() / w.sum().clamp_min(1e-8)
            for index, (lo, hi) in enumerate([(2, 14), (14, 17), (17, 21)]):
                cross_entropy = F.cross_entropy(
                    out[..., lo:hi].reshape(-1, hi - lo),
                    labels[:, section, index].reshape(-1),
                    reduction="none",
                )
                loss = loss + 0.3 * (cross_entropy * w.reshape(-1)).sum() / w.sum().clamp_min(1e-8)
            next_loss = (out[..., 21] - (next_rows[:, section] - 3) / 2).square()
            next_loss += F.binary_cross_entropy_with_logits(
                out[..., 22], next_up[:, section], reduction="none"
            )
            loss = loss + 0.3 * (next_loss * w).sum() / w.sum().clamp_min(1e-8)
            mark = time.monotonic()
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            backward_times.append(time.monotonic() - mark)
            sample_vram(samples, f"chunk{start // 2000}:backward")
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise AssertionError("nonfinite synthetic gradient")
            scaler.step(optimizer)
            scaler.update()
            teacher.update_parameters(model)
            sample_vram(samples, f"chunk{start // 2000}:optimizer_and_ema")
            LOG.info("chunk%s loss%s", start // 2000, float(loss.detach().cpu()))
        updates = int(optimizer.state[next(model.parameters())]["step"].item())
        if updates != 10 or teacher.n != 10:
            raise AssertionError("synthetic smoke skipped optimizer/EMA updates")
        result.update(
            {
                "passed": True,
                "optimizer_updates": updates,
                "ema_updates": teacher.n,
                "loss": float(loss.detach().cpu()),
                "forward_and_teacher_seconds": forward_times,
                "backward_seconds": backward_times,
                "state_shapes": [list(s.shape) for s in student_state],
                "teacher_state_shapes": [list(s.shape) for s in teacher_state],
            }
        )
        torch.save(
            {
                "model": model.state_dict(),
                "n_in": 211,
                "n_out": 23,
                "decay_init": args.decay_init,
                "synthetic_only": True,
                "source_sha256": result["source_sha256"],
                "optimizer_updates": updates,
                "seed": 3,
            },
            checkpoint_path,
        )
        result["checkpoint_sha256"] = sha256(checkpoint_path)
    except (AssertionError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        result.update({"passed": False, "failure": str(error)})
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["vram_samples"] = samples
        if torch.cuda.is_initialized():
            result["allocator_peak_bytes"] = torch.cuda.max_memory_allocated()
            result["allocator_reserved_peak_bytes"] = torch.cuda.max_memory_reserved()
        result["limitation"] = (
            "Stage samples can miss transient total-device peak; root's monitor required. Synthetic core profile excludes data-loader/features intermediates."
        )
        with args.output.open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
        LOG.info("Result written to %s", args.output)


if __name__ == "__main__":
    main()
