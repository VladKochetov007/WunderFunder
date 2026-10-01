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
from model import ARMS, ENDPOINT, EventContentModel

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
    """Profile three D1 arms with synthetic student+EMA; root alone launches GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
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
        "arm": args.arm,
        "precision": "outer fp16 autocast + GradScaler; S/Z FP32; loss readouts FP32",
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
        model = EventContentModel(211, 23, args.arm).to(device)
        teacher = FastEMA(model, 0.999)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2, fused=True)
        scaler = torch.amp.GradScaler("cuda")
        raw = torch.randn(32, 20_000, 112, device=device)
        # Same frozen component-clock workload for all arms; no competition data.
        row = torch.arange(20_000, device=device)
        batch = torch.arange(32, device=device)
        price_state = ((row // 8) % 17 - 8).float()
        raw[..., :22] = price_state[None, :, None] * torch.linspace(0.01, 0.22, 22, device=device)
        endpoint_gate = ((row % 16 == 0) | (row % 113 == 0))[None].expand(32, -1).clone()
        endpoint_gate[:, 0] = batch % 3 == 0
        raw[..., 110] = torch.where(endpoint_gate, ENDPOINT, (row % 16).float()[None] * 0.05)
        from wnn.features import event_features_torch

        ev = event_features_torch(raw)
        if ev.shape[-1] != 99:
            raise AssertionError("synthetic feature grammar changed")
        inputs = torch.cat([raw, ev], dim=-1)
        del ev
        code_gate = (raw[:, 1:, :22] != raw[:, :-1, :22]).any(-1)
        result["synthetic_clock_counts"] = {
            "endpoint": int(endpoint_gate.sum().item()),
            "raw_code_nonorigin": int(code_gate.sum().item()),
            "endpoint_nonorigin_samecode": int((endpoint_gate[:, 1:] & ~code_gate).sum().item()),
            "rawcode_nonendpoint": int((code_gate & ~endpoint_gate[:, 1:]).sum().item()),
            "endpoint_origins": int(endpoint_gate[:, 0].sum().item()),
        }
        del row, batch, price_state, endpoint_gate, code_gate
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
        attempts, skips = 0, 0
        update_records = []
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
            if student_state[1].dtype != torch.float32 or student_state[2].dtype != torch.float32:
                raise AssertionError("S/Z FP32 state policy violated")
            if (
                student_state[3].dtype != torch.int64
                or student_state[4].dtype != torch.int64
                or student_state[5].dtype != torch.float32
            ):
                raise AssertionError("counter/rawprice state grammar violated")
            if not torch.isfinite(out).all() or not torch.isfinite(target).all():
                raise AssertionError("nonfinite synthetic output")
            # Trainer loss reads cast AMP outputs to FP32, matching all aux terms.
            out, target = out.float(), target.float()
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
            attempts += 1
            parameter = next(model.parameters())
            before = int(optimizer.state.get(parameter, {}).get("step", 0))
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            teacher.update_parameters(model)
            after = int(optimizer.state[parameter]["step"].item())
            skipped = after == before
            skips += int(skipped)
            update_records.append(
                {
                    "attempt": attempts,
                    "before_step": before,
                    "after_step": after,
                    "skipped": skipped,
                    "scale_before": scale_before,
                    "scale_after": scaler.get_scale(),
                    "ema_step": teacher.n,
                }
            )
            sample_vram(samples, f"chunk{start // 2000}:optimizer_and_ema")
            LOG.info("chunk%s loss%s", start // 2000, float(loss.detach().cpu()))
        updates = int(optimizer.state[next(model.parameters())]["step"].item())
        if updates != 10 or teacher.n != 10:
            raise AssertionError("synthetic smoke skipped optimizer/EMA updates")
        result.update(
            {
                "passed": True,
                "optimizer_updates": updates,
                "optimizer_attempts": attempts,
                "amp_skips": skips,
                "update_records": update_records,
                "ema_updates": teacher.n,
                "loss": float(loss.detach().cpu()),
                "forward_and_teacher_seconds": forward_times,
                "backward_seconds": backward_times,
                "state_shapes": [list(s.shape) for s in student_state],
                "teacher_state_shapes": [list(s.shape) for s in teacher_state],
            }
        )
        # Final frozen student weights, explicit FP32 reset, saved input: no RNG ambiguity.
        backend_flags = {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        }
        model.eval()
        parity_inputs = inputs[:1, :2000].detach().clone().float()
        primary_flags = backend_flags
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            parity_output, parity_state = model(parity_inputs, None)
        if parity_output.dtype != torch.float32 or any(
            s.dtype != torch.float32
            for s in (parity_state[0], parity_state[1], parity_state[2], parity_state[5])
        ):
            raise AssertionError("final parity probe is not FP32")
        sample_vram(samples, "final_frozen_student_fp32_parity_reset")
        parity_tolerances = {
            "output_max_absolute": 3e-5,
            "floating_state_max_absolute": 0.003,
            "integer_counts_and_previous_rawprices_exact": True,
        }
        result["parity_artifact"] = {
            "input_shape": list(parity_inputs.shape),
            "reset": True,
            "final_student": True,
            "precision": "FP32; outerautocast disabled",
            "backend_flags": primary_flags,
            "tolerances": parity_tolerances,
        }
        # AMP reset diagnostic uses the same inherited training precision policy.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            amp_output, amp_state = model(parity_inputs, None)
        if not all(torch.isfinite(t).all() for t in (amp_output, *amp_state)):
            raise AssertionError("nonfinite saved parity output/state")
        if amp_state[1].dtype != torch.float32 or amp_state[2].dtype != torch.float32:
            raise AssertionError("AMP saved parity S/Z not FP32")
        if (
            not torch.equal(amp_state[3], parity_state[3])
            or not torch.equal(amp_state[4], parity_state[4])
            or not torch.equal(amp_state[5], parity_state[5])
        ):
            raise AssertionError("AMP saved parity changes counts/rawprices")
        result["amp_parity_diagnostic"] = {
            "output_dtype": str(amp_output.dtype),
            "state_dtypes": [str(t.dtype) for t in amp_state],
            "finite_and_rawcounts_gate": True,
            "numeric_tolerance": None,
        }
        sample_vram(samples, "final_frozen_student_amp_parity_reset")
        torch.save(
            {
                "model": model.state_dict(),
                "parity": {
                    "inputs": parity_inputs.cpu(),
                    "output": parity_output.cpu(),
                    "state": tuple(s.cpu() for s in parity_state),
                    "backend_flags": primary_flags,
                    "tolerances": parity_tolerances,
                },
                "amp_parity_diagnostic": {
                    "output": amp_output.cpu(),
                    "state": tuple(s.cpu() for s in amp_state),
                    "backend_flags": backend_flags,
                    "output_dtype": str(amp_output.dtype),
                    "state_dtypes": [str(t.dtype) for t in amp_state],
                    "numeric_tolerance": None,
                },
                "n_in": 211,
                "n_out": 23,
                "arm": args.arm,
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
