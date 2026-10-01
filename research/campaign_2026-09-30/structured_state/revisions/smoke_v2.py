from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import torch
from model import STATE_WIDTH, StructuredStateModel
from torch import Tensor

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    """Hash the exact artifact bytes used in this synthetic probe."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference(a: Tensor, b: Tensor) -> float:
    """Return the maximum absolute difference as an auditable scalar."""
    return float((a - b).abs().max().detach())


def require(errors: dict[str, float], tolerance: float) -> None:
    """Fail on nonfinite values or an exceeded preregistered tolerance."""
    for name, error in errors.items():
        if not torch.isfinite(torch.tensor(error)) or error > tolerance:
            raise AssertionError(f"{name}: {error} > {tolerance}")


def gradient_parity(dtype: torch.dtype) -> dict[str, float]:
    """Compare input, parameter, and nonzero initial-state gradients."""
    model = StructuredStateModel(11, 5).to(dtype=dtype)
    reference = copy.deepcopy(model)
    x = torch.randn(2, 65, 11, dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    states = tuple(torch.randn(2, STATE_WIDTH, dtype=dtype).requires_grad_() for _ in range(2))
    sr = tuple(s.detach().clone().requires_grad_() for s in states)
    target = torch.randn(2, 65, 5, dtype=dtype)
    out, final = model(x, states)
    ref_out, ref_final = reference.sequential(xr, sr)
    loss = (out - target).square().mean() + sum(s.square().mean() for s in final)
    ref_loss = (ref_out - target).square().mean() + sum(s.square().mean() for s in ref_final)
    loss.backward()
    ref_loss.backward()
    result = {"output": difference(out, ref_out), "input_grad": difference(x.grad, xr.grad)}
    for index, (state, ref_state) in enumerate(zip(states, sr, strict=True)):
        result[f"state_grad_{index}"] = difference(state.grad, ref_state.grad)
        result[f"final_state_{index}"] = difference(final[index], ref_final[index])
    max_param_grad = 0.0
    for (name, parameter), (ref_name, ref_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if name != ref_name or parameter.grad is None or ref_parameter.grad is None:
            raise AssertionError(f"missing/mismatched gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"nonfinite gradient: {name}")
        max_param_grad = max(max_param_grad, difference(parameter.grad, ref_parameter.grad))
    result["parameter_grad"] = max_param_grad
    require(result, 1e-8 if dtype == torch.float64 else 2e-3)
    return result


@torch.no_grad()
def invariants(model: StructuredStateModel) -> dict[str, object]:
    """Exercise partitioning, real streaming, future perturbation, and reset."""
    x = torch.randn(1, 2049, 11)
    incoming = tuple(torch.randn(1, STATE_WIDTH) for _ in range(2))
    out, final = model(x, incoming)
    ref, ref_final = model.sequential(x, incoming)
    parity = {"output": difference(out, ref)}
    parity.update(
        {
            f"state_{i}": difference(a, b)
            for i, (a, b) in enumerate(zip(final, ref_final, strict=True))
        }
    )
    require(parity, 5e-4)
    partitions: dict[str, object] = {}
    for length in [1, 17, 64, 2000]:
        state = tuple(s.clone() for s in incoming)
        outputs: list[Tensor] = []
        for start in range(0, x.shape[1], length):
            prediction, state = model(x[:, start : start + length], state)
            outputs.append(prediction)
        errors = {"output": difference(out, torch.cat(outputs, dim=1))}
        errors.update(
            {
                f"state_{i}": difference(a, b)
                for i, (a, b) in enumerate(zip(final, state, strict=True))
            }
        )
        require(errors, 5e-4)
        partitions[str(length)] = errors
    stream_state = tuple(s.clone() for s in incoming)
    streamed: list[Tensor] = []
    for row in x[:, :129].unbind(dim=1):
        prediction, stream_state = model.stream_step(row, stream_state)
        streamed.append(prediction)
    prefix, prefix_state = model(x[:, :129], incoming)
    stream_errors = {"output": difference(prefix, torch.stack(streamed, dim=1))}
    stream_errors.update(
        {
            f"state_{i}": difference(a, b)
            for i, (a, b) in enumerate(zip(prefix_state, stream_state, strict=True))
        }
    )
    require(stream_errors, 5e-4)
    perturbed = x.clone()
    perturbed[:, 1024:] += 10 * torch.randn_like(perturbed[:, 1024:])
    changed, _ = model(perturbed, incoming)
    causality = {"future_perturbation": difference(out[:, :1024], changed[:, :1024])}
    require(causality, 1e-4)
    a, _ = model(x[:, :65], model.reset())
    model(x[:, 65:130])
    b, _ = model(x[:, :65], model.reset())
    reset = {"output": difference(a, b)}
    require(reset, 0.0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp, amp_state = model(x[:, :65])
    ordinary, ordinary_state = model(x[:, :65])
    autocast = {"output": difference(amp, ordinary)}
    if any(s.dtype != torch.float32 for s in amp_state):
        raise AssertionError("state precision lost under autocast")
    autocast.update(
        {
            f"state_{i}": difference(a, b)
            for i, (a, b) in enumerate(zip(amp_state, ordinary_state, strict=True))
        }
    )
    require(autocast, 0.0)
    return {
        "real_reference": parity,
        "chunks": partitions,
        "stream": stream_errors,
        "causality": causality,
        "reset": reset,
        "autocast": autocast,
    }


@torch.no_grad()
def stability(model: StructuredStateModel) -> dict[str, object]:
    """Check 20k quiet rows and scalar/rotation contraction at extreme radii."""
    slow = copy.deepcopy(model)
    for block in slow.blocks:
        block.raw_half_life.fill_(80)
    prediction, final = slow(torch.zeros(1, 20_000, 11))
    if not torch.isfinite(prediction).all() or any(not s.isfinite().all() for s in final):
        raise AssertionError("20k slow-mode quiet sequence is not finite")
    results: dict[str, object] = {
        "quiet_prediction_absmax": float(prediction.abs().max()),
        "quiet_state_absmax": max(float(s.abs().max()) for s in final),
    }
    for raw in [-80.0, 80.0]:
        block = copy.deepcopy(model.blocks[0])
        block.raw_half_life.fill_(raw)
        initial = torch.randn(1, STATE_WIDTH)
        state = initial.clone()
        drive = torch.zeros_like(state)
        for _ in range(20_000):
            state = block.real_step(drive, state)
        log_rho, _ = block.dynamics()
        scalar_expected = initial[:, :64] * (20_000 * log_rho[:64]).exp()
        impulse_error = difference(state[:, :64], scalar_expected)
        initial_pair_norm = initial[:, 64:].reshape(1, 32, 2).norm(dim=-1)
        expected_pair_norm = initial_pair_norm * (20_000 * log_rho[64:]).exp()
        pair_error = difference(state[:, 64:].reshape(1, 32, 2).norm(dim=-1), expected_pair_norm)
        require({"scalar_impulse": impulse_error, "rotation_impulse": pair_error}, 5e-4)
        constant_drive = torch.randn_like(state) * 0.1
        constant = block.fft_states(constant_drive[:, None].expand(-1, 20_000, -1), initial)
        rho = log_rho.exp()
        bound_scalar = initial[:, :64].abs() + constant_drive[:, :64].abs() / (1 - rho[:64])
        bound_pair = initial_pair_norm + constant_drive[:, 64:].reshape(1, 32, 2).norm(dim=-1) / (
            1 - rho[64:]
        )
        violation = max(
            float((constant[..., :64].abs() - bound_scalar[:, None]).max()),
            float(
                (
                    constant[..., 64:].reshape(1, 20_000, 32, 2).norm(dim=-1) - bound_pair[:, None]
                ).max()
            ),
            0.0,
        )
        require({"constant_bound_violation": violation}, 5e-4)
        results[str(raw)] = {
            "half_life": float(block.half_lives()[0]),
            "scalar_impulse_error": impulse_error,
            "rotation_impulse_norm_error": pair_error,
            "constant_bound_violation": violation,
        }
    return results


def profile() -> dict[str, object]:
    """Measure CPU FFT chunk training, preserving a synthetic checkpoint."""
    model = StructuredStateModel(211, 20)
    x = torch.randn(2, 2000, 211)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    forward_times, backward_times = [], []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        out, state = model(x)
        middle = time.perf_counter()
        loss = out.square().mean() + 0.001 * sum(s.square().mean() for s in state)
        loss.backward()
        optimizer.step()
        end = time.perf_counter()
        forward_times.append(middle - start)
        backward_times.append(end - middle)
    checkpoint = ROOT / "synthetic_smoke.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "n_in": 211,
            "n_out": 20,
            "seed": 930,
            "synthetic_only": True,
            "optimizer_updates": 3,
            "decay_init": "multi",
            "source_sha256": sha256(ROOT / "model.py"),
        },
        checkpoint,
    )
    with torch.no_grad():
        start = time.perf_counter()
        model(torch.zeros(1, 20_000, 211))
        long_time = time.perf_counter() - start
        start = time.perf_counter()
        model.sequential(torch.zeros(1, 256, 211))
        reference_time = time.perf_counter() - start
    return {
        "batch": 2,
        "chunk": 2000,
        "forward_seconds": forward_times,
        "backward_and_update_seconds": backward_times,
        "full20k_forward_seconds": long_time,
        "reference256_forward_seconds": reference_time,
        "parameters": sum(p.numel() for p in model.parameters()),
        "macs_per_row_dense": 211 * 96
        + 2 * (96 * 128 + 128 * 96 + 2 * 96 * 96)
        + 192 * 96
        + 96 * 20,
        "state_bytes_per_sequence_fp32": 2 * 128 * 4,
        "batch32_one_two_block_state_timeline_mib": 32 * 2000 * 128 * 2 * 4 / 2**20,
        "batch32_each_block_scalar_plus_rotation_fft_mib": (32 * 64 * 2049 * 8 + 32 * 32 * 4096 * 8)
        / 2**20,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint": checkpoint.name,
    }


def main() -> None:
    """Run a deterministic, no-competition-data actor smoke and emit JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "smoke_results.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(930)
    started = time.time()
    results: dict[str, object] = {
        "synthetic_only": True,
        "seed": 930,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "threads": torch.get_num_threads(),
        "cuda_initialized": False,
        "source_sha256": {
            p.name: sha256(p)
            for p in [ROOT / "model.py", ROOT / "smoke.py", ROOT / "preregistration.md"]
        },
        "base_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scope": os.environ.get("WNN_SCOPE_UNIT"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        torch.manual_seed(930)
        multi = StructuredStateModel(11, 5, "multi")
        torch.manual_seed(930)
        single = StructuredStateModel(11, 5, "single")
        differing = [
            name
            for name, value in multi.state_dict().items()
            if not torch.equal(value, single.state_dict()[name])
        ]
        if differing != ["blocks.0.raw_half_life", "blocks.1.raw_half_life"]:
            raise AssertionError(f"arm differences: {differing}")
        results["matched_arm_differences"] = differing
        LOG.info("Checking FP64/FP32 gradients")
        results["gradient_fp64"] = gradient_parity(torch.float64)
        results["gradient_fp32"] = gradient_parity(torch.float32)
        LOG.info("Checking real recurrence, partitions, stream, causality, reset")
        results["multi_invariants"] = invariants(multi)
        results["single_invariants"] = invariants(single)
        LOG.info("Checking20k quiet stability and extreme radii")
        results["stability"] = stability(multi)
        LOG.info("Profiling optimized FFT primitives; no graph compile requested")
        results["profile"] = profile()
        if torch.cuda.is_initialized():
            raise AssertionError("CPU-only actor initialized CUDA")
        results["passed"] = True
    except (AssertionError, RuntimeError, ValueError) as error:
        results["passed"] = False
        results["failure"] = str(error)
        raise
    finally:
        results["elapsed_seconds"] = time.time() - started
        results["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        results["cuda_initialized"] = torch.cuda.is_initialized()
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        LOG.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
