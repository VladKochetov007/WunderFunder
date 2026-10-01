"""Independent CPU arithmetic, gradient, causality and chronology model checks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import resource
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
MODEL = ROOT / "research/campaign_2026-09-30/structured_state/model.py"


def reference(model: torch.nn.Module, x: torch.Tensor, initial: tuple) -> tuple:
    """Write scalar/2x2 updates directly, without actor recurrence helpers."""
    z = F.gelu(F.linear((x - model.mean) / model.std, model.inp.weight, model.inp.bias))
    u = z
    final = []
    for block, state in zip(model.blocks, initial, strict=True):
        life = torch.exp(torch.sigmoid(block.raw_half_life) * math.log(8192.0))
        log_rho = -math.log(2.0) / life
        scale = torch.sqrt(-torch.expm1(2 * log_rho))
        injection = F.linear(u, block.inject.weight) * torch.cat(
            [scale[:64], scale[64:].repeat_interleave(2)]
        )
        rho = torch.exp(log_rho)
        states = []
        for row in injection.unbind(1):
            scalar = state[:, :64] * rho[:64]
            pairs = state[:, 64:].reshape(-1, 32, 2)
            a, b = pairs.unbind(-1)
            cosine, sine = torch.cos(block.phase), torch.sin(block.phase)
            rotated = (
                torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1) * rho[64:, None]
            )
            state = torch.cat((scalar, rotated.flatten(1)), 1) + row
            states.append(state)
        timeline = torch.stack(states, 1)
        direct = F.linear(u, block.direct.weight)
        read = F.linear(timeline, block.readout.weight)
        gate = torch.sigmoid(F.linear(u, block.gate.weight, block.gate.bias))
        u = u + gate * F.gelu(read + direct)
        final.append(state)
    h = torch.cat((u, z), -1)
    pred = F.linear(
        F.gelu(F.linear(h, model.head[0].weight, model.head[0].bias)),
        model.head[2].weight,
        model.head[2].bias,
    )
    return pred, tuple(final)


def difference(left: torch.Tensor, right: torch.Tensor, atol: float, rtol: float) -> dict:
    """Record maximum absolute error and violations of the frozen mixed bound."""
    delta = (left - right).abs()
    bound = atol + rtol * right.abs()
    return {
        "max_abs": float(delta.max()),
        "violations": int((delta > bound).sum()),
        "max_bound_ratio": float((delta / bound).max()),
        "finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
    }


def run() -> dict:
    """Execute a predeclared CPU protocol on immutable actor source bytes."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    module_spec = importlib.util.spec_from_file_location("independent_actor", MODEL)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    contract = json.loads((AUDIT / "architecture_cpu_grading_contract.json").read_text())
    cases = []

    def record(name: str, action: object) -> None:
        try:
            details = action()
            cases.append(
                {
                    "name": name,
                    "status": "pass" if details.pop("passed") else "fail",
                    "details": details,
                }
            )
        except (AssertionError, RuntimeError, ValueError) as exc:
            cases.append(
                {
                    "name": name,
                    "status": "fail",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    torch.manual_seed(19301)
    multi = module.StructuredStateModel(9, decay_init="multi")
    torch.manual_seed(19301)
    single = module.StructuredStateModel(9, decay_init="single")
    mismatches = [
        name
        for name, value in multi.state_dict().items()
        if not name.endswith("raw_half_life") and not torch.equal(value, single.state_dict()[name])
    ]
    cases.append(
        {
            "name": "only_decay_initialization_differs",
            "status": "pass" if not mismatches else "fail",
            "details": {
                "mismatches": mismatches,
                "parameter_count": sum(p.numel() for p in multi.parameters()),
            },
        }
    )
    for arm, base in (("multi", multi), ("single", single)):

        def grad_case(base=base) -> dict:
            torch.manual_seed(7319)
            model = copy.deepcopy(base).double()
            other = copy.deepcopy(model)
            x = torch.randn(2, 97, 9, dtype=torch.float64, requires_grad=True)
            xx = x.detach().clone().requires_grad_()
            states = tuple(
                torch.randn(2, 128, dtype=torch.float64, requires_grad=True) for _ in range(2)
            )
            ss = tuple(s.detach().clone().requires_grad_() for s in states)
            y, h = model(x, states)
            yy, hh = reference(other, xx, ss)
            weight = torch.randn_like(y)
            state_weights = tuple(torch.randn_like(s) for s in h)
            loss = (y * weight).sum() + sum(
                (s * w).sum() for s, w in zip(h, state_weights, strict=True)
            )
            loss2 = (yy * weight).sum() + sum(
                (s * w).sum() for s, w in zip(hh, state_weights, strict=True)
            )
            loss.backward()
            loss2.backward()
            forward = [
                difference(y, yy, 1e-9, 1e-8),
                *[difference(a, b, 1e-9, 1e-8) for a, b in zip(h, hh, strict=True)],
            ]
            gradients = {"input": difference(x.grad, xx.grad, 1e-8, 1e-7)}
            gradients.update(
                {
                    f"state{i}": difference(a.grad, b.grad, 1e-8, 1e-7)
                    for i, (a, b) in enumerate(zip(states, ss, strict=True))
                }
            )
            gradients.update(
                {
                    name: difference(
                        param.grad, dict(other.named_parameters())[name].grad, 1e-8, 1e-7
                    )
                    for name, param in model.named_parameters()
                }
            )
            return {
                "passed": all(
                    v["finite"] and v["violations"] == 0 for v in forward + list(gradients.values())
                ),
                "forward": forward,
                "gradients": gradients,
            }

        record(f"{arm}_FP64_forward_all_gradients", grad_case)

        def partition_case(base=base) -> dict:
            torch.manual_seed(7301)
            x = torch.randn(1, 2049, 9)
            state = tuple(torch.randn(1, 128) * 0.3 for _ in range(2))
            results = {}
            with torch.no_grad():
                target, hs = reference(base, x, state)
                for chunk in (1, 17, 64, 2000, 2049):
                    incoming = tuple(s.clone() for s in state)
                    parts = []
                    for start in range(0, len(x[0]), chunk):
                        p, incoming = base(x[:, start : start + chunk], incoming)
                        parts.append(p)
                    values = [
                        difference(torch.cat(parts, 1), target, 0.00002, 0.0002),
                        *[
                            difference(a, b, 0.00002, 0.0002)
                            for a, b in zip(incoming, hs, strict=True)
                        ],
                    ]
                    results[str(chunk)] = values
                incoming = tuple(s.clone() for s in state)
                parts = []
                for row in x.unbind(1):
                    p, incoming = base.stream_step(row, incoming)
                    parts.append(p)
                results["stream"] = [
                    difference(torch.stack(parts, 1), target, 0.00002, 0.0002),
                    *[difference(a, b, 0.00002, 0.0002) for a, b in zip(incoming, hs, strict=True)],
                ]
            return {
                "passed": all(
                    v["finite"] and v["violations"] == 0
                    for values in results.values()
                    for v in values
                ),
                "partitions": results,
            }

        record(f"{arm}_FP32_partition_and_stream", partition_case)

        def causal_reset_case(base=base) -> dict:
            torch.manual_seed(99)
            x = torch.randn(2, 257, 9)
            changed = x.clone()
            changed[:, 121:] = torch.randn_like(changed[:, 121:]) * 10
            with torch.no_grad():
                p, _ = base(x)
                q, _ = base(changed)
                one, hs = base(x[:, :121])
                prefix = difference(p[:, :121], q[:, :121], 0.00002, 0.0002)
                fullprefix = difference(p[:, :121], one, 0.00002, 0.0002)
                restored, states = base(x, base.reset(2))
                fresh, fs = base(x, None)
                reset_exact = torch.equal(restored, fresh) and all(
                    torch.equal(a, b) for a, b in zip(states, fs, strict=True)
                )
                batches = [base(x[i : i + 1])[0] for i in range(2)]
                batch = difference(torch.cat(batches), p, 0.00002, 0.0002)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    mixed, mh = base(x)
                autocast_exact = torch.equal(mixed, p) and all(s.dtype == torch.float32 for s in mh)
            return {
                "passed": reset_exact
                and autocast_exact
                and all(v["finite"] and v["violations"] == 0 for v in [prefix, fullprefix, batch]),
                "suffix_perturbation": prefix,
                "full_prefix_vs_prefix_only": fullprefix,
                "batch_independence": batch,
                "reset_exact": reset_exact,
                "autocast_exact_fp32_states": autocast_exact,
            }

        record(f"{arm}_causality_reset_autocast", causal_reset_case)

    def long_impulse() -> dict:
        block = copy.deepcopy(multi.blocks[0])
        block.raw_half_life.data.fill_(50.0)
        length = 20000
        incoming = torch.zeros(1, 128)
        incoming[0, 0] = 1
        incoming[0, 64] = 1
        drive = torch.zeros(1, length, 128)
        with torch.no_grad():
            optimized = block.fft_states(drive, incoming)
            rho = math.exp(-math.log(2.0) / 8192.0)
            times = torch.arange(1, length + 1, dtype=torch.float64)
            analytic = torch.pow(torch.tensor(rho, dtype=torch.float64), times).float()
            streamed = incoming.clone()
            for _ in range(length):
                streamed = block.real_step(drive[:, 0], streamed)
            scalar = difference(optimized[0, :, 0], analytic, 0.00002, 0.0002)
            rotation_norm = difference(
                optimized[0, :, 64:66].norm(dim=-1), analytic, 0.00002, 0.0002
            )
            final = difference(streamed, optimized[:, -1], 0.00002, 0.0002)
        return {
            "passed": all(
                v["finite"] and v["violations"] == 0 for v in [scalar, rotation_norm, final]
            ),
            "FFT_scalar_vs_exact_impulse": scalar,
            "FFT_rotation_norm_vs_exact_impulse": rotation_norm,
            "stream_final_vs_FFT": final,
            "stream_scalar_final": float(streamed[0, 0]),
            "FFT_scalar_final": float(optimized[0, -1, 0]),
        }

    record("half_life8192_20000row_impulse", long_impulse)
    after = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    return {
        "status": "pass"
        if all(c["status"] == "pass" for c in cases) and before == after
        else "fail",
        "model_sha256_before": before,
        "model_sha256_after": after,
        "protocol_sha256": hashlib.sha256(
            (AUDIT / "architecture_cpu_grading_contract.json").read_bytes()
        ).hexdigest(),
        "grader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independent_reference_actor_recurrence_helpers_used": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "cases": cases,
        "protocol": contract,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    result = run()
    result["wall_seconds"] = time.monotonic() - started
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": [{"name": c["name"], "status": c["status"]} for c in result["cases"]],
                "cuda_initialized": result["cuda_initialized"],
                "wall_seconds": result["wall_seconds"],
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if result["status"] == "pass" and not result["cuda_initialized"] else 1)
