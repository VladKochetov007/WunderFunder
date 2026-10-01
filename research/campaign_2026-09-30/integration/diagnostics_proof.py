"""Synthetic CPU proof: skip counters do not affect optimizer or sample order."""

from __future__ import annotations

import ast
import hashlib
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from run_identity import EpochUpdateCounter, normalization_identity, sample_order_identity, scaler_source_identity

HERE = Path(__file__).resolve().parent


def trajectory(enabled: bool, instrumented: bool):
    torch.manual_seed(3)
    parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    scaler = torch.amp.GradScaler("cpu", enabled=enabled)
    counter = EpochUpdateCounter(torch.device("cpu"))
    records = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = parameter.square().sum()
        scaler.scale(loss).backward()
        if enabled and step == 1:
            parameter.grad[0] = torch.inf
        scaler.unscale_(optimizer)
        flags = [] if not enabled else [float(value) for value in scaler._found_inf_per_device(optimizer).values()]
        before = torch.get_rng_state().clone()
        if instrumented:
            counter.observe_unscaled(scaler, optimizer)
        assert torch.equal(before, torch.get_rng_state())
        torch.nn.utils.clip_grad_norm_([parameter], 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if instrumented:
            counter.ema_updates += 1  # existing trainer updates EMA even if AMP skipped
        records.append({"parameter": parameter.detach().clone(), "grad": parameter.grad.clone(), "lr": optimizer.param_groups[0]["lr"], "scale": scaler.get_scale(), "flags": flags, "optimizer_step": optimizer.state[parameter]["step"].clone()})
    counter.completed_batches += int(instrumented)
    result = counter.finish({}) if instrumented else None
    return records, result


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    identity = scaler_source_identity()
    assert torch.__version__.startswith("2.14.")
    cases = []
    for enabled in [False, True]:
        reference, _ = trajectory(enabled, False)
        instrumented, result = trajectory(enabled, True)
        for old, new in zip(reference, instrumented, strict=True):
            for key in old:
                if isinstance(old[key], torch.Tensor):
                    torch.testing.assert_close(old[key], new[key], rtol=0, atol=0, equal_nan=True)
                else:
                    assert old[key] == new[key]
        assert result["epoch"] == {"optimizer_attempts": 3, "amp_skipped_updates": int(enabled), "optimizer_completed_updates": 3 - int(enabled), "ema_updates": 3, "completed_batches": 1}
        cases.append({"actual_cpu_GradScaler_enabled": enabled, "trajectory_exact": True, "counts": result})
    disabled = SimpleNamespace(is_enabled=lambda: False, _found_inf_per_device=lambda _: (_ for _ in ()).throw(AssertionError("disabled scaler accessor called")))
    counter = EpochUpdateCounter(torch.device("cpu"))
    counter.observe_unscaled(disabled, None)
    assert counter.finish({})["epoch"]["amp_skipped_updates"] == 0
    cumulative = {}
    for flags, expected in [([0.0], 0), ([1.0], 1), ([0.0, 1.0], 1), ([1.0, 1.0], 1)]:
        counter = EpochUpdateCounter(torch.device("cpu"))
        values = {index: torch.tensor(value) for index, value in enumerate(flags)}
        snapshots = [value.clone() for value in values.values()]
        counter.observe_unscaled(SimpleNamespace(is_enabled=lambda: True, _found_inf_per_device=lambda _: values), None)
        assert all(torch.equal(a, b) for a, b in zip(values.values(), snapshots, strict=True))
        assert counter.finish(cumulative)["epoch"]["amp_skipped_updates"] == expected
    assert cumulative["optimizer_attempts"] == 4 and cumulative["amp_skipped_updates"] == 3
    module = torch.nn.Module()
    module.register_buffer("mean", torch.tensor([0.0, -0.0, 0.125, -0.25]))
    module.register_buffer("std", torch.tensor([1.0, 2.0, 3.0, 4.0]))
    before = {name: value.clone() for name, value in module.named_buffers()}
    norm = normalization_identity(module)
    for name, value in before.items():
        assert norm[name]["bytes_sha256"] == hashlib.sha256(value.numpy().tobytes()).hexdigest()
        assert torch.equal(value, getattr(module, name))
    module.mean[0] = 0.001
    assert normalization_identity(module)["mean"]["bytes_sha256"] != norm["mean"]["bytes_sha256"]
    orders = []
    for inspect_order in [False, True]:
        torch.manual_seed(3)
        loader = torch.utils.data.DataLoader(torch.arange(17), batch_size=4, shuffle=True, drop_last=True, num_workers=0)
        before = torch.get_rng_state().clone()
        if inspect_order:
            order_identity = sample_order_identity(loader)
        assert torch.equal(before, torch.get_rng_state())
        orders.append(torch.cat(list(loader)))
    assert torch.equal(*orders)
    source = (HERE / "run_identity.py").read_text()
    observe = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == "observe_unscaled")
    assert not any(isinstance(node, ast.Attribute) and node.attr in {"item", "cpu", "numpy", "synchronize"} for node in ast.walk(observe))
    assert not torch.cuda.is_initialized()
    cgroup = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
    scope = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    limits = {name: (scope / name).read_text().strip() for name in ["memory.max", "memory.swap.max", "cpu.max"]}
    assert limits["memory.max"] == "2147483648" and limits["memory.swap.max"] == "0"
    quota, period = map(int, limits["cpu.max"].split())
    assert quota <= period
    result = {"passed": True, "synthetic_only": True, "CUDA_initialized": False, "runtime_seconds": time.monotonic() - started, "peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "cgroup": cgroup, "verified_limits": limits, "scaler_source_identity": identity, "actual_scaler_trajectories": cases, "disabled_accessor_not_called": True, "synthetic_multidevice_any_skip_counts": True, "normalization_exact_bytes_and_nonmutation": True, "sample_order_rng_and_order_exact": True, "sample_order_identity": order_identity, "no_per_step_host_scalar_or_transfer": True, "helper_sha256": hashlib.sha256(source.encode()).hexdigest(), "diagnostics_manifest_sha256": hashlib.sha256((HERE / "diagnostics_manifest.json").read_bytes()).hexdigest()}
    (HERE / "diagnostics_proof_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "runtime_seconds": result["runtime_seconds"], "peak_rss_KiB": result["peak_rss_KiB"]}))


if __name__ == "__main__":
    main()
