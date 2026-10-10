from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from wnn.nn import RNNModel
from wnn.optim import orthogonalization_runtime_info
from wnn.training_dynamics import build_optimizer, build_scheduler


def _profile_args() -> argparse.Namespace:
    """Create the frozen reference-shaped Muon/OneCycle profile settings."""
    return argparse.Namespace(
        optim="muon",
        optim_switch="",
        head_lr_mult=1.0,
        muon_lr_mult=3.0,
        muon_momentum=0.95,
        muon_gatewise="none",
        muon_ortho_dtype_policy="native_bf16_if_supported_else_fp32_v1",
        muon_scope="hidden",
        zoneout=0.0,
        momentum=0.0,
        fused_adam=False,
        lr=1e-3,
        beta2=0.999,
        wd=0.01,
        sched="onecycle",
        warmup_frac=0.05,
        wsd_decay=0.2,
        wsd_final=0.0004,
        restarts=1,
    )


def _parameter_shapes(model: RNNModel) -> dict[str, list[int]]:
    """Capture shapes that establish the profile used the intended GRU geometry."""
    names = (
        "gru.weight_ih_l0",
        "gru.weight_hh_l0",
        "gru.weight_ih_l1",
        "gru.weight_hh_l1",
        "head.0.weight",
    )
    state = dict(model.named_parameters())
    return {name: list(state[name].shape) for name in names}


def run_profile(
    output_path: Path,
    seed: int = 9,
    allocator_cap_bytes: int = 4_300_000_000,
) -> dict[str, object]:
    """Run one synthetic, full-shape CUDA update to check Muon matrix math only."""
    if output_path.exists():
        raise FileExistsError(f"profile output already exists: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("the actual-device Muon profile requires CUDA")
    if allocator_cap_bytes <= 0:
        raise ValueError("allocator cap must be positive")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    total_vram_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    if allocator_cap_bytes > total_vram_bytes:
        raise ValueError(
            f"allocator cap {allocator_cap_bytes} exceeds device VRAM {total_vram_bytes}"
        )
    torch.cuda.set_per_process_memory_fraction(
        allocator_cap_bytes / total_vram_bytes, device=device
    )
    model = RNNModel(n_in=211, hidden=256, layers=2, dropout=0.0).to(device)
    args = _profile_args()
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, total=100)
    shapes = _parameter_shapes(model)
    expected_shapes = {
        "gru.weight_ih_l0": [768, 256],
        "gru.weight_hh_l0": [768, 256],
        "gru.weight_ih_l1": [768, 256],
        "gru.weight_hh_l1": [768, 256],
        "head.0.weight": [256, 512],
    }
    if shapes != expected_shapes:
        raise RuntimeError(f"profile model shape mismatch: {shapes}")

    group_summary = [
        {
            "muon": bool(group.get("muon", False)),
            "blocks": int(group.get("blocks", 1)),
            "lr_scale": float(group.get("lr_scale", 1.0)),
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            "orthogonalization_dtype": str(
                group.get("orthogonalization_dtype", "not_applicable")
            ).removeprefix("torch."),
        }
        for group in optimizer.param_groups
    ]
    muon_groups = [group for group in optimizer.param_groups if group.get("muon")]
    if not muon_groups or any(int(group.get("blocks", 0)) != 1 for group in muon_groups):
        raise RuntimeError("whole-matrix Muon groups were not constructed")
    if any(not math.isclose(float(group["lr_scale"]), 3.0) for group in muon_groups):
        raise RuntimeError("Muon learning-rate multiplier was not applied to every Muon group")

    batch_size, context = 16, 1000
    values = torch.randn(batch_size, context, 211, device=device)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    output, _ = model(values)
    loss = output.float().square().mean()
    loss.backward()
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise RuntimeError("synthetic profile produced a nonfinite gradient")
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    changed = sum(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    if changed == 0 or any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("Muon device step did not produce finite parameter updates")

    result: dict[str, object] = {
        "schema": "h230-muon-actual-device-profile-result-v1",
        "status": "passed",
        "scope": "one synthetic full-shape optimizer update; no data, fit, checkpoint, prediction, or score",
        "seed": seed,
        "allocator_cap_bytes": allocator_cap_bytes,
        "device_total_vram_bytes": total_vram_bytes,
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "runtime_policy": orthogonalization_runtime_info(
            args.muon_ortho_dtype_policy, device
        ),
        "model": {
            "raw_feature_width": 112,
            "model_input_width": 211,
            "hidden": 256,
            "layers": 2,
            "batch_size": batch_size,
            "chunk_rows": context,
            "parameter_shapes": shapes,
        },
        "optimizer": {
            "muon_gatewise": args.muon_gatewise,
            "muon_scope": args.muon_scope,
            "muon_lr_multiplier": args.muon_lr_mult,
            "muon_momentum": args.muon_momentum,
            "groups": group_summary,
        },
        "loss": float(loss.detach().cpu()),
        "changed_parameter_tensors": changed,
        "elapsed_seconds": elapsed,
        "vram_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
        "vram_reserved_peak_bytes": int(torch.cuda.max_memory_reserved(device)),
        "profile_pid": os.getpid(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    temporary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, output_path)
    return result


def main() -> int:
    """Parse the bounded profile output location and run the actual-device check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--allocator-cap-bytes", type=int, default=4_300_000_000)
    args = parser.parse_args()
    result = run_profile(args.output, args.seed, args.allocator_cap_bytes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
