"""Remote acceptance: ``torchrun --standalone --nproc_per_node=2 scripts/check_sequence_muon_ddp.py --output-receipt /path/to/muon-ddp.json``."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from wnn.training.sequence_distributed import configure_strict_fp32
from wnn.training.sequence_fit_utils import make_optimizer
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_objectives import (
    objective_numerators,
    scaled_objective,
    window_denominators,
)

WORLD_SIZE, ROWS, MICROBATCHES, UPDATE_COUNT, SCHEDULE_UPDATES = 2, 5, 2, 2, 4
KD_WEIGHT, HINT_WEIGHT = 0.35, 0.2
ATOL, RTOL, DENOM_ATOL = 1e-5, 1e-4, 1e-5
Batch = dict[str, Tensor]
Case = dict[str, object]
Training = tuple[DistributedDataParallel, nn.Module, Optimizer, Optimizer, LambdaLR, LambdaLR]
OPTIMIZER_CONFIG: dict[str, object] = {
    "optimizer": "muon",
    "lr": 3e-4,
    "weight_decay": 0.01,
    "muon_momentum": 0.95,
    "muon_ns_steps": 5,
    "muon_ns_eps": 1e-7,
    "adamw_beta1": 0.9,
    "adamw_beta2": 0.95,
    "adamw_eps": 1e-8,
    "schedule": "warmup_cosine_v2",
    "min_lr_ratio": 0.1,
    "warmup_fraction": 0.25,
}
MODEL_CONFIG: dict[str, object] = {
    "architecture": "transformer",
    "hidden": 16,
    "layers": 2,
    "heads": 2,
    "ff_multiplier": 2,
    "context_rows": 8,
    "dropout": 0.0,
    "hint_dim": 2,
}


def _toy_batch(scenario: str, update: int, rank: int, micro: int, device: torch.device) -> Batch:
    """Build one deterministic raw-112 FP32 microbatch with its declared mask pattern."""
    generator = torch.Generator(device="cpu").manual_seed(
        17_291 + update * 1_009 + rank * 101 + micro
    )
    x = torch.randn((1, ROWS, 112), generator=generator) * 0.05
    target = 0.2 + 1.4 * torch.rand((1, ROWS, 2), generator=generator)
    teacher = torch.randn((1, ROWS, 2), generator=generator) * 0.1
    hints = torch.randn((1, ROWS, 2), generator=generator) * 0.1
    if scenario == "empty_rank" and rank == 1:
        valid = [False] * ROWS
    elif rank == 1:
        valid = [micro % 2 == 0, False, False, micro % 2 == 1, False]
    else:
        valid = [True, True, False, True, True]
    return {
        "x": x.to(device),
        "target": target.to(device),
        "teacher": teacher.to(device),
        "mask": torch.tensor([valid], device=device),
        "hints": hints.to(device),
    }


def _semantic_groups(model: nn.Module, optimizer: Optimizer) -> dict[str, object]:
    """Check every trainable Transformer weight uses its required named optimizer role."""
    groups = {
        str(group.get("role", "")): set(group.get("param_names", []))
        for group in optimizer.param_groups
    }
    named = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    qkv = {name for name in named if name.endswith(".qkv.weight")}
    hidden = {
        name
        for name in named
        if name.endswith(
            (".attention_output.weight", ".feedforward_in.weight", ".feedforward_out.weight")
        )
    }
    adamw_matrix = named & {
        "input_projection.weight",
        "output_head.weight",
        "hint_projector.linear.weight",
    }
    adamw_matrix |= {name for name in named if name.endswith(".head_gate.weight")}
    expected = dict(
        muon_qkv=qkv,
        muon_hidden=hidden,
        adamw_matrix=adamw_matrix,
        adamw_no_decay=named - qkv - hidden - adamw_matrix,
    )
    flat_names = [name for group in optimizer.param_groups for name in group.get("param_names", [])]
    passed = (
        len(qkv) == 2
        and len(hidden) == 6
        and groups == expected
        and len(flat_names) == len(set(flat_names))
    )
    return {"passed": passed, "roles": sorted(groups)}


def _compare(left: object, right: object) -> tuple[bool, float]:
    """Compare nested values and return the largest absolute tensor difference."""
    if isinstance(left, Tensor):
        if not isinstance(right, Tensor) or left.shape != right.shape:
            return False, 0.0
        error = float((left - right).abs().max().item()) if left.numel() else 0.0
        return bool(torch.allclose(left, right, atol=ATOL, rtol=RTOL)), error
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or left.keys() != right.keys():
            return False, 0.0
        results = [_compare(left[key], right[key]) for key in left]
        return all(ok for ok, _ in results), max((err for _, err in results), default=0.0)
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return False, 0.0
        results = [_compare(a, b) for a, b in zip(left, right, strict=True)]
        return all(ok for ok, _ in results), max((err for _, err in results), default=0.0)
    return type(left) is type(right) and left == right, 0.0


def _backward_window(
    model: nn.Module, batches: list[Batch], denominator: Tensor, scale: int
) -> None:
    """Backpropagate one accumulation window with DDP sync only on its final microbatch."""
    for index, batch in enumerate(batches):
        sync = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and index < len(batches) - 1
            else nullcontext()
        )
        with sync:
            prediction, hints, _ = model(batch["x"])
            numerators = objective_numerators(
                prediction, hints, batch["target"], batch["teacher"], batch["hints"], batch["mask"]
            )
            scaled_objective(numerators, denominator, scale, KD_WEIGHT, HINT_WEIGHT).backward()


def _run_update(training: Training, batches: list[list[Batch]], rank: int) -> Case:
    """Apply one DDP update and compare it against a concatenated weighted reference."""
    model, reference, optimizer, reference_optimizer, scheduler, reference_scheduler = training
    local = batches[rank]
    records = [batch for rank_batches in batches for batch in rank_batches]
    combined = {key: torch.cat([batch[key] for batch in records]) for key in records[0]}
    local_den = torch.zeros(2, dtype=torch.float32, device=local[0]["x"].device)
    for batch in local:
        local_den += window_denominators(batch["target"], batch["mask"], batch["hints"])
    global_den = local_den.clone()
    dist.all_reduce(global_den, op=dist.ReduceOp.SUM)
    reference_den = window_denominators(combined["target"], combined["mask"], combined["hints"])
    den_error = float((global_den - reference_den).abs().max().item())
    optimizer.zero_grad(set_to_none=True)
    reference_optimizer.zero_grad(set_to_none=True)
    _backward_window(model, local, global_den, WORLD_SIZE)
    _backward_window(reference, [combined], reference_den, 1)
    gradients = {
        name: parameter.grad.detach().clone() if parameter.grad is not None else None
        for name, parameter in model.module.named_parameters()
    }
    reference_gradients = {
        name: parameter.grad.detach().clone() if parameter.grad is not None else None
        for name, parameter in reference.named_parameters()
    }
    grad_match, grad_error = _compare(gradients, reference_gradients)
    torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm=1.0)
    optimizer.step()
    reference_optimizer.step()
    scheduler.step()
    reference_scheduler.step()
    param_match, param_error = _compare(model.module.state_dict(), reference.state_dict())
    state_match, state_error = _compare(optimizer.state_dict(), reference_optimizer.state_dict())
    schedule_match, _ = _compare(scheduler.state_dict(), reference_scheduler.state_dict())
    groups = _semantic_groups(model.module, optimizer)
    ref_groups = _semantic_groups(reference, reference_optimizer)
    passed = (
        den_error <= DENOM_ATOL
        and grad_match
        and param_match
        and state_match
        and schedule_match
        and bool(groups["passed"])
        and groups == ref_groups
    )
    return dict(
        rank=rank,
        valid_rows=sum(int(batch["mask"].sum().item()) for batch in local),
        passed=passed,
        errors=dict(
            denominator=den_error,
            gradients=grad_error,
            parameters=param_error,
            optimizer_state=state_error,
        ),
        checks=dict(
            gradients=grad_match,
            parameters=param_match,
            optimizer_state=state_match,
            scheduler=schedule_match,
            semantic_groups=bool(groups["passed"]),
            reference_groups=groups == ref_groups,
        ),
        semantic_roles=groups["roles"],
    )


def _run_checks(
    device: torch.device, rank: int, control: dist.ProcessGroup, model_config: dict[str, object]
) -> list[Case]:
    """Run uneven-mask and empty-rank windows with two explicit Muon updates."""
    torch.manual_seed(17_291)
    torch.cuda.manual_seed_all(17_291)
    model = build_sequence_model(model_config).to(device=device, dtype=torch.float32)
    distributed = DistributedDataParallel(
        model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False
    )
    reference = build_sequence_model(model_config).to(device=device, dtype=torch.float32)
    reference.load_state_dict(distributed.module.state_dict())
    optimizer_config = OPTIMIZER_CONFIG | {"model": model_config}
    optimizer, scheduler = make_optimizer(model, optimizer_config, SCHEDULE_UPDATES)
    ref_optimizer, ref_scheduler = make_optimizer(reference, optimizer_config, SCHEDULE_UPDATES)
    training = (distributed, reference, optimizer, ref_optimizer, scheduler, ref_scheduler)
    cases = []
    for update, scenario in enumerate(("uneven", "empty_rank")):
        batches = [
            [_toy_batch(scenario, update, worker, micro, device) for micro in range(MICROBATCHES)]
            for worker in range(WORLD_SIZE)
        ]
        local = _run_update(training, batches, rank)
        gathered: list[Case | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local, group=control)
        results = [item for item in gathered if item is not None]
        cases.append(
            dict(
                scenario=scenario,
                update=update + 1,
                ranks=results,
                passed=len(results) == WORLD_SIZE and all(bool(item["passed"]) for item in results),
            )
        )
    return cases


def _receipt(cases: list[Case], device: torch.device, model_config: dict[str, object]) -> Case:
    """Bind results to the strict precision mode, CUDA runtime, and numeric tolerances."""
    return {
        "passed": all(bool(case["passed"]) for case in cases),
        "world_size": WORLD_SIZE,
        "explicit_updates": UPDATE_COUNT,
        "schedule_total_updates": SCHEDULE_UPDATES,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "precision": "strict_fp32_no_tf32_no_autocast",
        "model_config": model_config,
        "tolerance": {"atol": ATOL, "rtol": RTOL, "denominator_atol": DENOM_ATOL},
        "cases": cases,
    }


def _write_receipt(path: Path, receipt: Case) -> None:
    """Atomically publish finite JSON without replacing an existing receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        content = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Run two-rank CUDA parity and write a synchronized pass or failure receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-receipt", required=True, type=Path)
    parser.add_argument(
        "--attention-gate", choices=("none", "constant_half", "headwise"), default="none"
    )
    args = parser.parse_args()
    model_config = dict(MODEL_CONFIG)
    if args.attention_gate != "none":
        model_config["attention_gate"] = args.attention_gate
    world, rank, local_rank = (
        int(os.environ.get(name, default))
        for name, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))
    )
    if world != WORLD_SIZE or not 0 <= rank < world or not torch.cuda.is_available():
        parser.error("launch with torchrun on exactly two CUDA ranks")
    if local_rank >= torch.cuda.device_count():
        parser.error("each rank requires a visible CUDA device")
    torch.cuda.set_device(local_rank)
    configure_strict_fp32()
    device = torch.device("cuda", local_rank)
    timeout = timedelta(seconds=60)
    dist.init_process_group(
        "nccl", init_method="env://", rank=rank, world_size=world, timeout=timeout
    )
    control = dist.new_group(backend="gloo", timeout=timeout)
    try:
        cases: list[Case] = []
        local_error: Case | None = None
        try:
            cases = _run_checks(device, rank, control, model_config)
        except Exception as error:
            local_error = {"rank": rank, "error": f"{type(error).__name__}: {error}"}
        failures: list[Case | None] = [None] * WORLD_SIZE
        dist.all_gather_object(failures, local_error, group=control)
        if any(error is not None for error in failures):
            cases = [{"scenario": "execution_failure", "ranks": failures, "passed": False}]
        receipt = _receipt(cases, device, model_config)
        status: list[Case] = [{"ok": True, "error": ""}]
        if rank == 0:
            try:
                _write_receipt(args.output_receipt, receipt)
            except (OSError, TypeError, ValueError) as error:
                status[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(status, src=0, group=control)
        if not bool(status[0]["ok"]):
            raise OSError(str(status[0]["error"]))
        if not bool(receipt["passed"]):
            raise RuntimeError(f"Muon DDP check failed; see {args.output_receipt}")
    finally:
        dist.destroy_process_group(control)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
