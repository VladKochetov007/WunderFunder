"""Remote command: ``torchrun --standalone --nproc_per_node=2 scripts/check_sequence_ddp.py --output-receipt /path/to/ddp.json``."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from wnn.training.sequence_objectives import (
    objective_numerators,
    scaled_objective,
    window_denominators,
)

FEATURES = 3
OUTPUTS = 4
TARGETS = 2
HINTS = 2
CHUNK_COUNT = 4
WORLD_SIZE = 2
KD_WEIGHT = 0.35
HINT_WEIGHT = 0.2
ATOL = 2e-6
RTOL = 2e-5


@dataclass(frozen=True)
class ToyBatch:
    """One deterministic toy microbatch with the fields consumed by sequence objectives."""

    x: Tensor
    target: Tensor
    teacher: Tensor
    mask: Tensor
    hint_target: Tensor


def _toy_batch(rank: int, chunk: int, scenario: str, device: torch.device) -> ToyBatch:
    generator = torch.Generator(device="cpu").manual_seed(17_291 + rank * 101 + chunk)
    x = torch.randn((1, 3, FEATURES), generator=generator)
    target = 0.1 + 2.5 * torch.rand((1, 3, TARGETS), generator=generator)
    teacher = torch.randn((1, 3, TARGETS), generator=generator)
    hint_target = torch.randn((1, 3, HINTS), generator=generator)
    if scenario == "empty_rank" and rank == 1:
        valid = [False, False, False]
    elif scenario == "uneven" and rank == 1:
        valid = [chunk % 2 == 0, chunk % 2 == 1, False]
    else:
        valid = [True, True, True]
    return ToyBatch(
        x.to(device=device, dtype=torch.float32),
        target.to(device=device, dtype=torch.float32),
        teacher.to(device=device, dtype=torch.float32),
        torch.tensor([valid], device=device, dtype=torch.bool),
        hint_target.to(device=device, dtype=torch.float32),
    )


def _make_batches(scenario: str, device: torch.device) -> list[list[ToyBatch]]:
    return [
        [_toy_batch(rank, chunk, scenario, device) for chunk in range(CHUNK_COUNT)]
        for rank in range(WORLD_SIZE)
    ]


def _sum_denominators(batches: Sequence[ToyBatch], indices: Sequence[int]) -> Tensor:
    return sum(
        (
            window_denominators(batch.target, batch.mask, batch.hint_target)
            for batch in (batches[index] for index in indices)
        ),
        torch.zeros(2, dtype=torch.float32, device=batches[0].x.device),
    )


def _reference_gradients(
    model: DistributedDataParallel,
    batches: list[list[ToyBatch]],
    indices: Sequence[int],
    denominators: Tensor,
    alpha: float = 0.0,
) -> list[Tensor]:
    reference = nn.Linear(FEATURES, OUTPUTS).to(device=batches[0][0].x.device)
    reference.load_state_dict(model.module.state_dict())
    reference.zero_grad(set_to_none=True)
    reference_loss: Tensor | None = None
    for rank_batches in batches:
        for index in indices:
            batch = rank_batches[index]
            output = reference(batch.x)
            prediction, hint_prediction = output[..., :TARGETS], output[..., TARGETS:]
            numerators = objective_numerators(
                prediction,
                hint_prediction,
                batch.target,
                batch.teacher,
                batch.hint_target,
                batch.mask,
                blend_objective_alpha=alpha,
            )
            term = scaled_objective(
                numerators,
                denominators,
                1,
                0.0 if alpha else KD_WEIGHT,
                0.0 if alpha else HINT_WEIGHT,
            )
            reference_loss = term if reference_loss is None else reference_loss + term
    if reference_loss is None:
        raise RuntimeError("reference objective unexpectedly had no microbatches")
    reference_loss.backward()
    return [parameter.grad.detach().clone() for parameter in reference.parameters()]


def _max_absolute_error(got: Sequence[Tensor], ref: Sequence[Tensor]) -> float:
    errors = [float((a - b).abs().max()) for a, b in zip(got, ref, strict=True)]
    return max(errors, default=0.0)


def _run_window(
    model: DistributedDataParallel,
    all_batches: list[list[ToyBatch]],
    local_batches: Sequence[ToyBatch],
    indices: Sequence[int],
    alpha: float = 0.0,
) -> dict[str, object]:
    local_denominators = _sum_denominators(local_batches, indices)
    global_denominators = local_denominators.clone()
    dist.all_reduce(global_denominators, op=dist.ReduceOp.SUM)
    expected_denominators = sum(
        (_sum_denominators(rank_batches, indices) for rank_batches in all_batches),
        torch.zeros_like(global_denominators),
    )

    model.zero_grad(set_to_none=True)
    for micro_index, batch_index in enumerate(indices):
        final_microbatch = micro_index == len(indices) - 1
        sync_context = model.no_sync() if not final_microbatch else nullcontext()
        with sync_context:
            batch = local_batches[batch_index]
            output = model(batch.x)
            prediction, hint_prediction = output[..., :TARGETS], output[..., TARGETS:]
            numerators = objective_numerators(
                prediction,
                hint_prediction,
                batch.target,
                batch.teacher,
                batch.hint_target,
                batch.mask,
                blend_objective_alpha=alpha,
            )
            loss = scaled_objective(
                numerators,
                global_denominators,
                WORLD_SIZE,
                0.0 if alpha else KD_WEIGHT,
                0.0 if alpha else HINT_WEIGHT,
            )
            loss.backward()

    actual_gradients = [parameter.grad.detach().clone() for parameter in model.module.parameters()]
    expected_gradients = _reference_gradients(
        model, all_batches, indices, expected_denominators, alpha
    )
    max_absolute = _max_absolute_error(actual_gradients, expected_gradients)
    denominators_match = bool(torch.allclose(global_denominators, expected_denominators))
    gradients_match = all(
        torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True)
    )
    return {
        "rank": dist.get_rank(),
        "device": torch.cuda.get_device_name(),
        "indices": list(indices),
        "denominators": {
            "local": local_denominators.detach().cpu().tolist(),
            "global": global_denominators.detach().cpu().tolist(),
            "reference": expected_denominators.detach().cpu().tolist(),
        },
        "max_absolute_gradient_error": max_absolute,
        "passed": denominators_match and gradients_match,
    }


def _run_checks(device: torch.device, rank: int, alpha: float = 0.0) -> list[dict[str, object]]:
    torch.manual_seed(1_729)
    model = DistributedDataParallel(
        nn.Linear(FEATURES, OUTPUTS).to(device),
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=False,
    )
    results: list[dict[str, object]] = []
    for scenario in ("uneven", "empty_rank"):
        all_batches = _make_batches(scenario, device)
        for accumulation in (1, 3):
            windows = [
                list(range(start, min(start + accumulation, CHUNK_COUNT)))
                for start in range(0, CHUNK_COUNT, accumulation)
            ]
            window_results = []
            for indices in windows:
                local = _run_window(model, all_batches, all_batches[rank], indices, alpha)
                gathered: list[dict[str, object] | None] = [None] * WORLD_SIZE
                dist.all_gather_object(gathered, local)
                rank_results = [value for value in gathered if value is not None]
                window_results.append(
                    {
                        "indices": local["indices"],
                        "ranks": rank_results,
                        "passed": len(rank_results) == WORLD_SIZE
                        and all(bool(value["passed"]) for value in rank_results),
                    }
                )
            results.append(
                {
                    "scenario": scenario,
                    "accumulate_chunks": accumulation,
                    "windows": window_results,
                    "passed": all(bool(window["passed"]) for window in window_results),
                }
            )
    return results


def _write_receipt(path: Path, receipt: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        encoded = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_receipt(
    results: list[dict[str, object]], device: torch.device, alpha: float = 0.0
) -> dict[str, object]:
    passed = all(bool(result["passed"]) for result in results)
    return {
        "receipt_schema": "wnn.sequence-ddp-check.v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "world_size": WORLD_SIZE,
        "precision": "strict_fp32_no_tf32_no_autocast",
        "objective_weights": {
            "kd": 0.0 if alpha else KD_WEIGHT,
            "hints": 0.0 if alpha else HINT_WEIGHT,
        },
        **({"blend_objective_alpha": alpha, "curvature_multiplier": alpha**-2} if alpha else {}),
        "gradient_tolerance": {"atol": ATOL, "rtol": RTOL},
        "cases": results,
    }


def _publish(path: Path, receipt: dict[str, object], rank: int, device: torch.device) -> None:
    status: list[dict[str, object]] = [{"ok": True, "error": ""}]
    if rank == 0:
        try:
            _write_receipt(path, receipt)
        except (OSError, TypeError, ValueError) as error:
            status[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(status, src=0, device=device)
    if not status[0]["ok"]:
        raise OSError(str(status[0]["error"]))


def main() -> None:
    """Run the two-rank CUDA acceptance check and write its runtime receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-receipt", "--outputreceipt", required=True, type=Path)
    parser.add_argument("--blend-objective-alpha", type=float, default=0.0)
    args = parser.parse_args()
    if not math.isfinite(args.blend_objective_alpha) or not 0 <= args.blend_objective_alpha <= 1:
        parser.error("--blend-objective-alpha must be finite and in [0, 1]")
    world_size, rank, local_rank = (
        int(os.environ.get(name, default))
        for name, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))
    )
    if world_size != WORLD_SIZE or not 0 <= rank < world_size or not torch.cuda.is_available():
        parser.error("launch with torchrun on exactly two CUDA ranks")
    if local_rank >= torch.cuda.device_count():
        parser.error("each rank requires a visible CUDA device")

    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    try:
        results = _run_checks(device, rank, args.blend_objective_alpha)
        receipt = _runtime_receipt(results, device, args.blend_objective_alpha)
        _publish(args.output_receipt, receipt, rank, device)
        if not bool(receipt["passed"]):
            raise RuntimeError(f"DDP objective gradients failed: {args.output_receipt}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
