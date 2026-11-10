from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from wnn.training.sequence_distributed import DistributedContext, configure_strict_fp32, global_sum
from wnn.training.sequence_ensemble_config import MIXTURE_MODE
from wnn.training.sequence_objectives import (
    objective_numerators,
    scaled_objective,
    window_denominators,
)


def gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    intervals: list[tuple[int, int]],
    context: DistributedContext,
) -> list[torch.Tensor]:
    """Exercise the actual full-window WMSE and immediate per-chunk backward/no_sync path."""
    hints = y[..., :0]
    denominator = global_sum(
        sum(
            (window_denominators(y[:, a:b], mask[:, a:b], hints[:, a:b]) for a, b in intervals),
            torch.zeros(2, device=x.device),
        ),
        context,
    )
    for index, (start, stop) in enumerate(intervals):
        sync = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and index < len(intervals) - 1
            else nullcontext()
        )
        with sync:
            p = model(x[:, start:stop])
            nums = objective_numerators(
                p,
                hints[:, start:stop],
                y[:, start:stop],
                reference[:, start:stop],
                hints[:, start:stop],
                mask[:, start:stop],
                blend_objective_alpha=0.1,
                objective_mode=MIXTURE_MODE,
                context=context,
            )
            scaled_objective(
                nums, denominator, context.world_size, 0, 0, microchunks=len(intervals)
            ).backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def main() -> None:
    """Compare every DDP parameter gradient with concatenated single-rank known inputs; fit nothing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file-store", type=Path)
    args = parser.parse_args()
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(1)
    device = torch.device("cuda", local_rank) if args.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    configure_strict_fp32()
    if args.file_store is not None:
        dist.init_process_group(
            args.backend,
            init_method=args.file_store.absolute().as_uri(),
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=45),
        )
    else:
        dist.init_process_group(args.backend, timeout=timedelta(seconds=90))
    if dist.get_world_size() != 2:
        raise ValueError("this oracle requires exactly two ranks")
    context = DistributedContext(rank, local_rank, 2, device, True)
    single = DistributedContext(0, 0, 1, device, False)
    records = []
    try:
        for case in (
            "unequal",
            "empty_rank",
            "global_empty",
            "constant_target",
            "constant_prediction",
            "near_constant",
            "partial_window",
            "zero_rows_rank",
        ):
            torch.manual_seed(811)
            x = torch.randn((7, 11, 3), device=device)
            y = torch.randn((7, 11, 2), device=device) * 1.5
            reference = torch.randn_like(y) * 0.3
            mask = torch.ones((7, 11), dtype=torch.bool, device=device)
            indices = slice(0, 5) if rank == 0 else slice(5, 7)
            if case in ("empty_rank", "zero_rows_rank"):
                mask[5:] = False
            if case == "zero_rows_rank" and rank == 1:
                indices = slice(7, 7)
            if case == "global_empty":
                mask.zero_()
            if case == "constant_target":
                y.fill_(0.4)
            if case in ("constant_prediction", "near_constant"):
                reference.zero_()
            initial = torch.nn.Linear(3, 2).to(device)
            if case == "constant_prediction":
                x.zero_()
            if case == "near_constant":
                with torch.no_grad():
                    initial.weight.mul_(2**-18)
                    initial.bias.fill_(0.5)
            control = torch.nn.Linear(3, 2).to(device)
            control.load_state_dict(initial.state_dict())
            intervals = (
                [(0, 3), (3, 6), (6, 9), (9, 11)]
                if case != "partial_window"
                else [(0, 3), (3, 6), (6, 7)]
            )
            expected = gradients(control, x, y, reference, mask, intervals, single)
            parallel = DistributedDataParallel(
                initial, device_ids=[local_rank] if device.type == "cuda" else None
            )
            actual = gradients(
                parallel,
                x[indices],
                y[indices],
                reference[indices],
                mask[indices],
                intervals,
                context,
            )
            maximum = 0.0
            for got, want in zip(actual, expected, strict=True):
                torch.testing.assert_close(
                    got,
                    want,
                    atol=3e-6,
                    rtol=8e-5,
                    msg=lambda detail, case=case: f"{case}: {detail}",
                )
                maximum = max(maximum, float((got - want).abs().max()))
            records.append(
                {"case": case, "parameters_checked": len(actual), "maximum_absolute_error": maximum}
            )
            dist.barrier()
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                json.dump(
                    {
                        "schema": "wnn.ensemble-objective-ddp-oracle.v1",
                        "status": "passed",
                        "backend": args.backend,
                        "world_size": 2,
                        "optimizer_updates": 0,
                        "ddp_factor": "autograd SUM replication cancels DDP average; no extra factor",
                        "correlation_prediction_std_floor": 0.01,
                        "loss_dtype": "float32",
                        "torch": str(torch.__version__),
                        "cuda_runtime": torch.version.cuda,
                        "device": torch.cuda.get_device_name(device)
                        if device.type == "cuda"
                        else "cpu",
                        "device_uuid": str(
                            getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable")
                        )
                        if device.type == "cuda"
                        else None,
                        "loss_source_sha256": {
                            relative: hashlib.sha256(
                                (
                                    Path(objective_numerators.__code__.co_filename)
                                    .resolve()
                                    .parents[3]
                                    / relative
                                ).read_bytes()
                            ).hexdigest()
                            for relative in (
                                "src/wnn/training/sequence_ensemble_correlation.py",
                                "src/wnn/training/sequence_objectives.py",
                                "src/wnn/training/sequence_ensemble_config.py",
                                "src/wnn/training/sequence_distributed.py",
                                "src/wnn/training/sequence_blend.py",
                            )
                        },
                        "oracle_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "absolute_tolerance": 3e-6,
                        "relative_tolerance": 8e-5,
                        "cases": records,
                    },
                    stream,
                    indent=2,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
