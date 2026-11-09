"""Receiver-only two-rank gradient equivalence for the actual causal stem model."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from wnn.training.sequence_distributed import (
    DistributedContext,
    coordinated_call,
    initialize_distributed,
    shutdown_distributed,
)
from wnn.training.sequence_ensemble_config import STANDALONE_LABELS_MODE
from wnn.training.sequence_models import build_sequence_model, detach_state
from wnn.training.sequence_objectives import (
    objective_numerators,
    scaled_objective,
    window_denominators,
)

CHUNKS = ((0, 7), (7, 19), (19, 32))


def publish_receipt(path: Path, receipt: dict[str, object], context: DistributedContext) -> bool:
    """Publish on rank zero while propagating write failures to every worker."""
    if context.is_primary:
        with path.open("x") as stream:
            stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return True


def backward_chunks(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    denominator: torch.Tensor,
    context: DistributedContext,
) -> None:
    """Match detached causal history and no_sync accumulation before one reduction."""
    state = None
    for index, (start, stop) in enumerate(CHUNKS):
        sync = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and index < len(CHUNKS) - 1
            else nullcontext()
        )
        with sync:
            pred, hints, state = model(x[:, start:stop], state)
            state = detach_state(state)
            empty = torch.empty_like(hints)
            nums = objective_numerators(
                pred,
                hints,
                y[:, start:stop],
                torch.zeros_like(pred),
                empty,
                mask[:, start:stop],
                objective_mode=STANDALONE_LABELS_MODE,
                context=context,
            )
            scaled_objective(nums, denominator, context.world_size, 0.0, 0.0).backward()


def check_case(context: DistributedContext, empty_rank: bool) -> dict[str, object]:
    """Compare every parameter gradient, including stem weights, to the global batch."""
    config = {
        "architecture": "transformer",
        "hidden": 32,
        "layers": 2,
        "heads": 4,
        "ff_multiplier": 2,
        "context_rows": 16,
        "input_stem_kernel": 16,
        "dropout": 0.0,
        "hint_dim": 0,
        "precision_mode": "fp32",
        "attention_backend": "math",
    }
    torch.manual_seed(81103)
    base = build_sequence_model(config).to(context.device).train()
    with torch.no_grad():
        dict(base.named_parameters())["input_stem.weight"].fill_(0.01)
    model = DistributedDataParallel(base, device_ids=[context.local_rank], broadcast_buffers=False)
    reference = build_sequence_model(config).to(context.device).train()
    reference.load_state_dict(base.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(492)
    x = torch.randn(2, 32, 112, generator=generator).to(context.device)
    y = (torch.randn(2, 32, 2, generator=generator) * 1.2).to(context.device)
    mask = torch.ones(2, 32, dtype=torch.bool, device=context.device)
    mask[0, ::3] = False
    mask[1, :17] = False
    if empty_rank:
        mask[1] = False
    rank = context.rank
    denominator = window_denominators(
        y[rank : rank + 1], mask[rank : rank + 1], y.new_empty(1, 32, 0)
    )
    dist.all_reduce(denominator)
    model.zero_grad(set_to_none=True)
    backward_chunks(
        model, x[rank : rank + 1], y[rank : rank + 1], mask[rank : rank + 1], denominator, context
    )
    reference.zero_grad(set_to_none=True)
    single = DistributedContext(0, context.local_rank, 1, context.device, False)
    backward_chunks(reference, x, y, mask, denominator, single)
    expected = dict(reference.named_parameters())
    errors = []
    passed = True
    for name, parameter in base.named_parameters():
        actual, wanted = parameter.grad, expected[name].grad
        if actual is None or wanted is None:
            passed = False
            continue
        passed = (
            passed
            and bool(torch.isfinite(actual).all())
            and torch.allclose(actual, wanted, atol=2e-5, rtol=2e-4)
        )
        errors.append(float((actual - wanted).abs().max()))
    stem = dict(base.named_parameters())["input_stem.weight"].grad
    return {
        "empty_rank": empty_rank,
        "rank": rank,
        "passed": bool(passed),
        "max_absolute_gradient_error": max(errors, default=0.0),
        "stem_gradient_nonzero": stem is not None and bool(stem.abs().sum() > 0),
    }


def main() -> None:
    """Run only on two allocated CUDA ranks; fail if any rank/case differs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("WORLD_SIZE") != "2"
        or os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION", "0") != "0"
    ):
        raise ValueError("requires exactly two ranks and strict-FP32 process settings")
    context = initialize_distributed()
    try:
        local = [check_case(context, empty_rank) for empty_rank in (False, True)]
        gathered = [None, None]
        dist.all_gather_object(gathered, local)
        passed = all(
            case["passed"] and case["stem_gradient_nonzero"] for cases in gathered for case in cases
        )
        receipt = {
            "schema": "wnn.input-stem-ddp.v1",
            "world_size": 2,
            "passed": passed,
            "input_stem_kernel": 16,
            "cases": gathered,
            "torch": str(torch.__version__),
            "precision": "fp32",
            "atol": 2e-5,
            "rtol": 2e-4,
        }
        coordinated_call(
            lambda: publish_receipt(args.output, receipt, context), "stem receipt", context
        )
        if not passed:
            raise RuntimeError("causal input stem DDP equivalence failed")
    finally:
        shutdown_distributed(context)


if __name__ == "__main__":
    main()
