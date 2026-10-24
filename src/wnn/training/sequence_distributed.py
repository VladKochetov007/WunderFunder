from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    """CUDA and process-group identity for one torchrun worker."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    process_group: bool

    @property
    def is_primary(self) -> bool:
        """Return whether this worker owns shared artifacts and summaries."""
        return self.rank == 0


def configure_strict_fp32() -> None:
    """Disable TF32 and request IEEE FP32 matmul precision for fit and inference."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def initialize_distributed() -> DistributedContext:
    """Initialize one strict-FP32 CUDA worker and NCCL when multiple ranks run."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("invalid torchrun rank configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("sequence fitting requires CUDA; CPU training is disabled")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK exceeds the visible CUDA device count")
    torch.cuda.set_device(local_rank)
    configure_strict_fp32()
    process_group = world_size > 1
    if process_group:
        if not dist.is_available():
            raise RuntimeError("this PyTorch build does not provide distributed support")
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
    return DistributedContext(
        rank, local_rank, world_size, torch.device("cuda", local_rank), process_group
    )


def wrap_distributed(model: torch.nn.Module, context: DistributedContext) -> torch.nn.Module:
    """Wrap the model in one-process-per-GPU DDP when the world has multiple ranks."""
    if not context.process_group:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=True,
        find_unused_parameters=False,
    )


def local_sequence_indices(
    size: int, context: DistributedContext, seed: int, epoch: int
) -> list[int]:
    """Shard one deterministic shuffled permutation, padding only with the -1 sentinel."""
    if size < 1 or epoch < 0:
        raise ValueError("training data must be nonempty and epoch must be nonnegative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    permutation = torch.randperm(size, generator=generator).tolist()
    padding = (-len(permutation)) % context.world_size
    permutation.extend([-1] * padding)
    local = permutation[context.rank :: context.world_size]
    if len(local) != (size + context.world_size - 1) // context.world_size:
        raise RuntimeError("distributed sequence partition produced unequal rank counts")
    return local


def global_sum(values: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """Sum a small device tensor across ranks while preserving its input value."""
    result = values.clone()
    if context.process_group:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def all_ranks_true(local_value: bool, context: DistributedContext) -> bool:
    """Agree on a finite/error predicate before any rank takes a divergent path."""
    flag = torch.tensor(int(local_value), dtype=torch.int32, device=context.device)
    if context.process_group:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def coordinated_call(function: Callable[[], T], label: str, context: DistributedContext) -> T:
    """Run one setup operation and report any rank-local Python failure to every worker."""
    value: T | None = None
    local_error = ""
    try:
        value = function()
    except Exception as error:
        local_error = f"rank {context.rank}: {type(error).__name__}: {error}"
    if not all_ranks_true(not local_error, context):
        errors: list[str | None] = [None] * context.world_size
        if context.process_group:
            dist.all_gather_object(errors, local_error)
        else:
            errors[0] = local_error
        detail = "; ".join(message for message in errors if message)
        raise RuntimeError(f"{label} failed on at least one torchrun rank: {detail}")
    if value is None:
        raise RuntimeError(f"{label} returned no value")
    return value


def collective_output_setup(output: Path, resume: bool, context: DistributedContext) -> None:
    """Create a new run directory on rank zero or validate an explicit resume directory."""
    status: list[dict[str, str | bool]] = [{"ok": True, "error": ""}]
    if context.is_primary:
        try:
            if resume:
                if not output.is_dir():
                    raise FileNotFoundError(f"resume output directory does not exist: {output}")
            else:
                output.mkdir(parents=True, exist_ok=False)
        except (FileExistsError, FileNotFoundError, OSError) as error:
            status[0] = {"ok": False, "error": str(error)}
    if context.process_group:
        dist.broadcast_object_list(status, src=0, device=context.device)
    if not status[0]["ok"]:
        raise FileExistsError(str(status[0]["error"]))


def barrier(context: DistributedContext) -> None:
    """Wait for all workers when a real process group is active."""
    if context.process_group:
        dist.barrier()


def shutdown_distributed(context: DistributedContext) -> None:
    """Release an initialized NCCL process group at normal process completion."""
    if context.process_group and dist.is_initialized():
        dist.destroy_process_group()
