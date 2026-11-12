from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from wnn.training.sequence_checkpoint import capture_rng_state, restore_rng_state
from wnn.training.sequence_distributed import DistributedContext, coordinated_call
from wnn.training.sequence_weight_ema import SequenceWeightEMA


def _same(left: object, right: object) -> bool:
    """Compare strict serialization types and every tensor/RNG byte across CPU/GPU location."""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.stride() == right.stride()
            and torch.equal(
                left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                right.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            )
        )
    if isinstance(left, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.strides == right.strides
            and left.tobytes() == right.tobytes()
        )
    if isinstance(left, dict):
        return list(left) == list(right) and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def check_profile_reload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    context: DistributedContext,
    output: Path,
    weight_ema: SequenceWeightEMA | None = None,
) -> dict[str, object]:
    """Round-trip actual-shaped model/optimizer/schedule/RNG; write no resumable scientific schema."""

    def check() -> dict[str, object]:
        """Keep failed engineering evidence, removing only a successfully verified temporary file."""
        path = output / f".engineering-reload-rank{context.rank}.pt"
        expected = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng_state(context),
        }
        if weight_ema is not None:
            expected["weight_ema"] = weight_ema.state_dict()
        with path.open("xb") as stream:
            torch.save(expected, stream)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if not _same(expected, loaded):
            raise ValueError(
                "engineering state serialization changed tensor/RNG bytes or container types"
            )
        model.load_state_dict(loaded["model"], strict=True)
        optimizer.load_state_dict(loaded["optimizer"])
        scheduler.load_state_dict(loaded["scheduler"])
        restore_rng_state(loaded["rng"], context)
        if weight_ema is not None:
            weight_ema.load_state_dict(
                loaded["weight_ema"], model, int(loaded["weight_ema"]["updates"])
            )
        restored = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng_state(context),
        }
        if weight_ema is not None:
            restored["weight_ema"] = weight_ema.state_dict()
        if not _same(loaded, restored):
            raise ValueError(
                "actual-shaped engineering reload changed model/optimizer/scheduler/RNG"
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        path.unlink()
        return {
            "status": "passed",
            "rank": context.rank,
            "roundtrip_sha256": digest.hexdigest(),
            "model_optimizer_scheduler_rng_equal": True,
            "temporary_engineering_state_removed": True,
            "scientific_resume_schema_written": False,
        }

    local = coordinated_call(check, "actual profile-shaped engineering reload", context)
    if context.process_group:
        checks: list[dict[str, object] | None] = [None] * context.world_size
        dist.all_gather_object(checks, local)
    else:
        checks = [local]
    if any(item is None or item["status"] != "passed" for item in checks):
        raise RuntimeError("actual-shaped reload proof is missing a rank")
    return {"status": "passed", "by_rank": checks, "scientific_resume_schema_written": False}
