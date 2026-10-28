from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable

import torch
from attrs import define
from torch import nn

from wnn.training.sequence_models import detach_state


@define
class ChannelMoments:
    """Merge float64 channel moments without retaining activations or gradients."""

    count: int = 0
    calls: int = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    def observe(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        """Observe one step's pre-BN batch; normalization remains in eval mode."""
        del module
        values = inputs[0].detach().to(dtype=torch.float64)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("BN diagnostic requires step-backend [batch, channels] inputs")
        count = values.shape[0]
        variance, mean = torch.var_mean(values, dim=0, correction=0)
        m2 = variance * count
        if self.mean is None or self.m2 is None:
            self.mean, self.m2 = mean, m2
        else:
            delta = mean - self.mean
            total = self.count + count
            self.mean = self.mean + delta * (count / total)
            self.m2 = self.m2 + m2 + delta.square() * (self.count * count / total)
        self.count += count
        self.calls += 1

    def install(self, norm: nn.BatchNorm1d) -> None:
        """Use pooled unbiased variance, matching PyTorch running_var convention."""
        if self.count < 2 or self.mean is None or self.m2 is None:
            raise ValueError("BN recalibration observed insufficient inputs")
        variance = self.m2 / (self.count - 1)
        if not torch.isfinite(self.mean).all() or not torch.isfinite(variance).all():
            raise ValueError("nonfinite BN calibration moments")
        if norm.running_mean is None or norm.running_var is None:
            raise ValueError("BN must track running statistics")
        norm.running_mean.copy_(self.mean)
        norm.running_var.copy_(variance.clamp_min(0))
        if norm.num_batches_tracked is None:
            raise ValueError("BN counter missing")
        norm.num_batches_tracked.fill_(self.calls)


def tensor_hash(value: torch.Tensor) -> str:
    """Hash the actual tensor representation, including shape and dtype."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(tensor.shape), tensor.dtype)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_hashes(model: nn.Module) -> dict[str, str]:
    """Capture all parameters and buffers for exact no-other-change verification."""
    return {name: tensor_hash(value) for name, value in model.state_dict().items()}


def bn_order(model: nn.Module) -> list[tuple[str, nn.BatchNorm1d]]:
    """Require the known acyclic GALAM BN dependency order, rejecting new topology."""
    core = getattr(model, "galam", None)
    if core is None:
        raise TypeError("diagnostic requires a canonical GALAM sequence adapter")
    if getattr(model, "execution_backend", "step") != "step":
        raise ValueError("BN diagnostic requires the original step backend")
    names = ["galam.local_input_norm.norm"]
    for index, _ in enumerate(getattr(core, "global_mixers", ())):
        names.extend(
            (f"galam.global_mixers.{index}.norm1.norm", f"galam.global_mixers.{index}.norm2.norm")
        )
    modules = {
        name: module for name, module in model.named_modules() if isinstance(module, nn.BatchNorm1d)
    }
    if set(names) != set(modules):
        raise ValueError("unsupported BN dependency graph")
    if any(not module.track_running_stats for module in modules.values()):
        raise ValueError("stateless BN cannot be recalibrated")
    return [(name, modules[name]) for name in names]


def bn_snapshot(norm: nn.BatchNorm1d) -> dict[str, object]:
    """Return small human-readable buffer evidence alongside content hashes."""
    if norm.running_mean is None or norm.running_var is None or norm.num_batches_tracked is None:
        raise ValueError("BN running buffers missing")
    return {
        "mean": norm.running_mean.detach().cpu().tolist(),
        "variance": norm.running_var.detach().cpu().tolist(),
        "num_batches_tracked": int(norm.num_batches_tracked.item()),
        "momentum": norm.momentum,
        "eps": norm.eps,
    }


@torch.no_grad()
def recalibrate(
    model: nn.Module,
    batches: Callable[[], Iterable[torch.Tensor]],
    chunk_rows: int,
    device: torch.device,
    check_deadline: Callable[[], None],
) -> list[dict[str, object]]:
    """Recompute each BN's moments with already-calibrated upstream BNs in eval.

    Every layer receives a fresh pass through identical train-only membership.
    Eval mode avoids batch-composition feedback. Pooling time positions estimates
    a fixed inference statistic; this deliberately differs from per-time training
    normalization and is an inference-only hypothesis, not equivalent training.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    model.eval()
    before = state_hashes(model)
    reports: list[dict[str, object]] = []
    allowed: set[str] = set()
    for name, norm in bn_order(model):
        collector = ChannelMoments()
        old = bn_snapshot(norm)
        handle = norm.register_forward_pre_hook(collector.observe)
        try:
            for inputs in batches():
                state = None
                for start in range(0, inputs.shape[1], chunk_rows):
                    check_deadline()
                    _, _, state = model(inputs[:, start : start + chunk_rows].to(device), state)
                    state = detach_state(state)
        finally:
            handle.remove()
        collector.install(norm)
        reports.append(
            {
                "module": name,
                "rows": collector.count,
                "calls": collector.calls,
                "before": old,
                "after": bn_snapshot(norm),
            }
        )
        allowed.update(
            f"{name}.{suffix}" for suffix in ("running_mean", "running_var", "num_batches_tracked")
        )
    after = state_hashes(model)
    if before.keys() != after.keys() or any(
        before[name] != after[name] for name in before.keys() - allowed
    ):
        raise RuntimeError("recalibration changed a parameter or non-BN buffer")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("recalibration produced or received parameter gradients")
    return reports
