"""Read-only run identities and update counters; no training arithmetic changes."""

from __future__ import annotations

import hashlib
import inspect

import numpy as np
import torch


def tensor_identity(value: torch.Tensor) -> dict[str, object]:
    """Hash exact contiguous CPU bytes, including dtype and shape metadata."""
    value = value.detach().contiguous().cpu()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes_sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def normalization_identity(model: torch.nn.Module) -> dict[str, object]:
    """Normalization buffers are set before fitting and are not modified here."""
    return {"mean": tensor_identity(model.mean), "std": tensor_identity(model.std)}


def scaler_source_identity() -> dict[str, object]:
    """Bind the installed private AMP accessor whose result defines actual skips."""
    if not callable(getattr(torch.amp.GradScaler, "_found_inf_per_device", None)):
        raise RuntimeError("installed GradScaler lacks _found_inf_per_device")
    path = inspect.getsourcefile(torch.amp.GradScaler)
    if path is None:
        raise RuntimeError("GradScaler source is unavailable")
    with open(path, "rb") as source:
        sha = hashlib.sha256(source.read()).hexdigest()
    return {"torch_version": torch.__version__, "grad_scaler_source_sha256": sha, "accessor": "_found_inf_per_device after unscale_ and before update", "skip_rule": "any found_inf flag nonzero means the optimizer update was skipped"}


def group_identity(values) -> dict[str, object]:
    """Bind existing ordered memberships without persisting another ID array."""
    values = np.asarray(values, dtype="<i8")
    return {"count": len(values), "ordered_int64_le_sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def sample_order_identity(loader) -> dict[str, object]:
    """Capture existing RNGs before iteration; never construct or alter an order."""
    sampler = loader.sampler
    sampler_generator = getattr(sampler, "generator", None)
    loader_generator = getattr(loader, "generator", None)
    return {"sampler_class": type(sampler).__name__, "batch_size": loader.batch_size, "drop_last": loader.drop_last, "loader_generator": "explicit" if loader_generator is not None else "global CPU RNG", "loader_rng": tensor_identity(loader_generator.get_state() if loader_generator is not None else torch.get_rng_state()), "sampler_generator": "explicit" if sampler_generator is not None else "global CPU RNG", "sampler_rng": tensor_identity(sampler_generator.get_state() if sampler_generator is not None else torch.get_rng_state()), "num_workers": loader.num_workers, "persistent_workers": loader.persistent_workers, "sampler_replacement": getattr(sampler, "replacement", None), "sampler_num_samples": getattr(sampler, "num_samples", None)}


class EpochUpdateCounter:
    """Accumulate skip flags on device; one scalar synchronization per epoch."""

    def __init__(self, device: torch.device) -> None:
        self.attempts = self.ema_updates = self.completed_batches = 0
        self.skips = torch.zeros((), dtype=torch.int64, device=device)
        self.step_skips: list[torch.Tensor] = []
        self.last_step_skips: list[bool] = []

    @torch.no_grad()
    def observe_unscaled(self, scaler, optimizer) -> None:
        """Call immediately after the existing unscale_; never alter its flags."""
        self.attempts += 1
        if not scaler.is_enabled():
            self.step_skips.append(torch.zeros_like(self.skips))
            return
        flags = scaler._found_inf_per_device(optimizer)
        if not flags:
            raise RuntimeError("enabled AMP has no unscaled optimizer inf flags")
        # A nonnegative flag sum matches GradScaler.step's normal and fused paths.
        found = torch.zeros((), device=self.skips.device, dtype=torch.float32)
        for value in flags.values():
            found.add_(value.detach().to(device=self.skips.device, dtype=torch.float32))
        skipped = found.ne(0).to(torch.int64)
        self.step_skips.append(skipped)
        self.skips.add_(skipped)

    def finish(self, cumulative: dict[str, int]) -> dict[str, dict[str, int]]:
        """Read exactly one device scalar after the epoch's fitting loop."""
        skipped = int(self.skips.item())
        self.last_step_skips = [
            bool(value) for value in torch.stack(self.step_skips).detach().cpu().tolist()
        ] if self.step_skips else []
        self.step_skips = []
        epoch = {"optimizer_attempts": self.attempts, "amp_skipped_updates": skipped, "optimizer_completed_updates": self.attempts - skipped, "ema_updates": self.ema_updates, "completed_batches": self.completed_batches}
        for key, value in epoch.items():
            cumulative[key] = cumulative.get(key, 0) + value
        return {"epoch": epoch, "cumulative": cumulative.copy()}
