from __future__ import annotations

import copy
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

# PyTorch optimizer, scheduler and scaler state dictionaries have heterogeneous,
# implementation-defined nested values; preserve them without coercing their schema.
State = dict[str, Any]
FORMAT = "wnn.epoch-training-state"
VERSION = 1


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def capture_rng() -> State:
    """Capture all process RNGs without advancing them."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    """Restore RNGs last, after constructing and restoring every training object."""
    if state["cuda"] and len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("resume CUDA device count differs from the saved RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def loader_state(loader: Any) -> State:
    """Save epoch-boundary sampler and explicit generator states, never an iterator."""
    if loader.persistent_workers:
        raise ValueError("exact epoch resume does not support persistent loader workers")
    sampler = loader.sampler
    state = {
        "sampler_type": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
        "batch_size": loader.batch_size,
        "drop_last": loader.drop_last,
        "num_workers": loader.num_workers,
        "dataset_length": len(loader.dataset),
    }
    for name, owner in (("loader", loader), ("sampler", sampler)):
        generator = getattr(owner, "generator", None)
        state[f"{name}_generator"] = None if generator is None else generator.get_state()
    state["sampler_state"] = sampler.state_dict() if hasattr(sampler, "state_dict") else None
    state["sampler_epoch"] = getattr(sampler, "epoch", None)
    return state


def restore_loader(loader: Any, saved: Mapping[str, Any]) -> None:
    """Restore an exhausted nonpersistent loader's next-epoch sampling state."""
    current = loader_state(loader)
    for key in ("sampler_type", "batch_size", "drop_last", "num_workers", "dataset_length"):
        if current[key] != saved[key]:
            raise ValueError(f"resume loader {key} differs")
    for name, owner in (("loader", loader), ("sampler", loader.sampler)):
        generator = getattr(owner, "generator", None)
        value = saved[f"{name}_generator"]
        if (generator is None) != (value is None):
            raise ValueError(f"resume {name} generator mode differs")
        if generator is not None:
            generator.set_state(value)
    if saved["sampler_state"] is not None:
        loader.sampler.load_state_dict(saved["sampler_state"])
    if saved["sampler_epoch"] is not None:
        loader.sampler.set_epoch(saved["sampler_epoch"])


def _averaged_state(averaged: Any) -> State | None:
    if averaged is None:
        return None
    if hasattr(averaged, "n_averaged"):
        return {"kind": "torch", "state": averaged.state_dict()}
    return {
        "kind": "fast",
        "state": averaged.module.state_dict(),
        "n": averaged.n,
        "decay": averaged.decay,
    }


def _restore_averaged(averaged: Any, saved: State | None) -> None:
    if (averaged is None) != (saved is None):
        raise ValueError("resume averaging configuration differs")
    if saved is None:
        return
    kind = "torch" if hasattr(averaged, "n_averaged") else "fast"
    if kind != saved["kind"]:
        raise ValueError("resume averaging implementation differs")
    if kind == "torch":
        averaged.load_state_dict(saved["state"])
    else:
        if averaged.decay != saved["decay"]:
            raise ValueError("resume EMA decay differs")
        averaged.module.load_state_dict(saved["state"])
        averaged.n = saved["n"]


def capture_training_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    loader: Any,
    contract: State,
    next_epoch: int,
    progress: State,
    ema: Any = None,
    swa: Any = None,
    auxiliary: State | None = None,
) -> State:
    """Snapshot raw weights and full continuation state at an exhausted epoch boundary.

    The caller must finish evaluation and all scheduled updates first. Recurrent
    batch state is deliberately absent: the trainer resets it for each sequence.
    Normalization buffers are included in the raw and averaged model state dicts.
    """
    if not 1 <= next_epoch <= contract["epochs"]:
        raise ValueError("next_epoch is outside the fixed schedule horizon")
    return _cpu_copy(
        {
            "format": FORMAT,
            "version": VERSION,
            "boundary": "epoch",
            "contract": contract,
            "next_epoch": next_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": _averaged_state(ema),
            "swa": _averaged_state(swa),
            "loader": loader_state(loader),
            "auxiliary": auxiliary or {},
            "progress": progress,
            "rng": capture_rng(),
        }
    )


def load_training_state(path: Path, contract: State | None = None) -> State:
    """Read trusted local training state, rejecting inference snapshots and drift."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(state, dict)
        or state.get("format") != FORMAT
        or state.get("version") != VERSION
        or state.get("boundary") != "epoch"
    ):
        raise ValueError("resume requires a full epoch training state, not an inference checkpoint")
    required = {
        "contract",
        "next_epoch",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "ema",
        "swa",
        "loader",
        "auxiliary",
        "progress",
        "rng",
    }
    if required - state.keys():
        raise ValueError(f"incomplete training state: {sorted(required - state.keys())}")
    if contract is not None and state["contract"] != contract:
        changed = sorted(
            key
            for key in state["contract"].keys() | contract.keys()
            if state["contract"].get(key) != contract.get(key)
        )
        raise ValueError(f"incompatible resume contract fields: {changed}")
    if not 1 <= state["next_epoch"] <= state["contract"]["epochs"]:
        raise ValueError("invalid saved epoch boundary")
    return state


def restore_training_state(
    state: State,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    loader: Any,
    ema: Any = None,
    swa: Any = None,
) -> None:
    """Restore constructed objects; caller restores auxiliary state, then RNG last."""
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    _restore_averaged(ema, state["ema"])
    _restore_averaged(swa, state["swa"])
    restore_loader(loader, state["loader"])


def atomic_save_training_state(state: State, path: Path) -> None:
    """Publish a complete durable checkpoint, preserving the old file on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
