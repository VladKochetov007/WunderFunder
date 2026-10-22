"""Canonical model loading and tensor fingerprints used by fit audits."""

from __future__ import annotations

# Any is required for serialized checkpoint mappings inspected by tensor name.
import hashlib
from pathlib import Path
from typing import Any


def tensor_digest(tensor: Any) -> str:
    """Hash a state tensor after canonical FP32 conversion for float tensors."""
    import torch

    value = tensor.detach().cpu().contiguous()
    if value.is_floating_point():
        value = value.to(dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def non_head_hashes(state: dict[str, Any]) -> dict[str, str]:
    """Fingerprint all frozen trunk and preprocessing tensors, excluding head.*."""
    return {
        name: tensor_digest(value)
        for name, value in sorted(state.items())
        if not name.startswith("head.")
    }


def load_model_state(path: Path) -> tuple[dict[str, Any], Any]:
    """Load one canonical checkpoint and instantiate its exact saved architecture."""
    import torch

    from wnn.training.models import make_model

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    if args.get("arch") != "gru" or args.get("cell") != "gru":
        raise ValueError("the pinned parent must be a plain GRU for inp/gru head-only freezing")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict) or "head.2.weight" not in model_state:
        raise ValueError("checkpoint lacks the canonical GRU readout state")
    model = make_model(args, int(checkpoint["n_in"]), model_state["head.2.weight"].shape[0])
    model.load_state_dict(model_state, strict=True)
    return checkpoint, model
