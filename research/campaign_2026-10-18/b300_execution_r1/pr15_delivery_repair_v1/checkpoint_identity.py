"""Content identity for PR15 checkpoints saved as separate Torch archives.

The PR15 trainer saves one Python checkpoint object to ``model_ep2.pt`` and
``model.pt`` in separate ``torch.save`` calls. Zip serialization bytes can
differ while the loaded checkpoint payload is exactly the same. This module
compares the full loaded payload and separately retains both raw file hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Hash one checkpoint archive without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_node(value: Any, torch: Any) -> dict[str, Any]:
    """Return a canonical tensor descriptor including exact logical bytes."""
    if value.layout != torch.strided or value.is_quantized:
        raise TypeError("checkpoint identity supports dense, nonquantized tensors only")
    if getattr(value, "is_conj", lambda: False)() or getattr(value, "is_neg", lambda: False)():
        raise TypeError("checkpoint identity rejects unresolved tensor view flags")
    tensor = value.detach().to(device="cpu").contiguous()
    try:
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    except (RuntimeError, TypeError) as error:
        raise TypeError(f"unsupported checkpoint tensor dtype: {tensor.dtype}") from error
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "layout": str(tensor.layout),
        "requires_grad": bool(value.requires_grad),
        "content_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _canonical(value: Any, torch: Any) -> Any:
    """Normalize supported checkpoint values while preserving type and content."""
    if isinstance(value, torch.Tensor):
        return _tensor_node(value, torch)
    if value is None:
        return {"type": "NoneType"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint metadata contains a nonfinite float")
        return {"type": "float", "hex": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("checkpoint mappings must use string keys")
        attrs = vars(value) if hasattr(value, "__dict__") else {}
        return {
            "type": f"mapping:{type(value).__module__}.{type(value).__qualname__}",
            "items": [[key, _canonical(item, torch)] for key, item in value.items()],
            "attributes": _canonical(attrs, torch) if attrs else {"type": "dict", "items": []},
        }
    if isinstance(value, list):
        return {"type": "list", "items": [_canonical(item, torch) for item in value]}
    if isinstance(value, tuple):
        return {
            "type": f"tuple:{type(value).__module__}.{type(value).__qualname__}",
            "items": [_canonical(item, torch) for item in value],
        }

    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("checkpoint identity rejects object-dtype NumPy arrays")
        array = np.ascontiguousarray(value)
        return {
            "type": "numpy.ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "content_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if np is not None and isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise TypeError("checkpoint identity rejects object-dtype NumPy scalars")
        if isinstance(value, np.floating) and not np.isfinite(value):
            raise ValueError("checkpoint metadata contains a nonfinite NumPy scalar")
        return {
            "type": f"numpy.scalar:{value.dtype.str}",
            "content_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    raise TypeError(f"unsupported checkpoint value type: {type(value)!r}")


def payload_identity_sha256(checkpoint: Any, torch: Any) -> str:
    """Hash all checkpoint keys, metadata, tensor types/shapes and tensor bytes."""
    normalized = _canonical(checkpoint, torch)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CheckpointIdentity:
    """Immutable identities for the fixed and selected checkpoint files."""

    fixed_path: str
    selected_path: str
    fixed_file_sha256: str
    selected_file_sha256: str
    payload_sha256: str
    model_state_sha256: str
    epoch: int
    input_width: int

    def receipt(self) -> dict[str, Any]:
        """Return a JSON-safe receipt preserving both byte and payload identities."""
        return asdict(self)


def verify_checkpoint_pair(
    fixed_path: Path,
    selected_path: Path,
    *,
    expected_epoch: int = 2,
    expected_input_width: int = 271,
) -> CheckpointIdentity:
    """Require exact full-payload equality of fixed and selected Torch checkpoints.

    This does not replace recipe, source, full-budget, or prediction validation.
    Callers must keep those independent checks and bind predictions to the fixed
    checkpoint's raw file hash returned here.
    """
    import torch

    fixed_path = fixed_path.resolve(strict=True)
    selected_path = selected_path.resolve(strict=True)
    fixed = torch.load(fixed_path, map_location="cpu", weights_only=False)
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    if not isinstance(fixed, dict) or not isinstance(selected, dict):
        raise TypeError("PR15 checkpoint payloads must be dictionaries")
    for label, payload in (("fixed", fixed), ("selected", selected)):
        if payload.get("epoch") != expected_epoch:
            raise ValueError(f"{label} checkpoint is not fixed epoch {expected_epoch}")
        if payload.get("n_in") != expected_input_width:
            raise ValueError(f"{label} checkpoint input width is not {expected_input_width}")
        if not isinstance(payload.get("model"), Mapping):
            raise ValueError(f"{label} checkpoint has no model state mapping")
    fixed_payload_sha = payload_identity_sha256(fixed, torch)
    selected_payload_sha = payload_identity_sha256(selected, torch)
    if fixed_payload_sha != selected_payload_sha:
        raise ValueError(
            "selected model.pt payload differs from model_ep2.pt "
            f"(fixed={fixed_payload_sha}, selected={selected_payload_sha})"
        )
    return CheckpointIdentity(
        fixed_path=str(fixed_path),
        selected_path=str(selected_path),
        fixed_file_sha256=sha256_file(fixed_path),
        selected_file_sha256=sha256_file(selected_path),
        payload_sha256=fixed_payload_sha,
        model_state_sha256=payload_identity_sha256(fixed["model"], torch),
        epoch=expected_epoch,
        input_width=expected_input_width,
    )


def validate_provenance_file_hashes(
    provenance: Mapping[str, Any], identity: CheckpointIdentity
) -> None:
    """Bind each saved archive byte hash to its own trainer provenance entry."""
    checkpoint_hashes = provenance.get("checkpoint_sha256")
    expected = {
        "model_ep2.pt": identity.fixed_file_sha256,
        "model.pt": identity.selected_file_sha256,
    }
    if not isinstance(checkpoint_hashes, Mapping) or any(
        checkpoint_hashes.get(filename) != digest for filename, digest in expected.items()
    ):
        raise ValueError("checkpoint bytes differ from the trainer's per-file provenance")


def validate_prediction_pin(
    sidecar: Mapping[str, Any],
    prediction_path: Path,
    identity: CheckpointIdentity,
    *,
    name: str,
    split_manifest_sha256: str,
    evaluation_groups: list[int],
) -> dict[str, str]:
    """Keep the evaluation export pinned to exact epoch-2 bytes and row membership."""
    prediction_sha256 = sha256_file(prediction_path)
    expected = {
        "run": name,
        "epoch": identity.epoch,
        "campaign_export_scope": "evaluation",
        "campaign_split_manifest_sha256": split_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "checkpoint_sha256": identity.fixed_file_sha256,
        "row_groups": evaluation_groups,
        "prediction_sha256": prediction_sha256,
    }
    if any(sidecar.get(key) != value for key, value in expected.items()):
        raise ValueError("prediction sidecar is not pinned to the fixed checkpoint/fold")
    return {
        "checkpoint_file_sha256": identity.fixed_file_sha256,
        "prediction_sha256": prediction_sha256,
    }
