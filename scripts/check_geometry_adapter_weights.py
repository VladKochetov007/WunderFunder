"""Verify frozen-input geometry checkpoints and inference-only scale aliases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

# Any is limited to heterogeneous PyTorch checkpoint payloads and tensor APIs.
_RAW_EVENT_WIDTH = 211
_GEOMETRY_END = 271
_MASKED_FEATURES = {"moments_only": (20, 60), "dictionary_only": (0, 20)}
_MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Hash a bounded checkpoint file without loading its payload."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def same_value(left: Any, right: Any, torch: Any) -> bool:
    """Compare nested checkpoint values exactly, including tensor bit patterns."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            return False
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        return torch.equal(
            left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(same_value(left[key], right[key], torch) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(same_value(a, b, torch) for a, b in zip(left, right, strict=True))
        )
    comparison = left == right
    return bool(comparison) if isinstance(comparison, bool) else bool(comparison.all())


def load_checkpoint(path: Path, torch: Any) -> dict[str, Any]:
    """Load one local checkpoint after enforcing the expected payload bound."""
    if not path.is_file() or path.stat().st_size > _MAX_CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint is missing or exceeds {_MAX_CHECKPOINT_BYTES} bytes: {path}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or not isinstance(value.get("model"), dict):
        raise ValueError(f"checkpoint lacks a model state mapping: {path}")
    return value


def verify_extension(
    parent: dict[str, Any], child: dict[str, Any], variant: str, torch: Any
) -> dict[str, object]:
    """Check inherited weights and the exact zero mask in a 271-input adapter."""
    parent_state, child_state = parent["model"], child["model"]
    if parent_state.keys() != child_state.keys():
        raise ValueError("parent and child model tensor keys differ")
    base, extended = parent_state.get("inp.weight"), child_state.get("inp.weight")
    if (
        not torch.is_tensor(base)
        or not torch.is_tensor(extended)
        or base.ndim != 2
        or extended.ndim != 2
        or base.shape[1] != _RAW_EVENT_WIDTH
        or extended.shape[1] != _GEOMETRY_END
    ):
        raise ValueError("expected a 211-input parent and 271-input child")
    if not same_value(base, extended[:, :_RAW_EVENT_WIDTH], torch):
        raise ValueError("the inherited raw/event input columns changed")

    checked = []
    for name, old in parent_state.items():
        new = child_state[name]
        if name == "inp.weight":
            continue
        if name in {"mean", "std"} and old.ndim == new.ndim == 1:
            unchanged = old.shape[0] == _RAW_EVENT_WIDTH and same_value(
                old, new[:_RAW_EVENT_WIDTH], torch
            )
        else:
            unchanged = same_value(old, new, torch)
        if not unchanged:
            raise ValueError(f"inherited tensor changed: {name}")
        checked.append(name)

    start, end = _MASKED_FEATURES[variant]
    masked = extended[:, _RAW_EVENT_WIDTH + start : _RAW_EVENT_WIDTH + end]
    nonzero = int(torch.count_nonzero(masked).item())
    if nonzero:
        raise ValueError(f"masked {variant} input block contains {nonzero} nonzero values")
    return {
        "kind": "frozen_parent_geometry_extension",
        "variant": variant,
        "inherited_raw_event_columns": _RAW_EVENT_WIDTH,
        "unchanged_model_tensors": checked,
        "masked_geometry_feature_offsets": [start, end],
        "masked_input_columns": [_RAW_EVENT_WIDTH + start, _RAW_EVENT_WIDTH + end],
        "masked_nonzero_values": nonzero,
    }


def verify_scale_alias(
    source: dict[str, Any], alias: dict[str, Any], alpha: float, torch: Any
) -> dict[str, object]:
    """Prove a scale alias changes only geometry columns by the stated factor."""
    if source.keys() != alias.keys() or not same_value(
        {key: value for key, value in source.items() if key != "model"},
        {key: value for key, value in alias.items() if key != "model"},
        torch,
    ):
        raise ValueError("scale alias changed checkpoint metadata outside model state")
    old_state, new_state = source["model"], alias["model"]
    if old_state.keys() != new_state.keys():
        raise ValueError("source and alias model tensor keys differ")
    old, new = old_state.get("inp.weight"), new_state.get("inp.weight")
    if (
        not torch.is_tensor(old)
        or not torch.is_tensor(new)
        or old.shape != new.shape
        or old.ndim != 2
        or old.shape[1] != _GEOMETRY_END
    ):
        raise ValueError("expected matching 271-input source and alias weights")
    if not same_value(old[:, :_RAW_EVENT_WIDTH], new[:, :_RAW_EVENT_WIDTH], torch):
        raise ValueError("scale alias changed raw/event input columns")
    expected = old[:, _RAW_EVENT_WIDTH:] * alpha
    actual = new[:, _RAW_EVENT_WIDTH:]
    if alpha == 0.0:
        matches = int(torch.count_nonzero(actual).item()) == 0
    else:
        matches = same_value(expected, actual, torch)
    if not matches:
        raise ValueError("geometry input columns do not equal source columns times alpha")
    for name, value in old_state.items():
        if name != "inp.weight" and not same_value(value, new_state[name], torch):
            raise ValueError(f"scale alias changed non-input model tensor: {name}")
    return {
        "kind": "inference_only_geometry_scale_alias",
        "alpha": alpha,
        "changed_input_columns": [_RAW_EVENT_WIDTH, _GEOMETRY_END],
        "raw_event_columns_unchanged": True,
        "all_other_checkpoint_fields_unchanged": True,
    }


def main() -> None:
    """Write a bounded bitwise checkpoint-inheritance proof after fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("extension", "scale-alias"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(_MASKED_FEATURES))
    parser.add_argument("--alpha", type=float, choices=(-0.5, -0.25, 0.0, 0.25, 0.5))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "extension" and (args.variant is None or args.alpha is not None):
        raise ValueError("extension mode requires --variant and forbids --alpha")
    if args.mode == "scale-alias" and (args.alpha is None or args.variant is not None):
        raise ValueError("scale-alias mode requires --alpha and forbids --variant")
    source_path, candidate_path = (
        args.source.resolve(strict=True),
        args.candidate.resolve(strict=True),
    )

    import torch

    source = load_checkpoint(source_path, torch)
    candidate = load_checkpoint(candidate_path, torch)
    check = (
        verify_extension(source, candidate, args.variant, torch)
        if args.mode == "extension"
        else verify_scale_alias(source, candidate, args.alpha, torch)
    )
    receipt = {
        "schema": "wnn.geometry-adapter-weight-proof.v1",
        "passed": True,
        "source_checkpoint": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "candidate_checkpoint": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "check": check,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
