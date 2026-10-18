#!/usr/bin/env python3
"""Bounded config redaction and tensor-schema summaries for W18 checkpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from w18_metadata_core import (
    _OMIT_KEYS,
    _SECRET_RE,
    MAX_DEPTH,
    MAX_ITEMS,
    MAX_METADATA_BYTES,
    require,
)


# Any is necessary for heterogeneous mappings returned by Torch's restricted loader.
def safe_metadata(value: Any, *, depth: int = 0, seen_items: list[int] | None = None) -> Any:
    """Retain bounded primitive config values while removing secret-shaped keys."""
    if seen_items is None:
        seen_items = [0]
    require(depth <= MAX_DEPTH, "metadata nesting exceeds limit")
    seen_items[0] += 1
    require(seen_items[0] <= MAX_ITEMS, "metadata item count exceeds limit")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:4096]
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, child in value.items():
            name = str(key)
            if not _SECRET_RE.search(name):
                output[name[:256]] = safe_metadata(child, depth=depth + 1, seen_items=seen_items)
        return output
    if isinstance(value, (list, tuple)):
        require(len(value) <= MAX_ITEMS, "metadata sequence exceeds limit")
        return [safe_metadata(child, depth=depth + 1, seen_items=seen_items) for child in value]
    return {"unsupported_type": type(value).__name__}


def tensor_metadata(
    value: Any,
    tensor_check: Any,
    path: str = "",
    *,
    depth: int = 0,
    seen_items: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Describe tensor names, shapes and dtypes without reading tensor values."""
    if seen_items is None:
        seen_items = [0]
    require(depth <= MAX_DEPTH, "tensor-schema nesting exceeds limit")
    seen_items[0] += 1
    require(seen_items[0] <= MAX_ITEMS, "tensor-schema item count exceeds limit")
    if tensor_check(value):
        return [
            {"path": path, "shape": [int(dim) for dim in value.shape], "dtype": str(value.dtype)}
        ]
    found: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            if not _SECRET_RE.search(name):
                found.extend(
                    tensor_metadata(
                        child,
                        tensor_check,
                        f"{path}.{name[:256]}" if path else name[:256],
                        depth=depth + 1,
                        seen_items=seen_items,
                    )
                )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(
                tensor_metadata(
                    child,
                    tensor_check,
                    f"{path}[{index}]",
                    depth=depth + 1,
                    seen_items=seen_items,
                )
            )
    return found


def summarize_payload(payload: Any, tensor_check: Any) -> dict[str, Any]:
    """Summarize safe configuration and tensor schema, never tensor values."""
    require(isinstance(payload, Mapping), "checkpoint root must be a mapping")
    metadata: dict[str, Any] = {}
    tensor_index: list[dict[str, Any]] = []
    omitted: list[str] = []
    metadata_items = [0]
    schema_items = [0]
    require(len(payload) <= MAX_ITEMS, "top-level checkpoint key count exceeds limit")
    for key, value in payload.items():
        name = str(key)
        if _SECRET_RE.search(name):
            omitted.append(name)
        elif name.lower() in _OMIT_KEYS or tensor_check(value):
            omitted.append(name)
            tensor_index.extend(tensor_metadata(value, tensor_check, name, seen_items=schema_items))
        else:
            metadata[name] = safe_metadata(value, seen_items=metadata_items)
            tensor_index.extend(tensor_metadata(value, tensor_check, name, seen_items=schema_items))
    encoded = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode()
    require(len(encoded) <= MAX_METADATA_BYTES, "safe metadata output exceeds limit")
    require(len(tensor_index) <= MAX_ITEMS, "tensor schema count exceeds limit")
    return {
        "top_level_keys": sorted(str(key) for key in payload),
        "metadata": metadata,
        "tensor_schema": tensor_index,
        "redacted_or_weight_payload_keys": sorted(set(omitted)),
    }
