from __future__ import annotations

import math
from typing import Any

import torch

# The approved metadata tree contains heterogeneous primitives/containers/tensors.
Metadata = Any
PROJECTION_VERSION = "metadata-json-v1"


def canonical_metadata(value: Metadata) -> dict:
    """Project metadata to typed JSON without accessing tensor numerical values.

    The controller approval builder and audit must call this identical function.
    Tensor descriptors identify metadata ONLY; independently verified full-file
    hashes bind contents. Depth16,4096nodes,1MiB text and1024entry container caps
    prevent broad/accidental recursive scans. No tensor copy, scan or hash occurs.
    """
    nodes = 0
    text_bytes = 0
    ancestors: set[int] = set()

    def visit(item: Metadata, path: tuple[str, ...], depth: int) -> dict:
        nonlocal nodes, text_bytes
        nodes += 1
        if depth > 16 or nodes > 4096:
            raise ValueError("Metadata projection recursion/node budget exceeded")
        if isinstance(item, torch.Tensor):
            if not path or path[-1] not in {"cached_propensity", "scored_mask"}:
                raise ValueError("Unexpected tensor-valued metadata field")
            if item.device.type != "cpu" or len(item.shape) > 16:
                raise ValueError("Tensor metadata must be bounded CPU descriptor")
            return dict(
                kind="tensor_metadata",
                tensor_class=f"{type(item).__module__}.{type(item).__qualname__}",
                shape=list(item.shape),
                dtype=str(item.dtype),
                device=str(item.device),
                layout=str(item.layout),
                requires_grad=bool(item.requires_grad),
                detached=item.grad_fn is None and not item.requires_grad,
            )
        if item is None or type(item) in (str, int, bool, float):
            if type(item) is float and not math.isfinite(item):
                raise ValueError("Nonfinite scalar metadata")
            if type(item) is str:
                text_bytes += len(item.encode("utf-8"))
                if text_bytes > 1048576:
                    raise ValueError("Metadata text budget exceeded")
            return dict(kind="scalar", scalar_type=type(item).__name__, value=item)
        if type(item) not in (dict, list, tuple):
            raise ValueError(f"Unexpected metadata type: {type(item).__name__}")
        if len(item) > 1024 or id(item) in ancestors:
            raise ValueError("Metadata container too large or cyclic")
        ancestors.add(id(item))
        try:
            if type(item) is dict:
                if not all(type(key) is str for key in item):
                    raise ValueError("Metadata dictionary keys must be strings")
                entries = []
                for key in sorted(item):
                    text_bytes += len(key.encode("utf-8"))
                    if text_bytes > 1048576:
                        raise ValueError("Metadata text budget exceeded")
                    entries.append([key, visit(item[key], (*path, key), depth + 1)])
                return dict(kind="mapping", entries=entries)
            return dict(
                kind="sequence",
                sequence_type=type(item).__name__,
                items=[visit(child, path, depth + 1) for child in item],
            )
        finally:
            ancestors.remove(id(item))

    return dict(projection_version=PROJECTION_VERSION, identity=visit(value, (), 0))
