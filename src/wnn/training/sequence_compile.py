"""Explicit optional compilation for the Transformer block modules."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn


def compile_transformer_blocks(blocks: nn.ModuleList, *, enabled: bool) -> bool:
    """Compile blocks in place, preserving module parameter and checkpoint paths.

    PyTorch defers compilation until a block is first called and caches compiled
    specializations internally. Static-shape compilation avoids CUDA graph capture;
    graph breaks may run eager segments and must be recorded by the profile.
    """
    if type(enabled) is not bool:
        raise TypeError("enabled must be a bool")
    if not enabled:
        return False
    if len(blocks) == 0:
        raise RuntimeError("cannot compile an empty Transformer block list")

    compile_methods: list[Callable[..., object]] = []
    for index, block in enumerate(blocks):
        compile_method = getattr(block, "compile", None)
        if not callable(compile_method):
            raise RuntimeError(f"Transformer block {index} does not support torch.compile")
        compile_methods.append(compile_method)

    for compile_method in compile_methods:
        compile_method(backend="inductor", mode="default", fullgraph=False, dynamic=None)
    return True
