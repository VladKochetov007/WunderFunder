"""Focused CPU tests for optional in-place Transformer block compilation."""

from __future__ import annotations

import pytest
from torch import nn

from wnn.training.sequence_compile import compile_transformer_blocks


class _CompileProbe(nn.Module):
    """Record compile requests without invoking a compiler or model forward."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.compile_options: dict[str, object] | None = None

    def compile(self, **kwargs: object) -> None:
        self.compile_options = kwargs


def test_disabled_compilation_is_an_in_place_no_op() -> None:
    block = _CompileProbe()
    before = tuple(block.state_dict())

    result = compile_transformer_blocks(nn.ModuleList([block]), enabled=False)

    assert result is False
    assert block.compile_options is None
    assert tuple(block.state_dict()) == before


def test_enabled_compilation_uses_in_place_fullgraph_default_inductor() -> None:
    blocks = nn.ModuleList([_CompileProbe(), _CompileProbe()])
    before = tuple(blocks.state_dict())

    result = compile_transformer_blocks(blocks, enabled=True)

    assert result is True
    assert all(
        block.compile_options
        == {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": False,
            "dynamic": None,
        }
        for block in blocks
    )
    assert tuple(blocks.state_dict()) == before


def test_unsupported_block_fails_before_mutating_any_block() -> None:
    first = _CompileProbe()
    unsupported = _CompileProbe()
    unsupported.compile = None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="block 1 does not support torch.compile"):
        compile_transformer_blocks(nn.ModuleList([first, unsupported]), enabled=True)

    assert first.compile_options is None


@pytest.mark.parametrize("enabled", [0, 1, None, "true"])
def test_enabled_flag_requires_a_boolean(enabled: object) -> None:
    with pytest.raises(TypeError, match="enabled must be a bool"):
        compile_transformer_blocks(nn.ModuleList([_CompileProbe()]), enabled=enabled)  # type: ignore[arg-type]
