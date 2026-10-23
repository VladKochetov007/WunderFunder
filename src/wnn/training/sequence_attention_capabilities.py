"""Shape-bound eligibility probes for explicit sequence-attention backends."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

import torch
from torch import Tensor

from wnn.training.sequence_attention_masks import (
    _additive_window_mask,
    _trim_window_keys,
    _window_mask,
)

_MAX_EFFICIENT_CAPABILITIES = 128
_EFFICIENT_MASK_FORMATS: OrderedDict[tuple[object, ...], str] = OrderedDict()


def _sdpa_params(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor, dropout_p: float
) -> object:
    """Build an SDPA probe value across supported PyTorch constructor signatures."""
    try:
        return torch.backends.cuda.SDPAParams(query, key, value, mask, dropout_p, False, False)
    except TypeError:
        return torch.backends.cuda.SDPAParams(query, key, value, mask, dropout_p, False)


def _can_use(
    checker: Callable[..., bool],
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor,
    dropout_p: float,
) -> bool:
    """Ask PyTorch whether one forced SDPA backend accepts this exact mask."""
    return bool(checker(_sdpa_params(query, key, value, mask, dropout_p), debug=False))


def _cache_key(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    past_rows: int,
    context_rows: int,
    dropout_p: float,
) -> tuple[object, ...]:
    """Key results by bounded tensor layout and relative cache geometry."""
    return (
        query.device.type,
        query.device.index,
        query.dtype,
        tuple(query.shape),
        tuple(query.stride()),
        tuple(key.shape),
        tuple(key.stride()),
        tuple(value.shape),
        tuple(value.stride()),
        past_rows,
        context_rows,
        float(dropout_p),
    )


def _efficient_mask_format(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    past_rows: int,
    context_rows: int,
    dropout_p: float,
) -> tuple[str, Tensor]:
    """Select boolean or additive exact mask after a cached eligibility probe."""
    key_id = _cache_key(query, key, value, past_rows, context_rows, dropout_p)
    selected = _EFFICIENT_MASK_FORMATS.get(key_id)
    if selected is not None:
        _EFFICIENT_MASK_FORMATS.move_to_end(key_id)
        return selected, _mask_for_format(selected, query, key, past_rows, context_rows)

    checker = torch.backends.cuda.can_use_efficient_attention
    for name in ("boolean", "additive"):
        mask = _mask_for_format(name, query, key, past_rows, context_rows)
        if _can_use(checker, query, key, value, mask, dropout_p):
            _remember(key_id, name)
            return name, mask
    _remember(key_id, "unsupported")
    raise RuntimeError("efficient SDPA rejected boolean and additive exact rolling masks")


def _mask_for_format(
    name: str, query: Tensor, key: Tensor, past_rows: int, context_rows: int
) -> Tensor:
    """Construct the requested exact mask with its padded backing stride."""
    if name == "boolean":
        return _window_mask(query.shape[-2], key.shape[-2], past_rows, context_rows, query.device)
    if name == "additive":
        return _additive_window_mask(
            query.shape[-2],
            key.shape[-2],
            past_rows,
            context_rows,
            query.device,
            query.dtype,
        )
    raise RuntimeError("efficient SDPA rejected boolean and additive exact rolling masks")


def _remember(key: tuple[object, ...], value: str) -> None:
    """Insert one bounded capability result in the LRU cache."""
    _EFFICIENT_MASK_FORMATS[key] = value
    if len(_EFFICIENT_MASK_FORMATS) > _MAX_EFFICIENT_CAPABILITIES:
        _EFFICIENT_MASK_FORMATS.popitem(last=False)


def attention_backend_capabilities(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_start: int,
    key_start: int,
    context_rows: int,
    dropout_p: float = 0.0,
) -> dict[str, dict[str, bool | str]]:
    """Report exact-mask eligibility for math, efficient, flash, cuDNN, and flex."""
    key, value, past_rows = _trim_window_keys(
        query, key, value, query_start, key_start, context_rows
    )
    report: dict[str, dict[str, bool | str]] = {
        "math": {
            "available": True,
            "eligible": True,
            "selectable": True,
            "reason": "always available",
        }
    }
    if not query.is_cuda:
        for backend in ("efficient", "flash", "cudnn"):
            report[backend] = {
                "available": False,
                "eligible": False,
                "selectable": backend == "efficient",
                "mask_formats": "",
                "reason": "requires CUDA tensors",
            }
    else:
        cuda = torch.backends.cuda
        checks = (
            ("efficient", cuda.can_use_efficient_attention, True),
            ("flash", cuda.can_use_flash_attention, False),
            ("cudnn", cuda.can_use_cudnn_attention, False),
        )
        for name, checker, selectable in checks:
            try:
                formats = [
                    mask_format
                    for mask_format in ("boolean", "additive")
                    if _can_use(
                        checker,
                        query,
                        key,
                        value,
                        _mask_for_format(mask_format, query, key, past_rows, context_rows),
                        dropout_p,
                    )
                ]
                eligible = bool(formats)
                reason = (
                    f"eligible with exact {', '.join(formats)} mask"
                    if eligible
                    else "backend rejected boolean and additive exact masks"
                )
            except (RuntimeError, TypeError, AttributeError) as error:
                formats = []
                eligible = False
                reason = f"{type(error).__name__}: {error}"
            report[name] = {
                "available": True,
                "eligible": eligible,
                "selectable": selectable,
                "mask_formats": ",".join(formats),
                "reason": reason,
            }
    try:
        from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask
        from torch.nn.attention.flex_attention import flex_attention as _flex_attention
    except ImportError:
        flex_available = False
    else:
        flex_available = _create_block_mask is not None and _flex_attention is not None
    flex_eligible = flex_available and query.is_cuda and dropout_p == 0.0
    report["flex"] = {
        "available": flex_available,
        "eligible": flex_eligible,
        "selectable": True,
        "reason": (
            "bounded Flex Triton path can be attempted"
            if flex_eligible
            else "requires FlexAttention, CUDA tensors, and zero attention dropout"
        ),
    }
    return report
