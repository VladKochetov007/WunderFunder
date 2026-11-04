from __future__ import annotations

from uuid import UUID


def normalize_gpu_uuid(value: object) -> str:
    """Normalize canonical CUDA/NVML UUID text without changing stored metadata."""
    if not isinstance(value, str):
        raise ValueError("GPU UUID must be a string")
    raw = value[4:] if value[:4].casefold() == "gpu-" else value
    try:
        normalized = str(UUID(raw))
    except (AttributeError, ValueError) as error:
        raise ValueError("GPU UUID must use canonical 8-4-4-4-12 hexadecimal form") from error
    if normalized != raw.casefold():
        raise ValueError("GPU UUID must use canonical 8-4-4-4-12 hexadecimal form")
    return normalized
