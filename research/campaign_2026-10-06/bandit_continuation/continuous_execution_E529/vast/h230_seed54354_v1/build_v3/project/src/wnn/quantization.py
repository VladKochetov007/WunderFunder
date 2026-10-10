from __future__ import annotations

import hashlib

import numpy as np

PRESETS = ("recurrent", "recurrent_head")


def component_panel(evaluation_groups: tuple[int, ...], components: dict[int, int]) -> list[int]:
    """Select one feature-only representative from each of 64 hashed components."""
    members: dict[int, list[int]] = {}
    for group in evaluation_groups:
        members.setdefault(components[group], []).append(group)
    if len(members) < 64:
        raise ValueError("E411 needs at least 64 distinct evaluation components")

    def rank(value: str) -> str:
        return hashlib.sha256(f"271828:{value}".encode()).hexdigest()

    selected = sorted(members, key=lambda component: rank(f"component:{component}"))[:64]
    return [
        min(members[component], key=lambda group: rank(f"group:{component}:{group}"))
        for component in selected
    ]


def quantize_rows(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode a matrix as signed bytes and float32 per-row scales.

    This is the scalar representation used by native ``PackedQ8`` before its
    tile permutation. Rounding is nearest-even; biases remain unquantized.
    """
    values = np.asarray(weights, dtype=np.float32)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("expected a nonempty finite matrix")
    maximum = np.max(np.abs(values), axis=1)
    scales = np.maximum(maximum / np.float32(127), np.nextafter(np.float32(0), np.float32(1)))
    scales = np.where(maximum == 0, np.float32(0), scales)
    ratios = values / np.where(scales == 0, np.float32(1), scales)[:, None]
    codes = np.clip(np.rint(ratios), -127, 127).astype(np.int8)
    return codes, scales


def represented_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Return the exact float32 matrix represented by valid q8 rows."""
    if (
        codes.dtype != np.int8
        or codes.ndim != 2
        or scales.dtype != np.float32
        or scales.shape != (codes.shape[0],)
        or not np.isfinite(scales).all()
        or (scales < 0).any()
        or (codes == -128).any()
        or ((scales == 0) & (codes != 0).any(axis=1)).any()
    ):
        raise ValueError("invalid signed-byte matrix representation")
    return np.ascontiguousarray(codes.astype(np.float32) * scales[:, None])


def quantize_model(
    weights: dict[str, np.ndarray], preset: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Create fixed recurrent-only or recurrent-plus-head weight-only variants.

    The second preset quantizes head.0 independently for hidden/projection
    columns, matching the planned q16x-style split. Input projection, final
    output layer, normalization and every bias remain float32. Returned codes
    and scales are an audit artifact, not a submission engine format.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown q8 preset: {preset}")
    hidden = int(weights["inp.bias"].size)
    represented = {
        name: np.ascontiguousarray(value, dtype=np.float32) for name, value in weights.items()
    }
    recurrent = sorted(
        name for name in weights if name.startswith(("gru.weight_ih_", "gru.weight_hh_"))
    )
    if not recurrent:
        raise ValueError("checkpoint has no recurrent matrices")
    selected = recurrent + (["head.0.weight"] if preset == "recurrent_head" else [])
    encoded, details = {}, {}
    for name in selected:
        original = represented[name]
        if name == "head.0.weight" and original.shape[1] != 2 * hidden:
            raise ValueError("head.0 must concatenate hidden and projection features")
        blocks = np.split(original, [hidden], axis=1) if name == "head.0.weight" else [original]
        restored, storage = [], 0
        for index, block in enumerate(blocks):
            codes, scales = quantize_rows(block)
            encoded[f"{name}.{index}.codes"] = codes
            encoded[f"{name}.{index}.scales"] = scales
            restored.append(represented_rows(codes, scales))
            # Native tile padding: four rows and sixteen columns per panel.
            storage += ((block.shape[0] + 3) // 4) * (((block.shape[1] + 15) // 16) * 64 + 16)
        represented[name] = np.ascontiguousarray(np.concatenate(restored, axis=1))
        error = represented[name].astype(np.float64) - original
        details[name] = {
            "shape": list(original.shape),
            "blocks": len(blocks),
            "max_abs_error": float(np.abs(error).max()),
            "rms_error": float(np.sqrt(np.mean(error**2))),
            "float32_bytes": original.nbytes,
            "q8_native_padded_bytes": storage,
        }
    return (
        represented,
        encoded,
        {
            "preset": preset,
            "scheme": "symmetric_per_row_int8_nearest_even_v1",
            "selected": details,
            "unquantized": sorted(set(weights) - set(selected)),
            "engine_format": False,
        },
    )


def merge_frozen_heads(
    parent: dict[str, np.ndarray], tuned: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Fuse equal parent/tuned heads only when every trunk byte is identical."""
    if parent.keys() != tuned.keys():
        raise ValueError("parent and tuned checkpoint fields differ")
    for name in parent:
        if not name.startswith("head.") and not np.array_equal(parent[name], tuned[name]):
            raise ValueError(f"head tuning changed the frozen trunk: {name}")
    result = dict(parent)
    result["head.0.weight"] = np.concatenate([parent["head.0.weight"], tuned["head.0.weight"]])
    result["head.0.bias"] = np.concatenate([parent["head.0.bias"], tuned["head.0.bias"]])
    result["head.2.weight"] = np.concatenate(
        [parent["head.2.weight"], tuned["head.2.weight"]], axis=1
    ) * np.float32(0.5)
    result["head.2.bias"] = (parent["head.2.bias"] + tuned["head.2.bias"]) * np.float32(0.5)
    return {name: np.ascontiguousarray(value) for name, value in result.items()}
