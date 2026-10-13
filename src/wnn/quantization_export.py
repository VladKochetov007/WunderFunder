from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wnn.quantization import (
    LEGACY_PRESETS,
    PRESETS,
    bits_for_matrix,
    quantize_model,
    represented_rows,
    represented_rows_q4,
)

METADATA = "__q8_metadata__"
PACKED_METADATA = "__quantized_rows_metadata__"


def pack_q8_codes(codes: np.ndarray) -> np.ndarray:
    """Store signed Q8 codes as flat row-major bytes without changing any bit.

    Args:
        codes: Nonempty two-dimensional int8 codes.

    Returns:
        A flat uint8 array preserving the signed-byte bits.

    Raises:
        ValueError: If the matrix is empty, has another dtype, or contains -128.
    """
    if codes.dtype != np.int8 or codes.ndim != 2 or 0 in codes.shape or (codes == -128).any():
        raise ValueError("expected a nonempty signed Q8 matrix without -128")
    return np.ascontiguousarray(codes).view(np.uint8).reshape(-1).copy()


def unpack_q8_codes(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Restore flat row-major bytes as signed Q8 codes of the declared shape.

    Args:
        packed: Flat uint8 code bytes.
        shape: Positive `(rows, columns)` matrix shape.

    Returns:
        A copied row-major int8 code matrix.

    Raises:
        ValueError: If the byte count, dtype, shape, or code range is invalid.
    """
    if (
        packed.dtype != np.uint8
        or packed.ndim != 1
        or len(shape) != 2
        or min(shape) <= 0
        or packed.size != shape[0] * shape[1]
    ):
        raise ValueError("invalid packed Q8 codes or matrix shape")
    codes = packed.view(np.int8).reshape(shape).copy()
    if (codes == -128).any():
        raise ValueError("packed Q8 codes contain invalid -128")
    return codes


def pack_q4_codes(codes: np.ndarray) -> np.ndarray:
    """Pack signed Q4 values low nibble first in flattened row-major order.

    The low nibble contains the first code and the high nibble the second. An
    odd final code count is padded with a zero high nibble. -8 is forbidden.

    Args:
        codes: Nonempty two-dimensional int8 codes in [-7, 7].

    Returns:
        A flat uint8 array containing two signed nibbles per byte.

    Raises:
        ValueError: If the matrix is empty, has another dtype, or contains -8.
    """
    if (
        codes.dtype != np.int8
        or codes.ndim != 2
        or 0 in codes.shape
        or (codes < -7).any()
        or (codes > 7).any()
    ):
        raise ValueError("expected a nonempty signed Q4 matrix in [-7, 7]")
    flat = codes.reshape(-1).astype(np.int16)
    nibbles = np.zeros((flat.size + 1) // 2 * 2, dtype=np.uint8)
    nibbles[: flat.size] = np.bitwise_and(flat, 0x0F).astype(np.uint8)
    return nibbles[0::2] | (nibbles[1::2] << 4)


def unpack_q4_codes(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Unpack low-nibble-first Q4 bytes, rejecting -8 and nonzero pad nibbles.

    Args:
        packed: Flat uint8 code bytes.
        shape: Positive `(rows, columns)` matrix shape.

    Returns:
        A copied row-major int8 code matrix.

    Raises:
        ValueError: If the byte count, dtype, padding, or code range is invalid.
    """
    if (
        packed.dtype != np.uint8
        or packed.ndim != 1
        or len(shape) != 2
        or min(shape) <= 0
        or packed.size != (shape[0] * shape[1] + 1) // 2
    ):
        raise ValueError("invalid packed Q4 codes or matrix shape")
    nibbles = np.empty(packed.size * 2, dtype=np.uint8)
    nibbles[0::2] = packed & 0x0F
    nibbles[1::2] = packed >> 4
    size = shape[0] * shape[1]
    if size % 2 and nibbles[-1] != 0:
        raise ValueError("odd Q4 code count requires a zero high-nibble pad")
    nibbles = nibbles[:size]
    codes = np.where(nibbles < 8, nibbles, nibbles.astype(np.int16) - 16).astype(np.int8)
    if (codes == -8).any():
        raise ValueError("packed Q4 codes contain invalid -8")
    return codes.reshape(shape)


def write_q8_export(
    path: Path,
    weights: dict[str, np.ndarray],
    preset: str,
    encoded: dict[str, np.ndarray] | None = None,
) -> None:
    """Save byte matrices, exact float32 scales and unchanged tensors in one NPZ.

    Existing checkpoint codes take precedence when supplied, so effective QAT
    weights are never requantized during transport. Native constructors receive
    the restored float tensors plus the preserved ``q8_codes`` mapping.
    """
    if path.exists():
        raise FileExistsError(path)
    if preset not in LEGACY_PRESETS or METADATA in weights:
        raise ValueError("invalid preset or reserved metadata field")
    selected = sorted(
        name for name in weights if name.startswith(("gru.weight_ih_", "gru.weight_hh_"))
    )
    if not selected:
        raise ValueError("no recurrent matrices")
    if preset == "recurrent_head":
        selected.append("head.0.weight")
    if encoded is None:
        _, encoded, _ = quantize_model(weights, preset)
    expected = {
        f"{name}.{block}.{suffix}"
        for name in selected
        for block in range(2 if name == "head.0.weight" else 1)
        for suffix in ("codes", "scales")
    }
    if set(encoded) != expected:
        raise ValueError("preserved code fields do not match the quantization preset")
    for name in selected:
        blocks = [
            represented_rows(encoded[f"{name}.{block}.codes"], encoded[f"{name}.{block}.scales"])
            for block in range(2 if name == "head.0.weight" else 1)
        ]
        if np.concatenate(blocks, axis=1).shape != weights[name].shape:
            raise ValueError(f"preserved code shape disagrees with {name}")
    metadata = {"version": 1, "preset": preset, "selected": selected, "format": "wnn_q8_rows_v1"}
    payload = {
        name: np.ascontiguousarray(value, dtype=np.float32)
        for name, value in weights.items()
        if name not in selected
    }
    payload.update(encoded)
    payload[METADATA] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8)
    with path.open("xb") as output:
        np.savez_compressed(output, **payload)


def read_q8_export(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    """Restore exactly represented tensors and the original byte/scales mapping."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive[METADATA].tobytes())
        if (
            metadata.get("version") != 1
            or metadata.get("format") != "wnn_q8_rows_v1"
            or metadata.get("preset") not in LEGACY_PRESETS
        ):
            raise ValueError("unsupported q8 export format")
        encoded_names = {
            f"{name}.{block}.{suffix}"
            for name in metadata["selected"]
            for block in range(2 if name == "head.0.weight" else 1)
            for suffix in ("codes", "scales")
        }
        encoded = {name: np.ascontiguousarray(archive[name]) for name in encoded_names}
        weights = {
            name: np.ascontiguousarray(archive[name])
            for name in archive.files
            if name not in encoded_names and name != METADATA
        }
    for name in metadata["selected"]:
        if name in weights:
            raise ValueError("export contains both quantized and float matrix values")
        weights[name] = np.ascontiguousarray(
            np.concatenate(
                [
                    represented_rows(
                        encoded[f"{name}.{block}.codes"], encoded[f"{name}.{block}.scales"]
                    )
                    for block in range(2 if name == "head.0.weight" else 1)
                ],
                axis=1,
            )
        )
    return weights, encoded, metadata["preset"]


def write_packed_quantized_export(
    path: Path,
    weights: dict[str, np.ndarray],
    preset: str,
    encoded: dict[str, np.ndarray] | None = None,
) -> None:
    """Write a versioned Q8/Q4 archive with exact scales and packed code bytes.

    Q4 codes use low-nibble-first row-major packing; Q8 codes remain one signed
    byte each. Supplied QAT codes are preserved and never requantized.

    Args:
        path: New NPZ path. Existing paths are never overwritten.
        weights: Model matrices, with effective values when `encoded` is supplied.
        preset: Versioned policy name from `wnn.quantization.PRESETS`.
        encoded: Optional exact `.codes`/`.scales` arrays from QAT export.

    Raises:
        FileExistsError: If `path` already exists.
        ValueError: If the policy, fields, scales, codes, or represented weights disagree.
    """
    if path.exists():
        raise FileExistsError(path)
    if preset not in PRESETS or PACKED_METADATA in weights or METADATA in weights:
        raise ValueError("invalid preset or reserved metadata field")
    effective_weights, generated, report = quantize_model(weights, preset)
    codes = generated if encoded is None else encoded
    source_weights = weights if encoded is not None else effective_weights
    if set(codes) != set(generated):
        raise ValueError("preserved code fields do not match the quantization preset")
    selected = report["selected"]
    payload = {
        name: np.ascontiguousarray(value, dtype=np.float32)
        for name, value in source_weights.items()
        if name not in selected
    }
    matrices: dict[str, object] = {}
    for name, details in selected.items():
        blocks = []
        restored = []
        for index, bits in enumerate(details["block_bits"]):
            code_name = f"{name}.{index}.codes"
            scale_name = f"{name}.{index}.scales"
            block_codes = np.ascontiguousarray(codes[code_name])
            block_scales = np.ascontiguousarray(codes[scale_name])
            original_block = (
                np.split(
                    np.asarray(source_weights[name], dtype=np.float32),
                    [int(source_weights["inp.bias"].size)],
                    axis=1,
                )[index]
                if name == "head.0.weight"
                else np.asarray(source_weights[name], dtype=np.float32)
            )
            if bits == 4:
                block = represented_rows_q4(block_codes, block_scales)
                packed = pack_q4_codes(block_codes)
            else:
                block = represented_rows(block_codes, block_scales)
                packed = pack_q8_codes(block_codes)
            if block.shape != original_block.shape:
                raise ValueError(f"preserved code shape disagrees with {code_name}")
            restored.append(block)
            packed_name = f"{name}.{index}.packed_codes"
            payload[packed_name] = packed
            payload[scale_name] = block_scales
            blocks.append(
                {
                    "shape": list(block.shape),
                    "bits": bits,
                    "packed_codes": packed_name,
                    "scales": scale_name,
                }
            )
        if name in source_weights and not np.array_equal(
            np.concatenate(restored, axis=1), np.asarray(source_weights[name], dtype=np.float32)
        ):
            raise ValueError(f"effective weights disagree with preserved codes for {name}")
        matrices[name] = {"shape": list(source_weights[name].shape), "blocks": blocks}
    metadata = {
        "version": 2,
        "format": "wnn_quantized_rows_v2",
        "preset": preset,
        "policy": report,
        "matrices": matrices,
        "unquantized": sorted(set(source_weights) - set(selected)),
    }
    payload[PACKED_METADATA] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8
    )
    with path.open("xb") as output:
        np.savez_compressed(output, **payload)


def read_packed_quantized_export(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Read packed Q8/Q4 rows and restore the exact native code/scale mapping.

    Args:
        path: Packed archive produced by `write_packed_quantized_export`.

    Returns:
        Effective float32 model weights, exact expanded code/scale arrays, and
        versioned archive metadata.

    Raises:
        ValueError: If the archive version, fields, shapes, scales, or codes are invalid.
    """
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive[PACKED_METADATA].tobytes())
        if (
            metadata.get("version") != 2
            or metadata.get("format") != "wnn_quantized_rows_v2"
            or metadata.get("preset") not in PRESETS
        ):
            raise ValueError("unsupported packed quantized export format")
        matrices = metadata.get("matrices")
        unquantized = metadata.get("unquantized")
        if not isinstance(matrices, dict) or not isinstance(unquantized, list):
            raise ValueError("invalid packed quantized export metadata")
        if set(matrices) & set(unquantized):
            raise ValueError("quantized and float archive fields overlap")
        expected = {PACKED_METADATA, *unquantized}
        for matrix in matrices.values():
            if not isinstance(matrix, dict) or not isinstance(matrix.get("blocks"), list):
                raise ValueError("invalid packed matrix metadata")
            for block in matrix["blocks"]:
                expected.update((block["packed_codes"], block["scales"]))
        if set(archive.files) != expected:
            raise ValueError("packed quantized archive fields do not match metadata")
        weights = {name: np.ascontiguousarray(archive[name]) for name in unquantized}
        encoded: dict[str, np.ndarray] = {}
        for name, matrix in matrices.items():
            shape = tuple(map(int, matrix["shape"]))
            expected_bits = bits_for_matrix(name, metadata["preset"])
            expected_blocks = 2 if name == "head.0.weight" else 1
            if expected_bits not in (4, 8) or len(matrix["blocks"]) != expected_blocks:
                raise ValueError("packed matrix selection disagrees with quantization preset")
            blocks = []
            for index, block in enumerate(matrix["blocks"]):
                block_shape = tuple(map(int, block["shape"]))
                bits = block["bits"]
                if bits != expected_bits:
                    raise ValueError("packed matrix bit width disagrees with quantization preset")
                packed = np.ascontiguousarray(archive[block["packed_codes"]])
                codes = (
                    unpack_q4_codes(packed, block_shape)
                    if bits == 4
                    else unpack_q8_codes(packed, block_shape)
                    if bits == 8
                    else None
                )
                if codes is None:
                    raise ValueError("unsupported packed quantized bit width")
                scales = np.ascontiguousarray(archive[block["scales"]])
                represented = (
                    represented_rows_q4(codes, scales)
                    if bits == 4
                    else represented_rows(codes, scales)
                )
                code_name = f"{name}.{index}.codes"
                scale_name = f"{name}.{index}.scales"
                encoded[code_name] = codes
                encoded[scale_name] = scales
                blocks.append(represented)
            restored = np.concatenate(blocks, axis=1)
            if restored.shape != shape:
                raise ValueError(f"packed block shapes do not reconstruct {name}")
            if name == "head.0.weight":
                if shape[1] != 2 * int(weights["inp.bias"].size):
                    raise ValueError("packed head.0 shape does not match hidden/projection split")
                if blocks[0].shape[1] != int(weights["inp.bias"].size):
                    raise ValueError("packed head.0 blocks do not split at the hidden width")
            weights[name] = restored
        return weights, encoded, metadata
