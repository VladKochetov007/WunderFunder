"""NumPy-only reader for the versioned QAT package archive."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

METADATA_KEY = "__quantized_rows_metadata__"
SUPPORTED_PRESETS = {"recurrent_w4", "input_w4_recurrent_w8"}


def _unpack_codes(packed: np.ndarray, shape: tuple[int, int], bits: int) -> np.ndarray:
    if (
        packed.dtype != np.uint8
        or packed.ndim != 1
        or len(shape) != 2
        or any(isinstance(size, bool) or size <= 0 for size in shape)
    ):
        raise ValueError("invalid packed code array or matrix shape")
    rows, columns = shape
    count = rows * columns
    if bits == 4:
        if packed.size != (count + 1) // 2:
            raise ValueError("invalid packed Q4 byte count")
        nibbles = np.empty(packed.size * 2, dtype=np.uint8)
        nibbles[0::2] = packed & 0x0F
        nibbles[1::2] = packed >> 4
        if count % 2 and nibbles[-1] != 0:
            raise ValueError("odd Q4 code count requires a zero high-nibble pad")
        nibbles = nibbles[:count]
        codes = np.where(nibbles < 8, nibbles, nibbles.astype(np.int16) - 16).astype(
            np.int8
        )
        if (codes == -8).any():
            raise ValueError("Q4 archive contains invalid -8 code")
    elif bits == 8:
        if packed.size != count:
            raise ValueError("invalid packed Q8 byte count")
        codes = packed.view(np.int8).copy()
        if (codes == -128).any():
            raise ValueError("Q8 archive contains invalid -128 code")
    else:
        raise ValueError("unsupported code width")
    return codes.reshape(shape)


def _expected_bits(matrix_name: str, preset: str) -> int | None:
    if matrix_name.startswith("gru.weight_ih_"):
        return 4
    if matrix_name.startswith("gru.weight_hh_"):
        return 4 if preset == "recurrent_w4" else 8
    if matrix_name == "head.0.weight":
        return 8
    return None


def read_qat_archive(
    path: Path, expected_preset: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Restore effective weights and exact signed code/scale rows.

    Q4 bytes are low-nibble-first in flattened row-major order. An odd number
    of codes requires a zero high-nibble pad; the unused signed value -8 is
    rejected. Q8 bytes preserve signed-int8 bit patterns and reject -128.
    Scales are retained without recomputation.
    """
    with np.load(path, allow_pickle=False) as archive:
        if METADATA_KEY not in archive.files:
            raise ValueError("QAT archive is missing versioned metadata")
        raw_metadata = archive[METADATA_KEY]
        if raw_metadata.dtype != np.uint8 or raw_metadata.ndim != 1:
            raise ValueError("QAT metadata must be a flat uint8 JSON payload")
        try:
            metadata = json.loads(raw_metadata.tobytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("QAT metadata is not valid JSON") from error
        if (
            not isinstance(metadata, dict)
            or metadata.get("version") != 2
            or metadata.get("format") != "wnn_quantized_rows_v2"
            or metadata.get("preset") not in SUPPORTED_PRESETS
        ):
            raise ValueError("unsupported QAT archive version or policy")
        preset = metadata["preset"]
        if expected_preset is not None and preset != expected_preset:
            raise ValueError("QAT archive preset differs from the package manifest")
        matrices = metadata.get("matrices")
        unquantized = metadata.get("unquantized")
        if (
            not isinstance(matrices, dict)
            or not matrices
            or not isinstance(unquantized, list)
            or any(not isinstance(name, str) for name in unquantized)
            or len(set(unquantized)) != len(unquantized)
            or set(matrices) & set(unquantized)
        ):
            raise ValueError("invalid QAT matrix/float field metadata")
        selected_names = set(matrices)
        if "head.0.weight" not in selected_names:
            raise ValueError("QAT archive is missing the split first-head matrix")
        input_names = {
            name for name in selected_names if re.fullmatch(r"gru\.weight_ih_l[0-9]+", name)
        }
        recurrent_names = {
            name for name in selected_names if re.fullmatch(r"gru\.weight_hh_l[0-9]+", name)
        }
        if (
            {name.rsplit("_l", 1)[1] for name in input_names}
            != {name.rsplit("_l", 1)[1] for name in recurrent_names}
            or not input_names
            or selected_names != input_names | recurrent_names | {"head.0.weight"}
        ):
            raise ValueError("QAT archive matrix identities do not match a complete GRU policy")
        expected_fields = {METADATA_KEY, *unquantized}
        if METADATA_KEY in unquantized:
            raise ValueError("QAT metadata key cannot be an unquantized weight")
        named_payload_fields = list(unquantized)
        for matrix in matrices.values():
            if not isinstance(matrix, dict) or not isinstance(matrix.get("blocks"), list):
                raise ValueError("invalid QAT matrix record")
            for block in matrix["blocks"]:
                if not isinstance(block, dict):
                    raise ValueError("invalid QAT block record")
                code_field = block.get("packed_codes")
                scale_field = block.get("scales")
                if not isinstance(code_field, str) or not isinstance(scale_field, str):
                    raise ValueError("QAT block lacks code or scale field names")
                named_payload_fields.extend((code_field, scale_field))
                expected_fields.update((code_field, scale_field))
        if len(named_payload_fields) != len(set(named_payload_fields)):
            raise ValueError("QAT archive metadata reuses a field name")
        if set(archive.files) != expected_fields:
            raise ValueError("QAT archive fields differ from versioned metadata")

        weights: dict[str, np.ndarray] = {}
        for name in unquantized:
            value = archive[name]
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"invalid float tensor in QAT archive: {name}")
            weights[name] = np.ascontiguousarray(value)
        encoded: dict[str, np.ndarray] = {}
        for name, matrix in matrices.items():
            bits = _expected_bits(name, preset)
            shape_value = matrix.get("shape")
            blocks = matrix["blocks"]
            expected_blocks = 2 if name == "head.0.weight" else 1
            if (
                bits is None
                or not isinstance(shape_value, list)
                or len(shape_value) != 2
                or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0
                       for size in shape_value)
                or len(blocks) != expected_blocks
            ):
                raise ValueError(f"QAT matrix selection/shape is invalid: {name}")
            shape = (shape_value[0], shape_value[1])
            restored_blocks: list[np.ndarray] = []
            for index, block in enumerate(blocks):
                block_shape_value = block.get("shape")
                if (
                    not isinstance(block_shape_value, list)
                    or len(block_shape_value) != 2
                    or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0
                           for size in block_shape_value)
                    or block.get("bits") != bits
                ):
                    raise ValueError(f"QAT block metadata is invalid: {name}.{index}")
                block_shape = (block_shape_value[0], block_shape_value[1])
                packed = archive[block["packed_codes"]]
                codes = _unpack_codes(packed, block_shape, bits)
                scales = np.ascontiguousarray(archive[block["scales"]])
                if (
                    scales.dtype != np.float32
                    or scales.shape != (block_shape[0],)
                    or not np.isfinite(scales).all()
                    or (scales < 0).any()
                    or ((scales == 0)[:, None] & (codes != 0)).any()
                ):
                    raise ValueError(f"QAT scales are invalid: {name}.{index}")
                encoded[f"{name}.{index}.codes"] = codes
                encoded[f"{name}.{index}.scales"] = scales
                restored_blocks.append(
                    np.ascontiguousarray(codes.astype(np.float32) * scales[:, None])
                )
            restored = np.concatenate(restored_blocks, axis=1)
            if restored.shape != shape:
                raise ValueError(f"QAT blocks do not reconstruct matrix: {name}")
            if name == "head.0.weight":
                hidden = weights.get("inp.bias")
                if hidden is None or hidden.ndim != 1 or shape[1] != 2 * hidden.size:
                    raise ValueError("QAT first-head split does not match hidden width")
                if restored_blocks[0].shape[1] != hidden.size:
                    raise ValueError("QAT first-head split is not at the hidden width")
            weights[name] = restored
        if set(weights) != set(unquantized) | set(matrices):
            raise ValueError("QAT archive reconstructed duplicate or missing matrices")
    return weights, encoded, metadata
