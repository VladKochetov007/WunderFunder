from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wnn.quantization import PRESETS, quantize_model, represented_rows

METADATA = "__q8_metadata__"


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
    if preset not in PRESETS or METADATA in weights:
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
            or metadata.get("preset") not in PRESETS
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
