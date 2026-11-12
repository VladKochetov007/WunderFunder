#!/usr/bin/env python3
"""Server-only Torch reference for the packaged NumPy LSTM readout."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

NPZ_SHAPES: dict[str, tuple[int, ...]] = {
    "mean": (112,),
    "std": (112,),
    "inp.weight": (192, 112),
    "inp.bias": (192,),
    "gru.weight_ih_l0": (768, 192),
    "gru.weight_hh_l0": (768, 192),
    "gru.bias_ih_l0": (768,),
    "gru.bias_hh_l0": (768,),
    "head.0.weight": (192, 384),
    "head.0.bias": (192,),
    "head.2.weight": (2, 192),
    "head.2.bias": (2,),
}
PARITY_ATOL = 1e-4


def _find_checkpoint(package_root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Load only the recipe-pinned root ``lstm.npz`` checkpoint member."""
    path = package_root / "lstm.npz"
    if not path.is_file() or path.is_symlink():
        raise ValueError("recipe-pinned root lstm.npz is missing or not a regular file")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(NPZ_SHAPES):
            raise ValueError("lstm.npz keys differ from the fixed 192-wide checkpoint schema")
        arrays = {name: np.asarray(archive[name]) for name in NPZ_SHAPES}
    for name, shape in NPZ_SHAPES.items():
        if arrays[name].shape != shape or arrays[name].dtype != np.float32:
            raise ValueError(f"checkpoint tensor {name!r} has the wrong shape or dtype")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"checkpoint tensor {name!r} contains non-finite values")
    return path, arrays


def _torch_reference(
    arrays: dict[str, np.ndarray],
    features: np.ndarray,
    *,
    chunk_rows: int,
    one_step: bool,
) -> np.ndarray:
    """Evaluate the exact packed equations with torch.nn.LSTM state carried by chunks."""
    import torch
    from torch import nn
    from torch.nn import functional as functional

    if torch.cuda.is_available():
        raise RuntimeError("Torch parity reference must be CPU-only")
    device = torch.device("cpu")
    tensors = {
        name: torch.from_numpy(value.copy()).to(device=device) for name, value in arrays.items()
    }
    recurrent = nn.LSTM(
        192, 192, num_layers=1, batch_first=True, device=device, dtype=torch.float32
    )
    with torch.no_grad():
        recurrent.weight_ih_l0.copy_(tensors["gru.weight_ih_l0"])
        recurrent.weight_hh_l0.copy_(tensors["gru.weight_hh_l0"])
        recurrent.bias_ih_l0.copy_(tensors["gru.bias_ih_l0"])
        recurrent.bias_hh_l0.copy_(tensors["gru.bias_hh_l0"])
        normalized = (
            torch.from_numpy(np.asarray(features, dtype=np.float32).copy()) - tensors["mean"]
        ) / tensors["std"]
        encoded = functional.gelu(
            functional.linear(normalized, tensors["inp.weight"], tensors["inp.bias"]),
            approximate="none",
        )
        states: tuple[torch.Tensor, torch.Tensor] | None = None
        hidden_outputs: list[torch.Tensor] = []
        ranges = (
            [(row, row + 1) for row in range(len(features))]
            if one_step
            else [
                (start, min(start + chunk_rows, len(features)))
                for start in range(0, len(features), chunk_rows)
            ]
        )
        for start, stop in ranges:
            output, states = recurrent(encoded[start:stop].unsqueeze(0), states)
            hidden_outputs.append(output.squeeze(0))
        hidden = torch.cat(hidden_outputs, dim=0)
        joined = torch.cat((hidden, encoded), dim=1)
        head = functional.gelu(
            functional.linear(joined, tensors["head.0.weight"], tensors["head.0.bias"]),
            approximate="none",
        )
        result = functional.linear(head, tensors["head.2.weight"], tensors["head.2.bias"])
    return result.cpu().numpy().astype(np.float32, copy=False)


def run_lstm_reference_parity(
    package_root: Path,
    prediction_model: Any,
    datapoint_type: type,
    sequence_id: int,
    features: np.ndarray,
    need: np.ndarray,
    *,
    rows: int = 2_000,
) -> dict[str, Any]:
    """Compare actual ZIP LSTM outputs with one-step and chunked Torch references."""
    checkpoint, arrays = _find_checkpoint(package_root)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    count = min(rows, len(features))
    if count < 1 or features.shape[1:] != (112,) or need.shape != (len(features),):
        raise ValueError("Torch parity requires nonempty [rows,112] features and a row mask")
    source = np.asarray(features[:count], dtype=np.float32)
    request = np.asarray(need[:count], dtype=bool)
    prediction_model.reset()
    actual = np.empty((count, 2), dtype=np.float32)
    for step, row in enumerate(source):
        value = prediction_model.predict(
            datapoint_type(sequence_id, step, bool(request[step]), row)
        )
        components = getattr(prediction_model, "last_components", None)
        if not isinstance(components, tuple) or len(components) != 2:
            raise TypeError("PredictionModel.last_components must contain (base, lstm)")
        recurrent_value = np.asarray(components[1], dtype=np.float32)
        if recurrent_value.shape != (2,) or not np.isfinite(recurrent_value).all():
            raise ValueError(f"invalid LSTM component at parity row {step}")
        if request[step]:
            output = np.asarray(value, dtype=np.float32)
            if output.shape != (2,) or not np.isfinite(output).all():
                raise ValueError(f"invalid callback output at parity row {step}")
        elif value is not None:
            raise ValueError("callback must return None when need_prediction is false")
        actual[step] = recurrent_value

    chunked = _torch_reference(arrays, source, chunk_rows=257, one_step=False)
    one_step = _torch_reference(arrays, source, chunk_rows=1, one_step=True)
    chunk_error = float(np.max(np.abs(actual - chunked)))
    step_error = float(np.max(np.abs(actual - one_step)))
    reference_chunk_error = float(np.max(np.abs(chunked - one_step)))
    if max(chunk_error, step_error, reference_chunk_error) > PARITY_ATOL:
        raise ValueError(
            "NumPy/Torch LSTM parity exceeded 1e-4: "
            f"numpy_chunked={chunk_error:.8g}, numpy_one_step={step_error:.8g}, "
            f"torch_chunked_vs_step={reference_chunk_error:.8g}"
        )

    prediction_model.reset()
    replay = np.empty_like(actual)
    for step, row in enumerate(source):
        prediction_model.predict(datapoint_type(sequence_id, step, bool(request[step]), row))
        components = getattr(prediction_model, "last_components", None)
        replay[step] = np.asarray(components[1], dtype=np.float32)
    if not np.array_equal(actual, replay):
        raise ValueError("PredictionModel.reset did not reproduce the same LSTM prefix")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    checkpoint_hash = digest.hexdigest()
    return {
        "passed": True,
        "checkpoint_member": checkpoint.relative_to(package_root).as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "rows": count,
        "numpy_vs_torch_chunked_max_abs": chunk_error,
        "numpy_vs_torch_one_step_max_abs": step_error,
        "torch_chunked_vs_one_step_max_abs": reference_chunk_error,
        "reset_replay_bitwise_equal": True,
        "torch_version": __import__("torch").__version__,
        "torch_cpu_only": True,
        "tolerance": PARITY_ATOL,
        "scope": "server-only FP32 LSTM equations and reset; not quality or runtime acceptance",
    }
