from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


INPUT_WIDTH = 112
HIDDEN_WIDTH = 192
GATE_WIDTH = 4 * HIDDEN_WIDTH

TENSOR_SHAPES = {
    "mean": (INPUT_WIDTH,),
    "std": (INPUT_WIDTH,),
    "inp.weight": (HIDDEN_WIDTH, INPUT_WIDTH),
    "inp.bias": (HIDDEN_WIDTH,),
    "gru.weight_ih_l0": (GATE_WIDTH, HIDDEN_WIDTH),
    "gru.weight_hh_l0": (GATE_WIDTH, HIDDEN_WIDTH),
    "gru.bias_ih_l0": (GATE_WIDTH,),
    "gru.bias_hh_l0": (GATE_WIDTH,),
    "head.0.weight": (HIDDEN_WIDTH, 2 * HIDDEN_WIDTH),
    "head.0.bias": (HIDDEN_WIDTH,),
    "head.2.weight": (2, HIDDEN_WIDTH),
    "head.2.bias": (2,),
}


class LSTM192CPU:
    """Run the pinned one-layer raw112 LSTM checkpoint one row at a time."""

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        """Validate and retain the exact FP32 checkpoint tensors.

        Args:
            weights: Data-only arrays exported from the pinned PR53 checkpoint.

        Raises:
            ValueError: If a tensor name, shape, dtype, value, or normalizer is invalid.
        """
        import numpy as np
        from scipy.special import erf

        if set(weights) != set(TENSOR_SHAPES):
            raise ValueError("LSTM tensor names differ from the pinned checkpoint schema")
        arrays: dict[str, np.ndarray] = {}
        for name, shape in TENSOR_SHAPES.items():
            value = weights[name]
            if not isinstance(value, np.ndarray) or value.shape != shape:
                raise ValueError(f"unexpected LSTM tensor shape: {name}")
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"LSTM tensor must contain finite FP32 values: {name}")
            arrays[name] = np.ascontiguousarray(value, dtype=np.float32)
        if np.any(arrays["std"] <= 0):
            raise ValueError("LSTM normalizer standard deviations must be positive")

        self.mean = arrays["mean"]
        self.std = arrays["std"]
        self.input_weight = arrays["inp.weight"]
        self.input_bias = arrays["inp.bias"]
        self.weight_ih = arrays["gru.weight_ih_l0"]
        self.weight_hh = arrays["gru.weight_hh_l0"]
        self.bias_ih = arrays["gru.bias_ih_l0"]
        self.bias_hh = arrays["gru.bias_hh_l0"]
        self.head_weight = arrays["head.0.weight"]
        self.head_bias = arrays["head.0.bias"]
        self.output_weight = arrays["head.2.weight"]
        self.output_bias = arrays["head.2.bias"]
        self._erf = erf
        self.reset()

    def reset(self) -> None:
        """Clear hidden/cell state at a sequence boundary; training chunks carry state."""
        import numpy as np

        self.hidden = np.zeros(HIDDEN_WIDTH, dtype=np.float32)
        self.cell = np.zeros(HIDDEN_WIDTH, dtype=np.float32)

    def step(self, row: np.ndarray) -> np.ndarray:
        """Advance the recurrent state and return the two unbounded predictions.

        Args:
            row: One chronological raw input row with 112 FP32 features.

        Returns:
            A float32 vector with the checkpoint's two prediction outputs.

        Raises:
            ValueError: If the raw input row has the wrong shape or nonfinite values.
        """
        import numpy as np

        if row.shape != (INPUT_WIDTH,) or row.dtype != np.float32:
            raise ValueError("LSTM expects one FP32 raw112 input row")
        if not np.isfinite(row).all():
            raise ValueError("LSTM input row must be finite")

        normalized = (row - self.mean) / self.std
        encoded = np.matmul(self.input_weight, normalized) + self.input_bias
        encoded = _gelu(encoded, self._erf)

        # Fit-time 256-row chunks detach gradients but preserve recurrent values.
        input_gates = np.matmul(self.weight_ih, encoded) + self.bias_ih
        hidden_gates = np.matmul(self.weight_hh, self.hidden) + self.bias_hh
        input_gate, forget_gate, candidate, output_gate = np.split(input_gates + hidden_gates, 4)
        input_gate = _sigmoid(input_gate)
        forget_gate = _sigmoid(forget_gate)
        candidate = np.tanh(candidate)
        output_gate = _sigmoid(output_gate)
        self.cell = forget_gate * self.cell + input_gate * candidate
        self.hidden = output_gate * np.tanh(self.cell)

        head_input = np.concatenate((self.hidden, encoded))
        head_hidden = np.matmul(self.head_weight, head_input) + self.head_bias
        head_hidden = _gelu(head_hidden, self._erf)
        return np.asarray(
            np.matmul(self.output_weight, head_hidden) + self.output_bias,
            dtype=np.float32,
        )


def _gelu(values: np.ndarray, erf: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Evaluate source-equivalent exact-erf GELU in FP64, then cast to FP32."""
    import numpy as np

    values64 = values.astype(np.float64)
    return np.asarray(
        np.float64(0.5) * values64 * (np.float64(1.0) + erf(values64 / np.sqrt(2.0))),
        dtype=np.float32,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply a numerically stable sigmoid while preserving FP32 outputs."""
    import numpy as np

    result = np.empty_like(values, dtype=np.float32)
    positive = values >= np.float32(0.0)
    result[positive] = np.float32(1.0) / (np.float32(1.0) + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (np.float32(1.0) + negative_exp)
    return result
