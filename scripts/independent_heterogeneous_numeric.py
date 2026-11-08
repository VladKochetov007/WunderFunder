"""Independent FP32 mixture arithmetic and two-pass weighted Pearson."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

MIXTURES: dict[str, tuple[tuple[str, float], ...]] = {
    "TL3": (("w18", 0.34), ("tiedlead_s5", 0.33), ("tiedlead_s6", 0.33)),
    "PR51_replacement": (("fbc", 0.34), ("tiedlead_s5", 0.33), ("tiedlead_s6", 0.33)),
    "GRU_family": (("fin256_s3", 0.5), ("fin256_s4", 0.5)),
    "heterogeneous_teacher_pool": (
        ("fin256_s3", 0.35),
        ("fin256_s4", 0.35),
        ("galam", 0.1),
        ("lstm", 0.1),
        ("tcn", 0.1),
    ),
    "TL3_plus_GALAM": (("TL3", 0.9), ("galam", 0.1)),
    "TL3_plus_LSTM": (("TL3", 0.9), ("lstm", 0.1)),
    "TL3_plus_TCN": (("TL3", 0.9), ("tcn", 0.1)),
    "TL3_heterogeneous": (("TL3", 0.7), ("galam", 0.1), ("lstm", 0.1), ("tcn", 0.1)),
    "GRU_student_mean": (("control_s1301", 0.5), ("control_s1302", 0.5)),
    "heterogeneous_student_mean": (("hetero_s1301", 0.5), ("hetero_s1302", 0.5)),
    "TL3_plus_GRU_students": (("TL3", 0.8), ("GRU_student_mean", 0.2)),
    "TL3_plus_heterogeneous_students": (("TL3", 0.8), ("heterogeneous_student_mean", 0.2)),
}


def compose(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Construct the frozen recursive mixtures before any output clipping.

    This is independently written arithmetic: the production mixer is not imported.
    The literal tuple order freezes the FP32 accumulation order, including nesting.
    """
    output = dict(values)
    if not output:
        raise ValueError("at least one source is required")
    shape = next(iter(output.values())).shape
    if (
        len(shape) != 2
        or shape[1] != 2
        or any(
            x.shape != shape or x.dtype != np.float32 or not np.isfinite(x).all()
            for x in output.values()
        )
    ):
        raise ValueError("raw selected predictions must be finite FP32 [rows,2]")
    for name, terms in MIXTURES.items():
        # Every nested dependency occurs earlier in this frozen definition.
        first, coefficient = terms[0]
        prediction = np.multiply(output[first], np.float32(coefficient), dtype=np.float32)
        for member, coefficient in terms[1:]:
            term = np.multiply(output[member], np.float32(coefficient), dtype=np.float32)
            np.add(prediction, term, out=prediction)
        output[name] = prediction
    return output


def clipped_pair(target: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the declared float32 input/clip convention, then use float64 moments."""
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("selected targets and predictions must have equal [rows,2] shapes")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("selected values must be finite before clipping")
    y = np.clip(np.asarray(target, dtype=np.float32), -2.0, 2.0).astype(np.float64)
    p = np.clip(np.asarray(prediction, dtype=np.float32), -2.0, 2.0).astype(np.float64)
    return y, p


class TwoPassPearson:
    """Accumulate weighted means first, then centered variances/covariance."""

    def __init__(self, names: Sequence[str]) -> None:
        """Allocate a bounded pair of target moments per declared output."""
        if len(names) != len(set(names)) or not names:
            raise ValueError("output names must be unique and nonempty")
        self.names = tuple(names)
        self.first = {name: np.zeros((3, 2), dtype=np.float64) for name in names}
        self.centered = {name: np.zeros((3, 2), dtype=np.float64) for name in names}
        self.rows = [0, 0]

    def add(self, target: np.ndarray, outputs: Mapping[str, np.ndarray], pass_index: int) -> None:
        """Accumulate one selected physical sequence in pass zero or one."""
        if set(outputs) != set(self.names) or pass_index not in (0, 1):
            raise ValueError("output membership or pass differs")
        self.rows[pass_index] += len(target)
        for name in self.names:
            y, p = clipped_pair(target, outputs[name])
            w = np.abs(y)
            if pass_index == 0:
                self.first[name] += np.stack((w.sum(0), (w * y).sum(0), (w * p).sum(0)))
            else:
                first = self.first[name]
                safe = np.where(first[0] > 0, first[0], 1.0)
                yc, pc = y - first[1] / safe, p - first[2] / safe
                self.centered[name] += np.stack(
                    ((w * yc * yc).sum(0), (w * pc * pc).sum(0), (w * yc * pc).sum(0))
                )

    def result(self) -> dict[str, dict[str, float | int]]:
        """Return the same two-target average, including the declared constant fallback."""
        if self.rows[0] <= 0 or self.rows[0] != self.rows[1]:
            raise ValueError("two complete nonempty passes are required")
        result = {}
        for name in self.names:
            total, moments = self.first[name][0], self.centered[name]
            values = []
            for target in range(2):
                if total[target] < 1e-8:
                    values.append(0.0)
                    continue
                sy, sp = np.sqrt(np.maximum(moments[:2, target], 0) / total[target])
                value = 0.0 if min(sy, sp) <= 1e-8 else moments[2, target] / total[target] / sy / sp
                values.append(float(np.clip(value, -1.0, 1.0)))
            result[name] = {
                "t0": values[0],
                "t1": values[1],
                "wp": 0.5 * sum(values),
                "selected_rows": self.rows[0],
            }
        return result
