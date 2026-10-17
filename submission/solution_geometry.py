"""Native causal moment-geometry callback for a fixed exported recurrent model.

`geometry.json` supplies fitting-only constants and the full input normalization vectors.
Those vectors are installed in the exported state dict so FastRnn folds the same normalization
into its input projection. Disabled feature channels are supplied at their fitted mean, which
produces the training-time post-normalization zero mask.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
_V3_FLAGS = {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}
_AVX2_FLAGS = {"avx", "avx2", "fma"}


def _cpu_flags() -> set[str]:
    """Return CPU feature flags when Linux exposes `/proc/cpuinfo`."""
    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8") as source:
            for line in source:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _native_level() -> str:
    """Choose the strongest packaged native build supported by this CPU."""
    flags = _cpu_flags()
    if flags >= _V3_FLAGS and (HERE / "v3").is_dir():
        return "v3"
    if flags >= _AVX2_FLAGS and (HERE / "avx2").is_dir():
        return "avx2"
    return "v2"


_LEVEL = _native_level()
sys.path.insert(0, str(HERE / _LEVEL))
import wnn_native  # noqa: E402


def _load_weights(path: Path, config: dict[str, object]) -> dict[str, np.ndarray]:
    """Load FP32 tensors and install the fitting-only feature normalizers."""
    with np.load(path, allow_pickle=False) as archive:
        weights = {
            key: np.ascontiguousarray(archive[key], dtype=np.float32) for key in archive.files
        }
    if "mean" not in weights or "std" not in weights:
        raise ValueError("weights must include mean and std tensors")
    mean = np.asarray(config["input_mean"], dtype=np.float32)
    scale = np.asarray(config["input_scale"], dtype=np.float32)
    if mean.shape != weights["mean"].shape or scale.shape != weights["std"].shape:
        raise ValueError("geometry input_mean/input_scale must match the model input width")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("geometry input normalizers must be finite with positive scales")
    weights["mean"] = np.ascontiguousarray(mean)
    weights["std"] = np.ascontiguousarray(scale)
    return weights


class PredictionModel:
    """No-argument competition callback backed by the native causal feature builder."""

    def __init__(self) -> None:
        config = json.loads((HERE / "geometry.json").read_text(encoding="utf-8"))
        weights = _load_weights(HERE / "weights.npz", config)
        dictionary_enabled = bool(config.get("dictionary_enabled", False))
        dictionary_levels = None
        if dictionary_enabled:
            dictionary_levels = np.ascontiguousarray(
                np.asarray(config["dictionary_levels"], dtype=np.float32)
            )
        self.model = wnn_native.GeometryPredictor(
            weights,
            np.ascontiguousarray(np.asarray(config["beta"], dtype=np.float64)),
            np.ascontiguousarray(np.asarray(config["intercept"], dtype=np.float64)),
            np.ascontiguousarray(np.asarray(config["atoms"], dtype=np.float64)),
            dictionary_levels=dictionary_levels,
            dictionary_enabled=dictionary_enabled,
            arm=str(config.get("arm", "scaled")),
            fast_activations=True,
        )
        # Bind the complete native callback; state reset and need_prediction handling stay native.
        self.predict = self.model.predict
        print(
            f"[wnn] level={_LEVEL} input_width={self.model.input_width} "
            f"arm={config.get('arm', 'scaled')} precision={self.model.precision}",
            file=sys.stderr,
            flush=True,
        )

    def predict(self, data_point: object) -> np.ndarray | None:
        """Advance sequence state and return a prediction only when the scorer requests one."""
        return self.model.predict(data_point)
