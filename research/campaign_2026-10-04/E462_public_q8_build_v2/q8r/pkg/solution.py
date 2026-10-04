"""Fixed q8 package callback for an E462 converted public-combo member."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METADATA = "__q8_metadata__"
EXPECTED_FORMAT = "wnn_q8_rows_v1"
_V3_FLAGS = {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}
_AVX2_FLAGS = {"avx", "avx2", "fma"}


def _cpu_flags() -> set[str]:
    """Return CPU feature flags for selecting a packaged native build."""
    try:
        with Path("/proc/cpuinfo").open() as source:
            for line in source:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        return set()
    return set()


def _native_level() -> str:
    """Select a present native build, respecting the verification override."""
    forced = os.environ.get("WNN_NATIVE_LEVEL")
    if forced in ("v2", "v3") and (HERE / forced).is_dir():
        return forced
    flags = _cpu_flags()
    if flags >= _V3_FLAGS and (HERE / "v3").is_dir():
        return "v3"
    if flags >= _AVX2_FLAGS and (HERE / "avx2").is_dir():
        return "avx2"
    return "v2"


def _read_export(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    """Read a q8 export and reconstruct represented float matrices for native construction."""
    with np.load(path, allow_pickle=False) as archive:
        if METADATA not in archive.files:
            raise ValueError("q8 export is missing metadata")
        metadata = json.loads(archive[METADATA].tobytes())
        if metadata.get("version") != 1 or metadata.get("format") != EXPECTED_FORMAT:
            raise ValueError("unsupported q8 export metadata")
        preset = metadata.get("preset")
        if preset not in ("recurrent", "recurrent_head"):
            raise ValueError("unsupported q8 preset")
        selected = metadata.get("selected")
        if not isinstance(selected, list) or not selected:
            raise ValueError("invalid q8 selected-matrix list")
        encoded_names = {
            f"{name}.{block}.{suffix}"
            for name in selected
            for block in range(2 if name == "head.0.weight" else 1)
            for suffix in ("codes", "scales")
        }
        encoded = {name: np.ascontiguousarray(archive[name]) for name in encoded_names}
        weights = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32)
            for name in archive.files
            if name != METADATA and name not in encoded_names
        }
    for name in selected:
        blocks = []
        for block in range(2 if name == "head.0.weight" else 1):
            codes = encoded[f"{name}.{block}.codes"]
            scales = encoded[f"{name}.{block}.scales"]
            if codes.dtype != np.int8 or codes.ndim != 2:
                raise ValueError(f"invalid q8 code array for {name}")
            if scales.dtype != np.float32 or scales.shape != (codes.shape[0],):
                raise ValueError(f"invalid q8 scales for {name}")
            if not np.isfinite(scales).all() or np.any(scales < 0) or np.any(codes == -128):
                raise ValueError(f"invalid q8 values for {name}")
            if np.any((scales == 0) & np.any(codes != 0, axis=1)):
                raise ValueError(f"zero scale has nonzero codes for {name}")
            blocks.append(np.ascontiguousarray(codes.astype(np.float32) * scales[:, None]))
        represented = np.concatenate(blocks, axis=1)
        if name == "head.0.weight" and represented.shape[1] != 2 * int(weights["inp.bias"].size):
            raise ValueError("merged head q8 blocks do not match hidden/projection widths")
        weights[name] = represented
    return weights, encoded, preset


def _load_native():
    """Import the selected native library only after placing its package directory first."""
    level = _native_level()
    sys.path.insert(0, str(HERE / level))
    import wnn_native

    return wnn_native, level


class PredictionModel:
    """Competition callback holding one fixed q8 predictor member."""

    def __init__(self) -> None:
        members = json.loads((HERE / "members.json").read_text())
        if len(members) != 1 or members[0].get("coef") != 1.0:
            raise ValueError("E462 requires exactly one unit-weight member")
        weights, codes, preset = _read_export(HERE / members[0]["weights"])
        precision = "q8r" if preset == "recurrent" else "q8x"
        override = os.environ.get("WNN_PRECISION")
        if override:
            if override not in ("q8r_portable", "q8x_portable", precision):
                raise ValueError("unsupported E462 precision override")
            precision = override
        native, self.level = _load_native()
        self.model = native.EnsemblePredictor(
            [weights],
            [np.float32(1.0)],
            fast_activations=True,
            precision=precision,
            q8_codes=[codes],
            budget_s=None,
        )
        if self.model.precision != precision:
            raise RuntimeError(f"native precision fallback: {precision} -> {self.model.precision}")
        reported = self.model.quantization_codes()[0]
        if reported.keys() != codes.keys() or any(
            reported[name].dtype != codes[name].dtype
            or not np.array_equal(reported[name].view(np.uint8), codes[name].view(np.uint8))
            for name in codes
        ):
            raise RuntimeError("native constructor changed q8 code/scale bytes")
        self.predict = self.model.predict
        print(
            f"[wnn] level={self.level} members=1 precision={self.model.precision}",
            file=sys.stderr,
            flush=True,
        )

    def predict(self, data_point: object) -> object:
        """Forward to the native stateful callback."""
        return self.model.predict(data_point)
