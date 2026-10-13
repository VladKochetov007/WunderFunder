"""Fixed four-member Q8 callback with one exact-code QAT member."""

import json
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qat_archive import read_qat_archive  # noqa: E402

PRECISION = "q8vnni"
_V3_FLAGS = {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}
_AVX2_FLAGS = {"avx", "avx2", "fma"}


def _cpu_flags():
    try:
        with open("/proc/cpuinfo", encoding="ascii") as source:
            for line in source:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _level():
    forced = os.environ.get("WNN_NATIVE_LEVEL")
    if forced in ("v2", "avx2", "v3"):
        return forced
    flags = _cpu_flags()
    if flags >= _V3_FLAGS:
        return "v3"
    if flags >= _AVX2_FLAGS:
        return "avx2"
    return "v2"


_LEVEL = _level()
sys.path.insert(0, str(HERE / _LEVEL))
import wnn_native  # noqa: E402


def _load_regular(name):
    path = (HERE / name).resolve(strict=True)
    if not path.is_relative_to(HERE) or path.suffix != ".npz":
        raise ValueError("invalid packaged weight path")
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: np.ascontiguousarray(archive[key], dtype=np.float32)
            for key in archive.files
        }


def _load_members():
    raw_members = json.loads((HERE / "members.json").read_text(encoding="utf-8"))
    if not isinstance(raw_members, list) or len(raw_members) != 4:
        raise ValueError("the fixed QAT blend requires exactly four members")
    weights = []
    coefficients = []
    exact_index = None
    exact_codes = None
    for index, item in enumerate(raw_members):
        if not isinstance(item, dict):
            raise ValueError("invalid member record")
        coefficient = item.get("coef")
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
            raise ValueError("invalid member coefficient")
        coefficient = np.float32(coefficient)
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError("invalid member coefficient")
        coefficients.append(coefficient)
        archive_name = item.get("qat_archive")
        if archive_name is not None:
            if exact_index is not None or not isinstance(archive_name, str):
                raise ValueError("expected exactly one valid QAT archive member")
            archive_path = (HERE / archive_name).resolve(strict=True)
            if not archive_path.is_relative_to(HERE) or archive_path.suffix != ".npz":
                raise ValueError("invalid packaged QAT archive path")
            preset = item.get("qat_preset")
            if not isinstance(preset, str):
                raise ValueError("QAT member lacks its versioned preset")
            weights_for_member, exact_codes, metadata = read_qat_archive(
                archive_path, expected_preset=preset
            )
            if metadata.get("format") != "wnn_quantized_rows_v2":
                raise ValueError("unsupported QAT member archive")
            weights.append(weights_for_member)
            exact_index = index
        else:
            weights.append(_load_regular(item["weights"]))
    if exact_index is None or exact_codes is None:
        raise ValueError("the fixed QAT blend must contain one exact-code member")
    if not np.isclose(np.sum(coefficients, dtype=np.float64), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("fixed member coefficients must sum to one")
    return weights, coefficients, exact_index, exact_codes


class PredictionModel:
    def __init__(self):
        weights, coefficients, exact_index, exact_codes = _load_members()

        # Export native-generated rows for unchanged members. The QAT member's
        # map then replaces only its constructor-generated rows, without scale
        # derivation or requantization of the packed training codes.
        native_seed = wnn_native.EnsemblePredictor(
            weights,
            coefficients,
            fast_activations=True,
            precision=PRECISION,
            budget_s=None,
        )
        export_codes = getattr(native_seed, "quantization_codes", None)
        if not callable(export_codes):
            raise RuntimeError("native library lacks exact Q8 code export required by QAT")
        code_maps = export_codes()
        if not isinstance(code_maps, list) or len(code_maps) != len(weights):
            raise RuntimeError("native code export did not return one map per member")
        code_maps[exact_index] = exact_codes
        del native_seed

        self.model = wnn_native.EnsemblePredictor(
            weights,
            coefficients,
            fast_activations=True,
            precision=PRECISION,
            q8_codes=code_maps,
            budget_s=None,
        )
        self.predict = self.model.predict
        print(
            f"[wnn] level={_LEVEL} members={len(weights)} precision={self.model.precision}",
            file=sys.stderr,
            flush=True,
        )

    def predict(self, data_point):  # shadowed by the bound native method in __init__
        return self.model.predict(data_point)
