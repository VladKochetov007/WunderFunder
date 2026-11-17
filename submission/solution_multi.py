"""Submission: blend of native streaming GRUs.

members.json lists weight files (model state dicts of wnn.nn.RNNModel, head sliced to the two
targets) with blend coefficients; the organizer baseline enters as an exact RNNModel rewrite
(scripts/convert_baseline.py) with its output scale folded into its coefficient. All members run
in the Rust extension shipped as v3/ (x86-64-v3) and v2/ (x86-64-v2); v3 is loaded only when the
CPU reports every flag it needs.
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


def _cpu_flags() -> set[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _native_dir() -> Path:
    forced = os.environ.get("WNN_NATIVE_LEVEL")
    if forced in ("v2", "v3"):
        return HERE / forced
    return HERE / ("v3" if _cpu_flags() >= _V3_FLAGS else "v2")


sys.path.insert(0, str(_native_dir()))
import wnn_native  # noqa: E402


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(HERE / name) as w:
        return {k: np.ascontiguousarray(w[k], dtype=np.float32) for k in w.files}


class PredictionModel:
    def __init__(self) -> None:
        spec = json.loads((HERE / "members.json").read_text())
        precision_path = HERE / "precision.txt"
        precision = precision_path.read_text().strip() if precision_path.is_file() else "f32"
        self.members = []
        for member in spec:
            layout = member.get("input_layout", {})
            if not isinstance(layout, dict):
                raise ValueError("member input_layout must be an object")
            feature_mode = layout.get("representation_mode", "none")
            kwargs = {
                "fast_activations": layout.get("fast_activations", True),
                "precision": precision,
                "feature_mode": feature_mode,
            }
            if feature_mode == "i1_history":
                prior = layout.get("history_prior")
                mask_indices = layout.get("post_normalization_mask_indices")
                scale = prior.get("scale") if isinstance(prior, dict) else None
                prior_semantics = {
                    "schema": "i1_causal_history_v1",
                    "scale_estimator": "within_sequence_first_difference_rms_v1",
                    "scale_denominator": "all_adjacent_transitions_within_selected_sequences",
                    "scale_floor": float(np.float32(1e-4)),
                    "fast_half_life_rows": 32.0,
                    "slow_half_life_rows": 512.0,
                    "block_rows": 256,
                    "alpha_definition": "1 - 2**(-1/half_life_rows)",
                    "mean_initialization": "first_observation",
                    "energy_initialization": "zero",
                    "energy_delta": "current_i1_row_minus_previous_i1_row",
                    "first_row_history": "zero",
                }
                if (
                    precision != "f32"
                    or layout.get("schema") != "wnn.native_input_layout.v1"
                    or layout.get("input_width") != 367
                    or layout.get("layout")
                    != [
                        "raw112",
                        "event99",
                        "i1_residual_h32_52",
                        "i1_residual_h512_52",
                        "i1_log_energy_h512_52",
                    ]
                    or layout.get("mask_point") != "after stored model normalization"
                    or layout.get("fast_activations") is not False
                    or not isinstance(prior, dict)
                    or any(prior.get(key) != value for key, value in prior_semantics.items())
                    or type(prior.get("transition_count")) is not int
                    or prior.get("transition_count", 0) < 1
                    or type(prior.get("sequence_count")) is not int
                    or prior.get("sequence_count", 0) < 1
                    or not isinstance(scale, list)
                    or len(scale) != 52
                    or any(type(value) not in (int, float) for value in scale)
                    or not np.isfinite(np.asarray(scale, dtype=np.float32)).all()
                    or any(value <= 0 for value in scale)
                    or not isinstance(mask_indices, list)
                    or any(type(index) is not int or not 0 <= index < 367 for index in mask_indices)
                    or len(set(mask_indices)) != len(mask_indices)
                ):
                    raise ValueError("member has an unsupported i1_history input contract")
                kwargs["history_scale"] = prior.get("scale")
                kwargs["post_normalization_mask_indices"] = mask_indices
            elif "history_prior" in layout or "post_normalization_mask_indices" in layout:
                raise ValueError("history metadata requires representation_mode='i1_history'")
            self.members.append(wnn_native.Predictor(_load(member["weights"]), **kwargs))
        self.coef = [np.float32(m["coef"]) for m in spec]
        self.seq_ix = None

    def predict(self, data_point: object) -> np.ndarray | None:
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            for m in self.members:
                m.reset()
        row = np.ascontiguousarray(data_point.state, dtype=np.float32)
        out = None
        for m, c in zip(self.members, self.coef, strict=True):
            v = c * m.step(row)
            out = v if out is None else out + v
        if not data_point.need_prediction:
            return None
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=2.0, neginf=-2.0)
        return out.astype(np.float32)
