"""Synthetic tests for the independent E487 delta/result verifier."""

from __future__ import annotations

import copy
import unittest

import numpy as np
from verify_probe_v4 import PRECISIONS, SOURCES, _compare, verify_payload

NAMES = sorted(
    [
        "gru.weight_hh_l0",
        "gru.weight_hh_l1",
        "gru.weight_ih_l0",
        "gru.weight_ih_l1",
        "inp.weight",
        "inp.bias",
        "gru.bias_ih_l0",
        "head.weight",
    ]
)


class ArchiveFixture:
    """Minimal in-memory equivalent of a NumPy NPZ archive."""

    def __init__(self, values):
        self.values = values
        self.files = list(values)

    def __getitem__(self, key):
        return self.values[key]


def fixture():
    arrays = {"parameter_names": np.asarray(NAMES)}
    crossed = {}
    base = {name: np.full((2,), 0.01 + index * 0.001) for index, name in enumerate(NAMES)}
    for source in SOURCES:
        for precision in PRECISIONS:
            key = f"{source}__{precision}"
            values = {name: value.copy() for name, value in base.items()}
            if precision == "fp32_diagnostic":
                values["head.weight"] += 0.0002
            if source == "custom" and precision == "bf16_source":
                values["gru.weight_hh_l0"][0] += 3e-5
            crossed[key] = values
            for name, value in values.items():
                arrays[f"{key}__{name}"] = value
    shared = {}
    for precision in PRECISIONS:
        label_prefix = f"shared_{precision}_"
        values = {name: value.copy() for name, value in base.items()}
        if precision == "fp32_diagnostic":
            values["head.weight"] += 0.0002
        shared[f"{precision}_native"] = values
        shared[f"{precision}_custom"] = values
        for replay in ("native", "custom"):
            label = label_prefix + replay
            for name, value in values.items():
                arrays[f"{label}__{name}"] = value
    shared["bf16_source_frozen_e474"] = shared["bf16_source_custom"]
    for name, value in shared["bf16_source_frozen_e474"].items():
        arrays[f"shared_bf16_source_frozen_e474__{name}"] = value

    captures = {
        source: {
            "raw_gradient_norm": 2.0,
            "raw_gradient_sha256": f"raw-{source}",
            "input_gradient_sha256": f"input-{source}",
            "loss": 1.0,
        }
        for source in SOURCES
    }
    receipts = {}
    for source in SOURCES:
        for precision in PRECISIONS:
            key = f"{source}__{precision}"
            receipts[key] = {
                "raw_gradient_sha256": f"raw-{source}",
                "clipped_gradient_sha256": f"clip-{source}",
                "parameter_sha256_before": "common-initial-state",
                "optimizer_state_finite": True,
            }
    gates = {
        precision: _compare(crossed[f"native__{precision}"], crossed[f"custom__{precision}"])
        for precision in PRECISIONS
    }
    sensitivities = {}
    for source in SOURCES:
        values = {}
        for name in NAMES:
            bf16 = crossed[f"{source}__bf16_source"][name]
            fp32 = crossed[f"{source}__fp32_diagnostic"][name]
            diff = fp32 - bf16
            values[name] = {
                "bf16_delta_rms": float(np.sqrt(np.mean(bf16**2))),
                "fp32_delta_rms": float(np.sqrt(np.mean(fp32**2))),
                "precision_change_rms": float(np.sqrt(np.mean(diff**2))),
                "precision_change_max_abs": float(np.abs(diff).max()),
            }
        sensitivities[source] = values

    def shared_receipt(precision, replay_name, state_hash):
        return {
            "raw_gradient_sha256": "shared-gradient-hash",
            "clipped_gradient_sha256": "shared-clipped-hash",
            "state_sha256_after": state_hash,
            "optimizer_state_sha256_after": f"opt-{precision}",
            "optimizer_state_finite": True,
        }

    shared_control = {}
    for precision in PRECISIONS:
        replays = {
            "native": shared_receipt(precision, "native", f"state-{precision}"),
            "custom": shared_receipt(precision, "custom", f"state-{precision}"),
            "frozen_e474": None,
        }
        if precision == "bf16_source":
            replays["frozen_e474"] = shared_receipt(precision, "frozen_e474", f"state-{precision}")
        shared_control[precision] = {
            "shared_gradient_sha256": "shared-gradient-hash",
            "same_gradient_delta_bytes_exact": True,
            "same_gradient_model_state_bytes_exact": True,
            "same_gradient_optimizer_state_hash_equal": True,
            "clip_activated": True,
            "frozen_e474_equal": True if precision == "bf16_source" else None,
            "replays": replays,
        }
    finite = True
    strict_pass = all(item["passed"] for item in gates.values())
    result = {
        "kind": "E487 one-batch CUDA Muon orthogonalization precision diagnostic",
        "status": "diagnostic_complete" if strict_pass else "diagnostic_complete_gate_failures_recorded",
        "fit_only": True,
        "test_rows_loaded": False,
        "seed": 271836,
        "initialization_bytes_match_after_device_transfer": True,
        "raw_feature_validation": {
            "all_24_fit_groups_checked_before_causal_sanitization": True,
            "fit_row_groups_in_manifest_order": list(range(24)),
            "raw_feature_sha256_in_manifest_order": [f"raw-feature-{i}" for i in range(24)],
        },
        "resources": {"allocator_limit_bytes": 2 * 1024**3, "observed_gpu_stop_bytes": 3_000_000_000},
        "gradient_capture": captures,
        "crossed_replay_arithmetic": {
            "cell_receipts": receipts,
            "strict_native_custom_update_gates": gates,
            "bf16_fp32_delta_sensitivity": sensitivities,
            "all_crossed_finite": finite,
            "gate_failures_recorded_without_aborting": True,
            "delta_arrays": {"crossed_replay_cells": 4, "shared_gradient_replay_cells": 5},
        },
        "shared_gradient_control": shared_control,
        "strict_precision_pass": strict_pass,
    }
    return result, ArchiveFixture(arrays)


class VerifyProbeTests(unittest.TestCase):
    """Verify synthetic known deltas and fail closed on receipt drift."""

    def test_recompute_known_gate_formula_and_preserve_failure_status(self):
        result, archive = fixture()
        proof = verify_payload(result, archive)
        self.assertEqual(proof["status"], "passed")
        self.assertFalse(proof["strict_precision_pass"])
        metrics = proof["crossed_update_gates_recomputed"]["bf16_source"]["metrics"]
        matrix = metrics["matrix:gru.weight_hh_l0"]
        self.assertAlmostEqual(matrix["max_abs_error"], 3e-5, delta=1e-15)
        self.assertFalse(matrix["pass_max_abs"])
        self.assertTrue(proof["crossed_update_gates_recomputed"]["fp32_diagnostic"]["passed"])

    def test_rejects_delta_drift_instead_of_trusting_saved_gate_booleans(self):
        result, archive = fixture()
        altered = {key: value.copy() for key, value in archive.values.items()}
        altered["custom__fp32_diagnostic__head.weight"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "stored strict gate arithmetic differs"):
            verify_payload(result, ArchiveFixture(altered))

    def test_rejects_raw_or_clip_digest_mismatch(self):
        result, archive = fixture()
        changed = copy.deepcopy(result)
        changed["crossed_replay_arithmetic"]["cell_receipts"]["custom__bf16_source"][
            "raw_gradient_sha256"
        ] = "wrong"
        with self.assertRaisesRegex(ValueError, "raw gradient digest mismatch"):
            verify_payload(changed, archive)

    def test_rejects_crossed_delta_cells_from_different_initial_states(self):
        """All update deltas must be measured from the same exact pre-step bytes."""
        result, archive = fixture()
        changed = copy.deepcopy(result)
        changed["crossed_replay_arithmetic"]["cell_receipts"]["custom__fp32_diagnostic"][
            "parameter_sha256_before"
        ] = "different-initial-state"
        with self.assertRaisesRegex(ValueError, "do not share identical pre-step parameters"):
            verify_payload(changed, archive)

    def test_requires_exact_null_marker_for_fp32_frozen_helper(self):
        """Serializer's expected null entry is accepted, but not silently skipped if populated."""
        result, archive = fixture()
        proof = verify_payload(result, archive)
        self.assertEqual(proof["status"], "passed")
        changed = copy.deepcopy(result)
        changed["shared_gradient_control"]["fp32_diagnostic"]["replays"]["unexpected"] = None
        with self.assertRaisesRegex(ValueError, "replay inventory differs"):
            verify_payload(changed, archive)
        changed = copy.deepcopy(result)
        changed["shared_gradient_control"]["fp32_diagnostic"]["replays"]["frozen_e474"] = {}
        with self.assertRaisesRegex(ValueError, "frozen-E474 marker must be null"):
            verify_payload(changed, archive)

    def test_rejects_nonfinite_delta_even_when_source_gate_receipt_says_finite(self):
        result, archive = fixture()
        altered = {key: value.copy() for key, value in archive.values.items()}
        altered["native__bf16_source__inp.weight"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite or nonnumeric"):
            verify_payload(result, ArchiveFixture(altered))

    def test_fp32_subtraction_precedes_float64_rms_accumulation(self):
        """Gate differences must match the actor's FP32 delta subtraction."""
        reference = {name: np.ones(2, dtype=np.float32) for name in NAMES}
        candidate = {name: value.copy() for name, value in reference.items()}
        candidate["gru.weight_hh_l0"][0] = np.float32(2**-25)
        actual = _compare(reference, candidate)["metrics"]["matrix:gru.weight_hh_l0"]
        fp32_error = candidate["gru.weight_hh_l0"] - reference["gru.weight_hh_l0"]
        widened_error = candidate["gru.weight_hh_l0"].astype(np.float64) - reference[
            "gru.weight_hh_l0"
        ].astype(np.float64)
        self.assertEqual(actual["max_abs_error"], float(np.abs(fp32_error).max()))
        self.assertNotEqual(actual["max_abs_error"], float(np.abs(widened_error).max()))


if __name__ == "__main__":
    unittest.main()
