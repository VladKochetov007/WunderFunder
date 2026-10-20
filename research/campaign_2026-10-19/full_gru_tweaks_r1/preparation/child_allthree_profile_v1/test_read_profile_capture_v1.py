"""Focused synthetic checks for the bounded profile capture reader."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from read_profile_capture_v1 import read_capture


def _receipt(raw: bytes) -> dict[str, object]:
    """Return a small exact-byte receipt fixture."""
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def _fixture() -> dict[str, object]:
    """Build a no-payload valid-shaped observation for the fixed profile."""
    from read_profile_capture_v1 import (
        CACHE_ACCEPTANCE,
        CACHE_METADATA,
        CONTRACT_SHA256,
        DICTIONARY_ACCEPTANCE,
        DICTIONARY_CACHE,
        INVOCATION,
        PARENT_ACCEPTANCE,
        PARENT_CHECKPOINT,
        PARENT_STOP,
        PROBE_SHA256,
        SOURCE_COMMIT,
        SPLIT,
        SUPPORT_SHA256,
        UNIT,
    )

    state = {
        "schema": "fg19-child-allthree-profile-initial-state.v1",
        "parent_run": "fg19_parent_h256x2_f0_s1001_fit_r2_v1",
        "parent_invocation_id": "5186b3016ff54b1fa754c10a1f87b3e4",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT,
        "parent_acceptance_sha256": PARENT_ACCEPTANCE,
        "parent_exact_stop_sha256": PARENT_STOP,
        "dictionary_cache_sha256": DICTIONARY_CACHE,
        "split_manifest_sha256": SPLIT,
        "source_commit": SOURCE_COMMIT,
        "parent_acceptance_status": "ACCEPTED",
        "initialization": {
            "parent_state_fingerprint": "a" * 64,
            "initial_child_state_fingerprint": "b" * 64,
            "parent_input_width": 211,
            "child_input_width": 467,
            "parent_output_width": 23,
            "child_output_width": 25,
            "parent_trunk_and_prefix_equal": True,
            "new_input_columns_zero": True,
            "new_logvariance_rows_zero": True,
            "normalization_prefix_equal": True,
            "normalization_suffix_mean_zero_std_one": True,
            "finite_initial_state": True,
        },
        "profile": {
            "sequence_length": 20_000,
            "batch": 16,
            "chunk": 2_000,
            "warmup_batches": 2,
            "measured_batches": 4,
            "batch_count": 6,
            "updates_per_batch": 10,
            "optimizer_attempts": 60,
            "input_representation": "ema_sparse",
            "objective": "nll",
        },
    }
    evidence = {
        "optimizer_attempts": 60,
        "optimizer_completed_updates": 60,
        "amp_skipped_updates": 0,
        "loss_checks": 60,
        "gradient_stats_steps": 60,
        "gradient_stats_nonfinite_steps": 0,
        "grad_scaler_enabled": False,
        "loss_finite": True,
        "gradient_finite": True,
        "counts_consistent": True,
        "valid_for_throughput": True,
        "gradient_check_basis": "complete pre-clip GradStats gradient norms",
    }
    log_raw = (
        "CHILD_PROFILE_INITIAL_STATE "
        + json.dumps(state)
        + '\nPROFILE {"step_wall":1.0}\nPROFILE_UPDATE_EVIDENCE '
        + json.dumps(evidence)
        + "\n"
    ).encode()
    admission = json.dumps(
        {
            "unit": UNIT,
            "run_name": "fg19_child_allthree_f0_s1001_profile_v1",
            "source_commit": SOURCE_COMMIT,
            "source_files_verified": 167,
            "source_manifest_sha256": "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
            "parent_checkpoint_sha256": PARENT_CHECKPOINT,
            "parent_acceptance_sha256": PARENT_ACCEPTANCE,
            "parent_exact_stop_sha256": PARENT_STOP,
            "split_sha256": SPLIT,
            "private_stage_sha256": {
                "DICTIONARY_ACCEPTANCE.json": DICTIONARY_ACCEPTANCE,
                "PARENT_EXACT_STOP.json": PARENT_STOP,
                "PARENT_TERMINAL_ACCEPTANCE.json": PARENT_ACCEPTANCE,
                "child_allthree_profile_contract_v1.py": CONTRACT_SHA256,
                "child_allthree_profile_probe_v1.py": PROBE_SHA256,
                "propensity_terminal_support.py": SUPPORT_SHA256,
            },
            "cache": {
                "metadata_sha256": CACHE_METADATA,
                "acceptance_sha256": CACHE_ACCEPTANCE,
            },
            "dictionary": {"sha256": DICTIONARY_CACHE},
        }
    ).encode()
    launch = json.dumps(
        {
            "unit": UNIT,
            "schema": "fg19-child-allthree-profile-launch.v1",
            "command": [
                "systemd-run",
                f"--unit={UNIT}",
                "child_allthree_profile_probe_v1.py",
            ],
            "submission": {
                "stderr_b64": base64.b64encode(
                    f"Running as unit: {UNIT}; invocation ID: {INVOCATION}\n".encode()
                ).decode()
            },
        }
    ).encode()
    return {
        "schema": "fg19-child-allthree-profile-capture.v1",
        "expected_unit": UNIT,
        "expected_invocation_id": INVOCATION,
        "terminal_success": True,
        "unit_query": {"returncode": 0, "timed_out": False},
        "unit_properties": {
            "InvocationID": INVOCATION,
            "ActiveState": "active",
            "SubState": "exited",
            "MainPID": "0",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "Result": "success",
        },
        "receipts": {
            "ADMISSION.json": _receipt(admission),
            "LAUNCH.json": _receipt(launch),
            "unit.log": _receipt(log_raw),
        },
        "staged_code_pins": {
            "child_allthree_profile_contract_v1.py": {"matches_expected": True},
            "child_allthree_profile_probe_v1.py": {"matches_expected": True},
            "propensity_terminal_support.py": {"matches_expected": True},
        },
        "run_directory_stat_only": {"exists": True, "symlink": False, "entries": []},
        "gpu": {},
        "parent_state": {"populated": {}},
        "own_cgroup_state": {},
    }


class ProfileCaptureReaderTests(unittest.TestCase):
    """Check exact receipt handling and strict profile evidence counters."""

    def test_valid_synthetic_profile_is_only_an_acceptance_candidate(self) -> None:
        """Recognize a source-shaped synthetic record without opening payloads."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_text(json.dumps(_fixture()), encoding="utf-8")
            result = read_capture(path)
        self.assertTrue(result["capture_consistent"])
        self.assertTrue(result["terminal_success"])
        self.assertTrue(result["acceptance_candidate_for_root_review"])

    def test_initial_state_types_are_strict(self) -> None:
        """Reject integer substitutions for proof booleans and widths."""
        from read_profile_capture_v1 import _exact_profile_state

        fixture = _fixture()
        receipts = fixture["receipts"]
        assert isinstance(receipts, dict)
        log_receipt = receipts["unit.log"]
        assert isinstance(log_receipt, dict)
        log_raw = base64.b64decode(log_receipt["base64"])
        marker = next(
            line
            for line in log_raw.decode().splitlines()
            if line.startswith("CHILD_PROFILE_INITIAL_STATE ")
        )
        state = json.loads(marker.removeprefix("CHILD_PROFILE_INITIAL_STATE "))
        self.assertTrue(_exact_profile_state(state))
        state["initialization"]["new_input_columns_zero"] = 1
        self.assertFalse(_exact_profile_state(state))
        state["initialization"]["new_input_columns_zero"] = True
        state["initialization"]["parent_input_width"] = True
        self.assertFalse(_exact_profile_state(state))

    def test_counter_mutation_and_receipt_corruption_are_rejected(self) -> None:
        """Reject a bool substituted for the update count and mismatched bytes."""
        from read_profile_capture_v1 import _valid_update_evidence

        fixture = _fixture()
        receipts = fixture["receipts"]
        assert isinstance(receipts, dict)
        log_receipt = receipts["unit.log"]
        assert isinstance(log_receipt, dict)
        log_raw = base64.b64decode(log_receipt["base64"])
        log = log_raw.decode()
        evidence_line = next(
            line for line in log.splitlines() if line.startswith("PROFILE_UPDATE_EVIDENCE ")
        )
        evidence = json.loads(evidence_line.removeprefix("PROFILE_UPDATE_EVIDENCE "))
        evidence["optimizer_attempts"] = True
        self.assertFalse(_valid_update_evidence(evidence))
        log_receipt["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_capture(path)


if __name__ == "__main__":
    unittest.main()
