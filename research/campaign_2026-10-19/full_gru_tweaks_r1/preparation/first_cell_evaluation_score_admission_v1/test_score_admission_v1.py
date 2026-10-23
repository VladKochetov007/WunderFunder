"""Synthetic metadata contract tests for conditional score admission only."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.pycache_prefix = None

DIRECTORY = Path(__file__).resolve().parent
UNIT = "computer-vision-homework-fg19-first-cell-export-v1.service"
INVOCATION = "0123456789abcdef0123456789abcdef"
ACCEPTANCE_SHA = "a" * 64


def load_module(name: str, path: Path) -> types.ModuleType:
    """Load a local pure template module without writing bytecode."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load local template: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BODY = load_module(
    "score_admission_body_fixture", DIRECTORY / "SCORE_ADMISSION_BODY_TEMPLATE_V1.py"
)


def accepted_export() -> dict[str, object]:
    """Build a tiny metadata-only fake for the producer acceptance contract."""
    outputs = {}
    for arm, _mode in BODY.ARMS:
        outputs[arm] = {
            "prediction": {
                "path": "predictions.npy",
                "bytes": BODY.PREDICTION_DATA_BYTES + 128,
                "npy_header_bytes": 128,
                "data_bytes": BODY.PREDICTION_DATA_BYTES,
                "sha256": "b" * 64,
                "dtype": "float32-little-endian",
                "shape": BODY.PREDICTION_SHAPE.copy(),
            },
            "checkpoint": {
                "relative_path": f"artifacts/{arm}/model_ep5.pt",
                "bytes": 4_279_970,
                "sha256": "c" * 64,
                "loaded_sha256": "c" * 64,
            },
        }
    return {
        "schema": "fg19.first-cell-export-acceptance.v1",
        "status": "ACCEPTED_FIXED_FIVE_PREDICTION_EXPORTS_NOT_SCORED",
        "unit": UNIT,
        "invocation_id": INVOCATION,
        "native_terminal": {
            "active_state": "active",
            "sub_state": "exited",
            "main_pid": 0,
            "exec_main_code": 1,
            "exec_main_status": 0,
        },
        "binding": {"bytes": BODY.EXPORT_BINDING_BYTES, "sha256": BODY.EXPORT_BINDING_SHA256},
        "source_review": {"bytes": BODY.SOURCE_REVIEW_BYTES, "sha256": BODY.SOURCE_REVIEW_SHA256},
        "evaluation_groups": 468,
        "membership_sha256": "d" * 64,
        "quality_result": None,
        "stop_status": "PENDING_EXACT_STOP",
        "array_payloads_read_or_hashed_by_ROOT": False,
        "outputs": outputs,
    }


def exact_stop() -> dict[str, object]:
    """Build a synthetic exact stop linked to the fake accepted export."""
    return {
        "schema": "fg19-first-cell-export-exact-owned-stop.v1",
        "captured_utc": "2026-10-21T12:00:00+00:00",
        "unit": UNIT,
        "invocation_id": INVOCATION,
        "acceptance_sha256": ACCEPTANCE_SHA,
        "before": {
            "Id": UNIT,
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "exited",
            "MainPID": "0",
            "InvocationID": INVOCATION,
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "Result": "success",
        },
        "after": {
            "Id": UNIT,
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": "0",
            "InvocationID": "",
        },
        "populated_before": {},
        "populated_after": {},
        "stop_returncode": 0,
        "stop_stdout": "",
        "stop_stderr": "",
        "scope": (
            "Only the one successful exited export unit. All old native/support and foreign units "
            "excluded; no file or payload mutation."
        ),
    }


class ScoreAdmissionMetadataTests(unittest.TestCase):
    """Exercise only local synthetic export-acceptance and stop metadata."""

    def test_exact_accepted_export_and_stop_pass(self) -> None:
        self.assertEqual(
            BODY.validate_export_authority(
                accepted_export(),
                exact_stop(),
                acceptance_sha256=ACCEPTANCE_SHA,
                binding_sha256=BODY.EXPORT_BINDING_SHA256,
                review_sha256=BODY.SOURCE_REVIEW_SHA256,
                expected_unit=UNIT,
            ),
            INVOCATION,
        )

    def test_nonzero_or_bool_native_exit_is_rejected(self) -> None:
        for bad_value in (1, True):
            with self.subTest(bad_value=bad_value):
                acceptance = accepted_export()
                acceptance["native_terminal"]["exec_main_status"] = bad_value
                with self.assertRaises(ValueError):
                    BODY.validate_export_authority(
                        acceptance,
                        exact_stop(),
                        acceptance_sha256=ACCEPTANCE_SHA,
                        binding_sha256=BODY.EXPORT_BINDING_SHA256,
                        review_sha256=BODY.SOURCE_REVIEW_SHA256,
                        expected_unit=UNIT,
                    )

    def test_receipt_mismatch_or_premature_score_is_rejected(self) -> None:
        invalid_cases = []
        wrong_stop = exact_stop()
        wrong_stop["acceptance_sha256"] = "e" * 64
        invalid_cases.append((accepted_export(), wrong_stop))
        scored = accepted_export()
        scored["quality_result"] = {"score": 0.1}
        invalid_cases.append((scored, exact_stop()))
        for acceptance, stop in invalid_cases:
            with (
                self.subTest(acceptance=acceptance["quality_result"]),
                self.assertRaises(ValueError),
            ):
                BODY.validate_export_authority(
                    acceptance,
                    stop,
                    acceptance_sha256=ACCEPTANCE_SHA,
                    binding_sha256=BODY.EXPORT_BINDING_SHA256,
                    review_sha256=BODY.SOURCE_REVIEW_SHA256,
                    expected_unit=UNIT,
                )

    def test_exact_stop_requires_zero_result_empty_streams_and_matching_unit(self) -> None:
        invalid_stops = []
        bad_returncode = exact_stop()
        bad_returncode["stop_returncode"] = True
        invalid_stops.append(bad_returncode)
        noisy_stop = exact_stop()
        noisy_stop["stop_stdout"] = "unexpected"
        invalid_stops.append(noisy_stop)
        wrong_unit = exact_stop()
        wrong_unit["before"]["Id"] = "another-unit.service"
        invalid_stops.append(wrong_unit)
        for stop in invalid_stops:
            with self.assertRaises(ValueError):
                BODY.validate_export_authority(
                    accepted_export(),
                    stop,
                    acceptance_sha256=ACCEPTANCE_SHA,
                    binding_sha256=BODY.EXPORT_BINDING_SHA256,
                    review_sha256=BODY.SOURCE_REVIEW_SHA256,
                    expected_unit=UNIT,
                )

    def test_unbound_renderer_refuses_to_build_submission_body(self) -> None:
        renderer = load_module(
            "score_admission_renderer_fixture", DIRECTORY / "render_score_admission_v1.py"
        )
        with self.assertRaisesRegex(RuntimeError, "still unbound"):
            renderer.render()


if __name__ == "__main__":
    unittest.main()
