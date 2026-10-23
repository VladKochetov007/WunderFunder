from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "accept_export_capture_v9_test", HERE / "ACCEPT_EXPORT_CAPTURE_V10.py"
)
assert SPEC is not None and SPEC.loader is not None
ACCEPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPT)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(raw: bytes) -> dict[str, object]:
    return {"bytes": len(raw), "sha256": sha(raw), "base64": base64.b64encode(raw).decode()}


def fixture_bundle(root: Path) -> dict[str, object]:
    launch_root = root / "export_v5"
    capture_root = launch_root / "terminal_capture_v1"
    launch_root.mkdir()
    capture_root.mkdir()
    unit = ACCEPT.UNIT
    invocation = "78fa25d0abc44d199e3696d172dbc4f9"
    admission_raw = json.dumps(
        {"schema": "fg19.first-cell-export-admission.v1", "unit": unit}, sort_keys=True
    ).encode()
    launch_raw = json.dumps(
        {
            "schema": "fg19.first-cell-export-launch.v1",
            "unit": unit,
            "invocation_id": invocation,
            "admission_sha256": sha(admission_raw),
        },
        sort_keys=True,
    ).encode()
    receipt_bytes = {"ADMISSION.json": admission_raw, "LAUNCH.json": launch_raw}
    launch_stdout_raw = json.dumps(
        {
            "unit": unit,
            "outcome": "submitted",
            "admission_base64": base64.b64encode(admission_raw).decode(),
            "launch_base64": base64.b64encode(launch_raw).decode(),
        },
        sort_keys=True,
    ).encode()
    launch_transport_path = launch_root / "TRANSPORT.json"
    launch_transport = {
        "schema": "fg19.first-cell-export-root-ssh-transport.v1",
        "started_utc": "2026-10-21T10:22:17.960420+00:00",
        "finished_utc": "2026-10-21T10:22:20.997424+00:00",
        "body_bytes": ACCEPT.V6_BODY["bytes"],
        "body_sha256": ACCEPT.V6_BODY["sha256"],
        "source_review_bytes": ACCEPT.V6_SOURCE_REVIEW["bytes"],
        "source_review_sha256": ACCEPT.V6_SOURCE_REVIEW["sha256"],
        "ssh_returncode": 0,
        "timed_out": False,
        "stdout_bytes": len(launch_stdout_raw),
        "stdout_sha256": sha(launch_stdout_raw),
        "stderr_bytes": 0,
        "stderr_sha256": sha(b""),
        "persisted_before_parse": True,
        "retry_performed": False,
    }
    receipt_pins = {
        name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in receipt_bytes.items()
    }
    launch_acceptance = {
        "schema": "fg19.first-cell-export-launch-acceptance.v1",
        "status": "ACCEPTED_SINGLE_EXPORT_LAUNCH_NOT_QUALITY",
        "created_utc": "2026-10-21T10:22:20.998612+00:00",
        "unit": unit,
        "invocation_id": invocation,
        "receipts": receipt_pins,
        "body": ACCEPT.V6_BODY,
        "source_review": ACCEPT.V6_SOURCE_REVIEW,
        "raw_transport": {"path": str(launch_transport_path), **launch_transport},
        "scientific_fits_started": 0,
        "quality_result": None,
        "note": "ROOT acceptance binds exact stdin bytes; no worker stdin self-read or worker body-SHA claim.",
    }

    template_raw = (
        f'UNIT = "{unit}"\n'
        'INVOCATION = "UNBOUND_ACTUAL_INVOCATION"\n'
        "ADMISSION_BYTES = 0\n"
        'ADMISSION_SHA256 = "UNBOUND_ACTUAL_ADMISSION"\n'
        "LAUNCH_BYTES = 0\n"
        'LAUNCH_SHA256 = "UNBOUND_ACTUAL_LAUNCH"\n'
        "\n\n"
        "def reader():\n"
        "    return UNIT, INVOCATION, ADMISSION_BYTES, ADMISSION_SHA256, LAUNCH_BYTES, LAUNCH_SHA256\n"
    ).encode()
    literals = {
        "INVOCATION": invocation,
        "ADMISSION_BYTES": len(admission_raw),
        "ADMISSION_SHA256": sha(admission_raw),
        "LAUNCH_BYTES": len(launch_raw),
        "LAUNCH_SHA256": sha(launch_raw),
    }
    status_body_raw = ACCEPT.bind_reader_template(template_raw, literals)
    source_review_raw = b"fixed synthetic reviewed reader source\n"
    ast_binding = {
        "schema": "fg19.first-cell-export-reader-actual-binding.v1",
        "status": "EXACT_FIVE_LITERAL_BINDING_AST_OTHERWISE_IDENTICAL",
        "template_bytes": len(template_raw),
        "template_sha256": sha(template_raw),
        "source_review_bytes": len(source_review_raw),
        "source_review_sha256": sha(source_review_raw),
        "actual_reader_bytes": len(status_body_raw),
        "actual_reader_sha256": sha(status_body_raw),
        "actual_literal_values": literals,
        "all_other_AST_identical": True,
    }
    status_stdout_raw = json.dumps(
        {
            "schema": "fg19.first-cell-export-bounded-status.v1",
            "captured_utc": "2026-10-21T10:32:05.786909+00:00",
            "unit": unit,
            "invocation_id": invocation,
            "systemd": {"Id": unit, "InvocationID": invocation},
            "array_payloads_read_or_hashed": False,
            "ADMISSION.json": encoded(admission_raw),
            "LAUNCH.json": encoded(launch_raw),
            "log_tail": encoded(b""),
            "terminal_metadata": {},
        },
        sort_keys=True,
    ).encode()
    status_transport_path = capture_root / "TRANSPORT.json"
    status_transport = {
        "schema": "fg19.first-cell-export-status-root-transport.v1",
        "started_utc": "2026-10-21T10:32:05.085932+00:00",
        "finished_utc": "2026-10-21T10:32:06.016059+00:00",
        "body_bytes": len(status_body_raw),
        "body_sha256": sha(status_body_raw),
        "ssh_returncode": 0,
        "timed_out": False,
        "stdout_bytes": len(status_stdout_raw),
        "stdout_sha256": sha(status_stdout_raw),
        "stderr_bytes": 0,
        "stderr_sha256": sha(b""),
        "persisted_before_parse": True,
        "retry_performed": False,
    }
    return {
        "launch_root": launch_root,
        "capture_root": capture_root,
        "launch_stdout_raw": launch_stdout_raw,
        "launch_transport": launch_transport,
        "launch_transport_path": launch_transport_path,
        "launch_acceptance": launch_acceptance,
        "launch_receipts": receipt_bytes,
        "status_stdout_raw": status_stdout_raw,
        "status_stderr_raw": b"",
        "status_transport": status_transport,
        "status_transport_path": status_transport_path,
        "status_body_raw": status_body_raw,
        "status_ast_binding": ast_binding,
        "status_template_raw": template_raw,
        "status_source_review_raw": source_review_raw,
    }


def validate(bundle: dict[str, object]) -> dict[str, object]:
    observation, _, _, pins = ACCEPT.validate_capture_pair(
        **bundle,
        expected_launch_root=bundle["launch_root"],
    )
    return {"observation": observation, "pins": pins}


class AcceptExportCaptureV9TwoTransportTests(unittest.TestCase):
    def test_v5_review_and_fixture_pin_correction_remain_jointly_bound(self) -> None:
        review = json.loads(
            (HERE / "ROOT_CAPTURE_EXPORT_STATUS_SOURCE_REVIEW_V5.json").read_bytes()
        )
        correction = json.loads(
            (HERE / "ROOT_CAPTURE_EXPORT_STATUS_V5_REVIEW_PIN_CORRECTION_V1.json").read_bytes()
        )
        ACCEPT.validate_status_review_pair(review, correction)
        correction["fixture_pin_correction"]["sha256_observed_for_exact_path"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not bind the preserved review"):
            ACCEPT.validate_status_review_pair(review, correction)

    def test_two_distinct_raw_first_captures_join_to_same_launch(self) -> None:
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            bundle = fixture_bundle(Path(temporary))
            result = validate(bundle)
            self.assertEqual(result["observation"]["unit"], ACCEPT.UNIT)
            self.assertEqual(
                result["pins"]["launch_transport"]["path"],
                str(bundle["launch_root"] / "TRANSPORT.json"),
            )
            self.assertEqual(
                result["pins"]["status_raw"]["path"],
                str(bundle["capture_root"] / "STDOUT.raw.json"),
            )
            self.assertNotEqual(
                result["pins"]["launch_transport"]["path"],
                result["pins"]["status_body"]["path"],
            )

    def test_status_must_be_a_distinct_child_and_use_its_own_transport(self) -> None:
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            bundle = fixture_bundle(Path(temporary))
            bundle["capture_root"] = bundle["launch_root"]
            with self.assertRaisesRegex(ValueError, "distinct single-component child"):
                validate(bundle)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            bundle = fixture_bundle(Path(temporary))
            bundle["status_transport"] = bundle["launch_transport"]
            with self.assertRaisesRegex(ValueError, "status-capture transport exact field set"):
                validate(bundle)

    def test_reader_body_and_status_raw_must_join_the_status_transport(self) -> None:
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            bundle = fixture_bundle(Path(temporary))
            bundle["status_body_raw"] += b"# changed\n"
            with self.assertRaisesRegex(ValueError, "status reader body/AST binding"):
                validate(bundle)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            bundle = fixture_bundle(Path(temporary))
            bundle["status_stdout_raw"] += b" "
            with self.assertRaisesRegex(ValueError, "status-capture transport does not bind"):
                validate(bundle)

    def test_launch_reply_and_acceptance_cannot_borrow_status_transport(self) -> None:
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            bundle = fixture_bundle(Path(temporary))
            bundle["launch_stdout_raw"] += b" "
            with self.assertRaisesRegex(ValueError, "root SSH transport does not bind"):
                validate(bundle)


if __name__ == "__main__":
    unittest.main()
