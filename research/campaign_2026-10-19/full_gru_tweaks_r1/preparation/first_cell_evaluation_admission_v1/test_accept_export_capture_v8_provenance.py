from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "accept_export_capture_v8_test", HERE / "ACCEPT_EXPORT_CAPTURE_V8.py"
)
assert SPEC is not None and SPEC.loader is not None
ACCEPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPT)


def valid_review() -> dict[str, object]:
    rows = []
    for path, (size, digest, status) in ACCEPT.EXPORT_REVIEWED_FILES.items():
        row = {"path": path, "bytes": size, "sha256": digest}
        if status is not None:
            row["status"] = status
        rows.append(row)
    return {
        "schema": "fg19.first-cell-export-source-review.v4",
        "status": "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
        "reviewed_files": rows,
    }


def valid_transport() -> dict[str, object]:
    return {
        "schema": "fg19.first-cell-export-root-ssh-transport.v1",
        "started_utc": "2026-10-21T10:22:17.960420+00:00",
        "finished_utc": "2026-10-21T10:22:20.997424+00:00",
        "body_bytes": ACCEPT.V6_BODY["bytes"],
        "body_sha256": ACCEPT.V6_BODY["sha256"],
        "source_review_bytes": ACCEPT.V6_SOURCE_REVIEW["bytes"],
        "source_review_sha256": ACCEPT.V6_SOURCE_REVIEW["sha256"],
        "ssh_returncode": 0,
        "timed_out": False,
        "stdout_bytes": 12,
        "stdout_sha256": hashlib.sha256(b"raw response").hexdigest(),
        "stderr_bytes": 0,
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "persisted_before_parse": True,
        "retry_performed": False,
    }


def valid_launch_acceptance(
    transport_path: Path,
) -> tuple[dict[str, object], dict[str, tuple[int, str]]]:
    transport = valid_transport()
    receipt_pins = {
        "ADMISSION.json": (12_345, "a" * 64),
        "LAUNCH.json": (2_345, "b" * 64),
    }
    acceptance = {
        "schema": "fg19.first-cell-export-launch-acceptance.v1",
        "status": "ACCEPTED_SINGLE_EXPORT_LAUNCH_NOT_QUALITY",
        "created_utc": "2026-10-21T10:22:20.998612+00:00",
        "unit": ACCEPT.UNIT,
        "invocation_id": "78fa25d0abc44d199e3696d172dbc4f9",
        "receipts": {
            name: {"bytes": size, "sha256": digest} for name, (size, digest) in receipt_pins.items()
        },
        "body": ACCEPT.V6_BODY,
        "source_review": ACCEPT.V6_SOURCE_REVIEW,
        "raw_transport": {"path": str(transport_path), **transport},
        "scientific_fits_started": 0,
        "quality_result": None,
        "note": "ROOT acceptance binds exact stdin bytes; no worker stdin self-read or worker body-SHA claim.",
    }
    return acceptance, receipt_pins


class AcceptExportCaptureV8ProvenanceTests(unittest.TestCase):
    def test_review_requires_exact_twelve_pinned_unique_rows(self) -> None:
        pins = ACCEPT.validate_export_review_rows(valid_review())
        self.assertEqual(len(pins), 12)
        review = valid_review()
        review["reviewed_files"] = review["reviewed_files"][:-1]
        with self.assertRaisesRegex(ValueError, "exact twelve-row"):
            ACCEPT.validate_export_review_rows(review)

    def test_review_rejects_duplicate_and_mutated_rows(self) -> None:
        review = valid_review()
        rows = review["reviewed_files"]
        rows[-1] = dict(rows[0])
        with self.assertRaisesRegex(ValueError, "malformed/duplicate"):
            ACCEPT.validate_export_review_rows(review)
        review = valid_review()
        review["reviewed_files"][0]["bytes"] += 1
        with self.assertRaisesRegex(ValueError, "exact pinned V4 review"):
            ACCEPT.validate_export_review_rows(review)

    def test_root_acceptance_joins_v6_body_review_receipts_and_transport(self) -> None:
        transport_path = Path("/root/capture/TRANSPORT.json")
        acceptance, receipts = valid_launch_acceptance(transport_path)
        transport = valid_transport()
        ACCEPT.validate_root_launch_acceptance(
            acceptance,
            transport,
            transport_path,
            {"bytes": 12, "sha256": hashlib.sha256(b"raw response").hexdigest()},
            receipts,
            ACCEPT.UNIT,
            "78fa25d0abc44d199e3696d172dbc4f9",
        )

    def test_root_acceptance_rejects_body_review_receipt_or_transport_drift(self) -> None:
        transport_path = Path("/root/capture/TRANSPORT.json")
        for field, replacement in (
            ("body", {**ACCEPT.V6_BODY, "sha256": "0" * 64}),
            ("source_review", {**ACCEPT.V6_SOURCE_REVIEW, "bytes": 1}),
        ):
            acceptance, receipts = valid_launch_acceptance(transport_path)
            acceptance[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                ACCEPT.validate_root_launch_acceptance(
                    acceptance,
                    valid_transport(),
                    transport_path,
                    {"bytes": 12, "sha256": hashlib.sha256(b"raw response").hexdigest()},
                    receipts,
                    ACCEPT.UNIT,
                    "78fa25d0abc44d199e3696d172dbc4f9",
                )
        acceptance, receipts = valid_launch_acceptance(transport_path)
        receipts["LAUNCH.json"] = (2_346, "b" * 64)
        with self.assertRaisesRegex(ValueError, "receipt pin differs"):
            ACCEPT.validate_root_launch_acceptance(
                acceptance,
                valid_transport(),
                transport_path,
                {"bytes": 12, "sha256": hashlib.sha256(b"raw response").hexdigest()},
                receipts,
                ACCEPT.UNIT,
                "78fa25d0abc44d199e3696d172dbc4f9",
            )
        acceptance, receipts = valid_launch_acceptance(transport_path)
        acceptance["raw_transport"]["stdout_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "raw transport differs"):
            ACCEPT.validate_root_launch_acceptance(
                acceptance,
                valid_transport(),
                transport_path,
                {"bytes": 12, "sha256": hashlib.sha256(b"raw response").hexdigest()},
                receipts,
                ACCEPT.UNIT,
                "78fa25d0abc44d199e3696d172dbc4f9",
            )

    def test_private_source_validator_compiles_exact_bytes_without_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            core = directory / "core.py"
            contract = directory / "contract.py"
            core_raw = b'TOKEN = "pinned"\n'
            contract_raw = (
                b"from first_cell_score_core import TOKEN\n"
                b"def _validate_data_identity(binding, manifest):\n"
                b"    if binding.get('ok') is not True or manifest.get('token') != TOKEN:\n"
                b"        raise ValueError('identity mismatch')\n"
            )
            core.write_bytes(core_raw)
            contract.write_bytes(contract_raw)
            pins = {
                "first_cell_score_core": (
                    "core.py",
                    len(core_raw),
                    hashlib.sha256(core_raw).hexdigest(),
                ),
                "first_cell_score_contract": (
                    "contract.py",
                    len(contract_raw),
                    hashlib.sha256(contract_raw).hexdigest(),
                ),
            }
            ACCEPT.run_pinned_source_validator(directory, pins, {"ok": True}, {"token": "pinned"})
            self.assertFalse(any(directory.rglob("__pycache__")))
            self.assertFalse(any(directory.rglob("*.pyc")))
            self.assertNotIn("first_cell_score_core", sys.modules)
            self.assertNotIn("first_cell_score_contract", sys.modules)

    def test_private_source_validator_rejects_preloaded_module_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            name = "first_cell_score_core"
            sys.modules[name] = types.ModuleType(name)
            try:
                with self.assertRaisesRegex(ValueError, "already imported"):
                    ACCEPT.run_pinned_source_validator(
                        Path(temporary),
                        {
                            "first_cell_score_core": ("core.py", 1, "0" * 64),
                            "first_cell_score_contract": ("contract.py", 1, "1" * 64),
                        },
                        {"ok": True},
                        {"token": "pinned"},
                    )
            finally:
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
