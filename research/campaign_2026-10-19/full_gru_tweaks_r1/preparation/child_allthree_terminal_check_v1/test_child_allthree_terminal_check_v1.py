"""Focused pure fixtures for strict AllThree terminal metadata."""

from __future__ import annotations

import unittest

import child_allthree_terminal_check_v1 as checker


def terminal_fixture() -> dict[str, object]:
    """Return a tiny structurally valid native exit receipt."""
    return {
        "schema": "fg19.child-allthree-fit-native-terminal-evidence.v1",
        "run_name": checker.RUN,
        "unit": checker.UNIT,
        "invocation_id": checker.INVOCATION,
        "active_state": "active",
        "sub_state": "exited",
        "main_pid": 0,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "raw_path": checker.NATIVE_RAW_PATH,
        "raw_bytes": 1,
        "raw_sha256": "a" * 64,
        "captured_utc": "2026-10-21T06:00:00+00:00",
        "native_exit_utc": "2026-10-21T05:59:59+00:00",
    }


class NativeTerminalMetadataTests(unittest.TestCase):
    def test_accepts_exact_typed_exit_metadata(self) -> None:
        result = checker._check_native_terminal(terminal_fixture())

        self.assertEqual(result["raw_path"], checker.NATIVE_RAW_PATH)
        self.assertEqual(result["state"], "active/exited")

    def test_rejects_boolean_pid_and_byte_count(self) -> None:
        for field in ("main_pid", "raw_bytes"):
            with self.subTest(field=field):
                receipt = terminal_fixture()
                receipt[field] = False
                with self.assertRaises(ValueError):
                    checker._check_native_terminal(receipt)

    def test_rejects_unbound_raw_path(self) -> None:
        receipt = terminal_fixture()
        receipt["raw_path"] = "some/other/capture.json"

        with self.assertRaises(ValueError):
            checker._check_native_terminal(receipt)

    def test_rejects_malformed_raw_digest_and_timezone(self) -> None:
        for field, value in (
            ("raw_sha256", "A" * 64),
            ("captured_utc", "2026-10-21T06:00:00"),
        ):
            with self.subTest(field=field):
                receipt = terminal_fixture()
                receipt[field] = value
                with self.assertRaises(ValueError):
                    checker._check_native_terminal(receipt)

    def test_rejects_extra_schema_fields(self) -> None:
        receipt = terminal_fixture()
        receipt["quality_score"] = None

        with self.assertRaises(ValueError):
            checker._check_native_terminal(receipt)

    def test_rejects_exit_after_capture(self) -> None:
        receipt = terminal_fixture()
        receipt["native_exit_utc"] = "2026-10-21T06:00:01+00:00"

        with self.assertRaises(ValueError):
            checker._check_native_terminal(receipt)


if __name__ == "__main__":
    unittest.main()
