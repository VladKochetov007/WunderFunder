"""Focused V2 regression for original dictionary stats and V1 failure binding."""

from __future__ import annotations

import ast
import json
import unittest

import prepare_child_ema_terminal_verifier_v2 as packet

EXPECTED_DICTIONARY_PATH = "/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
EXPECTED_DICTIONARY_BYTES = 22_557
EXPECTED_DICTIONARY_SHA = "9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659"


def dictionary_predicate() -> object:
    """Compile only the pure dictionary metadata predicate from V2 source."""
    raw = packet.stable_read(packet.PAYLOAD_INPUTS["child_ema_terminal_check_v2.py"], 50_000)
    module = ast.parse(raw)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_check_dictionary_stat_record"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(isolated, "child_ema_terminal_check_v2.py", "exec"), namespace)
    return namespace["_check_dictionary_stat_record"]


class EmaDictionaryStatV2Tests(unittest.TestCase):
    """Bind only actual accepted dictionary metadata and its exact stat contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.admission = json.loads(
            packet.stable_read(packet.LOCAL_INPUTS["ADMISSION.json"], 20_000)
        )
        cls.record = cls.admission["dictionary"]
        cls.predicate = staticmethod(dictionary_predicate())

    def test_original_five_field_record_passes_without_mtime(self) -> None:
        self.assertEqual(set(self.record), {"bytes", "device", "inode", "path", "sha256"})
        self.assertNotIn("mtime_ns", self.record)
        self.predicate(
            self.record,
            (self.record["bytes"], self.record["device"], self.record["inode"]),
            EXPECTED_DICTIONARY_PATH,
            EXPECTED_DICTIONARY_BYTES,
            EXPECTED_DICTIONARY_SHA,
        )

    def test_actual_stat_drift_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "EMA dictionary stat identity differs"):
            self.predicate(
                self.record,
                (self.record["bytes"], self.record["device"], self.record["inode"] + 1),
                EXPECTED_DICTIONARY_PATH,
                EXPECTED_DICTIONARY_BYTES,
                EXPECTED_DICTIONARY_SHA,
            )

    def test_accepted_digest_drift_is_rejected(self) -> None:
        altered = dict(self.record, sha256="0" * 64)
        with self.assertRaisesRegex(
            ValueError, "EMA dictionary accepted path/hash identity differs"
        ):
            self.predicate(
                altered,
                (altered["bytes"], altered["device"], altered["inode"]),
                EXPECTED_DICTIONARY_PATH,
                EXPECTED_DICTIONARY_BYTES,
                EXPECTED_DICTIONARY_SHA,
            )

    def test_unrecorded_mtime_key_is_rejected(self) -> None:
        altered = dict(self.record, mtime_ns=1)
        with self.assertRaisesRegex(ValueError, "EMA dictionary stat record schema differs"):
            self.predicate(
                altered,
                (altered["bytes"], altered["device"], altered["inode"]),
                EXPECTED_DICTIONARY_PATH,
                EXPECTED_DICTIONARY_BYTES,
                EXPECTED_DICTIONARY_SHA,
            )

    def test_exact_v1_failure_capture_and_traceback_are_bound(self) -> None:
        payload = {
            name: packet.stable_read(path, 100_000)
            for name, path in packet.PAYLOAD_INPUTS.items()
            if name in {"V1_FAILURE_STDOUT.raw.json", "V1_FAILURE_UNIT_LOG.txt"}
        }
        result = packet.validate_v1_failure_binding(payload)
        self.assertEqual(result["v1_checker_sha256"], packet.V1_CHECKER_SHA)
        self.assertEqual(result["v1_body_sha256"], packet.V1_BODY_SHA)
        self.assertEqual(result["failure"], "EMA dictionary stat identity differs")

    def test_v1_failure_capture_and_log_drift_are_rejected(self) -> None:
        payload = {
            name: packet.stable_read(path, 100_000)
            for name, path in packet.PAYLOAD_INPUTS.items()
            if name in {"V1_FAILURE_STDOUT.raw.json", "V1_FAILURE_UNIT_LOG.txt"}
        }
        altered_capture = dict(payload)
        altered_capture["V1_FAILURE_STDOUT.raw.json"] += b" "
        with self.assertRaisesRegex(RuntimeError, "V1 failure evidence digest differs"):
            packet.validate_v1_failure_binding(altered_capture)
        altered_log = dict(payload)
        altered_log["V1_FAILURE_UNIT_LOG.txt"] += b" "
        with self.assertRaisesRegex(RuntimeError, "V1 failure evidence digest differs"):
            packet.validate_v1_failure_binding(altered_log)


if __name__ == "__main__":
    unittest.main()
