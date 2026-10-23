"""Tiny local metadata fixtures for the score result capture boundary."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "SCORE_RESULT_CAPTURE_TEMPLATE_V1.py"
SOURCE_TEMPLATE = (
    HERE.parent / "child_allthree_terminal_result_capture_v2" / "RESULT_CAPTURE_TEMPLATE.py"
)
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("score_result_capture_v1", TEMPLATE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load score result capture template")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _functions(source: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(source)
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _operational_dump(function: ast.FunctionDef) -> str:
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


class ScoreResultCaptureTests(unittest.TestCase):
    def test_unbound_constants_fail_before_filesystem_or_systemd_access(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "has not bound exact score admission bytes"):
            MODULE.main()

    def test_reused_bounded_helpers_match_reviewed_reader_ast(self) -> None:
        candidate = _functions(TEMPLATE.read_text(encoding="utf-8"))
        reviewed = _functions(SOURCE_TEMPLATE.read_text(encoding="utf-8"))
        for name in (
            "_read_regular",
            "_checked_cgroup",
            "_cgroup_populated",
            "_cgroup_text",
            "_gpu_identity",
        ):
            with self.subTest(helper=name):
                self.assertEqual(
                    _operational_dump(candidate[name]),
                    _operational_dump(reviewed[name]),
                )

    def test_synthetic_receipt_identity_and_admission_digest_join(self) -> None:
        invocation = "a" * 32
        export_invocation = "b" * 32
        staged = {
            name: {"bytes": byte_count, "sha256": digest}
            for name, (byte_count, digest) in MODULE.SCORER_CODE_PINS.items()
        }
        admission = {
            "schema": "fg19.first-cell-score-admission.v1",
            "status": "FRESH_GATES_PASSED_SUBMISSION_PENDING",
            "unit": MODULE.SCORE_UNIT,
            "result_path": str(MODULE.RESULT_PATH),
            "export_unit": MODULE.EXPORT_UNIT,
            "export_invocation_id": export_invocation,
            "staged_files": staged,
            "result_create_only_and_absent_before_mutation": True,
            "export_acceptance": {
                "bytes": 10,
                "sha256": "d" * 64,
                "staged_path": "accepted_export/EXPORT_ACCEPTANCE.json",
            },
            "export_exact_stop": {
                "bytes": 11,
                "sha256": "e" * 64,
                "staged_path": "accepted_export/EXPORT_EXACT_STOP.json",
            },
        }
        admission_raw = json.dumps(admission, sort_keys=True).encode()
        launch = {
            "schema": "fg19.first-cell-score-launch.v1",
            "unit": MODULE.SCORE_UNIT,
            "invocation_id": invocation,
            "export_unit": MODULE.EXPORT_UNIT,
            "export_invocation_id": export_invocation,
            "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
            "submission": {"outcome": "submitted", "returncode": 0},
        }
        launch_raw = json.dumps(launch, sort_keys=True).encode()
        joined = MODULE._validate_receipt_metadata(admission_raw, launch_raw, invocation)
        self.assertEqual(joined["admission_sha256"], hashlib.sha256(admission_raw).hexdigest())
        self.assertEqual(joined["export_invocation_id"], export_invocation)

        launch["invocation_id"] = "c" * 32
        wrong_launch = json.dumps(launch, sort_keys=True).encode()
        with self.assertRaisesRegex(ValueError, "launch identity"):
            MODULE._validate_receipt_metadata(admission_raw, wrong_launch, invocation)
        del admission["export_exact_stop"]
        missing_export_stop = json.dumps(admission, sort_keys=True).encode()
        with self.assertRaisesRegex(ValueError, "export_exact_stop"):
            MODULE._validate_receipt_metadata(missing_export_stop, launch_raw, invocation)

    def test_strict_terminal_tuple_and_result_read_gate(self) -> None:
        success = {
            "Result": "success",
            "ActiveState": "active",
            "SubState": "exited",
            "MainPID": "0",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
        }
        self.assertTrue(MODULE._strict_terminal_success(success))
        self.assertFalse(MODULE._strict_terminal_success({**success, "ExecMainStatus": "1"}))
        self.assertFalse(MODULE._strict_terminal_success({**success, "Result": "timeout"}))

        missing = Path(tempfile.gettempdir()) / "score-result-capture-must-not-open-on-failure.json"
        capture = MODULE._capture_result(False, missing)
        self.assertEqual(capture, {"read": False, "reason": "unit_not_strict_success"})

        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "RESULT.json"
            raw = b'{"schema":"synthetic","status":"fixture-only"}\n'
            result_path.write_bytes(raw)
            captured = MODULE._capture_result(True, result_path)
            self.assertTrue(captured["read"])
            self.assertEqual(captured["bytes"], len(raw))
            self.assertEqual(captured["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(base64.b64decode(captured["base64"], validate=True), raw)

    def test_log_tail_is_bounded_and_rejects_final_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unit.log"
            raw = b"x" * (MODULE.MAX_LOG_TAIL_BYTES + 8) + b"FAILURE"
            path.write_bytes(raw)
            captured = MODULE._capture_log_tail(path)
            tail = base64.b64decode(captured["tail_base64"], validate=True)
            self.assertEqual(captured["bytes"], len(raw))
            self.assertEqual(len(tail), MODULE.MAX_LOG_TAIL_BYTES)
            self.assertEqual(tail[-7:], b"FAILURE")
            self.assertEqual(captured["tail_offset_bytes"], len(raw) - len(tail))
            link = Path(temporary) / "unit-link.log"
            link.symlink_to(path)
            with self.assertRaises((OSError, ValueError)):
                MODULE._capture_log_tail(link)

    def test_reader_has_no_stdin_submission_or_payload_import_boundary(self) -> None:
        source = TEMPLATE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertTrue(
            imported.issubset(
                {
                    "base64",
                    "datetime",
                    "hashlib",
                    "json",
                    "os",
                    "re",
                    "stat",
                    "subprocess",
                    "pathlib",
                    "__future__",
                }
            )
        )
        self.assertNotIn("sys.stdin", source)
        self.assertNotIn("systemd-run", source)
        self.assertNotIn("predictions.npy", source)
        self.assertNotIn("targets.npy", source)
        self.assertNotIn("torch", imported)
        self.assertNotIn("numpy", imported)
        self.assertNotIn("wnn", imported)


if __name__ == "__main__":
    unittest.main()
