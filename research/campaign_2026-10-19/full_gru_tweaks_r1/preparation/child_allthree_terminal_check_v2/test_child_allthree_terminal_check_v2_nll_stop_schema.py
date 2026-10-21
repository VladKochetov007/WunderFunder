from __future__ import annotations

import ast
import copy
import hashlib
import json
import unittest
from pathlib import Path

PACKET = Path(__file__).parent
PREPARATION = PACKET.parent
V1_CHECKER = (
    PREPARATION / "child_allthree_terminal_check_v1" / "child_allthree_terminal_check_v1.py"
)
V2_CHECKER = PACKET / "child_allthree_terminal_check_v2.py"
NLL_STOP = PREPARATION.parent / "records/child_nll_fit_v1/exact_stop/EXACT_STOP.json"
NLL_STOP_BYTES = 3_140
NLL_STOP_SHA = "c63be79b27d131acb0a38d2ad641fa7d72344cc344c26ab94ca77faf3b4a74f1"
NLL_SCHEMA = "fg19-child-nll-exact-owned-stop.v1"
NLL_UNIT = "computer-vision-homework-fg19-child-nll-fit-v1.service"
NLL_INVOCATION = "c833c613fb92426981ab255a5faf4452"
NLL_ACCEPTANCE_SHA = "d94457b11ce22a368cdc93df02e473c9a82cde6ddebd26cc6957bafc143750c5"


def _strict_stop_identity(value: object) -> bool:
    """Check only the pinned stop schema/identity fields in a metadata fixture."""
    if not isinstance(value, dict):
        return False
    before = value.get("before")
    if not isinstance(before, dict):
        return False
    return (
        value.get("schema") == NLL_SCHEMA
        and value.get("unit") == NLL_UNIT
        and value.get("invocation_id") == NLL_INVOCATION
        and value.get("acceptance_sha256") == NLL_ACCEPTANCE_SHA
        and before.get("InvocationID") == NLL_INVOCATION
        and before.get("ActiveState") == "active"
        and before.get("SubState") == "exited"
        and before.get("MainPID") == "0"
        and before.get("ExecMainCode") == "1"
        and before.get("ExecMainStatus") == "0"
    )


def _strict_schema_literal(source: bytes) -> str:
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_check_native_predecessor"
    ]
    if len(functions) != 1:
        raise AssertionError("expected one native-predecessor checker function")
    calls = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_check_stop_receipt"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "control_v3"
    ]
    if len(calls) != 1:
        raise AssertionError("expected one strict NLL stop-receipt call")
    schema = [keyword.value for keyword in calls[0].keywords if keyword.arg == "strict_schema"]
    if len(schema) != 1 or not isinstance(schema[0], ast.Constant):
        raise AssertionError("strict stop schema must be an explicit literal")
    if type(schema[0].value) is not str:
        raise AssertionError("strict stop schema literal must be a string")
    return schema[0].value


class NllStopSchemaTests(unittest.TestCase):
    """Check the one schema-literal repair and the frozen stop metadata only."""

    def test_checker_diff_is_exact_schema_literal_replacement(self) -> None:
        v1 = V1_CHECKER.read_bytes()
        v2 = V2_CHECKER.read_bytes()
        expected = b'strict_schema="fg19-child-nll-exact-owned-stop.v1"'
        previous = b'strict_schema="fg19.child-nll-exact-owned-stop.v1"'
        self.assertEqual(v1.count(previous), 1)
        self.assertEqual(v2.count(expected), 1)
        self.assertEqual(v2.replace(expected, previous, 1), v1)
        self.assertEqual(len(v2), len(v1))
        self.assertEqual(_strict_schema_literal(v2), NLL_SCHEMA)

    def test_actual_frozen_stop_metadata_matches_exact_hyphen_schema(self) -> None:
        raw = NLL_STOP.read_bytes()
        self.assertEqual(len(raw), NLL_STOP_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), NLL_STOP_SHA)
        actual = json.loads(raw)
        self.assertTrue(_strict_stop_identity(actual))

    def test_wrong_schema_and_missing_strict_identity_are_rejected(self) -> None:
        actual = json.loads(NLL_STOP.read_bytes())
        wrong_schema = copy.deepcopy(actual)
        wrong_schema["schema"] = "fg19.child-nll-exact-owned-stop.v1"
        self.assertFalse(_strict_stop_identity(wrong_schema))

        missing_unit = copy.deepcopy(actual)
        del missing_unit["unit"]
        self.assertFalse(_strict_stop_identity(missing_unit))

        missing_invocation = copy.deepcopy(actual)
        del missing_invocation["invocation_id"]
        self.assertFalse(_strict_stop_identity(missing_invocation))


if __name__ == "__main__":
    unittest.main()
