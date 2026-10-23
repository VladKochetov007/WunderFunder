"""Narrow static contract checks for the V5 one-key repair."""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

HERE = Path(__file__).absolute().parent
PREPARATION = HERE.parent
V4 = PREPARATION / "child_allthree_terminal_verifier_v4" / "REMOTE_SUBMISSION_V4.py"
V5 = HERE / "REMOTE_SUBMISSION_V5.py"
BINDING = HERE / "V5_BINDING.json"
READER = PREPARATION / "child_allthree_terminal_result_capture_v2" / "RESULT_CAPTURE_TEMPLATE.py"
V4_REVIEW = (
    PREPARATION / "child_allthree_terminal_result_capture_v2" / "PRODUCER_PAIR_REVIEW_V4.json"
)

UNIT = "computer-vision-homework-fg19-child-allthree-verify-v2.service"
NATIVE_UNIT = "computer-vision-homework-fg19-child-allthree-fit-v1.service"
INVOCATION = "94d3a7132a434d448345b0206bd3daa3"
RUN_NAME = "fg19_child_allthree_f0_s1001_fit_v1"
SUPPORT_INVOCATION = "a" * 32


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_bytes(), filename=str(path))


def top_level_assignment(module: ast.Module, name: str) -> ast.expr:
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in statement.targets
        ):
            return statement.value
    raise AssertionError(f"top-level assignment not found: {name}")


def string_dict_keys(node: ast.expr) -> list[str]:
    if not isinstance(node, ast.Dict):
        raise AssertionError("expected literal dict")
    keys = [key.value for key in node.keys if isinstance(key, ast.Constant)]
    if len(keys) != len(node.keys) or any(not isinstance(key, str) for key in keys):
        raise AssertionError("dict keys must be string literals")
    return keys


def function(module: ast.Module, name: str) -> ast.FunctionDef:
    for statement in module.body:
        if isinstance(statement, ast.FunctionDef) and statement.name == name:
            return statement
    raise AssertionError(f"function not found: {name}")


def assigned_dict_in(function_node: ast.FunctionDef, name: str) -> ast.Dict:
    for statement in ast.walk(function_node):
        if (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name for target in statement.targets
            )
            and isinstance(statement.value, ast.Dict)
        ):
            return statement.value
    raise AssertionError(f"dict assignment not found in {function_node.name}: {name}")


def dict_entries(node: ast.Dict) -> dict[str, ast.expr]:
    entries: dict[str, ast.expr] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise AssertionError("expected string-literal dict key")
        entries[key.value] = value
    return entries


def evaluate_identity(node: ast.expr, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    raise AssertionError(f"identity expression is not a literal or known name: {ast.dump(node)}")


def get_fields(node: ast.AST, base_name: str) -> set[str]:
    result: set[str] = set()
    for part in ast.walk(node):
        if (
            isinstance(part, ast.Call)
            and isinstance(part.func, ast.Attribute)
            and part.func.attr == "get"
            and isinstance(part.func.value, ast.Name)
            and part.func.value.id == base_name
            and part.args
            and isinstance(part.args[0], ast.Constant)
            and isinstance(part.args[0].value, str)
        ):
            result.add(part.args[0].value)
    return result


def producer_guard(main: ast.FunctionDef) -> ast.expr:
    for part in ast.walk(main):
        if (
            isinstance(part, ast.Call)
            and isinstance(part.func, ast.Name)
            and part.func.id == "require"
            and len(part.args) == 2
            and isinstance(part.args[1], ast.Constant)
            and part.args[1].value
            == "durable support records do not bind the successful submission"
        ):
            return part.args[0]
    raise AssertionError("durable reply guard not found")


def evaluate_guard(expression: ast.expr, namespace: dict[str, Any]) -> bool:
    allowed = {
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Compare,
        ast.Eq,
        ast.Call,
        ast.Attribute,
        ast.Name,
        ast.Constant,
        ast.Load,
    }
    if any(type(part) not in allowed for part in ast.walk(expression)):
        raise AssertionError("unexpected syntax in durable reply guard")
    code = compile(ast.Expression(expression), "<durable-reply-guard>", "eval")
    safe_globals = {"__builtins__": {"isinstance": isinstance, "dict": dict}}
    return bool(eval(code, safe_globals, namespace))


def top_level_json_object_call(main: ast.FunctionDef) -> ast.Dict:
    for part in ast.walk(main):
        if not (
            isinstance(part, ast.Call)
            and isinstance(part.func, ast.Attribute)
            and part.func.attr == "dumps"
            and isinstance(part.func.value, ast.Name)
            and part.func.value.id == "json"
            and part.args
            and isinstance(part.args[0], ast.Dict)
        ):
            continue
        return part.args[0]
    raise AssertionError("outer JSON reply mapping not found")


class VerifierV5ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v4_raw = V4.read_bytes()
        cls.v5_raw = V5.read_bytes()
        cls.v4_ast = ast.parse(cls.v4_raw, filename=str(V4))
        cls.v5_ast = ast.parse(cls.v5_raw, filename=str(V5))
        cls.binding = json.loads(BINDING.read_text(encoding="utf-8"))

    def test_v5_is_exactly_v4_plus_one_admission_unit_line(self) -> None:
        needle = (
            b"    admission = {\n"
            b'        "schema": "fg19.child-allthree-terminal-verifier-admission.v1",\n'
        )
        insertion = b'        "unit": UNIT,\n'
        self.assertEqual(self.v4_raw.count(needle), 1)
        self.assertEqual(self.v5_raw, self.v4_raw.replace(needle, needle + insertion, 1))
        self.assertEqual(self.binding["body"]["bytes"], len(self.v5_raw))
        self.assertEqual(self.binding["body"]["sha256"], sha256(self.v5_raw))
        self.assertEqual(self.binding["predecessor_v4_candidate"]["sha256"], sha256(self.v4_raw))
        insertion_offset = self.v5_raw.index(insertion, self.v5_raw.index(needle) + len(needle))
        reconstructed = (
            self.v5_raw[:insertion_offset] + self.v5_raw[insertion_offset + len(insertion) :]
        )
        self.assertEqual(reconstructed, self.v4_raw)
        inverse = self.binding["inverse_byte_proof"]
        self.assertEqual(inverse["inserted_text"], insertion.decode("ascii"))
        self.assertEqual(inverse["inserted_bytes"], len(insertion))
        self.assertEqual(inverse["byte_offset"], insertion_offset)
        self.assertEqual(inverse["restores_exact_v4_bytes"], True)

    def test_all_fifteen_embedded_payload_pins_are_unchanged(self) -> None:
        payload_names: list[str] | None = None
        for constant in ("PAYLOAD", "FIXED_PAYLOAD_PINS", "MAX_INPUT_BYTES"):
            before = top_level_assignment(self.v4_ast, constant)
            after = top_level_assignment(self.v5_ast, constant)
            self.assertEqual(
                ast.dump(before, include_attributes=False),
                ast.dump(after, include_attributes=False),
            )
            names = string_dict_keys(before)
            if payload_names is None:
                payload_names = names
            else:
                self.assertEqual(set(names), set(payload_names))
        assert payload_names is not None
        self.assertEqual(len(payload_names), 15)
        pins = self.binding["payload_pin_preservation"]["payload_pins"]
        self.assertEqual([item["name"] for item in pins], payload_names)
        v4_pins = ast.literal_eval(top_level_assignment(self.v4_ast, "FIXED_PAYLOAD_PINS"))
        self.assertEqual(
            pins,
            [
                {"name": name, "bytes": v4_pins[name]["bytes"], "sha256": v4_pins[name]["sha256"]}
                for name in payload_names
            ],
        )

    def test_producer_identity_satisfies_reader_and_reply_contract_fields(self) -> None:
        main = function(self.v5_ast, "main")
        admission_fields = dict_entries(assigned_dict_in(main, "admission"))
        # make_launch_record returns the exact durable launch identity mapping.
        launch_function = function(self.v5_ast, "make_launch_record")
        launch_return = next(
            node.value
            for node in ast.walk(launch_function)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        )
        launch_fields = dict_entries(launch_return)
        admission_identity = {
            key: evaluate_identity(
                admission_fields[key],
                {
                    "UNIT": UNIT,
                    "RUN_NAME": RUN_NAME,
                    "NATIVE_UNIT": NATIVE_UNIT,
                    "INVOCATION": INVOCATION,
                },
            )
            for key in (
                "schema",
                "status",
                "run_name",
                "unit",
                "native_unit",
                "native_invocation_id",
            )
        }
        launch_identity = {
            key: evaluate_identity(
                launch_fields[key],
                {
                    "UNIT": UNIT,
                    "NATIVE_UNIT": NATIVE_UNIT,
                    "INVOCATION": INVOCATION,
                    "invocation_id": SUPPORT_INVOCATION,
                },
            )
            for key in ("schema", "unit", "invocation_id", "native_unit", "native_invocation_id")
        }
        reader = parse(READER)
        reader_main = function(reader, "main")
        reader_admission = get_fields(reader_main, "admission")
        reader_launch = get_fields(reader_main, "launch")
        self.assertEqual(
            reader_admission,
            {"schema", "status", "run_name", "native_unit", "native_invocation_id"},
        )
        self.assertEqual(
            reader_launch,
            {"schema", "unit", "invocation_id", "native_unit", "native_invocation_id"},
        )
        reader_raw = READER.read_bytes()
        self.assertEqual(self.binding["reader_contract_reference"]["bytes"], len(reader_raw))
        self.assertEqual(self.binding["reader_contract_reference"]["sha256"], sha256(reader_raw))
        self.assertTrue(reader_admission <= admission_fields.keys())
        self.assertTrue(reader_launch <= launch_fields.keys())
        self.assertEqual(
            set(self.binding["identity_contract"]["admission_fields"]),
            set(admission_fields),
        )
        self.assertEqual(
            set(self.binding["identity_contract"]["launch_fields"]),
            set(launch_fields),
        )
        self.assertEqual(
            admission_identity,
            {
                "schema": "fg19.child-allthree-terminal-verifier-admission.v1",
                "status": "FRESH_GATES_PASSED_SUPPORT_SUBMISSION_PENDING",
                "run_name": RUN_NAME,
                "unit": UNIT,
                "native_unit": NATIVE_UNIT,
                "native_invocation_id": INVOCATION,
            },
        )
        self.assertEqual(
            launch_identity,
            {
                "schema": "fg19.child-allthree-terminal-verifier-launch.v1",
                "unit": UNIT,
                "invocation_id": SUPPORT_INVOCATION,
                "native_unit": NATIVE_UNIT,
                "native_invocation_id": INVOCATION,
            },
        )

    def test_actual_durable_reply_guard_accepts_correct_unit_and_rejects_missing_or_wrong(
        self,
    ) -> None:
        main = function(self.v5_ast, "main")
        admission_fields = dict_entries(assigned_dict_in(main, "admission"))
        admission = {
            key: evaluate_identity(
                admission_fields[key],
                {
                    "UNIT": UNIT,
                    "RUN_NAME": RUN_NAME,
                    "NATIVE_UNIT": NATIVE_UNIT,
                    "INVOCATION": INVOCATION,
                },
            )
            for key in (
                "schema",
                "status",
                "run_name",
                "unit",
                "native_unit",
                "native_invocation_id",
            )
        }
        launch = {
            "unit": UNIT,
            "native_unit": NATIVE_UNIT,
            "native_invocation_id": INVOCATION,
            "invocation_id": SUPPORT_INVOCATION,
            "submission": {"outcome": "submitted", "returncode": 0},
        }
        expression = producer_guard(main)
        actual_guard_fields = {
            "durable_admission": sorted(get_fields(expression, "durable_admission")),
            "durable_launch": sorted(get_fields(expression, "durable_launch")),
            "durable_submission": sorted(get_fields(expression, "durable_submission")),
        }
        self.assertEqual(
            actual_guard_fields,
            {
                "durable_admission": ["schema", "unit"],
                "durable_launch": [
                    "invocation_id",
                    "native_invocation_id",
                    "native_unit",
                    "unit",
                ],
                "durable_submission": ["outcome", "returncode"],
            },
        )
        self.assertEqual(
            self.binding["identity_contract"]["producer_reply_guard_fields"],
            actual_guard_fields,
        )
        self.assertTrue(
            evaluate_guard(
                expression,
                {
                    "durable_admission": admission,
                    "durable_launch": launch,
                    "durable_submission": launch["submission"],
                    "UNIT": UNIT,
                    "NATIVE_UNIT": NATIVE_UNIT,
                    "INVOCATION": INVOCATION,
                    "support_invocation": SUPPORT_INVOCATION,
                },
            )
        )
        for bad_admission in (
            {key: value for key, value in admission.items() if key != "unit"},
            {**admission, "unit": "wrong.service"},
            {**admission, "schema": "wrong"},
        ):
            self.assertFalse(
                evaluate_guard(
                    expression,
                    {
                        "durable_admission": bad_admission,
                        "durable_launch": launch,
                        "durable_submission": launch["submission"],
                        "UNIT": UNIT,
                        "NATIVE_UNIT": NATIVE_UNIT,
                        "INVOCATION": INVOCATION,
                        "support_invocation": SUPPORT_INVOCATION,
                    },
                )
            )
        for field, value in (
            ("unit", "wrong.service"),
            ("native_unit", "wrong.service"),
            ("native_invocation_id", "0" * 32),
            ("invocation_id", "0" * 32),
        ):
            self.assertFalse(
                evaluate_guard(
                    expression,
                    {
                        "durable_admission": admission,
                        "durable_launch": {**launch, field: value},
                        "durable_submission": launch["submission"],
                        "UNIT": UNIT,
                        "NATIVE_UNIT": NATIVE_UNIT,
                        "INVOCATION": INVOCATION,
                        "support_invocation": SUPPORT_INVOCATION,
                    },
                )
            )
        for field, value in (("outcome", "unknown"), ("returncode", 1)):
            bad_submission = {**launch["submission"], field: value}
            self.assertFalse(
                evaluate_guard(
                    expression,
                    {
                        "durable_admission": admission,
                        "durable_launch": {**launch, "submission": bad_submission},
                        "durable_submission": bad_submission,
                        "UNIT": UNIT,
                        "NATIVE_UNIT": NATIVE_UNIT,
                        "INVOCATION": INVOCATION,
                        "support_invocation": SUPPORT_INVOCATION,
                    },
                )
            )

    def test_outer_reply_mapping_and_rejection_are_pinned(self) -> None:
        main = function(self.v5_ast, "main")
        reply = top_level_json_object_call(main)
        reply_keys = string_dict_keys(reply)
        self.assertEqual(reply_keys, ["unit", "outcome", "admission_base64", "launch_base64"])
        review_raw = V4_REVIEW.read_bytes()
        review = json.loads(review_raw)
        self.assertEqual(review["status"], "REJECT_SOURCE_ONLY_NO_B_ATTEMPT")
        self.assertEqual(self.binding["preserved_v4_negative_review"]["bytes"], len(review_raw))
        self.assertEqual(self.binding["preserved_v4_negative_review"]["sha256"], sha256(review_raw))


if __name__ == "__main__":
    unittest.main(verbosity=2)
