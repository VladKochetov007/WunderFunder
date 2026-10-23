"""Focused synthetic checks for the V4 receipt-path collision guard."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

CALLER = Path(__file__).with_name("ROOT_ACCEPT_SCORE_RESULT_V4.py")


def preflight_expression() -> ast.expr:
    tree = ast.parse(CALLER.read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    matches = []
    for node in ast.walk(main):
        if not isinstance(node, ast.If):
            continue
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_fail"
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and child.args[0].value
            == "ROOT score acceptance already exists; never repeat this action"
            for statement in node.body
            for child in ast.walk(statement)
        ):
            matches.append(node.test)
    if len(matches) != 1:
        raise AssertionError("expected one exact output collision preflight")
    return matches[0]


class OutputPreflightTests(unittest.TestCase):
    def evaluate_guard(self, record: Path) -> bool:
        expression = ast.Expression(preflight_expression())
        ast.fix_missing_locations(expression)
        return bool(eval(compile(expression, str(CALLER), "eval"), {"RECORD": record}))

    def test_absent_outputs_are_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(self.evaluate_guard(Path(directory)))

    def test_existing_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory)
            (record / "SCORE_ACCEPTANCE.json").write_bytes(b"fixture")
            self.assertTrue(self.evaluate_guard(record))

    def test_dangling_binding_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory)
            dangling = record / "ROOT_ACCEPTANCE_BINDING.json"
            dangling.symlink_to(record / "missing-target")
            self.assertFalse(dangling.exists())
            self.assertTrue(dangling.is_symlink())
            self.assertTrue(self.evaluate_guard(record))


if __name__ == "__main__":
    unittest.main()
