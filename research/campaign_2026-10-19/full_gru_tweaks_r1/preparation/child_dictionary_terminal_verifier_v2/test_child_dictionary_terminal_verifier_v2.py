from __future__ import annotations

import ast
import unittest
from pathlib import Path

BODY = Path(__file__).with_name("REMOTE_SUBMISSION.py")


def _body_tree() -> ast.Module:
    """Parse the frozen local V2 body without importing or executing it."""
    return ast.parse(BODY.read_text(encoding="utf-8"))


def _literal_dict(node: ast.expr) -> dict[str, ast.expr]:
    """Return a literal string-key dictionary or fail the focused fixture."""
    if not isinstance(node, ast.Dict):
        raise AssertionError("expected a literal dictionary")
    result: dict[str, ast.expr] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise AssertionError("expected literal string keys")
        result[key.value] = value
    return result


class DictionaryTerminalVerifierV2Tests(unittest.TestCase):
    def test_body_has_no_worker_self_read_or_self_cap(self) -> None:
        tree = _body_tree()
        names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
        self.assertNotIn("__file__", names)
        main = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        main_names = [node.id for node in ast.walk(main) if isinstance(node, ast.Name)]
        self.assertNotIn("body_raw", main_names)
        self.assertNotIn("body_stat", main_names)
        self.assertFalse(
            any(isinstance(node, ast.Constant) and node.value == 250_000 for node in ast.walk(main))
        )

    def test_admission_uses_root_review_and_raw_transport_authority(self) -> None:
        tree = _body_tree()
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "verifier_admission"
                for target in node.targets
            )
        )
        admission = _literal_dict(assignment.value)
        self.assertNotIn("rendered_body", admission)
        authority = _literal_dict(admission["rendered_body_authority"])
        self.assertEqual(
            {
                "schema",
                "owner",
                "source_review",
                "source_review_scope",
                "root_preparation",
                "raw_transport",
            },
            set(authority),
        )
        self.assertIsInstance(authority["source_review_scope"], ast.Constant)
        self.assertIn("root prelaunch gate", authority["source_review_scope"].value)
        self.assertIsInstance(authority["source_review"], ast.Name)
        self.assertEqual("source_review_pin", authority["source_review"].id)
        self.assertEqual("root_preparation_pin", authority["root_preparation"].id)
        transport = _literal_dict(authority["raw_transport"])
        self.assertEqual(
            {
                "owner",
                "method",
                "external_binding_required",
                "worker_self_read",
                "worker_body_sha256_claimed",
            },
            set(transport),
        )
        self.assertIsInstance(transport["worker_self_read"], ast.Constant)
        self.assertFalse(transport["worker_self_read"].value)
        self.assertIsInstance(transport["worker_body_sha256_claimed"], ast.Constant)
        self.assertFalse(transport["worker_body_sha256_claimed"].value)


if __name__ == "__main__":
    unittest.main()
