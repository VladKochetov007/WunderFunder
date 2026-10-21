"""Focused static fixtures for the V4 verifier wrapper contract."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parent
TEMPLATE = PACKET / "REMOTE_SUBMISSION_V4_TEMPLATE.py"
V3_BODY = PACKET.parent / "child_allthree_terminal_verifier_v3/REMOTE_SUBMISSION_V3.py"


def _tree(path: Path) -> ast.Module:
    """Parse packet source without importing or executing it."""
    return ast.parse(path.read_text(encoding="utf-8"))


def _assign(tree: ast.Module, name: str) -> object:
    """Read one literal top-level assignment from static source."""
    values = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(values) != 1:
        raise AssertionError(f"expected one {name} assignment")
    return ast.literal_eval(values[0])


class AllThreeTerminalVerifierV4ContractTest(unittest.TestCase):
    """Guard the fixed V3-to-V4 adaptation and reply ordering."""

    def test_template_has_only_the_three_create_time_slots(self) -> None:
        text = TEMPLATE.read_text(encoding="utf-8")
        tree = ast.parse(text)
        self.assertEqual(_assign(tree, "PAYLOAD"), {})
        self.assertEqual(_assign(tree, "FIXED_PAYLOAD_PINS"), {})
        self.assertEqual(_assign(tree, "FINAL_REVIEW_BINDING"), {})
        self.assertEqual(text.count("PAYLOAD = {}"), 1)
        self.assertEqual(text.count("FIXED_PAYLOAD_PINS = {}"), 1)
        self.assertEqual(text.count("FINAL_REVIEW_BINDING = {}"), 1)

    def test_support_identity_is_v2_and_failure_unit_is_not_an_action_target(self) -> None:
        text = TEMPLATE.read_text(encoding="utf-8")
        tree = _tree(TEMPLATE)
        private_directory = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "D" for target in node.targets)
        )
        self.assertEqual(
            ast.get_source_segment(text, private_directory.value),
            'B / "research_tmp/fg19_child_allthree_verify_v2"',
        )
        self.assertEqual(
            _assign(tree, "UNIT"),
            "computer-vision-homework-fg19-child-allthree-verify-v2.service",
        )
        self.assertNotIn("child_allthree_terminal_check_v1", text)
        self.assertNotIn("fg19_child_allthree_verify_v1", text)
        self.assertNotIn("computer-vision-homework-fg19-child-allthree-verify-v1.service", text)
        self.assertEqual(
            _assign(tree, "ROOT_BINDING_SCHEMA"),
            "fg19.child-allthree-terminal-verifier-render-binding.v4",
        )
        self.assertIn("child_allthree_terminal_check_v2.py", _assign(tree, "MAX_INPUT_BYTES"))

    def test_unchanged_receipt_map_and_helper_structure_match_frozen_v3(self) -> None:
        v3 = _tree(V3_BODY)
        v4 = _tree(TEMPLATE)
        v3_receipts = next(
            node.value
            for node in v3.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "READONLY_RECEIPTS"
                for target in node.targets
            )
        )
        v4_receipts = next(
            node.value
            for node in v4.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "READONLY_RECEIPTS"
                for target in node.targets
            )
        )
        self.assertEqual(
            ast.dump(v4_receipts, include_attributes=False),
            ast.dump(v3_receipts, include_attributes=False),
        )
        v3_functions = {node.name for node in v3.body if isinstance(node, ast.FunctionDef)}
        v4_functions = {node.name for node in v4.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(v4_functions, v3_functions)
        for name in (
            "MAX_SOURCE_BYTES",
            "MAX_METADATA_BYTES",
            "MAX_MODEL_BYTES",
            "MAX_STATE_BYTES",
            "MAX_ARTIFACT_BYTES",
            "SUPPORT_MAX_BYTES",
            "PARENT_TARGET_BYTES",
            "PARENT_RESERVE_BYTES",
        ):
            self.assertEqual(_assign(v4, name), _assign(v3, name), name)

    def test_outer_reply_is_exact_bounded_and_after_durable_launch(self) -> None:
        tree = _tree(TEMPLATE)
        main = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        prints = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        self.assertEqual(len(prints), 1)
        printed = prints[0].args[0]
        self.assertIsInstance(printed, ast.Call)
        self.assertIsInstance(printed.func, ast.Attribute)
        self.assertEqual(printed.func.attr, "dumps")
        reply = printed.args[0]
        self.assertIsInstance(reply, ast.Dict)
        self.assertEqual(
            [key.value for key in reply.keys if isinstance(key, ast.Constant)],
            ["unit", "outcome", "admission_base64", "launch_base64"],
        )
        stable_calls = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "stable_file"
        ]
        self.assertEqual(len(stable_calls), 2)
        self.assertTrue(all(ast.literal_eval(node.args[1]) == 1_000_000 for node in stable_calls))
        self.assertLess(max(node.lineno for node in stable_calls), prints[0].lineno)
        source = TEMPLATE.read_text(encoding="utf-8")
        main_source = "\n".join(source.splitlines()[main.lineno - 1 : main.end_lineno])
        self.assertIn("len(admission_raw) + len(launch_raw) <= 4_000_000", main_source)
        self.assertIn("support_invocation", main_source)
        self.assertIn('"outcome": durable_submission["outcome"]', main_source)

    def test_render_gate_binds_v2_review_root_proof_and_native_bytes(self) -> None:
        tree = _tree(TEMPLATE)
        binding = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "check_render_binding"
        )
        source = ast.get_source_segment(TEMPLATE.read_text(encoding="utf-8"), binding)
        self.assertIsNotNone(source)
        assert source is not None
        self.assertIn("private_import_boundary_proof", source)
        self.assertIn("fg19.child-allthree-terminal-import-boundary-proof.v2", source)
        self.assertEqual(
            _assign(tree, "ROOT_BINDING_SCHEMA"),
            "fg19.child-allthree-terminal-verifier-render-binding.v4",
        )
        self.assertIn("native_terminal", source)


if __name__ == "__main__":
    unittest.main()
