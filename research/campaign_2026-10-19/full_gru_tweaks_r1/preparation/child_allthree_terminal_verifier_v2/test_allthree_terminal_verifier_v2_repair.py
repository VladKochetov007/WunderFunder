from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path

V1_BODY = Path(__file__).parents[1] / "child_allthree_terminal_verifier_v1" / "REMOTE_SUBMISSION.py"
V2_BODY = Path(__file__).with_name("REMOTE_SUBMISSION_V2.py")
INSERTED_LINE = (
    b'        "--setenv=FG19_APPROVED_PATH_COMPAT_EXCEPTION='
    b'pathlib._local.PosixPath:exact-dictionary-path-only",\n'
)
INSERTED_ARGUMENT = (
    "--setenv=FG19_APPROVED_PATH_COMPAT_EXCEPTION="
    "pathlib._local.PosixPath:exact-dictionary-path-only"
)
GPU_ARGUMENT = "--setenv=CUDA_VISIBLE_DEVICES="


def _read_source(path: Path) -> bytes:
    """Read a bounded local source file for a byte-only comparison."""
    data = path.read_bytes()
    if len(data) > 2 * 1024 * 1024:
        raise AssertionError(f"source exceeds the local fixture bound: {path}")
    return data


def _find_command_list(tree: ast.Module) -> ast.List:
    """Find the unique systemd-run command list without importing the body."""
    candidates = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "command"
        and isinstance(node.value, ast.List)
        and node.value.elts
        and isinstance(node.value.elts[0], ast.Constant)
        and node.value.elts[0].value == "systemd-run"
    ]
    if len(candidates) != 1:
        raise AssertionError("expected one literal systemd-run argv list")
    return candidates[0]


def _assert_single_repair(candidate: bytes, original: bytes) -> None:
    """Require exactly the reviewed insertion and its byte-exact inverse."""
    if candidate.count(INSERTED_LINE) != 1:
        raise AssertionError("candidate must contain one exact compatibility argument")
    if candidate.replace(INSERTED_LINE, b"", 1) != original:
        raise AssertionError("candidate has a change beyond the single insertion")


class ExactRepairTests(unittest.TestCase):
    """Check the single argv fix without importing or executing either worker."""

    def test_inverse_bytes_match_frozen_v1(self) -> None:
        original = _read_source(V1_BODY)
        candidate = _read_source(V2_BODY)
        self.assertEqual(original.count(b'"--setenv=CUDA_VISIBLE_DEVICES=",\n'), 1)
        _assert_single_repair(candidate, original)
        self.assertEqual(len(candidate), len(original) + len(INSERTED_LINE))

    def test_ast_diff_is_one_systemd_argv_item(self) -> None:
        original = ast.parse(_read_source(V1_BODY), filename=str(V1_BODY))
        candidate = ast.parse(_read_source(V2_BODY), filename=str(V2_BODY))
        _find_command_list(original)
        after_command = _find_command_list(candidate)
        matching_indices = [
            index
            for index, element in enumerate(after_command.elts)
            if isinstance(element, ast.Constant) and element.value == INSERTED_ARGUMENT
        ]
        self.assertEqual(len(matching_indices), 1)
        index = matching_indices[0]
        previous = after_command.elts[index - 1]
        self.assertIsInstance(previous, ast.Constant)
        self.assertEqual(previous.value, GPU_ARGUMENT)
        normalized = copy.deepcopy(candidate)
        _find_command_list(normalized).elts.pop(index)
        self.assertEqual(
            ast.dump(normalized, include_attributes=False),
            ast.dump(original, include_attributes=False),
        )
        self.assertNotIn(
            INSERTED_ARGUMENT,
            [
                node.value
                for node in ast.walk(original)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            ],
        )

    def test_missing_and_duplicate_argument_fixtures_fail(self) -> None:
        original = _read_source(V1_BODY)
        candidate = _read_source(V2_BODY)
        missing = candidate.replace(INSERTED_LINE, b"", 1)
        duplicate = candidate.replace(INSERTED_LINE, INSERTED_LINE * 2, 1)
        for malformed in (missing, duplicate):
            with (
                self.subTest(argument_count=malformed.count(INSERTED_LINE)),
                self.assertRaises(AssertionError),
            ):
                _assert_single_repair(malformed, original)


if __name__ == "__main__":
    unittest.main()
