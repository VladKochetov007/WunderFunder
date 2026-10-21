from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path

V2_BODY = (
    Path(__file__).parents[1] / "child_allthree_terminal_verifier_v2" / "REMOTE_SUBMISSION_V2.py"
)
V3_BODY = Path(__file__).with_name("REMOTE_SUBMISSION_V3.py")
EXPECTED_NAMES = {
    "ADMISSION.json",
    "LAUNCH.json",
    "LAUNCH_ACCEPTANCE.json",
    "ALLTHREE_LAUNCH_STDOUT.raw.json",
    "TRANSPORT.json",
}
NATIVE_NAMES = {"ADMISSION.json", "LAUNCH.json"}
ROOT_ONLY_NAMES = {
    "ALLTHREE_LAUNCH_STDOUT.raw.json",
    "LAUNCH_ACCEPTANCE.json",
    "TRANSPORT.json",
}


def _read_source(path: Path) -> bytes:
    """Read only the bounded local worker source bytes for static fixtures."""
    data = path.read_bytes()
    if len(data) > 2 * 1024 * 1024:
        raise AssertionError(f"source exceeds the local fixture bound: {path}")
    return data


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {name} function")
    return matches[0]


def _literal_string_set(function: ast.FunctionDef, name: str) -> set[str]:
    assignments = [
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Set):
        raise AssertionError(f"{name} must be one literal set")
    values = assignments[0].value.elts
    if not all(isinstance(item, ast.Constant) and type(item.value) is str for item in values):
        raise AssertionError(f"{name} must contain literal strings only")
    result = {item.value for item in values}
    if len(result) != len(values):
        raise AssertionError(f"{name} must not contain duplicates")
    return result


def _expected_names(function: ast.FunctionDef) -> set[str]:
    assignments = [
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "expected"
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
        raise AssertionError("expected receipt map must be one literal dict")
    keys = assignments[0].value.keys
    if not all(isinstance(key, ast.Constant) and type(key.value) is str for key in keys):
        raise AssertionError("expected receipt names must be literal strings")
    result = {key.value for key in keys}
    if len(result) != len(keys):
        raise AssertionError("expected receipt names must not be duplicated")
    return result


def _assert_source_class_contract(tree: ast.Module) -> None:
    function = _function(tree, "check_fit_receipts")
    native = _literal_string_set(function, "native_names")
    root_only = _literal_string_set(function, "root_only_names")
    expected = _expected_names(function)
    if native != NATIVE_NAMES or root_only != ROOT_ONLY_NAMES:
        raise AssertionError("receipt names must retain their exact native/root-only classes")
    if native & root_only or native | root_only != expected or expected != EXPECTED_NAMES:
        raise AssertionError("receipt source classes must be a disjoint exact partition")

    stable_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "stable_file"
    ]
    if len(stable_calls) != 1:
        raise AssertionError("only the two native receipt paths may be read from EVIDENCE")
    native_branches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "name"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.In)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Name)
        and node.test.comparators[0].id == "native_names"
    ]
    if len(native_branches) != 1:
        raise AssertionError("native file read must be guarded by the native-name branch")
    branch = native_branches[0]
    if stable_calls[0] not in list(ast.walk(branch)) or stable_calls[0] in list(
        ast.walk(ast.Module(body=branch.orelse, type_ignores=[]))
    ):
        raise AssertionError("stable_file may only run in the native-name branch")


class ReceiptSourceClassTests(unittest.TestCase):
    """Check source boundaries without importing or executing either worker."""

    def test_v3_is_one_function_change_with_exact_inverse_bytes(self) -> None:
        original = _read_source(V2_BODY)
        candidate = _read_source(V3_BODY)
        v2_tree = ast.parse(original, filename=str(V2_BODY))
        v3_tree = ast.parse(candidate, filename=str(V3_BODY))
        v2_function = _function(v2_tree, "check_fit_receipts")
        v3_function = _function(v3_tree, "check_fit_receipts")
        original_lines = original.splitlines(keepends=True)
        candidate_lines = candidate.splitlines(keepends=True)
        old_block = b"".join(original_lines[v2_function.lineno - 1 : v2_function.end_lineno])
        new_block = b"".join(candidate_lines[v3_function.lineno - 1 : v3_function.end_lineno])
        self.assertEqual(candidate.count(new_block), 1)
        self.assertEqual(candidate.replace(new_block, old_block, 1), original)

        normalized = copy.deepcopy(v3_tree)
        normalized_function = _function(normalized, "check_fit_receipts")
        normalized.body[normalized.body.index(normalized_function)] = copy.deepcopy(v2_function)
        self.assertEqual(
            ast.dump(normalized, include_attributes=False),
            ast.dump(v2_tree, include_attributes=False),
        )

    def test_exact_native_and_root_only_source_classes(self) -> None:
        tree = ast.parse(_read_source(V3_BODY), filename=str(V3_BODY))
        _assert_source_class_contract(tree)

    def test_wrong_class_and_missing_name_fixtures_fail_closed(self) -> None:
        source = _read_source(V3_BODY)
        tree = ast.parse(source, filename=str(V3_BODY))
        function = _function(tree, "check_fit_receipts")
        _assert_source_class_contract(tree)

        native_assignment = next(
            node
            for node in function.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "native_names"
        )
        native_assignment.value.elts[0].value = "TRANSPORT.json"
        with self.assertRaises(AssertionError):
            _assert_source_class_contract(tree)

        wrong_root_only_tree = ast.parse(source, filename=str(V3_BODY))
        wrong_root_only_function = _function(wrong_root_only_tree, "check_fit_receipts")
        root_only_assignment = next(
            node
            for node in wrong_root_only_function.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "root_only_names"
        )
        root_only_assignment.value.elts[0].value = "UNEXPECTED.json"
        with self.assertRaises(AssertionError):
            _assert_source_class_contract(wrong_root_only_tree)

        missing_name_tree = ast.parse(source, filename=str(V3_BODY))
        missing_function = _function(missing_name_tree, "check_fit_receipts")
        expected_assignment = next(
            node
            for node in missing_function.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "expected"
        )
        expected_assignment.value.keys.pop()
        with self.assertRaises(AssertionError):
            _assert_source_class_contract(missing_name_tree)


if __name__ == "__main__":
    unittest.main()
