"""Static source checks for the V3 optional-result capture contract."""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path

PREPARATION = Path(__file__).resolve().parents[1]
V2_TEMPLATE = PREPARATION / "child_allthree_terminal_result_capture_v2/RESULT_CAPTURE_TEMPLATE.py"
V3_TEMPLATE = PREPARATION / "child_allthree_terminal_result_capture_v3/RESULT_CAPTURE_TEMPLATE.py"
V5_PRODUCER = PREPARATION / "child_allthree_terminal_verifier_v5/REMOTE_SUBMISSION_V5.py"
V5_PRODUCER_SHA256 = "4d555fd69954d94415c135e467eadd081aa6fd915686197e5c7e628e935b9374"


def _module(path: Path) -> ast.Module:
    """Parse one source file without importing or executing it."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    """Return one top-level function by name."""
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    matches = [node for node in functions if node.name == name]
    assert len(matches) == 1, f"expected exactly one {name}() function"
    return matches[0]


def _normalize_v3(module: ast.Module) -> ast.Module:
    """Reverse the documented optional-file delta before comparing reader ASTs."""
    normalized = copy.deepcopy(module)
    first = normalized.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        first.value.value = "NORMALIZED_MODULE_DOCSTRING"

    normalized.body = [
        node
        for node in normalized.body
        if not isinstance(node, ast.FunctionDef) or node.name != "_capture_optional_verification"
    ]
    main = _function(normalized, "main")
    main.body = [
        node
        for node in main.body
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "directory_identity"
                for target in node.targets
            )
        )
    ]
    terminal_gate = next(
        node
        for node in main.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "terminal_success"
    )
    terminal_gate.body = [
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id="_capture_file", ctx=ast.Load()),
                args=[
                    ast.Name(id="files", ctx=ast.Load()),
                    ast.Constant(value="VERIFICATION.json"),
                ],
                keywords=[],
            )
        )
    ]
    ast.fix_missing_locations(normalized)
    return normalized


def _runner_literals(module: ast.Module) -> list[str]:
    """Collect literal source fragments used to construct the worker runner."""
    main = _function(module, "main")
    runner = next(
        node
        for node in main.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "runner" for target in node.targets)
    )
    return [
        node.value
        for node in ast.walk(runner.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def main() -> None:
    """Validate the narrow V3 source delta and producer result-output contract."""
    v2 = _module(V2_TEMPLATE)
    if (
        isinstance(v2.body[0], ast.Expr)
        and isinstance(v2.body[0].value, ast.Constant)
        and isinstance(v2.body[0].value.value, str)
    ):
        v2.body[0].value.value = "NORMALIZED_MODULE_DOCSTRING"
    v3 = _module(V3_TEMPLATE)
    assert ast.dump(v2, include_attributes=False) == ast.dump(
        _normalize_v3(v3), include_attributes=False
    ), "V3 changed source outside the documented optional VERIFICATION capture"

    optional = _function(v3, "_capture_optional_verification")
    assert any(
        isinstance(handler.type, ast.Name) and handler.type.id == "FileNotFoundError"
        for node in ast.walk(optional)
        if isinstance(node, ast.Try)
        for handler in node.handlers
    ), "optional capture must catch only a missing-file condition"
    optional_constants = {
        node.value
        for node in ast.walk(optional)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "VERIFICATION.json" in optional_constants
    assert "not_found" in optional_constants
    assert any(
        isinstance(node, ast.Compare)
        and any(isinstance(operator, ast.NotEq) for operator in node.ops)
        for node in ast.walk(optional)
    ), "absence capture must reject a changed output-directory identity"

    main = _function(v3, "main")
    required_capture_names = {
        element.value
        for node in ast.walk(main)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple)
        for element in node.iter.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }
    assert {"VERIFIER_ADMISSION.json", "VERIFIER_LAUNCH.json", "unit.log"}.issubset(
        required_capture_names
    ), "all three required files must remain required captures"

    producer_bytes = V5_PRODUCER.read_bytes()
    assert hashlib.sha256(producer_bytes).hexdigest() == V5_PRODUCER_SHA256
    producer = ast.parse(producer_bytes, filename=str(V5_PRODUCER))
    runner_fragments = _runner_literals(producer)
    assert any("result = checker.verify_terminal" in fragment for fragment in runner_fragments)
    assert any(
        "print(json.dumps(result, sort_keys=True, allow_nan=False))\n" in fragment
        for fragment in runner_fragments
    ), "the worker result must be emitted as its JSON stdout line"
    assert all("VERIFICATION.json" not in fragment for fragment in runner_fragments)
    assert any(
        fragment == "))\nprint(json.dumps(result, sort_keys=True, allow_nan=False))\n"
        for fragment in runner_fragments
    ), "the checker-result print is the runner's final statement"
    assert all(
        "os.open(path, os.O_RDONLY" in fragment or "open(" not in fragment
        for fragment in runner_fragments
    ), "runner file opens are limited to read-only source prehashes"
    assert all(
        "write(" not in fragment and "write_json" not in fragment for fragment in runner_fragments
    )
    assert all(
        flag not in fragment
        for fragment in runner_fragments
        for flag in ("os.O_WRONLY", "os.O_RDWR", "os.O_CREAT", "os.O_TRUNC")
    ), "runner source prehashes do not open writable files"

    producer_main = _function(producer, "main")
    command = next(
        node
        for node in producer_main.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets)
    )
    command_literals = {
        node.value
        for node in ast.walk(command.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "--property=StandardOutput=append:" in command_literals
    assert "--property=StandardError=append:" in command_literals
    assert "unit.log" in command_literals

    print("PASS: V3 optional-file delta and print-to-unit.log contract are source-only")


if __name__ == "__main__":
    main()
