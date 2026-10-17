"""Parse the frozen E540 command lines without importing Torch or reading data."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def normalize_entry_argv(script_name: str, argv: list[str]) -> list[str]:
    """Remove the explicitly logged script token before in-process ``main`` calls."""
    expected = f"scripts/{Path(script_name).name}"
    if not argv or argv[0] != expected:
        raise ValueError(f"E540 argv must begin with its declared entry point {expected!r}")
    return argv[1:]


def _assigns_name(statement: ast.stmt, name: str) -> bool:
    targets = []
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
    return any(isinstance(target, ast.Name) and target.id == name for target in targets)


def _is_parser_assignment(statement: ast.stmt, name: str) -> bool:
    if not _assigns_name(statement, name):
        return False
    value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "ArgumentParser"
    )


def _is_parse_assignment(statement: ast.stmt, parser_name: str) -> bool:
    if not _assigns_name(statement, "args"):
        return False
    value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == parser_name
        and value.func.attr == "parse_args"
    )


def _modular_parser_statements(path: Path, module: ast.Module) -> list[ast.stmt] | None:
    """Extract current modular declarations without importing Torch or any data code."""
    is_modular = any(
        isinstance(node, ast.ImportFrom) and node.module == "wnn.training.cli"
        for node in module.body
    )
    if not is_modular:
        return None
    package = path.resolve().parents[1] / "src/wnn/training"
    statements: list[ast.stmt] = []
    # Use actual feature-choice definitions, including expanded moment/age modes.
    features = ast.parse((package.parent / "research_features.py").read_text())
    for node in features.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id
            in {"MOMENT_FEATURE_MODES", "FIELD_AGE_FEATURE_MODES", "RESEARCH_FEATURE_MODES"}
            for target in node.targets
        ):
            statements.append(node)
    for source in sorted(package.glob("cli*.py")):
        tree = ast.parse(source.read_text(), filename=str(source))
        statements.extend(node for node in tree.body if isinstance(node, ast.FunctionDef))
    statements.extend(ast.parse("args = build_parser().parse_args()").body)
    return statements


def _extract_main_parser(path: Path, *, parser_name: str) -> list[ast.stmt]:
    """Read the actual parser-building statements from a script's ``main``."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modular = _modular_parser_statements(path, module)
    if modular is not None:
        return modular
    main = next(
        (
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
        ),
        None,
    )
    if main is None:
        raise ValueError(f"CLI source has no top-level main(): {path}")
    start = next(
        (
            index
            for index, statement in enumerate(main.body)
            if _is_parser_assignment(statement, parser_name)
        ),
        None,
    )
    if start is None:
        raise ValueError(f"CLI parser construction not found in {path}")
    end = next(
        (
            index
            for index in range(start, len(main.body))
            if _is_parse_assignment(main.body[index], parser_name)
        ),
        None,
    )
    if end is None:
        raise ValueError(f"CLI parse_args() not found in {path}")
    return main.body[start : end + 1]


def _dynamics_argument_adder(path: Path) -> Any:
    """Compile the actual stdlib-only optimizer argument registration function."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "add_arguments"
        ),
        None,
    )
    if function is None:
        raise ValueError(f"optimizer argument registration not found in {path}")
    namespace: dict[str, Any] = {"argparse": argparse}
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"
    )
    exec(code, namespace)
    return namespace["add_arguments"]


def parse_project_cli(
    source_path: Path,
    argv: list[str],
    *,
    parser_name: str,
    dynamics_path: Path | None = None,
) -> argparse.Namespace:
    """Execute only the real parser declarations with stdlib-only dependencies."""
    statements = _extract_main_parser(source_path, parser_name=parser_name)
    namespace: dict[str, Any] = {
        "argparse": argparse,
        "Path": Path,
        "NumericFeatureMap": SimpleNamespace(MODES=("linear", "ple")),
        "QAT_MODES": (
            "none",
            "recurrent",
            "recurrent_head",
            "recurrent_w4",
            "input_w4_recurrent_w8",
        ),
        "VALUE_MODES": ("unchanged", "neutral", "safe"),
        "INDICATOR_MODES": ("none", "missing", "current"),
        "RESEARCH_FEATURE_MODES": ("none", "volume28", "volume28_current"),
        "PRIVILEGED_MODES": ("none", "current", "future"),
        "PrefixAdapterGRUModel": SimpleNamespace(
            MODES=("conditional", "unconditional", "diagonal")
        ),
    }
    if dynamics_path is not None:
        namespace["add_dynamics_arguments"] = _dynamics_argument_adder(dynamics_path)
    else:
        namespace["add_dynamics_arguments"] = lambda parser: None

    parser_module = ast.Module(body=statements, type_ignores=[])
    code = compile(ast.fix_missing_locations(parser_module), str(source_path), "exec")
    previous_argv = sys.argv
    try:
        sys.argv = [str(source_path), *argv]
        exec(code, namespace)
    finally:
        sys.argv = previous_argv
    return namespace["args"]
