from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
REQUIRED_FILES = (
    Path("src/wnn/training/__init__.py"),
    Path("scripts/train_rnn.py"),
    Path("scripts/check_training_structure.py"),
)


def inspect_module(path: Path, max_lines: int, max_function_lines: int) -> list[str]:
    """Check source size, signatures, public docs, and hidden state without importing code."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"{path}: cannot read Python source: {exc}"]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        line = exc.lineno or 0
        column = exc.offset or 0
        return [f"{path}:{line}:{column}: invalid Python syntax: {exc.msg}"]

    errors: list[str] = []
    if len(source.splitlines()) > max_lines:
        errors.append(f"{path}: module exceeds {max_lines} lines; split by responsibility")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append(f"{path}:{node.lineno}: pass state explicitly")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end_lineno = node.end_lineno or node.lineno
        if end_lineno - node.lineno + 1 > max_function_lines:
            errors.append(
                f"{path}:{node.lineno}: function {node.name!r} exceeds {max_function_lines} lines"
            )
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        missing = [
            arg.arg
            for arg in arguments
            if arg.arg not in {"self", "cls"} and arg.annotation is None
        ]
        missing_parts = []
        if missing:
            missing_parts.append(f"parameter annotations missing for {', '.join(missing)}")
        if node.returns is None:
            missing_parts.append("return annotation missing")
        if missing_parts:
            errors.append(f"{path}:{node.lineno}: {'; '.join(missing_parts)}")
        if not node.name.startswith("_") and ast.get_docstring(node) is None:
            errors.append(f"{path}:{node.lineno}: document the public function {node.name!r}")
    return errors


def discover_sources(root: Path) -> tuple[list[Path], list[str]]:
    """Find the trainer modules and report incomplete repository layouts."""
    root = root.resolve()
    if not root.is_dir():
        return [], [f"{root}: repository root does not exist or is not a directory"]

    training_dir = root / "src/wnn/training"
    if not training_dir.is_dir():
        return [], [f"{training_dir}: required training source directory is missing"]

    paths = sorted(training_dir.glob("*.py"))
    if not paths:
        return [], [f"{training_dir}: no Python modules found"]

    errors = []
    for relative_path in REQUIRED_FILES:
        path = root / relative_path
        if not path.is_file():
            errors.append(f"{path}: required quality-check source is missing")
        elif path not in paths:
            paths.append(path)
    return sorted(paths), errors


def main() -> int:
    """Check the modular trainer contract using only the standard library."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-lines", type=int, default=300)
    parser.add_argument("--max-function-lines", type=int, default=200)
    args = parser.parse_args()
    if args.max_lines < 1 or args.max_function_lines < 1:
        parser.error("line limits must be positive integers")

    logging.basicConfig(level=logging.INFO, format="{message}", style="{")
    paths, errors = discover_sources(args.root)
    errors.extend(
        message
        for path in paths
        for message in inspect_module(path, args.max_lines, args.max_function_lines)
    )
    for message in errors:
        LOGGER.error(message)
    LOGGER.info(
        json.dumps(
            {
                "files": len(paths),
                "errors": len(errors),
                "max_module_lines": args.max_lines,
                "max_function_lines": args.max_function_lines,
            }
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
