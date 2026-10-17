from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def inspect_module(path: Path, max_lines: int, max_function_lines: int) -> list[str]:
    """Lint size, signatures and hidden-state constructs without importing code."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    errors: list[str] = []
    if len(source.splitlines()) > max_lines:
        errors.append(f"{path}: module exceeds {max_lines} lines; split by responsibility")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append(f"{path}:{node.lineno}: pass state explicitly")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.end_lineno - node.lineno + 1 > max_function_lines:
            errors.append(f"{path}:{node.lineno}: function exceeds {max_function_lines} lines")
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
        if missing or node.returns is None:
            errors.append(f"{path}:{node.lineno}: annotate parameters and return type")
        if not node.name.startswith("_") and ast.get_docstring(node) is None:
            errors.append(f"{path}:{node.lineno}: document the public function")
    return errors


def main() -> int:
    """Check the modular trainer contract using only the standard library."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-lines", type=int, default=300)
    parser.add_argument("--max-function-lines", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="{message}", style="{")
    paths = sorted((args.root / "src/wnn/training").glob("*.py"))
    paths.extend([args.root / "scripts/train_rnn.py", Path(__file__).resolve()])
    errors = [
        message
        for path in paths
        for message in inspect_module(path, args.max_lines, args.max_function_lines)
    ]
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
