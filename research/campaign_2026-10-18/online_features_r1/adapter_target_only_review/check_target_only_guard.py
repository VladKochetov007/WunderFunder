from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _added_target_gate(patch_text: str):
    """Extract the proposed target-only compatibility predicate from its diff."""
    lines = patch_text.splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.startswith('+    if objective == "target-only" and (')
    )
    predicate_lines: list[str] = []
    for line in lines[start:]:
        if not line.startswith("+"):
            raise ValueError("target-only predicate is not contiguous in the patch")
        code = line[1:]
        predicate_lines.append(code)
        if code.strip() == "):":
            break
    else:
        raise ValueError("target-only predicate has no closing parenthesis")
    source = "def rejected(args, objective):\n" + "\n".join(predicate_lines)
    source += "\n        return True\n    return False\n"
    namespace: dict[str, object] = {}
    exec(compile(ast.parse(source), "<proposed-target-only-gate>", "exec"), namespace)
    return namespace["rejected"]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    """Find one top-level function in a source module AST."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"function {name!r} is absent")


def _names(node: ast.AST) -> set[str]:
    """Return names referenced within a function AST."""
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def main() -> None:
    """Exercise the changed gate against teacher paths without importing Torch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    args = parser.parse_args()
    gate = _added_target_gate(args.patch.read_text(encoding="utf-8"))

    def case(**values: object) -> SimpleNamespace:
        defaults: dict[str, object] = {
            "loss": "wmse",
            "distill": "",
            "td_alpha": 0.0,
            "ema_teacher": 0.0,
            "ce_mse": 0.0,
        }
        defaults.update(values)
        return SimpleNamespace(**defaults)

    cache_target_passes = not gate(case(distill_cache_root="/validated/cache"), "target-only")
    initializer_teacher_passes = not gate(
        case(distill_cache_root="/validated/cache", final_distill_initializer=True),
        "target-only",
    )
    family_loss_passes = not gate(
        case(distill_cache_root="/validated/cache", distill_family_weight=0.1),
        "target-only",
    )
    non_wmse_rejected = gate(case(loss="mse"), "target-only")

    root = args.repo_root.resolve(strict=True)
    losses_tree = ast.parse((root / "src/wnn/training/losses.py").read_text(encoding="utf-8"))
    main_loss_names = _names(_function(losses_tree, "main_loss"))
    main_loss_has_teacher_route = {
        "teach",
        "initializer_teacher",
        "family_targets",
    }.issubset(main_loss_names)
    if not all(
        (
            cache_target_passes,
            initializer_teacher_passes,
            family_loss_passes,
            non_wmse_rejected,
            main_loss_has_teacher_route,
        )
    ):
        raise AssertionError("source counterexample no longer matches the reviewed patch")
    sys.stdout.write(
        json.dumps(
            {
                "status": "pass",
                "cache_target_passes_proposed_gate": cache_target_passes,
                "initializer_teacher_passes_proposed_gate": initializer_teacher_passes,
                "family_loss_passes_proposed_gate": family_loss_passes,
                "non_wmse_rejected": non_wmse_rejected,
                "main_loss_contains_teacher_and_family_routes": main_loss_has_teacher_route,
                "torch_imported": False,
                "payloads_read": False,
                "scope": "stdlib AST and mock-argument check; no trainer/model execution",
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
