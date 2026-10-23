from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast


def _preflight_function(source: str) -> ast.FunctionDef:
    """Extract configure_feature_adapter's no-Torch objective preflight."""
    tree = ast.parse(source)
    configure = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "configure_feature_adapter"
    )
    prefix: list[ast.stmt] = []
    for statement in configure.body:
        prefix.append(statement)
        if (
            isinstance(statement, ast.If)
            and isinstance(statement.test, ast.UnaryOp)
            and isinstance(statement.test.op, ast.Not)
            and isinstance(statement.test.operand, ast.Name)
            and statement.test.operand.id == "adapter_only"
        ):
            break
    else:
        raise ValueError("objective preflight does not terminate at adapter-only return")
    return ast.FunctionDef(
        name="preflight",
        args=configure.args,
        body=prefix,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )


def _run_preflight(preflight: object, **values: object) -> str:
    """Return an exception class name, or empty string for accepted arguments."""
    defaults: dict[str, object] = {
        "research_feature_adapter_only": True,
        "research_feature_adapter_objective": "target-only",
        "loss": "wmse",
        "distill": "",
        "distill_cache_root": None,
        "final_distill_initializer": False,
        "distill_family_weight": 0.0,
        "td_alpha": 0.0,
        "ema_teacher": 0.0,
        "ce_mse": 0.0,
    }
    defaults.update(values)
    run = SimpleNamespace(options=SimpleNamespace(args=SimpleNamespace(**defaults)))
    try:
        cast(Callable[[object], None], preflight)(run)
    except (TypeError, ValueError) as error:
        return type(error).__name__
    return ""


def main() -> None:
    """Check target-only guard cases and default auxiliary-gate placement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--integration", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve(strict=True)
    adapter_path = root / "src/wnn/training/feature_adapter.py"
    losses_path = root / "src/wnn/training/losses.py"
    adapter_source = adapter_path.read_text(encoding="utf-8")
    losses_source = losses_path.read_text(encoding="utf-8")
    patch_source = args.patch.read_text(encoding="utf-8")
    integration = json.loads(args.integration.read_text(encoding="utf-8"))

    preflight_ast = _preflight_function(adapter_source)
    future_annotations = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations")],
        level=0,
    )
    module = ast.Module(body=[future_annotations, preflight_ast], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(adapter_path), "exec"), namespace)
    preflight = namespace["preflight"]

    blocked_options = {
        "distill": "gpt",
        "distill_cache_root": "/validated/cache",
        "final_distill_initializer": True,
        "distill_family_weight": 0.1,
        "td_alpha": 0.2,
        "ema_teacher": 0.1,
        "ce_mse": 0.1,
        "loss": "mse",
    }
    blocked_results = {
        name: _run_preflight(preflight, **{name: value}) for name, value in blocked_options.items()
    }
    all_teacher_modifiers_rejected = all(blocked_results.values())
    target_only_without_adapter_rejected = bool(
        _run_preflight(preflight, research_feature_adapter_only=False)
    )
    valid_target_only_accepted = not _run_preflight(preflight)
    inherited_default_accepted = not _run_preflight(
        preflight,
        research_feature_adapter_only=False,
        research_feature_adapter_objective="inherited",
    )

    losses_tree = ast.parse(losses_source)
    auxiliary = next(
        node
        for node in losses_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "auxiliary_loss"
    )
    early_guard = next(
        (
            node
            for node in auxiliary.body
            if isinstance(node, ast.If)
            and any(isinstance(item, ast.Return) for item in ast.walk(node))
        ),
        None,
    )
    guard_text = ast.unparse(early_guard.test) if early_guard is not None else ""
    guard_has_both_flags = (
        "research_feature_adapter_only" in guard_text
        and "research_feature_adapter_objective" in guard_text
        and "target-only" in guard_text
    )
    guard_lambda = None
    if early_guard is not None:
        guard_lambda = ast.Expression(
            body=ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="args")],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=early_guard.test,
            )
        )
        ast.fix_missing_locations(guard_lambda)
    gate = (
        cast(Callable[[object], bool], eval(compile(guard_lambda, str(losses_path), "eval"), {}))
        if guard_lambda is not None
        else None
    )
    default_auxiliary_gate_false = (
        not gate(SimpleNamespace(research_feature_adapter_only=False)) if gate else False
    )
    inherited_adapter_gate_false = (
        not gate(
            SimpleNamespace(
                research_feature_adapter_only=True,
                research_feature_adapter_objective="inherited",
            )
        )
        if gate
        else False
    )
    target_only_auxiliary_gate_true = (
        gate(
            SimpleNamespace(
                research_feature_adapter_only=True,
                research_feature_adapter_objective="target-only",
            )
        )
        if gate
        else False
    )
    aux_declarations_follow_guard = bool(
        early_guard is not None
        and next(
            index for index, statement in enumerate(auxiliary.body) if statement is early_guard
        )
        < next(
            index
            for index, statement in enumerate(auxiliary.body)
            if isinstance(statement, ast.Assign)
            and any(isinstance(node, ast.Name) and node.id == "aux" for node in ast.walk(statement))
        )
    )

    patch_only_auxiliary_gate = (
        "@@ -119,6 +119,11 @@" in patch_source
        and "def main_loss" not in patch_source
        and "def auxiliary_loss" in patch_source
    )
    source_pins = integration.get("source_sha256", {})
    pins_match = (
        all(
            isinstance(source_pins.get(relative), str)
            and hashlib.sha256((root / relative).read_bytes()).hexdigest() == source_pins[relative]
            for relative in (
                "src/wnn/training/cli_features.py",
                "src/wnn/training/feature_adapter.py",
                "src/wnn/training/losses.py",
            )
        )
        and integration.get("independent_recheck") == "pending"
        and integration.get("fit_admitted") is False
    )
    passed = all(
        (
            all_teacher_modifiers_rejected,
            target_only_without_adapter_rejected,
            valid_target_only_accepted,
            inherited_default_accepted,
            guard_has_both_flags,
            default_auxiliary_gate_false,
            inherited_adapter_gate_false,
            target_only_auxiliary_gate_true,
            aux_declarations_follow_guard,
            patch_only_auxiliary_gate,
            pins_match,
        )
    )
    result = {
        "status": "pass" if passed else "fail",
        "blocked_option_exception_types": blocked_results,
        "valid_target_only_accepted": valid_target_only_accepted,
        "target_only_without_adapter_rejected": target_only_without_adapter_rejected,
        "inherited_default_accepted": inherited_default_accepted,
        "auxiliary_early_guard_has_both_flags": guard_has_both_flags,
        "default_auxiliary_gate_false": default_auxiliary_gate_false,
        "inherited_adapter_gate_false": inherited_adapter_gate_false,
        "target_only_auxiliary_gate_true": target_only_auxiliary_gate_true,
        "auxiliary_guard_precedes_loss_state_reads": aux_declarations_follow_guard,
        "v2_patch_changes_only_auxiliary_loss_gate": patch_only_auxiliary_gate,
        "integration_still_unadmitted": pins_match,
        "torch_imported": False,
        "payloads_read": False,
        "scope": "stdlib AST extraction and mocked objective arguments; no trainer/model execution",
    }
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
