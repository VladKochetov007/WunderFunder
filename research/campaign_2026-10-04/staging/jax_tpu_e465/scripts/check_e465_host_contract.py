"""Run JAX-free E465 fold, counter, sequence-domain and call-shape checks."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src"))

from wnn_jax.e465_host_contract import (  # noqa: E402
    BATCH_SIZE,
    CHUNK_ROWS,
    EVALUATION_GROUPS,
    FIT_VALID_GROUPS,
    SEQUENCE_ROWS,
    TRAIN_GROUPS,
    validate_epoch_batches,
    validate_fold_membership,
    validate_sequence_domain,
)


def _assert_call_arity(path: Path, expected: dict[str, tuple[int, int]]) -> None:
    """Statically ensure local helpers are called with their declared shape."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in expected:
            required = len(node.args.args) - len(node.args.defaults)
            required += sum(default is None for default in node.args.kw_defaults)
            total = len(node.args.args) + len(node.args.kwonlyargs)
            definitions[node.name] = (required, total)
    if definitions != expected:
        raise AssertionError(f"helper signature drift: {definitions!r} != {expected!r}")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in expected or not any(
            isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and parent.name != name
            for parent in _ancestors(tree, node)
        ):
            continue
        positional = len(node.args)
        keyword_names = {item.arg for item in node.keywords if item.arg is not None}
        low, high = expected[name]
        if positional > high or positional + len(keyword_names) < low:
            raise AssertionError(
                f"{name} call at line {node.lineno} supplies {positional} positional "
                f"and {len(keyword_names)} named values; expected {low}..{high}"
            )


def _ancestors(tree: ast.AST, target: ast.AST):
    """Find function parents without an external AST utility dependency."""
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    current = target
    while current in parents:
        current = parents[current]
        yield current


def main() -> None:
    """Execute the small deterministic host-only contract fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worker = STAGE_ROOT / "src/wnn_jax/e465_tpu_worker.py"
    _assert_call_arity(
        worker,
        {
            "_replicate": (3, 3),
            "_new_training_state": (4, 4),
            "_zero_recurrent_states": (2, 2),
            "_step_update": (7, 7),
        },
    )

    train = tuple(group for group in range(TRAIN_GROUPS + 1) if group != 528)
    valid_fit = tuple(range(FIT_VALID_GROUPS))
    evaluation = tuple(range(FIT_VALID_GROUPS, FIT_VALID_GROUPS + EVALUATION_GROUPS))
    checked = validate_fold_membership(train, valid_fit, evaluation)
    # Equal numeric row-group IDs across TRAIN and VALID are legal because
    # the parquet namespaces are distinct. Duplicates within one split and
    # fit/evaluation overlap inside VALID are invalid.
    try:
        validate_fold_membership(train, (*valid_fit, valid_fit[0]), evaluation)
    except ValueError:
        pass
    else:
        raise AssertionError("same-split duplicate fit-VALID group was accepted")
    overlapped_evaluation = (valid_fit[-1], *evaluation[1:])
    try:
        validate_fold_membership(train, valid_fit, overlapped_evaluation)
    except ValueError:
        pass
    else:
        raise AssertionError("VALID evaluation overlap was accepted")
    membership = tuple(("train", group) for group in checked[0]) + tuple(
        ("valid", group) for group in checked[1]
    )
    epoch_prefix = membership[:-4]
    batches = tuple(
        epoch_prefix[start : start + BATCH_SIZE]
        for start in range(0, len(epoch_prefix), BATCH_SIZE)
    )
    contract = validate_epoch_batches(batches, membership)
    assert (contract.fit_groups, contract.full_batches, contract.dropped_groups) == (
        11_940,
        373,
        4,
    )
    assert contract.chunks_per_batch == 20 and contract.attempts_per_arm == 7_460
    assert contract.scheduled_steps[0] == 0 and contract.scheduled_steps[-1] == 7_459
    assert contract.unused_post_last_schedule_index == 7_460

    validate_sequence_domain(
        sequence_rows=SEQUENCE_ROWS,
        chunk_rows=CHUNK_ROWS,
        official_first_scored_row=99,
        all_valid_groups=1_873,
        evaluation_groups=len(evaluation),
    )
    for bad in (
        {
            "sequence_rows": 19_999,
            "chunk_rows": 1_000,
            "official_first_scored_row": 99,
            "all_valid_groups": 1_873,
            "evaluation_groups": 468,
        },
        {
            "sequence_rows": 20_000,
            "chunk_rows": 1_000,
            "official_first_scored_row": 100,
            "all_valid_groups": 1_873,
            "evaluation_groups": 468,
        },
    ):
        try:
            validate_sequence_domain(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid sequence/evaluation domain was accepted")
    try:
        validate_fold_membership((*train, 528), valid_fit, evaluation)
    except ValueError:
        pass
    else:
        raise AssertionError("forbidden TRAIN group 528 was accepted")
    try:
        validate_epoch_batches((*batches[:-1], batches[-1][:-1]), membership)
    except ValueError:
        pass
    else:
        raise AssertionError("short final batch was accepted")

    report = {
        "format": "wnn.e465-host-contract",
        "status": "passed",
        "checks": 9,
        "fit_groups": contract.fit_groups,
        "full_batches": contract.full_batches,
        "dropped_groups": contract.dropped_groups,
        "chunks_per_batch": contract.chunks_per_batch,
        "attempts_per_arm": contract.attempts_per_arm,
        "last_used_scheduler_index": contract.scheduled_steps[-1],
        "unused_post_last_scheduler_index": contract.unused_post_last_schedule_index,
        "jax_imported": False,
        "numpy_imported": False,
        "torch_imported": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
