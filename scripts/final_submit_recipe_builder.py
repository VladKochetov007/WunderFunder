"""Build an explicit all-labeled-data final-fit plan from the frozen TL3 graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.tl3_chronological_recipes import FINAL_WEIGHTS, W18_COEFFICIENTS, build_nodes
from wnn.final_submit_manifest import (
    FINAL_FIT_POLICY,
    FINAL_FIT_SCHEMA,
    TOTAL_SEQUENCES,
    TRAIN_SEQUENCES,
    VALID_SEQUENCES,
    build_final_fit_membership,
    validate_final_fit_membership,
)
from wnn.splits import file_sha256

PLAN_SCHEMA = "wnn.final-submit-recipes.v1"
FINAL_RUN_PREFIX = "tl3final_"
SEQUENCE_ROWS = 20_000

__all__ = [
    "FINAL_FIT_POLICY",
    "FINAL_FIT_SCHEMA",
    "TOTAL_SEQUENCES",
    "build_final_fit_membership",
    "materialize_plan",
    "validate_final_fit_membership",
]


def _replace_option(arguments: list[str], option: str, value: str) -> None:
    """Replace one existing option's value and reject malformed frozen argv."""
    indices = [index for index, token in enumerate(arguments) if token == option]
    if len(indices) != 1 or indices[0] + 1 >= len(arguments):
        raise ValueError(f"frozen TL3 argument must contain exactly one {option}")
    arguments[indices[0] + 1] = value


def _remove_option(arguments: list[str], option: str, *, takes_value: bool) -> None:
    """Remove one frozen option, rejecting duplicates and dangling values."""
    indices = [index for index, token in enumerate(arguments) if token == option]
    if len(indices) > 1:
        raise ValueError(f"frozen TL3 argument repeats {option}")
    if not indices:
        return
    index = indices[0]
    del arguments[index : index + (2 if takes_value else 1)]


def _final_node(
    node: dict[str, object],
    final_manifest: Path,
    source_validation_receipt: Path,
    run_root: Path,
    run_names: dict[str, str],
    source_validation_sha256: str,
) -> dict[str, object]:
    """Adapt a train-only node into non-overwriting all-labeled-data argv."""
    args = list(node["args"])
    for old, new in run_names.items():
        args = [argument.replace(old, new) for argument in args]
    _replace_option(args, "--name", run_names[str(node["run"])])
    _replace_option(args, "--train-source", "train+valid")
    _replace_option(args, "--train-seqs", str(TRAIN_SEQUENCES))
    _remove_option(args, "--eval-source", takes_value=True)
    _remove_option(args, "--full-valid", takes_value=False)
    _remove_option(args, "--split-manifest", takes_value=True)
    _remove_option(args, "--train-eval-seqs", takes_value=True)
    if "--distill-cache-root" in args:
        _replace_option(args, "--distill-cache-root", str(run_root / "final-fit-w18-mixed-targets"))
    args.extend(
        [
            "--include-heldout-valid",
            "--no-eval-during-fit",
            "--checkpoint-epoch",
            str(node["endpoint"]),
            "--final-fit-manifest",
            str(final_manifest),
            "--final-fit-source-validation-sha256",
            source_validation_sha256,
            "--final-fit-source-validation-receipt",
            str(source_validation_receipt),
        ]
    )
    batch = int(args[args.index("--batch") + 1])
    chunk = int(args[args.index("--chunk") + 1])
    epochs = int(node["epochs"])
    updates = (TOTAL_SEQUENCES // batch) * (SEQUENCE_ROWS // chunk) * epochs
    return {
        **node,
        "run": run_names[str(node["run"])],
        "args": args,
        "expected_updates": updates,
        "expected_sequences_per_epoch": TOTAL_SEQUENCES,
        "evaluation": "disabled; fixed declared endpoint",
        "neutral_propensity": True,
        "source_membership_schema": FINAL_FIT_SCHEMA,
        "distillation_cache_root": str(run_root / "final-fit-w18-mixed-targets")
        if node["id"] == "w18final_distilled_s5"
        else None,
    }


def materialize_plan(
    final_manifest: Path,
    run_root: Path,
    train_sha256: str,
    valid_sha256: str,
    source_validation_sha256: str,
) -> dict[str, object]:
    """Return all 17 frozen TL3 fits with bound data identity and fixed ancestry."""
    final_manifest = final_manifest.resolve()
    run_root = run_root.resolve()
    source_validation_receipt = final_manifest.parent / "SOURCE_VALIDATION_R1.json"
    if source_validation_receipt.is_symlink() or not source_validation_receipt.is_file():
        raise ValueError(
            "the pinned SOURCE_VALIDATION_R1.json receipt must sit beside the manifest"
        )
    if file_sha256(source_validation_receipt) != source_validation_sha256:
        raise ValueError("source-validation receipt bytes differ from the requested plan identity")
    membership = build_final_fit_membership(train_sha256, valid_sha256, source_validation_sha256)
    train_only_nodes = build_nodes(final_manifest, run_root)
    run_names = {str(node["run"]): f"{FINAL_RUN_PREFIX}{node['id']}" for node in train_only_nodes}
    nodes = [
        _final_node(
            node,
            final_manifest,
            source_validation_receipt,
            run_root,
            run_names,
            source_validation_sha256,
        )
        for node in train_only_nodes
    ]
    if len(nodes) != 17 or len({node["run"] for node in nodes}) != 17:
        raise ValueError("final-fit TL3 plan must contain 17 uniquely named fits")
    if sum(int(node["expected_updates"]) for node in nodes) != 312_000:
        raise ValueError("all-data TL3 optimizer-update inventory changed")
    remapped_weights = {run_names[f"tl3c_{name}"]: weight for name, weight in FINAL_WEIGHTS.items()}
    if list(W18_COEFFICIENTS) != list(nodes[-1]["parents"]):
        raise ValueError("W18 ancestor coefficient order differs from the frozen graph")
    plan: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "policy": FINAL_FIT_POLICY,
        "status": "trainer_bridge_ready",
        "fit_started": False,
        "membership": membership,
        "fit_sequences": TOTAL_SEQUENCES,
        "fitting_train_sequences": TRAIN_SEQUENCES,
        "fitting_valid_sequences": VALID_SEQUENCES,
        "independent_evaluation_sequences": 0,
        "independent_score_claim": "forbidden",
        "source_validation_receipt_sha256": source_validation_sha256,
        "source_validation_receipt": {
            "path": str(source_validation_receipt),
            "sha256": source_validation_sha256,
        },
        "train_diagnostics": "optional in-sample diagnostics only; never an independent validation score",
        "final_members": remapped_weights,
        "configured_epochs": sum(int(node["epochs"]) for node in nodes),
        "expected_total_updates": sum(int(node["expected_updates"]) for node in nodes),
        "nodes": nodes,
        "propensity_policy": "neutral; no fitted propensity model, propensity input, or mask weighting",
        "feature_ancestry": "all learned normalization, teacher, initializer, and cache ancestry must bind the same final-fit membership identity",
        "trainer_bridge": "separate final-fit schema branch; chronological-primary guard remains unchanged",
        "optional_source4_lstm": {
            "included": False,
            "recipe": "native LSTM hidden=192, layers=1, fixed 8 epochs",
            "reason": "no Source4 entrypoint/parser/checkpoint compatibility receipt is present in this worktree",
        },
    }
    plan_identity = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    plan["plan_sha256"] = hashlib.sha256(plan_identity).hexdigest()
    return plan


def main() -> None:
    """Write a final-fit recipe plan; this command never starts training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-fit-manifest",
        type=Path,
        required=True,
        help="new manifest path beside the pinned SOURCE_VALIDATION_R1.json receipt",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--valid-sha256", required=True)
    parser.add_argument("--source-validation-sha256", required=True)
    args = parser.parse_args()
    membership = build_final_fit_membership(
        args.train_sha256, args.valid_sha256, args.source_validation_sha256
    )
    with args.final_fit_manifest.open("x", encoding="utf-8") as stream:
        json.dump(membership, stream, indent=2)
        stream.write("\n")
    plan = materialize_plan(
        args.final_fit_manifest,
        args.run_root,
        args.train_sha256,
        args.valid_sha256,
        args.source_validation_sha256,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
