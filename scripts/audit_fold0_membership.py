#!/usr/bin/env python3
"""Independently verify fold-0 fit groups from frozen manifests and run provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# JSON artifact schemas are source-defined and differ across campaign receipts.
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def option_map(argv: list[str]) -> tuple[dict[str, str | bool], set[str]]:
    values: dict[str, str | bool] = {}
    flags: set[str] = set()
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            values[token] = argv[index + 1]
            index += 2
        else:
            values[token] = True
            flags.add(token)
            index += 1
    return values, flags


def inspect_spec(path: Path, split: dict[str, Any], split_hash: str) -> dict[str, Any]:
    packet = json.loads(path.read_text())
    fit_step = next(step for step in packet["steps"] if step["name"] == "fit")
    argv = fit_step["argv"]
    values, flags = option_map(argv)
    expected_manifest = Path(values["--split-manifest"])
    if not expected_manifest.is_absolute():
        repository = next(
            (parent for parent in path.parents if (parent / "AGENTS.md").is_file()), None
        )
        if repository is None:
            raise ValueError(f"cannot resolve repository root for {path}")
        expected_manifest = repository / expected_manifest
    observed_hash = sha256(expected_manifest)
    checks = {
        "uses_valid_as_fit_source": values.get("--train-source") == "valid",
        "full_valid_flag_present": "--full-valid" in flags,
        "eval_loader_disabled": "--no-eval-during-fit" in flags,
        "fixed_epoch_zero_checkpoint": (
            values.get("--checkpoint-policy") == "fixed" and values.get("--checkpoint-epoch") == "0"
        ),
        "does_not_include_heldout_valid": "--include-heldout-valid" not in values,
        "split_manifest_hash_matches": observed_hash == split_hash,
        "manifest_fit_valid_count": len(split["fit"]["valid"]),
        "manifest_evaluation_valid_count": len(split["evaluation"]["valid"]),
    }
    if not all(
        value is True
        for key, value in checks.items()
        if key.startswith("uses_")
        or key
        in {
            "full_valid_flag_present",
            "eval_loader_disabled",
            "fixed_epoch_zero_checkpoint",
            "does_not_include_heldout_valid",
            "split_manifest_hash_matches",
        }
    ):
        raise ValueError(f"fold-0 fit spec does not meet membership contract: {checks}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "run_id": packet["run_id"],
        "fit_argv": argv,
        "checks": checks,
        "interpretation": (
            "Campaign split binding resolves valid_fold to fold 0; train-source=valid then uses "
            "all valid groups minus evaluation_groups. --full-valid sets evaluation length only; "
            "--no-eval-during-fit prevents constructing that evaluation loader."
        ),
    }


def inspect_run(root: Path, run: str, split: dict[str, Any], split_hash: str) -> dict[str, Any]:
    provenance_path = root / "artifacts/runs" / run / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"requested run provenance is missing: {provenance_path}")
    provenance = json.loads(provenance_path.read_text())
    args = provenance.get("args", {})
    recorded_hash = provenance.get("split_manifest_sha256")
    if recorded_hash != split_hash or args.get("split_manifest_sha256") != split_hash:
        raise ValueError(f"{run}: split manifest hash mismatch")
    if args.get("valid_fold") != int(split["fold"]):
        raise ValueError(f"{run}: valid_fold does not match the manifest")

    expected_train = set(map(int, split["fit"]["train"]))
    expected_valid = set(map(int, split["fit"]["valid"]))
    eval_valid = set(map(int, split["evaluation"]["valid"]))
    fit = provenance.get("fit_row_groups")
    if not isinstance(fit, dict):
        raise ValueError(f"{run}: missing fit_row_groups")
    fitted_train = set(map(int, fit.get("train", [])))
    fitted_valid = set(map(int, fit.get("valid", [])))

    source = args.get("train_source")
    if source == "valid":
        if fitted_train or fitted_valid != expected_valid:
            raise ValueError(f"{run}: valid-source groups do not equal manifest fit-valid groups")
    elif source == "train+valid":
        if fitted_train != expected_train or fitted_valid != expected_valid:
            raise ValueError(f"{run}: train+valid groups do not equal manifest fit groups")
    elif source == "train":
        if fitted_train != expected_train or fitted_valid:
            raise ValueError(f"{run}: train groups do not equal manifest fit-train groups")
    else:
        raise ValueError(f"{run}: unexpected train source {source!r}")

    component_of = {
        (source_split, int(node["row_group"])): int(node["component"])
        for source_split, nodes in split["sequences"].items()
        for node in nodes
    }
    evaluation_components = {component_of[("valid", group)] for group in eval_valid}
    fit_components = {component_of[("train", group)] for group in fitted_train} | {
        component_of[("valid", group)] for group in fitted_valid
    }
    overlap = fit_components & evaluation_components
    if overlap:
        raise ValueError(f"{run}: fitted/evaluation components overlap: {sorted(overlap)[:8]}")

    head_ft_checks: dict[str, bool] | None = None
    if args.get("train_source") == "valid":
        head_ft_checks = {
            "full_valid_flag": args.get("full_valid") is True,
            "heldout_valid_not_included": args.get("include_heldout_valid") is False,
            "evaluation_loader_disabled": args.get("no_eval_during_fit") is True,
            "evaluation_groups_recorded_but_not_fit": set(
                map(int, provenance.get("evaluation_row_groups", []))
            )
            == eval_valid,
        }
        if not all(head_ft_checks.values()):
            raise ValueError(f"{run}: full-valid/evaluation provenance disagrees: {head_ft_checks}")

    return {
        "run": run,
        "status": "passed",
        "provenance_path": str(provenance_path),
        "provenance_sha256": sha256(provenance_path),
        "split_manifest_sha256": recorded_hash,
        "train_source": source,
        "resolved_valid_fold": args.get("valid_fold"),
        "full_valid": args.get("full_valid"),
        "include_heldout_valid": args.get("include_heldout_valid"),
        "no_eval_during_fit": args.get("no_eval_during_fit"),
        "fit_train_count": len(fitted_train),
        "fit_valid_count": len(fitted_valid),
        "evaluation_count": len(eval_valid),
        "recorded_evaluation_group_count": len(provenance.get("evaluation_row_groups", [])),
        "fit_evaluation_component_overlap": 0,
        "head_ft_checks": head_ft_checks,
        "fit_group_set_matches_manifest": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--spec", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    split_path = args.split.resolve()
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    split_hash = hashlib.sha256(split_bytes).hexdigest()
    sys.path.insert(0, str(root / "src"))
    from wnn.splits import load_campaign_split

    validated_split = load_campaign_split(split_path, root)
    if validated_split.sha256 != split_hash:
        raise ValueError("runtime split loader returned a different manifest identity")
    if set(validated_split.valid_groups) != set(map(int, split["fit"]["valid"])) or set(
        validated_split.evaluation_groups
    ) != set(map(int, split["evaluation"]["valid"])):
        raise ValueError("runtime split loader membership differs from the frozen JSON manifest")
    reports = [inspect_spec(path.resolve(), split, split_hash) for path in args.spec]
    reports.extend(inspect_run(root, run, split, split_hash) for run in args.run)
    result = {
        "schema": "wnn.independent-fold0-membership-audit.v1",
        "status": "passed_static_and_available_run_membership_checks",
        "split": str(split_path),
        "split_sha256": split_hash,
        "fold": split["fold"],
        "component_method": split["component_method"],
        "counts": split["counts"],
        "manifest_runtime_validation": {
            "passed": True,
            "loader": "wnn.splits.load_campaign_split",
            "actual_feature_graph_and_cached_fold_checked": True,
            "evaluation_components_disjoint_from_fit_membership": True,
        },
        "source_bindings": {
            "train_rnn.py": sha256(root / "scripts/train_rnn.py"),
            "splits.py": sha256(root / "src/wnn/splits.py"),
        },
        "specs": reports[: len(args.spec)],
        "runs": reports[len(args.spec) :],
        "limitations": [
            "A pending spec has no fitted membership until its terminal provenance is available.",
            "This checks direct fit membership and graph components; it does not make the exposed fold a fresh holdout.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "runs": len(args.run), "specs": len(args.spec)}))


if __name__ == "__main__":
    main()
