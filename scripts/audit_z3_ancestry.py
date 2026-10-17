#!/usr/bin/env python3
"""Independently bind the fold-0 Z3 recipe to clean source ancestry and targets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

# JSON artifact schemas are source-defined and differ across campaign receipts.
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    """Hash a file with bounded memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def row_groups_sha256(groups: list[int]) -> str:
    """Hash ordered row-group identities using little-endian int64 bytes."""
    return hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject unexpected top-level values."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def parse_fit_args(spec_path: Path) -> dict[str, Any]:
    """Run a spec's trainer parser through parse_args without entering training."""
    packet = read_json(spec_path)
    fit_step = next(step for step in packet["steps"] if step["name"] == "fit")
    argv = fit_step["argv"]
    source_index = next(
        index for index, value in enumerate(argv) if value.endswith("/scripts/train_rnn.py")
    )
    source = Path(argv[source_index]).resolve()
    sys.path.insert(0, str(source.parent.parent / "src"))
    for name in list(sys.modules):
        if name == "wnn" or name.startswith("wnn."):
            del sys.modules[name]
    module_spec = importlib.util.spec_from_file_location("z3_audited_trainer", source)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"could not load the trainer source {source}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not hasattr(module, "build_parser"):
        tree = ast.parse(source.read_text())
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        parse_args_index = next(
            index
            for index, node in enumerate(function.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets)
        )
        function.body = function.body[: parse_args_index + 1] + [
            ast.Return(ast.Name("args", ast.Load()))
        ]
        parser_module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(parser_module, str(source), "exec"), module.__dict__)
    previous_argv = sys.argv
    try:
        sys.argv = [str(source)] + argv[source_index + 1 :]
        parsed = vars(
            module.build_parser().parse_args() if hasattr(module, "build_parser") else module.main()
        )
    finally:
        sys.argv = previous_argv
    if parsed.get("fast"):
        for key in ("sync_free", "fused_adam", "pin_memory", "prop_cache"):
            parsed[key] = True
    return parsed


def normalized(value: Any) -> Any:
    """Convert argparse paths and nested values to stable JSON-compatible forms."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value


def compare_recipe_args(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    allowed_differences: set[str],
    label: str,
) -> dict[str, Any]:
    """Require every parsed training option except named membership/source changes."""
    differences: dict[str, dict[str, Any]] = {}
    for key, candidate_value in candidate.items():
        if key not in reference:
            raise ValueError(f"{label}: parsed option {key} is absent from its reference")
        reference_value = normalized(reference[key])
        candidate_value = normalized(candidate_value)
        if reference_value != candidate_value:
            if key not in allowed_differences:
                differences[key] = {
                    "reference": reference_value,
                    "candidate": candidate_value,
                }
            else:
                differences[key] = {
                    "reference": reference_value,
                    "candidate": candidate_value,
                }
    unexpected = set(differences) - allowed_differences
    if unexpected:
        raise ValueError(f"{label}: undeclared recipe differences: {sorted(unexpected)}")
    return differences


def find_packet(root: Path, run_id: str) -> Path:
    """Locate a queued packet by its immutable run identifier."""
    queue_root = (
        root / "research/campaign_2026-10-06/bandit_continuation/runtime/warm_queue_local_v1"
    )
    matches = []
    for path in queue_root.glob("*/*.json"):
        if read_json(path).get("run_id") == run_id:
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one queued packet for {run_id}, got {matches}")
    return matches[0]


def check_run_membership(
    root: Path,
    run: str,
    checkpoint_file: str,
    expected_checkpoint_sha256: str,
    split: Any,
    *,
    expected_fit: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Check a learned source's checkpoint bytes and component-safe membership."""
    run_root = root / "artifacts/runs" / run
    provenance_path = run_root / "provenance.json"
    checkpoint_path = run_root / checkpoint_file
    provenance = read_json(provenance_path)
    checkpoint_sha256 = sha256(checkpoint_path)
    args = provenance.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"{run}: provenance has no argument mapping")
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(f"{run}: checkpoint bytes differ from expected ancestry hash")
    checkpoint_records = provenance.get("checkpoint_sha256", {})
    if checkpoint_records.get(checkpoint_file) != checkpoint_sha256:
        raise ValueError(f"{run}: provenance checkpoint hash does not match file bytes")
    if provenance.get("split_manifest_sha256") != split.sha256:
        raise ValueError(f"{run}: source used a different split manifest")
    if args.get("split_manifest_sha256") != split.sha256:
        raise ValueError(f"{run}: recorded arguments used a different split manifest")

    memberships = provenance.get("fit_row_groups")
    if not isinstance(memberships, dict):
        raise ValueError(f"{run}: missing fitting row-group provenance")
    fitted = {name: list(map(int, memberships.get(name, []))) for name in ("train", "valid")}
    if expected_fit is not None and any(
        set(fitted[name]) != set(expected_fit[name]) for name in ("train", "valid")
    ):
        raise ValueError(f"{run}: fit membership differs from the expected parent set")
    split.assert_fit_groups(fitted["train"], fitted["valid"])
    fit_counts = {name: len(fitted[name]) for name in ("train", "valid")}
    return {
        "run": run,
        "checkpoint_file": checkpoint_file,
        "checkpoint_sha256": checkpoint_sha256,
        "provenance_sha256": sha256(provenance_path),
        "split_manifest_sha256": split.sha256,
        "fit_group_counts": fit_counts,
        "fit_row_group_hashes": {
            name: row_groups_sha256(fitted[name]) for name in ("train", "valid")
        },
        "fit_components_exclude_evaluation": True,
        "args": args,
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--recipe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    split_path = args.split.resolve()
    preflight = read_json(args.preflight.resolve())
    cache_root = args.cache.resolve(strict=True)
    cache_metadata_path = cache_root / "metadata.json"
    cache_metadata = read_json(cache_metadata_path)
    parser_source_root = root / "src"
    import sys

    sys.path.insert(0, str(parser_source_root))
    from wnn.splits import load_campaign_split

    split = load_campaign_split(split_path, root)
    if split.fold != 0:
        raise ValueError("expected the frozen fold-0 split")
    if sha256(split_path) != split.sha256:
        raise ValueError("split bytes changed during the audit")
    if sha256(cache_metadata_path) != preflight.get("cache_metadata_sha256"):
        raise ValueError("cache metadata bytes differ from parser preflight")
    if cache_metadata.get("split_manifest_sha256") != split.sha256:
        raise ValueError("teacher cache uses a different split manifest")
    if cache_metadata.get("fold") != split.fold:
        raise ValueError("teacher cache uses a different fold")
    expected_fit = {
        "train": list(split.train_groups),
        "valid": list(split.valid_groups),
    }
    expected_fit_hash = {name: row_groups_sha256(groups) for name, groups in expected_fit.items()}
    if cache_metadata.get("fit_groups") != expected_fit:
        raise ValueError("teacher cache fit groups differ from the frozen split")
    if cache_metadata.get("evaluation_valid_groups") != list(split.evaluation_groups):
        raise ValueError("teacher cache evaluation groups differ from the frozen split")

    parent_sha256 = str(preflight["parent_checkpoint_sha256"])
    parent = check_run_membership(
        root,
        "c20261003_combo_s4_f0",
        "model_ep3.pt",
        parent_sha256,
        split,
        expected_fit=expected_fit,
    )
    if parent["provenance_sha256"] != preflight.get("parent_provenance_sha256"):
        raise ValueError("matched parent provenance differs from parser preflight")
    if parent["args"].get("seed") != 4 or parent["args"].get("train_source") != "train+valid":
        raise ValueError("matched parent is not the declared seed-4 train+valid source")

    propensity = check_run_membership(
        root,
        "c20260929_prop_f0",
        "model.pt",
        str(parent["provenance"].get("propensity_checkpoint_sha256")),
        split,
        expected_fit={"train": [], "valid": list(split.valid_groups)},
    )
    prop_args = propensity["args"]
    if prop_args.get("heldout_fold") != 0 or prop_args.get("dev_all") is True:
        raise ValueError("propensity fit was not isolated from the fold-0 evaluation groups")
    if prop_args.get("no_eval_during_fit") is not True:
        raise ValueError("propensity fitting did not suppress evaluation during fit")
    if set(map(int, propensity["provenance"].get("evaluation_row_groups", []))) != set(
        split.evaluation_groups
    ):
        raise ValueError("propensity provenance does not record the fold-0 evaluation membership")

    recipe_dir = args.recipe_dir.resolve()
    student_spec = recipe_dir / "f0_z3_combo_s4_student_v2.json"
    student_hft_spec = recipe_dir / "f0_z3_combo_s4_student_hft_v2.json"
    parent_hft_spec = recipe_dir / "f0_z3_combo_s4_parent_hft_v2.json"
    preflight_specs = {item["spec"]: item for item in preflight.get("specs", [])}
    for spec_path in (student_spec, student_hft_spec, parent_hft_spec):
        record = preflight_specs.get(spec_path.name)
        if record is None or record.get("spec_sha256") != sha256(spec_path):
            raise ValueError(f"{spec_path.name}: current bytes differ from R2 parser preflight")

    student_reference = read_json(
        root / "artifacts/runs/z3_combo_s4_parent_student_v1/provenance.json"
    )["args"]
    student_candidate_args = parse_fit_args(student_spec)
    student_allowed = {
        "name",
        "valid_fold",
        "include_heldout_valid",
        "init",
        "split_manifest",
        "split_manifest_sha256",
        "distill_cache_root",
        "distill_cache_all_data",
    }
    student_differences = compare_recipe_args(
        student_reference,
        student_candidate_args,
        student_allowed,
        "Z3 student",
    )
    if student_candidate_args["seed"] != 4:
        raise ValueError("fold-0 Z3 student backbone must keep seed 4")
    if student_candidate_args["init"] != "c20261003_combo_s4_f0@ep3":
        raise ValueError("fold-0 Z3 student must start from the matched seed-4 parent")
    if (
        student_candidate_args["train_source"] != "train+valid"
        or student_candidate_args["valid_fold"] != 0
        or student_candidate_args["include_heldout_valid"]
        or student_candidate_args["split_manifest"] != split_path
        or Path(student_candidate_args["distill_cache_root"]).resolve() != cache_root
        or student_candidate_args["distill_cache_variant"] != "source"
        or not student_candidate_args["no_eval_during_fit"]
    ):
        raise ValueError("fold-0 Z3 student membership/cache flags differ from the audited sources")
    if student_candidate_args["distill_family_weight"] != 0.1:
        raise ValueError("fold-0 Z3 student family-head weight differs from exact Z3")

    all_data_hft_path = find_packet(root, "z3h_combo_s4_parent_student_hft_alldata_v1")
    all_data_hft_args = parse_fit_args(all_data_hft_path)
    hft_allowed = {"name", "init", "valid_fold", "split_manifest", "split_manifest_sha256"}
    student_hft_args = parse_fit_args(student_hft_spec)
    parent_hft_args = parse_fit_args(parent_hft_spec)
    student_hft_differences = compare_recipe_args(
        all_data_hft_args, student_hft_args, hft_allowed, "Z3 student HFT"
    )
    parent_hft_differences = compare_recipe_args(
        all_data_hft_args, parent_hft_args, hft_allowed, "Z3 parent HFT control"
    )
    hft_same = compare_recipe_args(
        parent_hft_args,
        student_hft_args,
        {"name", "init"},
        "matched HFT endpoints",
    )
    for label, fitted_args in (
        ("student HFT", student_hft_args),
        ("parent HFT", parent_hft_args),
    ):
        if (
            fitted_args["seed"] != 3
            or fitted_args["train_source"] != "valid"
            or not fitted_args["full_valid"]
            or not fitted_args["mask_weight"]
            or not fitted_args["no_eval_during_fit"]
            or fitted_args["valid_fold"] != 0
            or fitted_args["split_manifest"] != split_path
            or fitted_args["epochs"] != 1
            or fitted_args["checkpoint_epoch"] != 0
        ):
            raise ValueError(f"{label}: flags do not match fold-0 Z3h finish contract")

    for spec_path, required_steps in (
        (student_hft_spec, {"make_core", "fit", "predict", "predict_core"}),
        (parent_hft_spec, {"fit", "predict"}),
    ):
        packet = read_json(spec_path)
        present_steps = {step["name"] for step in packet["steps"]}
        if not required_steps.issubset(present_steps):
            raise ValueError(f"{spec_path.name}: prediction/core steps are incomplete")
        expected_outputs = set(packet.get("expected_outputs", []))
        needed_outputs = {
            str(root / "artifacts/runs" / packet["run_id"] / "model_ep0.pt"),
            str(root / "artifacts/runs" / packet["run_id"] / "provenance.json"),
            str(root / "cache" / f"pred_{packet['run_id']}_valid.npy"),
            str(root / "cache" / f"pred_{packet['run_id']}_valid.json"),
        }
        if spec_path == student_hft_spec:
            core_run = "f0_z3_combo_s4_student_v2_core"
            needed_outputs.update(
                {
                    str(root / "artifacts/runs" / core_run / "model_ep1.pt"),
                    str(root / "artifacts/runs" / core_run / "provenance.json"),
                    str(root / "cache" / f"pred_{core_run}_valid.npy"),
                    str(root / "cache" / f"pred_{core_run}_valid.json"),
                }
            )
        if not needed_outputs.issubset(expected_outputs):
            missing = sorted(needed_outputs - expected_outputs)
            raise ValueError(f"{spec_path.name}: expected output closure omits {missing}")

    source_records = cache_metadata.get("sources")
    coefficient_records = cache_metadata.get("source_coefficients")
    if not isinstance(source_records, list) or not isinstance(coefficient_records, list):
        raise ValueError("teacher cache source records are missing")
    coefficients = {item["run"]: item for item in coefficient_records}
    if len(coefficients) != len(coefficient_records):
        raise ValueError("teacher cache repeats source run names")
    if abs(float(cache_metadata.get("coefficient_sum", 0)) - 0.8125) > 1e-7:
        raise ValueError("teacher cache source-scale coefficient sum is not 0.8125")

    teacher_runs: dict[str, dict[str, Any]] = {}
    for record in source_records:
        run = str(record["run"])
        coefficient = coefficients.get(run)
        if coefficient is None:
            raise ValueError(f"teacher source {run} is absent from coefficient records")
        checkpoint_file = f"model_ep{int(record['epoch'])}.pt"
        audited = check_run_membership(
            root,
            run,
            checkpoint_file,
            str(record["checkpoint_sha256"]),
            split,
        )
        if audited["fit_group_counts"] != record.get("fit_group_counts"):
            raise ValueError(f"{run}: source fit counts differ from cache metadata")
        if audited["fit_row_group_hashes"] != record.get("fit_row_group_hashes"):
            raise ValueError(f"{run}: source fit groups differ from cache metadata")
        if record.get("family") != coefficient.get("family"):
            raise ValueError(f"{run}: teacher family differs from cache source declaration")

        closure_records = record.get("ancestor_closure")
        if not isinstance(closure_records, list) or not closure_records:
            raise ValueError(f"{run}: ancestor closure is missing")
        closure: dict[str, dict[str, Any]] = {}
        for ancestor in closure_records:
            ancestor_run = str(ancestor["run"])
            ancestor_file = str(ancestor["checkpoint_file"])
            checked = check_run_membership(
                root,
                ancestor_run,
                ancestor_file,
                str(ancestor["checkpoint_sha256"]),
                split,
            )
            if checked["fit_group_counts"] != ancestor.get("fit_group_counts"):
                raise ValueError(f"{ancestor_run}: ancestor fit counts differ from cache")
            if checked["fit_row_group_hashes"] != ancestor.get("fit_row_group_hashes"):
                raise ValueError(f"{ancestor_run}: ancestor fit hashes differ from cache")
            closure[ancestor_run] = checked
        if run not in closure:
            raise ValueError(f"{run}: own checkpoint is absent from its ancestor closure")
        for ancestor_run, checked in closure.items():
            source_args = checked["args"]
            init_spec = str(source_args.get("init") or "")
            if init_spec:
                parent_run, marker, epoch = init_spec.partition("@ep")
                parent_file = f"model_ep{int(epoch)}.pt" if marker else "model.pt"
                if (
                    parent_run not in closure
                    or closure[parent_run]["checkpoint_file"] != parent_file
                ):
                    raise ValueError(
                        f"{ancestor_run}: initializer is missing from ancestor closure"
                    )
            propensity_run = str(source_args.get("propensity") or "")
            if propensity_run and (
                propensity_run not in closure
                or closure[propensity_run]["checkpoint_file"] != "model.pt"
            ):
                raise ValueError(
                    f"{ancestor_run}: propensity source is missing from ancestor closure"
                )
            for token in filter(None, str(source_args.get("distill") or "").split(",")):
                teacher_run = token.partition("*")[0]
                if teacher_run not in coefficients:
                    raise ValueError(
                        f"{ancestor_run}: distillation source {teacher_run} is not in this cache"
                    )
        teacher_runs[run] = {
            "family": record["family"],
            "epoch": record["epoch"],
            "coefficient": coefficient["coefficient"],
            "checkpoint_sha256": audited["checkpoint_sha256"],
            "fit_group_counts": audited["fit_group_counts"],
            "fit_row_group_hashes": audited["fit_row_group_hashes"],
            "ancestor_runs": sorted(closure),
            "provenance_sha256": audited["provenance_sha256"],
        }

    # The normal validator checks exact row order, shape, dtype, finite values and bytes.
    from wnn.distillation_cache import validate_distillation_cache

    checked_cache = validate_distillation_cache(
        cache_root, root, split, "source", require_families=True
    )
    if checked_cache["metadata_sha256"] != preflight.get("cache_metadata_sha256"):
        raise ValueError("cache validator returned a different metadata identity")

    output = {
        "schema": "wnn.independent-z3-fold0-ancestry-audit.v1",
        "status": "passed",
        "split_sha256": split.sha256,
        "fit_group_counts": {name: len(groups) for name, groups in expected_fit.items()},
        "fit_row_group_hashes": expected_fit_hash,
        "evaluation_valid_group_count": len(split.evaluation_groups),
        "evaluation_component_count": len(
            {
                component
                for source, group, component in split.components
                if source == "valid" and group in set(split.evaluation_groups)
            }
        ),
        "parent": {
            "run": parent["run"],
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "provenance_sha256": parent["provenance_sha256"],
            "fit_group_counts": parent["fit_group_counts"],
            "propensity_checkpoint_sha256": propensity["checkpoint_sha256"],
        },
        "propensity": {
            "run": propensity["run"],
            "checkpoint_sha256": propensity["checkpoint_sha256"],
            "fit_group_counts": propensity["fit_group_counts"],
            "evaluation_row_groups_count": len(
                propensity["provenance"].get("evaluation_row_groups", [])
            ),
            "heldout_evaluation_used_for_fit": False,
        },
        "recipe_equivalence": {
            "status": "passed_all_parsed_flags_except_declared_membership_source_changes",
            "student_spec_sha256": sha256(student_spec),
            "student_allowed_changes_from_all_data_z3": student_differences,
            "all_data_hft_packet": str(all_data_hft_path),
            "student_hft_allowed_changes_from_all_data_z3h": student_hft_differences,
            "parent_hft_allowed_changes_from_all_data_z3h": parent_hft_differences,
            "parent_vs_student_hft_allowed_changes": hft_same,
            "finishing_seed": 3,
            "backbone_student_seed": 4,
            "prediction_and_provenance_steps_bound": True,
            "descriptive_note_ambiguity": (
                "R2 HFT notes say matched seed4; parsed argv and hypothesis bind finishing seed3."
            ),
        },
        "cache": {
            "root": str(cache_root),
            "metadata_sha256": checked_cache["metadata_sha256"],
            "split_sha256": cache_metadata["split_manifest_sha256"],
            "fit_group_counts": {
                name: len(groups) for name, groups in cache_metadata["fit_groups"].items()
            },
            "evaluation_valid_groups_count": len(cache_metadata["evaluation_valid_groups"]),
            "teacher_source_fit_excludes_evaluation_components": True,
            "teacher_sources": teacher_runs,
            "verified_arrays": checked_cache["verified_files"],
        },
        "hft_fit_semantics": {
            "valid_fold": 0,
            "train_source": "valid",
            "full_valid": True,
            "mask_weight": True,
            "no_eval_during_fit": True,
            "fit_groups": {"train": 0, "valid": len(split.valid_groups)},
            "evaluation_groups": len(split.evaluation_groups),
            "source_trace": "valid_pool = setdiff(all_valid, fold_groups); fold_groups = evaluation_groups",
            "conclusion": "held-out evaluation components are excluded from fit",
        },
        "limitations": [
            "Fold 0 is exposed development evidence, not a fresh holdout.",
            "This receipt audits source/cache ancestry and membership, not model quality.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "teacher_sources": len(teacher_runs)}))


if __name__ == "__main__":
    main()
