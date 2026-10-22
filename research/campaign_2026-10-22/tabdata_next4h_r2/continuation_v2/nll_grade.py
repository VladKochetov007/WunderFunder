#!/usr/bin/env python3
"""Grade fixed epoch-2 cold-label NLL versus WMSE panels on frozen folds.

This is a source-bound development-panel grader, not a public evaluation. It
loads the project's official Global WP and paired component-bootstrap helpers;
it does not copy either metric implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

RUN_AT_EPOCH2 = re.compile(r"^(?P<run>[A-Za-z0-9][A-Za-z0-9_.-]{0,127})@ep2$")
SEQUENCE_ROWS = 20_000
OUTPUT_WIDTH = 25
PREDICTION_WIDTH = 2


def _parse_run_at(value: str) -> tuple[str, int]:
    match = RUN_AT_EPOCH2.fullmatch(value)
    if match is None:
        raise ValueError(f"expected an explicit RUN@ep2 name, got {value!r}")
    return match.group("run"), 2


def _source_imports(project_root: Path):
    """Import project-owned split, metric, and prediction validation code."""
    sys.path.insert(0, str(project_root / "scripts"))
    sys.path.insert(0, str(project_root / "src"))
    from evaluate_bandit_continuation import (
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
        _official_mask,
        _read_json,
        _validate_prediction_artifact,
        compare_arrays,
    )
    from evaluate_bandit_continuation import (
        SEQUENCE_ROWS as OFFICIAL_SEQUENCE_ROWS,
    )
    from wnn.gmetric import METRIC_ID, global_wp
    from wnn.splits import file_sha256, load_campaign_split

    if OFFICIAL_SEQUENCE_ROWS != SEQUENCE_ROWS:
        raise ValueError("grader sequence length differs from the source evaluator")
    return {
        "BOOTSTRAP_REPLICATES": BOOTSTRAP_REPLICATES,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "_official_mask": _official_mask,
        "_read_json": _read_json,
        "_validate_prediction_artifact": _validate_prediction_artifact,
        "compare_arrays": compare_arrays,
        "METRIC_ID": METRIC_ID,
        "global_wp": global_wp,
        "file_sha256": file_sha256,
        "load_campaign_split": load_campaign_split,
    }


def _model_state(checkpoint: dict[str, Any], run: str) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{run}: checkpoint lacks its exported model state")
    for name, tensor in state.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise ValueError(f"{run}: malformed model-state entry {name!r}")
        if tensor.is_floating_point():
            if tensor.dtype != torch.float32:
                raise ValueError(f"{run}: non-FP32 model tensor {name}: {tensor.dtype}")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{run}: nonfinite model tensor {name}")
        elif tensor.is_complex():
            raise ValueError(f"{run}: complex model tensor {name}")
    weight = state.get("head.2.weight")
    bias = state.get("head.2.bias")
    if (
        not torch.is_tensor(weight)
        or not torch.is_tensor(bias)
        or weight.ndim != 2
        or weight.shape[0] != OUTPUT_WIDTH
        or bias.shape != (OUTPUT_WIDTH,)
    ):
        raise ValueError(f"{run}: expected the matched 25-output mean/logvariance head")
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError(f"{run}: output head is not strict FP32")
    return state


def _expected_trainer_identity(root: Path, file_sha256) -> dict[str, Any]:
    """Reproduce the canonical training_source_identity source-file manifest."""
    entrypoint = (root / "scripts" / "train_rnn.py").resolve(strict=True)
    source_root = entrypoint.parents[1]
    dependency_paths = [source_root / "pyproject.toml", source_root / "uv.lock"]
    missing_dependencies = [path.name for path in dependency_paths if not path.is_file()]
    paths = [entrypoint, *(path for path in dependency_paths if path.is_file())]
    paths.extend(sorted((source_root / "src" / "wnn" / "training").glob("*.py")))
    files = {path.relative_to(source_root).as_posix(): file_sha256(path) for path in paths}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "wnn.training-source.v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
        "missing_dependency_files": missing_dependencies,
    }


def _validate_arm(
    *,
    root: Path,
    run_at: str,
    objective: str,
    split: Any,
    expected_row_groups: list[int],
    dependencies: dict[str, Any],
) -> tuple[np.memmap, dict[str, Any], dict[str, Any]]:
    run, epoch = _parse_run_at(run_at)
    run_root = root / "artifacts" / "runs" / run
    checkpoint_path = run_root / f"model_ep{epoch}.pt"
    provenance_path = run_root / "provenance.json"
    prediction_path = root / "cache" / f"pred_{run}_valid.npy"
    sidecar_path = prediction_path.with_suffix(".json")
    for required in (checkpoint_path, provenance_path, prediction_path, sidecar_path):
        if not required.is_file():
            raise FileNotFoundError(f"{run}: required artifact missing: {required}")

    provenance = dependencies["_read_json"](provenance_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("args"), dict):
        raise TypeError(f"{run}: checkpoint has no serialized arguments")
    args = checkpoint["args"]
    embedded = checkpoint.get("provenance")
    if not isinstance(embedded, dict):
        raise TypeError(f"{run}: checkpoint has no embedded provenance")
    if embedded.get("args") != args or provenance.get("args") != args:
        raise ValueError(f"{run}: checkpoint and run provenance arguments disagree")
    if args.get("name") != run or checkpoint.get("epoch") != epoch:
        raise ValueError(f"{run}: checkpoint run name or epoch differs")
    if (
        args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != epoch
        or args.get("no_eval_during_fit") is not True
        or args.get("epochs") != 3
    ):
        raise ValueError(f"{run}: expected fixed epoch 2 from a three-epoch no-eval fit")
    if (
        args.get("split_manifest_sha256") != split.sha256
        or args.get("valid_fold") != split.fold
        or provenance.get("split_manifest_sha256") != split.sha256
    ):
        raise ValueError(f"{run}: checkpoint/provenance is bound to another source fold")
    if (
        args.get("precision") != "fp32"
        or args.get("arch") != "gru"
        or args.get("cell") != "gru"
        or args.get("enc") != "linear"
        or args.get("mean_variance_head") != 1
        or args.get("mean_variance_objective") != objective
    ):
        raise ValueError(
            f"{run}: architecture, precision, head, or objective differs from the panel"
        )
    if objective == "nll" and args.get("mean_variance_target_source") != "labels":
        raise ValueError(f"{run}: NLL candidate is not the cold labels-only arm")
    if args.get("loss") != "wmse" or args.get("loss_norm") != "chunk":
        raise ValueError(f"{run}: base loss or normalization differs from the frozen recipe")
    if args.get("train_source") != "train+valid":
        raise ValueError(f"{run}: expected the frozen train+fit-valid membership recipe")
    for key in (
        "init",
        "resume_state",
        "distill",
        "distill_cache_root",
        "final_distill_initializer",
        "propensity",
        "prop_input",
    ):
        if args.get(key):
            raise ValueError(f"{run}: cold-label panel forbids {key}")
    if provenance.get("initial_checkpoint_sha256") or provenance.get("distillation_cache"):
        raise ValueError(f"{run}: provenance shows an initializer or teacher cache")

    expected_trainer = _expected_trainer_identity(root, dependencies["file_sha256"])
    trainer = provenance.get("trainer_implementation")
    if not isinstance(trainer, dict):
        raise TypeError(f"{run}: provenance lacks trainer source identity")
    if (
        trainer.get("schema") != expected_trainer["schema"]
        or trainer.get("sha256") != expected_trainer["sha256"]
        or trainer.get("files") != expected_trainer["files"]
        or trainer.get("missing_dependency_files") != expected_trainer["missing_dependency_files"]
        or embedded.get("trainer_implementation") != trainer
    ):
        raise ValueError(
            f"{run}: trainer implementation identity differs from canonical source files"
        )

    history_path = run_root / "history.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"{run}: completed epoch history is missing")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    provenance_diagnostics = provenance.get("update_diagnostics")
    embedded_diagnostics = embedded.get("update_diagnostics")
    if (
        not isinstance(history, list)
        or len(history) != 3
        or [record.get("epoch") for record in history if isinstance(record, dict)] != [0, 1, 2]
        or any(
            not isinstance(record, dict)
            or not isinstance(record.get("update_diagnostics"), dict)
            or not isinstance(record.get("fit_summary"), dict)
            for record in history
        )
    ):
        raise ValueError(f"{run}: history.json must contain exactly epoch 0/1/2 records")
    for records, label in (
        (provenance_diagnostics, "run provenance"),
        (embedded_diagnostics, "checkpoint provenance"),
    ):
        if (
            not isinstance(records, list)
            or len(records) != 3
            or [record.get("epoch_index") for record in records if isinstance(record, dict)]
            != [0, 1, 2]
            or any(not isinstance(record, dict) for record in records)
        ):
            raise ValueError(f"{run}: {label} must contain exactly three epoch update diagnostics")
    epoch_binding = {
        "history_epochs": [int(record["epoch"]) for record in history],
        "update_diagnostic_epochs": [
            int(record["epoch_index"]) for record in provenance_diagnostics
        ],
        "trainer_implementation_sha256": trainer["sha256"],
    }

    fit = provenance.get("fit_row_groups")
    if not isinstance(fit, dict):
        raise TypeError(f"{run}: provenance lacks fit-row membership")
    train_groups = set(map(int, fit.get("train", [])))
    valid_groups = set(map(int, fit.get("valid", [])))
    if train_groups != set(map(int, split.train_groups)) or valid_groups != set(
        map(int, split.valid_groups)
    ):
        raise ValueError(f"{run}: fit membership is not the complete frozen train+valid set")
    split.assert_fit_groups(train_groups, valid_groups)
    evaluation_groups = provenance.get("evaluation_row_groups")
    if (
        evaluation_groups is None
        or set(map(int, evaluation_groups)) != set(map(int, split.evaluation_groups))
        or len(evaluation_groups) != len(split.evaluation_groups)
    ):
        raise ValueError(f"{run}: recorded evaluation membership differs from the split")
    checkpoint_sha256 = dependencies["file_sha256"](checkpoint_path)
    recorded_hashes = provenance.get("checkpoint_sha256", {})
    if recorded_hashes.get(checkpoint_path.name) != checkpoint_sha256:
        raise ValueError(f"{run}: provenance checkpoint hash differs from checkpoint bytes")
    _model_state(checkpoint, run)

    metadata = dependencies["_read_json"](sidecar_path)
    row_groups = list(map(int, metadata.get("row_groups", [])))
    if row_groups != expected_row_groups:
        raise ValueError(f"{run}: prediction sidecar row order differs from valid_targets")
    prediction, binding = dependencies["_validate_prediction_artifact"](
        prediction_path,
        metadata,
        run=run,
        epoch=epoch,
        split_sha256=split.sha256,
        checkpoint_sha256=checkpoint_sha256,
        row_groups=row_groups,
        shape=(len(row_groups), SEQUENCE_ROWS, PREDICTION_WIDTH),
    )
    if metadata.get("inference_precision") != "fp32":
        raise ValueError(f"{run}: prediction sidecar does not declare FP32 inference")
    binding["model_output_width"] = OUTPUT_WIDTH
    binding["inference_precision"] = metadata.get("inference_precision")
    binding.update(epoch_binding)
    return prediction, binding, args


def _grade_panel(
    *,
    root: Path,
    split_path: Path,
    candidate_at: str,
    control_at: str,
    n_boot: int,
    seed: int,
    dependencies: dict[str, Any],
) -> dict[str, Any]:
    split = dependencies["load_campaign_split"](split_path, root)
    target_path = root / "cache" / "valid_targets.npz"
    if not target_path.is_file():
        raise FileNotFoundError(f"target cache missing: {target_path}")
    with np.load(target_path, allow_pickle=False) as cache:
        row_groups = list(map(int, np.asarray(cache["row_group"]).tolist()))
        targets_all = np.asarray(cache["y"])
        is_scored_all = np.asarray(cache["mask"], dtype=bool)
    expected_groups = set(map(int, split.valid_groups)) | set(map(int, split.evaluation_groups))
    if (
        len(set(row_groups)) != len(row_groups)
        or set(row_groups) != expected_groups
        or targets_all.shape != (len(row_groups), SEQUENCE_ROWS, PREDICTION_WIDTH)
        or targets_all.dtype != np.float32
        or is_scored_all.shape != (len(row_groups), SEQUENCE_ROWS)
    ):
        raise ValueError("valid target cache differs from the frozen fold membership or shape")
    position = {group: index for index, group in enumerate(row_groups)}
    groups = list(map(int, split.evaluation_groups))
    indices = np.asarray([position[group] for group in groups], dtype=np.int64)
    targets = np.asarray(targets_all[indices], dtype=np.float32)
    is_scored = np.asarray(is_scored_all[indices], dtype=bool)
    mask = dependencies["_official_mask"](is_scored)
    component_of = {
        int(group): int(component)
        for source, group, component in split.components
        if source == "valid"
    }
    if any(group not in component_of for group in groups):
        raise ValueError("split component map does not cover all evaluation groups")
    components = np.asarray([component_of[group] for group in groups], dtype=np.int64)

    candidate, candidate_binding, candidate_args = _validate_arm(
        root=root,
        run_at=candidate_at,
        objective="nll",
        split=split,
        expected_row_groups=row_groups,
        dependencies=dependencies,
    )
    control, control_binding, control_args = _validate_arm(
        root=root,
        run_at=control_at,
        objective="wmse",
        split=split,
        expected_row_groups=row_groups,
        dependencies=dependencies,
    )
    if candidate_args.get("seed") != control_args.get("seed"):
        raise ValueError("paired NLL and WMSE arms use different random seeds")
    allowed_differences = {
        "name",
        "mean_variance_objective",
        "mean_variance_target_source",
    }
    all_keys = set(candidate_args) | set(control_args)
    mismatched = sorted(
        key
        for key in all_keys - allowed_differences
        if candidate_args.get(key) != control_args.get(key)
    )
    if mismatched:
        raise ValueError(f"paired arms differ outside objective/name/target-source: {mismatched}")
    candidate_eval = np.asarray(candidate[indices], dtype=np.float32)
    control_eval = np.asarray(control[indices], dtype=np.float32)
    paired = dependencies["compare_arrays"](
        targets,
        candidate_eval,
        control_eval,
        is_scored,
        components,
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "fold": int(split.fold),
        "split_manifest": str(split_path),
        "split_manifest_sha256": split.sha256,
        "valid_targets_sha256": dependencies["file_sha256"](target_path),
        "evaluation_groups": len(groups),
        "components": len(set(components.tolist())),
        "candidate": candidate_at,
        "candidate_binding": candidate_binding,
        "candidate_official": dependencies["global_wp"](targets, candidate_eval, mask),
        "control": control_at,
        "control_binding": control_binding,
        "control_official": dependencies["global_wp"](targets, control_eval, mask),
        "paired_component_bootstrap": paired,
        "cold_common_initialization": {
            "same_seed": int(candidate_args["seed"]),
            "arguments_match_except": sorted(allowed_differences),
            "verified_from_recipe": False,
            "evidence": "matched clean-init arguments and seed; initial tensors were not archived separately",
            "initial_weights_directly_observable_in_final_checkpoints": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--candidate", action="append", required=True, help="RUN@ep2; repeat per fold"
    )
    parser.add_argument(
        "--control", action="append", required=True, help="RUN@ep2; repeat per fold"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--plan-sha256", default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    root = args.project_root.resolve()
    if not root.is_dir():
        parser.error(f"project root is not a directory: {root}")
    if not (len(args.split_manifest) == len(args.candidate) == len(args.control)):
        parser.error("repeat --split-manifest, --candidate, and --control once per fold")
    if len(args.split_manifest) < 1:
        parser.error("at least one fold panel is required")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing result: {output}")
    dependencies = _source_imports(root)
    if (args.plan is None) != (args.plan_sha256 is None):
        parser.error("--plan and --plan-sha256 must be supplied together")
    plan_binding = None
    if args.plan is not None:
        plan_path = args.plan.resolve()
        if not plan_path.is_file():
            parser.error(f"plan packet is not a file: {plan_path}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", args.plan_sha256):
            parser.error("--plan-sha256 must be a 64-character hexadecimal digest")
        actual_plan_sha256 = dependencies["file_sha256"](plan_path)
        if actual_plan_sha256.lower() != args.plan_sha256.lower():
            parser.error("plan packet bytes differ from the externally supplied SHA256")
        plan_binding = {
            "path": str(plan_path),
            "sha256": actual_plan_sha256,
            "external_sha256_verified": True,
        }
    n_boot = args.n_boot or dependencies["BOOTSTRAP_REPLICATES"]
    seed = args.seed if args.seed is not None else dependencies["BOOTSTRAP_SEED"]
    if n_boot < 2:
        parser.error("--n-boot must be at least 2")

    panels = []
    seen_runs: set[str] = set()
    for split_arg, candidate_at, control_at in zip(
        args.split_manifest, args.candidate, args.control, strict=True
    ):
        candidate_run, _ = _parse_run_at(candidate_at)
        control_run, _ = _parse_run_at(control_at)
        if candidate_run == control_run or candidate_run in seen_runs or control_run in seen_runs:
            raise ValueError("each candidate/control run must appear in exactly one contrast")
        seen_runs.update((candidate_run, control_run))
        split_path = split_arg if split_arg.is_absolute() else root / split_arg
        panel = _grade_panel(
            root=root,
            split_path=split_path,
            candidate_at=candidate_at,
            control_at=control_at,
            n_boot=n_boot,
            seed=seed,
            dependencies=dependencies,
        )
        panels.append(panel)

    report = {
        "grader": "nll_grade.py",
        "created_utc": datetime.now(UTC).isoformat(),
        "project_root": str(root),
        "plan_binding": plan_binding,
        "metric_id": dependencies["METRIC_ID"],
        "source_sha256": {
            "scripts/evaluate_bandit_continuation.py": dependencies["file_sha256"](
                root / "scripts" / "evaluate_bandit_continuation.py"
            ),
            "scripts/predict_run.py": dependencies["file_sha256"](
                root / "scripts" / "predict_run.py"
            ),
            "src/wnn/training/mean_variance.py": dependencies["file_sha256"](
                root / "src" / "wnn" / "training" / "mean_variance.py"
            ),
            "src/wnn/training/losses.py": dependencies["file_sha256"](
                root / "src" / "wnn" / "training" / "losses.py"
            ),
            "src/wnn/evaluation.py": dependencies["file_sha256"](
                root / "src" / "wnn" / "evaluation.py"
            ),
            "src/wnn/gmetric.py": dependencies["file_sha256"](root / "src" / "wnn" / "gmetric.py"),
            "src/wnn/splits.py": dependencies["file_sha256"](root / "src" / "wnn" / "splits.py"),
        },
        "epoch": 2,
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "evidence_scope": "conditional on these fixed fits; component bootstrap covers evaluation-unit uncertainty only, with no seed-level estimate",
        "exposure": "frozen excluded-group development panels; repeated screening evidence, not public evaluation",
        "panels": panels,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "metric_id": report["metric_id"],
                "folds": [
                    {
                        "fold": panel["fold"],
                        "nll_wp": panel["candidate_official"]["wp"],
                        "wmse_wp": panel["control_official"]["wp"],
                        "delta": panel["paired_component_bootstrap"]["mean_diff"],
                    }
                    for panel in panels
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
