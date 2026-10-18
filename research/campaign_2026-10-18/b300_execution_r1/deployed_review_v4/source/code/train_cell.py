"""Thin entrypoint for one frozen a266 labels-only GRU fit cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from adapter_common import (
    _RECIPE,
    _SOURCE_MANIFEST,
    _TREE_MANIFEST,
    _json,
    _sha256,
    _write_json,
    adapter_bundle_identity,
    apply_thread_environment,
    verify_deployed_tree_manifest,
    verify_source_manifest,
)
from fit_receipt import extract_fit_receipt
from fit_recipe import build_training_argv
from fit_workspace import prepare_workspace


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=("wide", "deep"), required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("profile", "fit", "resume"), required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-state", type=Path)
    return parser.parse_args()


def main() -> None:
    """Bind workspace inputs, record argv, then invoke the canonical trainer."""
    args = _parse_args()
    thread_environment = apply_thread_environment()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    source_root = args.source_root.resolve(strict=True)
    run_dir = Path(os.environ["WNN_RUN_DIR"]).resolve(strict=True)
    data_root = args.data_root.resolve(strict=True)
    if run_dir.is_relative_to(source_root) or source_root.is_relative_to(run_dir):
        raise ValueError("WNN_RUN_DIR and frozen source root must be separate trees")
    for relative in (
        "scripts/train_rnn.py",
        "src/wnn/training/environment.py",
        "src/wnn/training/runner.py",
    ):
        if not (source_root / relative).is_file():
            raise FileNotFoundError(source_root / relative)
    sys.path.insert(0, str(source_root / "src"))
    os.environ["WNN_PROJECT_ROOT"] = str(run_dir)
    os.environ["WNN_LOADER_RAM_GB"] = "1"
    os.environ["WNN_LOADER_WORKERS"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

    recipe = _json(_RECIPE)
    run_name = args.run_name or f"tabdata_b300_r01_v1_{args.arm}_f{args.fold}_s{args.seed}"
    output_dir = run_dir / "artifacts" / "runs" / run_name
    resume_state = args.resume_state or output_dir / "training_state.pt"
    if args.mode == "resume":
        if not output_dir.is_dir() or resume_state.resolve(strict=True) != (
            output_dir / "training_state.pt"
        ).resolve(strict=True):
            raise ValueError("resume must continue inside the same WNN_RUN_DIR workspace")
    elif not args.prepare_only and output_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing trainer run: {output_dir}")
    training_argv = build_training_argv(
        recipe,
        arm=args.arm,
        fold=args.fold,
        seed=args.seed,
        run_name=run_name,
        split_manifest=args.split_manifest,
        mode=args.mode,
        resume_state=resume_state if args.mode == "resume" else None,
    )
    input_receipt = prepare_workspace(run_dir, data_root, args.split_manifest, args.fold)
    source_manifest = _json(_SOURCE_MANIFEST)
    if source_manifest.get("source_revision") != recipe["expected_base_source_revision"]:
        raise ValueError("portable source manifest and command builder pin different revisions")
    if recipe.get("source_manifest_sha256") != _sha256(_SOURCE_MANIFEST):
        raise ValueError("portable source manifest hash differs from the frozen command builder")
    if source_manifest.get("source_identity_sha256") != recipe.get(
        "expected_source_identity_sha256"
    ):
        raise ValueError("portable source aggregate differs from the frozen command builder")
    verified_source = verify_source_manifest(source_root, source_manifest)
    tree_manifest = _json(_TREE_MANIFEST)
    tree_manifest_sha256 = _sha256(_TREE_MANIFEST)
    if recipe.get("source_tree_manifest") != _TREE_MANIFEST.name:
        raise ValueError("portable deployed-tree manifest and command builder name disagree")
    if recipe.get("source_tree_manifest_sha256") != tree_manifest_sha256:
        raise ValueError("portable deployed-tree manifest differs from the frozen command builder")
    if tree_manifest.get("archive_sha256") != recipe.get("source_tree_archive_sha256"):
        raise ValueError("deployed-tree manifest archive hash differs from the command builder")
    if tree_manifest.get("deployed_tree_identity_sha256") != recipe.get(
        "expected_source_tree_identity_sha256"
    ):
        raise ValueError("deployed-tree aggregate differs from the command builder")
    if tree_manifest.get("source_revision") != recipe["expected_base_source_revision"]:
        raise ValueError("deployed-tree manifest and command builder pin different revisions")
    verified_tree = verify_deployed_tree_manifest(
        source_root, tree_manifest, manifest_sha256=tree_manifest_sha256
    )
    input_receipt["deployed_source_tree"] = verified_tree
    input_receipt["trainer_source_identity"] = verified_source
    adapter_bundle = adapter_bundle_identity()
    input_receipt["adapter_bundle_identity"] = adapter_bundle
    command = {
        "schema": "tabdata-b300-command.v1",
        "mode": args.mode,
        "arm": args.arm,
        "fold": args.fold,
        "seed": args.seed,
        "run_name": run_name,
        "source_root": str(source_root),
        "source_commit_expected": recipe["expected_base_source_revision"],
        "source_git_metadata_required": False,
        "source_manifest_sha256": _sha256(_SOURCE_MANIFEST),
        "trainer_source_identity": verified_source,
        "source_tree_manifest_sha256": tree_manifest_sha256,
        "deployed_source_tree": verified_tree,
        "data_root": str(data_root),
        "run_dir": str(run_dir),
        "python": sys.executable,
        "training_argv": training_argv,
        "training_argv_sha256": hashlib.sha256(
            json.dumps(training_argv, separators=(",", ":")).encode()
        ).hexdigest(),
        "adapter_sha256": _sha256(Path(__file__).resolve()),
        "adapter_bundle_identity": adapter_bundle,
        "runtime_environment": {
            "WNN_PROJECT_ROOT": str(run_dir),
            "WNN_LOADER_RAM_GB": "1",
            "WNN_LOADER_WORKERS": "0",
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "WNN_GPU_MEMORY_BYTES": os.environ.get("WNN_GPU_MEMORY_BYTES"),
            **thread_environment,
        },
    }
    _write_json(run_dir / "command.json", command)
    _write_json(run_dir / "adapter_preflight.json", input_receipt)
    if args.prepare_only:
        return

    from wnn.training.environment import TrainingEnvironment
    from wnn.training.runner import run_training

    environment = TrainingEnvironment(
        source_root=source_root,
        root=run_dir,
        argv=training_argv,
    )
    started = time.time()
    try:
        run_training(training_argv, environment=environment)
    except BaseException as error:
        _write_json(
            run_dir / "producer_receipt.json",
            {
                "schema": "tabdata-b300-producer.v1",
                "status": "trainer_failed",
                "run_name": run_name,
                "arm": args.arm,
                "fold": args.fold,
                "seed": args.seed,
                "elapsed_seconds": time.time() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "command_sha256": _sha256(run_dir / "command.json"),
            },
        )
        raise
    if args.mode == "profile":
        receipt = {
            "schema": "tabdata-b300-profile-producer.v1",
            "status": "profile_completed_check_controller_log",
            "run_name": run_name,
            "arm": args.arm,
            "fold": args.fold,
            "seed": args.seed,
            "expected_log_records": ["PROFILE", "PROFILE_UPDATE_EVIDENCE"],
            "elapsed_seconds": time.time() - started,
            "input_ancestry": input_receipt,
            "adapter_bundle_identity": adapter_bundle,
            "command_sha256": _sha256(run_dir / "command.json"),
        }
    else:
        receipt = extract_fit_receipt(
            run_dir,
            output_dir,
            run_name=run_name,
            arm=args.arm,
            fold=args.fold,
            seed=args.seed,
            input_receipt=input_receipt,
            adapter_bundle_identity=adapter_bundle,
        )
        receipt["elapsed_seconds"] = time.time() - started
        receipt["command_sha256"] = _sha256(run_dir / "command.json")
    _write_json(run_dir / "producer_receipt.json", receipt)


if __name__ == "__main__":
    main()
