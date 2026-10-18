"""Run the frozen feature-only predictor and bind its output to the R4 grader."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from adapter_common import (
    _TREE_MANIFEST,
    _json,
    _sha256,
    _write_json,
    adapter_bundle_identity,
    apply_thread_environment,
    verify_deployed_tree_manifest,
    verify_source_manifest,
)
from export_contract import _check_fit_binding, _load_grader, predictor_argv
from fit_workspace import prepare_export_workspace, require_fresh_export_outputs

_CODE_DIR = Path(__file__).resolve().parent
_BUILDER = _CODE_DIR / "command_builder.json"
_SOURCE_MANIFEST = _CODE_DIR / "a266_source_manifest.json"
_DEFAULT_GRADER = (
    _CODE_DIR.parents[3]
    / "research/campaign_2026-10-18/priority25_execution_r4/code/grade_endpoint.py"
)


def _verify_export_lineage(
    source_root: Path,
    run_dir: Path,
    split_manifest: Path,
    grader_path: Path,
    run_name: str,
    fold: int,
    seed: int,
) -> tuple[dict[str, Any], Path, dict[str, Any], list[int], dict[str, Any], dict[str, Any], str]:
    """Bind the pinned source tree, completed fit, split and adapter bundle."""
    builder = _json(_BUILDER)
    if builder.get("export_endpoint_epoch") != 5:
        raise ValueError("export builder no longer names the fixed epoch-5 endpoint")
    predictor = source_root / "scripts" / "predict_run.py"
    if builder.get("predictor_sha256") != _sha256(predictor):
        raise ValueError("predictor source bytes differ from the pinned a266 predictor")
    if builder.get("r4_grader_sha256") != _sha256(grader_path):
        raise ValueError("R4 grader bytes differ from the frozen export contract")
    source_identity = verify_source_manifest(source_root, _json(_SOURCE_MANIFEST))
    tree_manifest = _json(_TREE_MANIFEST)
    tree_sha = _sha256(_TREE_MANIFEST)
    if (
        builder.get("source_tree_manifest") != _TREE_MANIFEST.name
        or builder.get("source_tree_manifest_sha256") != tree_sha
        or tree_manifest.get("archive_sha256") != builder.get("source_tree_archive_sha256")
        or tree_manifest.get("deployed_tree_identity_sha256")
        != builder.get("expected_source_tree_identity_sha256")
        or tree_manifest.get("source_revision") != builder.get("expected_base_source_revision")
    ):
        raise ValueError("deployed-tree manifest differs from the frozen command builder")
    deployed_tree = verify_deployed_tree_manifest(
        source_root, tree_manifest, manifest_sha256=tree_sha
    )
    training_command = _json(run_dir / "command.json")
    fit_receipt, _, groups = _check_fit_binding(run_dir, split_manifest, run_name, fold, seed)
    bundle_identity = adapter_bundle_identity()
    if training_command.get("trainer_source_identity") != source_identity:
        raise ValueError("trainer source identity differs from the fit command receipt")
    if (
        training_command.get("deployed_source_tree") != deployed_tree
        or fit_receipt.get("input_ancestry", {}).get("deployed_source_tree") != deployed_tree
        or training_command.get("source_tree_manifest_sha256") != tree_sha
    ):
        raise ValueError("deployed source-tree identity differs from the fitted run ancestry")
    if (
        training_command.get("adapter_bundle_identity") != bundle_identity
        or fit_receipt.get("adapter_bundle_identity") != bundle_identity
    ):
        raise ValueError("producer/export adapter bundle differs from the fit-time code hashes")
    return builder, predictor, fit_receipt, groups, bundle_identity, deployed_tree, tree_sha


def run_export(
    *,
    source_root: Path,
    run_dir: Path,
    export_dir: Path | None = None,
    split_manifest: Path,
    run_name: str,
    fold: int,
    seed: int,
    grader_path: Path = _DEFAULT_GRADER,
) -> dict[str, Any]:
    """Call the exact a266 predictor, then validate through the frozen R4 grader."""
    source_root = source_root.resolve(strict=True)
    run_dir = run_dir.resolve(strict=True)
    if export_dir is None:
        output_root = run_dir
        make_export_workspace = False
    else:
        output_root = export_dir.parent.resolve(strict=True) / export_dir.name
        if output_root == run_dir:
            make_export_workspace = False
        else:
            if (
                output_root == source_root
                or output_root.is_relative_to(source_root)
                or source_root.is_relative_to(output_root)
            ):
                raise ValueError("export workspace and frozen source root must be separate trees")
            make_export_workspace = True
    workspace_receipt = None
    split_manifest = split_manifest.resolve(strict=True)
    grader_path = grader_path.resolve(strict=True)
    thread_environment = apply_thread_environment()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    import numpy as np

    (
        builder,
        predictor,
        fit_receipt,
        groups,
        bundle_identity,
        deployed_tree,
        tree_manifest_sha256,
    ) = _verify_export_lineage(
        source_root, run_dir, split_manifest, grader_path, run_name, fold, seed
    )

    if make_export_workspace:
        workspace_receipt = prepare_export_workspace(output_root, run_dir)
    prediction_path, receipt_path, command_path = require_fresh_export_outputs(
        output_root, run_name
    )

    argv = predictor_argv(predictor, run_name, split_manifest)
    environment = {
        "WNN_PROJECT_ROOT": str(output_root),
        "WNN_LOADER_RAM_GB": "1",
        "WNN_LOADER_WORKERS": "0",
        "NVIDIA_TF32_OVERRIDE": "0",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        **thread_environment,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    _write_json(
        command_path,
        {
            "schema": "tabdata-b300-export-command.v1",
            "run_name": run_name,
            "fit_run_dir": str(run_dir),
            "export_dir": str(output_root),
            "export_workspace": workspace_receipt,
            "fold": fold,
            "seed": seed,
            "epoch": 5,
            "source_root": str(source_root),
            "predictor": str(predictor),
            "predictor_sha256": builder["predictor_sha256"],
            "grader": str(grader_path),
            "grader_sha256": builder["r4_grader_sha256"],
            "argv": argv,
            "argv_sha256": hashlib.sha256(
                json.dumps(argv, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "environment": environment,
            "split_manifest_sha256": _sha256(split_manifest),
            "evaluation_row_groups": groups,
            "checkpoint_sha256": fit_receipt["outputs"][f"artifacts/runs/{run_name}/model_ep5.pt"][
                "sha256"
            ],
            "adapter_bundle_identity": bundle_identity,
            "deployed_source_tree": deployed_tree,
            "source_tree_manifest_sha256": tree_manifest_sha256,
            "WNN_GPU_MEMORY_BYTES": os.environ.get("WNN_GPU_MEMORY_BYTES"),
        },
    )

    started = time.time()
    try:
        os.environ.update(environment)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            (str(source_root / "src"), os.environ.get("PYTHONPATH", ""))
        ).rstrip(os.pathsep)
        sys.path.insert(0, str(source_root / "src"))
        sys.path.insert(0, str(source_root / "scripts"))
        original_argv = sys.argv
        sys.argv = argv
        sys.modules.pop("predict_run", None)
        predictor_module = importlib.import_module("predict_run")
        if Path(predictor_module.__file__).resolve() != predictor.resolve():
            raise RuntimeError("import resolved a different predictor module")
        try:
            predictor_module.main()
        finally:
            sys.argv = original_argv

        grader = _load_grader(grader_path)
        split = grader.load_campaign_split(split_manifest, output_root)
        if split.fold != fold or list(split.evaluation_groups) != groups:
            raise ValueError("R4 grader split membership differs from the requested fit cell")
        prediction, binding = grader.validate_prediction(output_root, run_name, 5, groups, split)
        receipt = {
            "schema": "tabdata-b300-export-producer.v1",
            "status": "completed_fixed_epoch5_feature_only_fp32",
            "run_name": run_name,
            "fit_run_dir": str(run_dir),
            "export_dir": str(output_root),
            "export_workspace": workspace_receipt,
            "fold": fold,
            "seed": seed,
            "epoch": 5,
            "prediction_shape": list(prediction.shape),
            "prediction_dtype": str(prediction.dtype),
            "prediction_sha256": binding["prediction_sha256"],
            "prediction_bytes": prediction_path.stat().st_size,
            "sidecar_sha256": binding["sidecar_sha256"],
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "fit_receipt_sha256": _sha256(run_dir / "producer_receipt.json"),
            "command_sha256": _sha256(command_path),
            "predictor_sha256": builder["predictor_sha256"],
            "r4_grader_sha256": builder["r4_grader_sha256"],
            "split_manifest_sha256": split.sha256,
            "evaluation_membership_sha256": hashlib.sha256(
                np.asarray(groups, dtype="<i8").tobytes()
            ).hexdigest(),
            "evaluation_row_groups": groups,
            "inference_precision": "fp32",
            "feature_only": True,
            "target_reads_during_export": False,
            "input_ancestry": fit_receipt["input_ancestry"],
            "adapter_bundle_identity": bundle_identity,
            "deployed_source_tree": deployed_tree,
            "elapsed_seconds": time.time() - started,
        }
        _write_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        if not receipt_path.exists():
            _write_json(
                receipt_path,
                {
                    "schema": "tabdata-b300-export-producer.v1",
                    "status": "export_failed",
                    "run_name": run_name,
                    "fit_run_dir": str(run_dir),
                    "export_dir": str(output_root),
                    "fold": fold,
                    "seed": seed,
                    "epoch": 5,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "command_sha256": _sha256(command_path),
                },
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--grader-script", type=Path, default=_DEFAULT_GRADER)
    args = parser.parse_args()
    run_export(
        source_root=args.source_root,
        run_dir=args.run_dir,
        export_dir=args.export_dir,
        split_manifest=args.split_manifest,
        run_name=args.run_name,
        fold=args.fold,
        seed=args.seed,
        grader_path=args.grader_script,
    )


if __name__ == "__main__":
    main()
