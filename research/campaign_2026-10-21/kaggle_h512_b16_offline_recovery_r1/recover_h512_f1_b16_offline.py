"""Recover SWA345 from saved B16 checkpoints without fitting or reading targets."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

from checkpoint_io import canonical_mean_state, copy_file, load_source_checkpoints
from fit_audit import validate_fit_artifacts
from recovery_contract import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_RECOVERY_ROOT,
    EXPECTED_CHECKPOINTS,
    FIT_NAME,
    INITIALIZATION_LIMITATION,
    INITIALIZATION_STATUS,
    PACKET,
    ROOT,
    SWA_NAME,
    require,
    safe_recovery_root,
    sha256,
    validate_inference_source,
    validate_worker_receipts,
    verify_pinned,
)

RECEIPT_NAME = "B16_OFFLINE_RECOVERY_RECEIPT_R1.json"
STAGE_RECEIPT_NAME = "B16_OFFLINE_RECOVERY_STAGE_R1.json"


def derive_swa(artifact_root: Path, recovery_root: Path, project_root: Path) -> dict[str, Any]:
    """Copy CP3/4/5 and derive the frozen equal-parameter mean in a new scratch root.

    Args:
        artifact_root: Collected Kaggle output root containing ``artifacts/runs``.
        recovery_root: New physical-disk output root; it must not already exist.
        project_root: Repository checkout containing the pinned helper and receipts.

    Returns:
        A create-only recovery receipt binding source, copied and derived bytes.

    Raises:
        FileExistsError: If the output root or receipt already exists.
        ValueError: If any source, fit, checkpoint or ancestry contract differs.
    """
    source_root = artifact_root.expanduser().resolve(strict=True)
    output_root = safe_recovery_root(recovery_root)
    require(
        not output_root.is_relative_to(source_root) and not source_root.is_relative_to(output_root),
        "Recovery and collected artifact roots must be separate",
    )
    require(not output_root.exists(), f"Recovery root already exists: {output_root}")
    provenance, diagnostics, source_fit_dir = validate_fit_artifacts(project_root, source_root)
    worker = validate_worker_receipts(project_root)
    source_checkpoints = load_source_checkpoints(source_fit_dir)
    mean_state = canonical_mean_state(project_root)
    state = mean_state([checkpoint["model"] for checkpoint in source_checkpoints])

    output_fit_dir = output_root / "artifacts/runs" / FIT_NAME
    output_swa_dir = output_root / "artifacts/runs" / SWA_NAME
    output_fit_dir.mkdir(parents=True)
    output_swa_dir.mkdir(parents=True)
    copied: dict[str, dict[str, str | int]] = {}
    for epoch, (size, digest) in EXPECTED_CHECKPOINTS.items():
        relative = f"artifacts/runs/{FIT_NAME}/model_ep{epoch}.pt"
        source = source_fit_dir / f"model_ep{epoch}.pt"
        target = output_root / relative
        copied[relative] = {"sha256": copy_file(source, target, digest), "size_bytes": size}
    for filename in ("provenance.json", "history.json"):
        source = source_fit_dir / filename
        target = output_fit_dir / filename
        copied[f"artifacts/runs/{FIT_NAME}/{filename}"] = {
            "sha256": copy_file(source, target),
            "size_bytes": target.stat().st_size,
        }

    cp5 = copy.deepcopy(source_checkpoints[-1])
    derivation: dict[str, Any] = {
        "schema": "wnn.kaggle-h512-b16-offline-swa345.v1",
        "status": "DERIVED_UNGRADED",
        "method": "equal_parameter_mean_epochs_3_4_5",
        "source_run": FIT_NAME,
        "destination_run": SWA_NAME,
        "source_revision": "68ff682977b822f38351aac0f621146f5f13b440",
        "source_checkpoints": [
            {
                "epoch": epoch,
                "path": f"artifacts/runs/{FIT_NAME}/model_ep{epoch}.pt",
                "sha256": EXPECTED_CHECKPOINTS[epoch][1],
            }
            for epoch in (3, 4, 5)
        ],
        "input_manifest": {
            "path": "research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2/fold1_profile_view.json",
            "sha256": provenance["split_manifest_sha256"],
        },
        "original_canonical_manifest": {
            "path": "research/campaign_2026-09-29/artifacts/splits/fold1.json",
            "sha256": "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae",
        },
        "accumulation": "float64 mean; cast to each source parameter dtype",
        "normalization_identity": provenance["normalization_identity"],
        "initial_parameter_sha256": None,
        "initialization_identity_status": INITIALIZATION_STATUS,
        "initialization_limitation": INITIALIZATION_LIMITATION,
        "training_run": False,
        "quality_evaluated": False,
    }
    cp5["model"] = state
    cp5["weight_derivation"] = derivation
    cp5["provenance"]["weight_derivation"] = derivation
    model_path = output_swa_dir / "model_ep5.pt"

    import torch

    torch.save(cp5, model_path)
    model_link = output_swa_dir / "model.pt"
    try:
        os.link(model_path, model_link)
    except OSError:
        shutil.copyfile(model_path, model_link)
    swa_sha = sha256(model_path)
    swa_provenance = {
        **cp5["provenance"],
        "checkpoint_sha256": {"model_ep5.pt": swa_sha, "model.pt": sha256(model_link)},
    }
    provenance_path = output_swa_dir / "provenance.json"
    provenance_path.write_text(json.dumps(swa_provenance, indent=2) + "\n", encoding="utf-8")
    derivation_path = output_swa_dir / "weight_derivation_receipt.json"
    derivation_receipt = {
        **derivation,
        "outputs": {
            "model_ep5.pt": {
                "path": f"artifacts/runs/{SWA_NAME}/model_ep5.pt",
                "sha256": swa_sha,
                "size_bytes": model_path.stat().st_size,
            },
            "provenance.json": {
                "path": f"artifacts/runs/{SWA_NAME}/provenance.json",
                "sha256": sha256(provenance_path),
                "size_bytes": provenance_path.stat().st_size,
            },
            "model.pt": {
                "path": f"artifacts/runs/{SWA_NAME}/model.pt",
                "sha256": sha256(model_link),
                "size_bytes": model_link.stat().st_size,
            },
        },
    }
    derivation_path.write_text(
        json.dumps(derivation_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema": "wnn.kaggle-h512-b16-offline-recovery.v1",
        "status": "OFFLINE_ARTIFACT_RECOVERY_COMPLETE_NOT_GRADED",
        "artifact_root": str(source_root),
        "recovery_root": str(output_root),
        "worker_terminal_sha256": sha256(project_root / PACKET / "B16_WORKER_TERMINAL_R3.json"),
        "collection_receipt_sha256": sha256(project_root / PACKET / "B16_COLLECTION_R3.json"),
        "fit_provenance_sha256": sha256(source_fit_dir / "provenance.json"),
        "fit_history_sha256": sha256(source_fit_dir / "history.json"),
        "source_commit": "68ff682977b822f38351aac0f621146f5f13b440",
        "recovery_code_sha256": {
            name: sha256(Path(__file__).resolve().with_name(name))
            for name in (
                "recover_h512_f1_b16_offline.py",
                "recovery_contract.py",
                "fit_audit.py",
                "checkpoint_io.py",
            )
        },
        "run_identity_source_sha256": provenance["run_identity_source_sha256"],
        "trainer_implementation": provenance["trainer_implementation"],
        "source_checkpoints": derivation["source_checkpoints"],
        "copied_fit_artifacts": copied,
        "swa_derivation_receipt_path": f"artifacts/runs/{SWA_NAME}/weight_derivation_receipt.json",
        "swa_derivation_receipt_sha256": sha256(derivation_path),
        "swa_checkpoint_sha256": swa_sha,
        "normalization_identity": provenance["normalization_identity"],
        "fit_membership": {
            "train_groups": len(provenance["fit_row_groups"]["train"]),
            "valid_fit_groups": len(provenance["fit_row_groups"]["valid"]),
            "evaluation_groups": len(provenance["evaluation_row_groups"]),
        },
        "optimizer_chunk_updates": provenance["cumulative_updates"]["optimizer_completed_updates"],
        "epoch_diagnostics": len(diagnostics),
        "initial_parameter_sha256": None,
        "initialization_identity_status": INITIALIZATION_STATUS,
        "initialization_limitation": INITIALIZATION_LIMITATION,
        "worker_terminal_status": worker["status"],
        "worker_terminal_error": worker["error"],
        "precision_evidence": {
            "fit_args_precision": provenance["args"]["precision"],
            "profile_tf32_override": worker["precision_audit"]["nvidia_tf32_override"],
            "profile_tf32_flags": worker["precision_audit"]["flags_after_profile"],
            "after_fit_runtime_flags": "not recorded because the worker exited at the missing initialization-identity postfit gate",
        },
        "training_run": False,
        "quality_evaluated": False,
    }
    receipt_path = output_root / RECEIPT_NAME
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def stage_prediction_root(recovery_root: Path, data_root: Path) -> dict[str, Any]:
    """Symlink only the frozen raw inputs and structural caches for the canonical predictor."""
    root = safe_recovery_root(recovery_root)
    source = data_root.expanduser().resolve(strict=True)
    require(
        (root / RECEIPT_NAME).is_file(),
        "Run the offline derivation before staging predictor inputs",
    )
    require((source / "datasets").is_dir(), "Data root lacks mounted train/valid datasets")
    raw_sources = {}
    expected_raw = {
        "train.parquet": (
            29_240_219_011,
            "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
        ),
        "valid.parquet": (
            5_615_509_128,
            "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
        ),
    }
    for name, (size, digest) in expected_raw.items():
        path = source / "datasets" / name
        require(path.stat().st_size == size, f"Raw source size differs: {name}")
        raw_sources[f"datasets/{name}"] = {
            "path": str(path.resolve(strict=True)),
            "size_bytes": size,
            "expected_sha256": digest,
            "verification": "portable registry validates bytes during canonical prediction preflight",
        }
    datasets = root / "datasets"
    cache = root / "cache"
    datasets.symlink_to(source / "datasets", target_is_directory=True)
    cache.mkdir()
    linked: dict[str, str] = {"datasets": str(source / "datasets")}
    for name in ("chains.npz", "chain_folds.npz"):
        source_path = source / "cache" / name
        if name == "chains.npz":
            expected = "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766"
        else:
            expected = "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f"
        verify_pinned(source_path, expected, name)
        target = cache / name
        target.symlink_to(source_path)
        linked[f"cache/{name}"] = str(source_path)
    registry_path = source / PACKET / "fold1_portable_inputs.json"
    verify_pinned(
        registry_path,
        "2dcd5cee727d1299f1fdab86b71116dc4a70008160c9495be8bf6f5c00b9dc67",
        "portable input registry",
    )
    stage = {
        "schema": "wnn.kaggle-h512-b16-prediction-root-stage.v1",
        "status": "STAGED_NOT_EXPORTED",
        "recovery_root": str(root),
        "data_root": str(source),
        "registry_path": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "raw_sources": raw_sources,
        "linked_inputs": linked,
        "structural_caches": {
            name: {"path": str((cache / name).resolve(strict=True)), "sha256": sha256(cache / name)}
            for name in ("chains.npz", "chain_folds.npz")
        },
    }
    receipt_path = root / STAGE_RECEIPT_NAME
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(stage, indent=2, sort_keys=True) + "\n")
    return stage


def main() -> None:
    """Verify the frozen predictor, derive SWA or stage the prediction inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    derive = commands.add_parser("derive", help="validate and derive SWA345 from saved CP3/4/5")
    derive.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    derive.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    derive.add_argument("--project-root", type=Path, default=ROOT)
    stage = commands.add_parser(
        "stage", help="link source data and graph caches for predict_run.py"
    )
    stage.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    stage.add_argument("--data-root", type=Path, default=ROOT)
    source = commands.add_parser(
        "verify-source", help="verify the clean frozen predictor checkout and dependency lock"
    )
    source.add_argument("--inference-source-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "derive":
        receipt = derive_swa(
            args.artifact_root, args.recovery_root, args.project_root.resolve(strict=True)
        )
    elif args.command == "stage":
        receipt = {"staged_inputs": stage_prediction_root(args.recovery_root, args.data_root)}
    else:
        receipt = {"inference_source": validate_inference_source(args.inference_source_root)}
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
