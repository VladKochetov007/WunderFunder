"""Accept recovered checkpoints and canonical feature-only exports for fixed grading."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fit_audit import validate_fit_artifacts
from recovery_checks import validate_export, verify_mean
from recovery_contract import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_RECOVERY_ROOT,
    DERIVED_SHA,
    EXPECTED_CHECKPOINTS,
    FIT_NAME,
    INITIALIZATION_LIMITATION,
    INITIALIZATION_STATUS,
    PACKET,
    ROOT,
    SWA_NAME,
    read_json,
    require,
    safe_recovery_root,
    sha256,
    validate_inference_source,
    verify_pinned,
)

RECEIPT_NAME = "B16_OFFLINE_RECOVERY_RECEIPT_R1.json"
ACCEPTANCE_NAME = "B16_OFFLINE_RECOVERY_ACCEPTANCE_R1.json"
STAGE_RECEIPT_NAME = "B16_OFFLINE_RECOVERY_STAGE_R1.json"


def accept(
    artifact_root: Path,
    recovery_root: Path,
    inference_source_root: Path,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Write a create-only acceptance receipt after source and export verification."""
    source_root = artifact_root.expanduser().resolve(strict=True)
    output_root = safe_recovery_root(recovery_root)
    provenance, diagnostics, source_fit_dir = validate_fit_artifacts(project_root, source_root)
    require((output_root / RECEIPT_NAME).is_file(), "Offline derivation receipt is missing")
    recovery = read_json(output_root / RECEIPT_NAME)
    require(
        recovery.get("schema") == "wnn.kaggle-h512-b16-offline-recovery.v1",
        "Recovery receipt schema differs",
    )
    require(
        recovery.get("status") == "OFFLINE_ARTIFACT_RECOVERY_COMPLETE_NOT_GRADED",
        "Recovery receipt is not complete",
    )
    require(
        recovery.get("artifact_root") == str(source_root),
        "Recovery receipt names a different input artifact root",
    )
    require(
        recovery.get("recovery_root") == str(output_root),
        "Recovery receipt names a different output root",
    )
    require(
        recovery.get("fit_provenance_sha256") == sha256(source_fit_dir / "provenance.json"),
        "Recovery receipt fit provenance changed",
    )
    require(
        recovery.get("source_commit") == "68ff682977b822f38351aac0f621146f5f13b440",
        "Recovery source revision changed",
    )
    code_dir = Path(__file__).resolve().parent
    recovery_code = recovery.get("recovery_code_sha256", {})
    require(
        recovery_code
        == {
            name: sha256(code_dir / name)
            for name in (
                "recover_h512_f1_b16_offline.py",
                "recovery_contract.py",
                "fit_audit.py",
                "checkpoint_io.py",
            )
        },
        "Recovery derivation source hashes changed",
    )
    require(
        recovery.get("initial_parameter_sha256") is None,
        "Recovery receipt contains an invented initialization hash",
    )
    require(
        recovery.get("initialization_identity_status") == INITIALIZATION_STATUS,
        "Recovery identity limitation changed",
    )
    stage_path = output_root / STAGE_RECEIPT_NAME
    stage = read_json(stage_path)
    require(
        stage.get("schema") == "wnn.kaggle-h512-b16-prediction-root-stage.v1",
        "Prediction-root stage receipt schema differs",
    )
    require(stage.get("status") == "STAGED_NOT_EXPORTED", "Prediction-root staging is not complete")
    for epoch, (_, digest) in EXPECTED_CHECKPOINTS.items():
        path = output_root / "artifacts/runs" / FIT_NAME / f"model_ep{epoch}.pt"
        require(sha256(path) == digest, f"Recovered CP{epoch} copy differs from collection")
    fit_provenance_path = output_root / "artifacts/runs" / FIT_NAME / "provenance.json"
    fit_history_path = output_root / "artifacts/runs" / FIT_NAME / "history.json"
    require(
        sha256(fit_provenance_path) == sha256(source_fit_dir / "provenance.json"),
        "Copied provenance differs from the source fit",
    )
    require(
        sha256(fit_history_path) == sha256(source_fit_dir / "history.json"),
        "Copied history differs from the source fit",
    )
    swa_derivation_path = (
        output_root / "artifacts/runs" / SWA_NAME / "weight_derivation_receipt.json"
    )
    swa_derivation = read_json(swa_derivation_path)
    require(
        sha256(swa_derivation_path) == recovery.get("swa_derivation_receipt_sha256"),
        "SWA derivation receipt changed",
    )
    expected_sources = [
        {
            "epoch": epoch,
            "path": f"artifacts/runs/{FIT_NAME}/model_ep{epoch}.pt",
            "sha256": digest,
        }
        for epoch, (_, digest) in EXPECTED_CHECKPOINTS.items()
    ]
    require(
        swa_derivation.get("method") == "equal_parameter_mean_epochs_3_4_5",
        "SWA method differs from the frozen trajectory mean",
    )
    require(
        swa_derivation.get("source_checkpoints") == expected_sources,
        "SWA source checkpoint identities differ",
    )
    output_paths = {
        "model_ep5.pt": f"artifacts/runs/{SWA_NAME}/model_ep5.pt",
        "provenance.json": f"artifacts/runs/{SWA_NAME}/provenance.json",
        "model.pt": f"artifacts/runs/{SWA_NAME}/model.pt",
    }
    output_records = swa_derivation.get("outputs", {})
    require(set(output_records) == set(output_paths), "SWA derivation output map is incomplete")
    for name, relative in output_paths.items():
        path = output_root / relative
        record = output_records[name]
        require(record.get("path") == relative, f"SWA output path differs: {name}")
        require(record.get("sha256") == sha256(path), f"SWA output hash differs: {name}")
        require(record.get("size_bytes") == path.stat().st_size, f"SWA output size differs: {name}")
    require(
        swa_derivation.get("training_run") is False
        and swa_derivation.get("quality_evaluated") is False,
        "SWA derivation makes a fit or score claim",
    )
    swa_model = verify_mean(project_root, source_fit_dir, output_root)
    swa_provenance_path = output_root / "artifacts/runs" / SWA_NAME / "provenance.json"
    swa_provenance = read_json(swa_provenance_path)
    require(
        swa_provenance.get("args") == provenance.get("args"),
        "SWA trainer arguments differ from CP5",
    )
    require(
        swa_provenance.get("split_manifest_sha256") == DERIVED_SHA, "SWA manifest ancestry differs"
    )
    require(
        "initial_parameter_sha256" not in swa_provenance,
        "SWA provenance contains a reconstructed initialization identity",
    )
    require(
        swa_provenance.get("weight_derivation")
        == {key: value for key, value in swa_derivation.items() if key != "outputs"},
        "SWA provenance derivation differs from its standalone receipt",
    )
    require(
        swa_derivation.get("initial_parameter_sha256") is None,
        "SWA receipt contains an invented initialization hash",
    )

    cp5_path = output_root / "artifacts/runs" / FIT_NAME / "model_ep5.pt"
    exports = {
        "cp5": validate_export(output_root, project_root, FIT_NAME, cp5_path),
        "swa345": validate_export(
            output_root,
            project_root,
            SWA_NAME,
            output_root / "artifacts/runs" / SWA_NAME / "model_ep5.pt",
        ),
    }
    portable_path = verify_pinned(
        project_root / PACKET / "fold1_portable_inputs.json",
        "2dcd5cee727d1299f1fdab86b71116dc4a70008160c9495be8bf6f5c00b9dc67",
        "portable input registry",
    )
    require(
        stage.get("registry_path") == str(portable_path), "Staged portable registry path differs"
    )
    require(
        stage.get("registry_sha256") == sha256(portable_path),
        "Staged portable registry hash differs",
    )
    inference_source = validate_inference_source(inference_source_root)
    require(
        (output_root / "datasets").is_symlink()
        and (output_root / "datasets").resolve(strict=True)
        == Path(stage["linked_inputs"]["datasets"]).resolve(strict=True),
        "Staged raw-data mount changed",
    )
    require(
        os.environ.get("WNN_PROJECT_ROOT") == str(output_root),
        "Acceptance must use the staged prediction root",
    )
    require(
        os.environ.get("WNN_PORTABLE_DATA_MANIFEST") == str(portable_path),
        "Acceptance must use the pinned portable registry",
    )
    raw_expected = {
        "datasets/train.parquet": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
        "datasets/valid.parquet": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
    }
    for relative, digest in raw_expected.items():
        require(
            stage.get("raw_sources", {}).get(relative, {}).get("expected_sha256") == digest,
            f"Staged source identity differs: {relative}",
        )
    cache_expected = {
        "chains.npz": "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766",
        "chain_folds.npz": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f",
    }
    for name, digest in cache_expected.items():
        cache_path = output_root / "cache" / name
        require(cache_path.is_symlink(), f"Staged structural cache is not a symlink: {name}")
        require(sha256(cache_path) == digest, f"Staged structural cache bytes changed: {name}")
        require(
            stage.get("structural_caches", {}).get(name, {}).get("sha256") == digest,
            f"Staged structural cache differs: {name}",
        )
    acceptance = {
        "schema": "wnn.kaggle-h512-f1-offline-recovery-acceptance.v1",
        "status": "ACCEPTED_FOR_FIXED_GRADE_WITH_INITIAL_IDENTITY_LIMITATION",
        "artifact_root": str(output_root),
        "source_artifact_root": str(source_root),
        "worker_terminal_path": (PACKET / "B16_WORKER_TERMINAL_R3.json").as_posix(),
        "worker_terminal_sha256": sha256(project_root / PACKET / "B16_WORKER_TERMINAL_R3.json"),
        "collection_receipt_path": (PACKET / "B16_COLLECTION_R3.json").as_posix(),
        "collection_receipt_sha256": sha256(project_root / PACKET / "B16_COLLECTION_R3.json"),
        "offline_recovery_receipt_path": RECEIPT_NAME,
        "offline_recovery_receipt_sha256": sha256(output_root / RECEIPT_NAME),
        "prediction_root_stage_receipt_sha256": sha256(stage_path),
        "fit_run": FIT_NAME,
        "fit_provenance_sha256": sha256(fit_provenance_path),
        "fit_history_sha256": sha256(fit_history_path),
        "checkpoint_sha256": {
            f"model_ep{epoch}.pt": digest for epoch, (_, digest) in EXPECTED_CHECKPOINTS.items()
        },
        "swa_derivation_receipt_path": swa_derivation_path.relative_to(output_root).as_posix(),
        "swa_derivation_receipt_sha256": sha256(swa_derivation_path),
        "swa_checkpoint_sha256": swa_model["sha256"],
        "swa_provenance_sha256": sha256(swa_provenance_path),
        "exports": exports,
        "original_split_manifest_sha256": "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae",
        "execution_split_manifest_sha256": DERIVED_SHA,
        "portable_registry_path": str(portable_path),
        "portable_registry_sha256": sha256(portable_path),
        "source_revision": "68ff682977b822f38351aac0f621146f5f13b440",
        "inference_source": inference_source,
        "recovery_code_sha256": {
            name: sha256(code_dir / name)
            for name in (
                "recover_h512_f1_b16_offline.py",
                "recovery_contract.py",
                "fit_audit.py",
                "checkpoint_io.py",
                "accept_h512_f1_b16_offline.py",
                "recovery_checks.py",
                "grade_h512_f1_b16_offline.py",
            )
        },
        "normalization_identity": provenance["normalization_identity"],
        "optimizer_chunk_updates": provenance["cumulative_updates"]["optimizer_completed_updates"],
        "epochs": len(diagnostics),
        "initial_parameter_sha256": None,
        "initialization_identity_status": INITIALIZATION_STATUS,
        "initialization_limitation": INITIALIZATION_LIMITATION,
        "quality_fit_claim": None,
        "grade_claim": None,
        "precision_evidence": {
            "fit_args_precision": provenance["args"]["precision"],
            "trainer_source_revision": "68ff682977b822f38351aac0f621146f5f13b440",
            "profile_tf32_override": "0",
            "profile_tf32_flags": {"matmul_tf32": False, "cudnn_tf32": False},
            "after_fit_runtime_flags": "not recorded because the worker exited at the missing initialization-identity postfit gate",
        },
    }
    acceptance_path = output_root / ACCEPTANCE_NAME
    with acceptance_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
    return acceptance


def main() -> None:
    """Validate recovered artifacts and feature exports without changing the frozen grader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    parser.add_argument("--inference-source-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = accept(
        args.artifact_root,
        args.recovery_root,
        args.inference_source_root,
        args.project_root.resolve(strict=True),
    )
    print(
        json.dumps(
            {"status": result["status"], "acceptance": str(args.recovery_root / ACCEPTANCE_NAME)}
        )
    )


if __name__ == "__main__":
    main()
