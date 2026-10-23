"""Grade the frozen five-endpoint F1 panel using separate offline acceptance."""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from fit_audit import validate_fit_artifacts
from recovery_contract import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_RECOVERY_ROOT,
    FIT_NAME,
    GRADE_PACKET,
    PACKET,
    ROOT,
    SWA_NAME,
    load_frozen_inputs,
    read_json,
    require,
    sha256,
    validate_inference_source,
    verify_pinned,
)

ACCEPTANCE_NAME = "B16_OFFLINE_RECOVERY_ACCEPTANCE_R1.json"
RECOVERY_RECEIPT_NAME = "B16_OFFLINE_RECOVERY_RECEIPT_R1.json"
STAGE_RECEIPT_NAME = "B16_OFFLINE_RECOVERY_STAGE_R1.json"
GRADE_NAME = "F1_FIXED_SCREEN_GRADE_OFFLINE_R1.json"


def _pinned_grade_module(project_root: Path):
    """Load the frozen grader only for its unchanged official five-endpoint score panel."""
    grade_dir = project_root / GRADE_PACKET
    verify_pinned(
        grade_dir / "f1_grade_contract.py",
        "87330be5429dca896365d786720c48252618f251d6984b194cbde67ecf085c90",
        "frozen F1 grade contract",
    )
    verify_pinned(
        grade_dir / "f1_grade_outputs.py",
        "a94c34b0e7acd1bb7270ea6def5843316b98d6a458573b4c2d22944cd56ad934",
        "frozen F1 grade output checks",
    )
    if str(grade_dir) not in sys.path:
        sys.path.insert(0, str(grade_dir))
    path = verify_pinned(
        grade_dir / "grade_h512_f1_screen.py",
        "7e2bb8ebed36b693f935d7247c455f4340b2acff735234ae28998a9491ed33be",
        "frozen five-endpoint scorer",
    )
    spec = importlib.util.spec_from_file_location("frozen_h512_f1_grade", path)
    require(spec is not None and spec.loader is not None, "Could not import the frozen score panel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tracked_inputs(
    project_root: Path,
    recovery_root: Path,
    plan: dict[str, Any],
    acceptance_path: Path,
    inference_source: dict[str, Any],
) -> dict[str, Path]:
    """Resolve all frozen and recovered files whose bytes must stay stable during scoring."""
    proposal, _, _, _ = load_frozen_inputs(project_root)
    del proposal
    tracked: dict[str, Path] = {
        "plan": project_root / GRADE_PACKET / "F1_FIXED_SCREEN_R2.json",
        "acceptance": acceptance_path,
        "recovery_receipt": recovery_root / RECOVERY_RECEIPT_NAME,
        "stage_receipt": recovery_root / STAGE_RECEIPT_NAME,
        "worker_terminal": project_root / PACKET / "B16_WORKER_TERMINAL_R3.json",
        "collection_receipt": project_root / PACKET / "B16_COLLECTION_R3.json",
        "canonical_manifest": project_root
        / "research/campaign_2026-09-29/artifacts/splits/fold1.json",
        "derived_manifest": project_root / PACKET / "fold1_profile_view.json",
        "portable_registry": project_root / PACKET / "fold1_portable_inputs.json",
        "proposal": project_root
        / PACKET
        / "H512_F1_B16_CONDITIONAL_PROFILE_FIT_PROPOSAL_20261021_R2.json",
        "targets": project_root / plan["targets"]["path"],
        "oldz4_prediction": project_root / plan["baseline"]["prediction"]["path"],
        "oldz4_sidecar": project_root / plan["baseline"]["sidecar"]["path"],
        "oldz4_checkpoint": project_root / plan["baseline"]["checkpoint"]["path"],
        "oldz4_provenance": project_root / plan["baseline"]["provenance"]["path"],
    }
    tracked["targets"] = verify_pinned(
        project_root / plan["targets"]["path"], plan["targets"]["sha256"], "official target cache"
    )
    baseline = plan["baseline"]
    for key in ("prediction", "sidecar", "checkpoint", "provenance"):
        tracked[f"oldz4_{key}"] = verify_pinned(
            project_root / baseline[key]["path"],
            baseline[key]["sha256"],
            f"frozen oldZ4 {key}",
        )
    for epoch in (3, 4, 5):
        tracked[f"fit_cp{epoch}"] = (
            recovery_root / "artifacts/runs" / FIT_NAME / f"model_ep{epoch}.pt"
        )
    tracked.update(
        {
            "fit_provenance": recovery_root / "artifacts/runs" / FIT_NAME / "provenance.json",
            "fit_history": recovery_root / "artifacts/runs" / FIT_NAME / "history.json",
            "swa_checkpoint": recovery_root / "artifacts/runs" / SWA_NAME / "model_ep5.pt",
            "swa_model_link": recovery_root / "artifacts/runs" / SWA_NAME / "model.pt",
            "swa_provenance": recovery_root / "artifacts/runs" / SWA_NAME / "provenance.json",
            "swa_derivation": recovery_root
            / "artifacts/runs"
            / SWA_NAME
            / "weight_derivation_receipt.json",
            "cp5_prediction": recovery_root / "cache" / f"pred_{FIT_NAME}_valid.npy",
            "cp5_sidecar": recovery_root / "cache" / f"pred_{FIT_NAME}_valid.json",
            "swa345_prediction": recovery_root / "cache" / f"pred_{SWA_NAME}_valid.npy",
            "swa345_sidecar": recovery_root / "cache" / f"pred_{SWA_NAME}_valid.json",
            "recovery_contract": Path(__file__).with_name("recovery_contract.py"),
            "checkpoint_io": Path(__file__).with_name("checkpoint_io.py"),
            "fit_audit": Path(__file__).with_name("fit_audit.py"),
            "recovery_checks": Path(__file__).with_name("recovery_checks.py"),
            "recovery_script": Path(__file__).with_name("recover_h512_f1_b16_offline.py"),
            "acceptance_script": Path(__file__).with_name("accept_h512_f1_b16_offline.py"),
            "grade_script": Path(__file__).resolve(),
        }
    )
    tracked.update(
        {
            "inference_project_manifest": Path(inference_source["project_manifest_path"]),
            "inference_dependency_lock": Path(inference_source["dependency_lock_path"]),
            "inference_predictor": Path(inference_source["predictor_path"]),
            "inference_trainer_entrypoint": Path(inference_source["trainer_entrypoint_path"]),
        }
    )
    grade_dir = project_root / GRADE_PACKET
    tracked["frozen_grade_script"] = verify_pinned(
        grade_dir / "grade_h512_f1_screen.py",
        "7e2bb8ebed36b693f935d7247c455f4340b2acff735234ae28998a9491ed33be",
        "frozen F1 score-panel code",
    )
    tracked["frozen_grade_contract"] = verify_pinned(
        grade_dir / "f1_grade_contract.py",
        "87330be5429dca896365d786720c48252618f251d6984b194cbde67ecf085c90",
        "frozen F1 grade contract",
    )
    tracked["frozen_grade_outputs"] = verify_pinned(
        grade_dir / "f1_grade_outputs.py",
        "a94c34b0e7acd1bb7270ea6def5843316b98d6a458573b4c2d22944cd56ad934",
        "frozen F1 output checks",
    )
    for relative, digest in plan["scorer_sources"].items():
        tracked[f"scorer:{relative}"] = verify_pinned(
            project_root / relative, digest, f"official scorer source {relative}"
        )
    return tracked


def grade(
    artifact_root: Path,
    recovery_root: Path,
    inference_source_root: Path,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Score the fixed panel after validating separate recovered-artifact acceptance."""
    started = time.monotonic()
    output_root = recovery_root.expanduser().resolve(strict=True)
    acceptance_path = output_root / ACCEPTANCE_NAME
    acceptance = read_json(acceptance_path)
    code_dir = Path(__file__).resolve().parent
    code_files = (
        "recover_h512_f1_b16_offline.py",
        "recovery_contract.py",
        "fit_audit.py",
        "checkpoint_io.py",
        "accept_h512_f1_b16_offline.py",
        "recovery_checks.py",
        "grade_h512_f1_b16_offline.py",
    )
    require(
        acceptance.get("recovery_code_sha256")
        == {name: sha256(code_dir / name) for name in code_files},
        "Recovery source hashes differ from acceptance",
    )
    require(
        acceptance.get("schema") == "wnn.kaggle-h512-f1-offline-recovery-acceptance.v1",
        "Offline recovery acceptance schema differs",
    )
    require(
        acceptance.get("status") == "ACCEPTED_FOR_FIXED_GRADE_WITH_INITIAL_IDENTITY_LIMITATION",
        "Offline recovery was not accepted for grading",
    )
    require(
        acceptance.get("artifact_root") == str(output_root),
        "Acceptance names a different artifact root",
    )
    require(
        acceptance.get("initial_parameter_sha256") is None,
        "Acceptance contains an invented initialization hash",
    )
    require(
        acceptance.get("prediction_root_stage_receipt_sha256")
        == sha256(output_root / STAGE_RECEIPT_NAME),
        "Accepted prediction-root staging receipt changed",
    )
    inference_source = validate_inference_source(inference_source_root)
    require(
        acceptance.get("inference_source") == inference_source,
        "Accepted predictor source or dependency lock changed",
    )
    require(acceptance.get("grade_claim") is None, "Acceptance contains an ungraded result")
    recovery_receipt = read_json(output_root / RECOVERY_RECEIPT_NAME)
    require(
        acceptance.get("offline_recovery_receipt_sha256")
        == sha256(output_root / RECOVERY_RECEIPT_NAME),
        "Offline recovery receipt changed after acceptance",
    )
    require(
        recovery_receipt.get("recovery_code_sha256")
        == {name: sha256(code_dir / name) for name in code_files[:4]},
        "Recovery derivation source hashes differ from its receipt",
    )
    source_root = Path(acceptance["source_artifact_root"]).resolve(strict=True)
    require(
        source_root == artifact_root.expanduser().resolve(strict=True),
        "CLI source artifact root differs from accepted source",
    )
    provenance, _, source_fit_dir = validate_fit_artifacts(project_root, source_root)
    require(
        acceptance.get("fit_provenance_sha256")
        == sha256(output_root / "artifacts/runs" / FIT_NAME / "provenance.json"),
        "Accepted copied provenance hash changed",
    )
    require(
        acceptance.get("fit_provenance_sha256") == sha256(source_fit_dir / "provenance.json"),
        "Accepted source provenance hash changed",
    )
    _, derived, _, plan = load_frozen_inputs(project_root)
    require(
        plan.get("schema") == "wnn.kaggle-h512-f1-fixed-screen.v1",
        "Frozen F1 score-plan schema changed",
    )
    require(plan.get("status") == "FROZEN_SCREEN_NOT_GRADED", "Frozen F1 score-plan status changed")
    worker = read_json(
        verify_pinned(
            project_root / PACKET / "B16_WORKER_TERMINAL_R3.json",
            "9b94418f06652e78ae7cdd4c02d7113b568c1f7fbece40ff3db4899b1c0dd57f",
            "failed B16 worker receipt",
        )
    )
    require(
        worker.get("input_identity", {}).get("derived_manifest_sha256")
        == acceptance.get("execution_split_manifest_sha256"),
        "Worker and acceptance F1 manifest differ",
    )
    require(
        acceptance.get("optimizer_chunk_updates") == 44_760
        and provenance.get("cumulative_updates", {}).get("optimizer_completed_updates") == 44_760,
        "Accepted optimizer update count changed",
    )

    grade_module = _pinned_grade_module(project_root)
    from f1_grade_contract import check_manifests
    from f1_grade_contract import read_json as read_grade_json

    _, checked_derived = check_manifests(project_root, plan, worker)
    require(checked_derived == derived, "Frozen manifest check returned different F1 membership")
    fit_dir = output_root / "artifacts/runs" / FIT_NAME
    swa_dir = output_root / "artifacts/runs" / SWA_NAME
    for key, run, checkpoint in (
        ("cp5", FIT_NAME, fit_dir / "model_ep5.pt"),
        ("swa345", SWA_NAME, swa_dir / "model_ep5.pt"),
    ):
        record = acceptance.get("exports", {}).get(key, {})
        prediction_path = output_root / record.get("prediction_path", "")
        sidecar_path = output_root / record.get("sidecar_path", "")
        require(record.get("run_name") == run, f"Accepted {key} endpoint name changed")
        require(
            sha256(checkpoint) == record.get("checkpoint_sha256"),
            f"Accepted {key} checkpoint hash changed",
        )
        require(
            sha256(prediction_path) == record.get("prediction_sha256"),
            f"Accepted {key} prediction hash changed",
        )
        require(
            sha256(sidecar_path) == record.get("sidecar_sha256"),
            f"Accepted {key} sidecar hash changed",
        )
        metadata = read_grade_json(sidecar_path)
        require(
            metadata.get("run") == run
            and metadata.get("campaign_split_manifest_sha256")
            == acceptance.get("execution_split_manifest_sha256"),
            f"Accepted {key} sidecar identity changed",
        )
        require(
            metadata.get("row_groups") == derived["evaluation"]["valid"],
            f"Accepted {key} group order changed",
        )

    tracked = _tracked_inputs(project_root, output_root, plan, acceptance_path, inference_source)
    before = {name: {"path": str(path), "sha256": sha256(path)} for name, path in tracked.items()}
    predictions = {
        "cp5": np.load(tracked["cp5_prediction"], mmap_mode="r", allow_pickle=False),
        "swa345": np.load(tracked["swa345_prediction"], mmap_mode="r", allow_pickle=False),
    }
    comparison = grade_module.score_panel(project_root, plan, derived, predictions, tracked)
    after = {name: {"path": str(path), "sha256": sha256(path)} for name, path in tracked.items()}
    require(before == after, "A source or artifact changed during fixed-panel grading")
    result = {
        "schema": "wnn.kaggle-h512-f1-fixed-screen-grade-offline-recovery.v1",
        "status": "GRADED_NOT_PROMOTED",
        "plan_sha256": sha256(project_root / GRADE_PACKET / "F1_FIXED_SCREEN_R2.json"),
        "inference_source": inference_source,
        "offline_acceptance_sha256": sha256(acceptance_path),
        "recovery_receipt_sha256": sha256(output_root / RECOVERY_RECEIPT_NAME),
        "initial_parameter_sha256": None,
        "initialization_identity_status": acceptance["initialization_identity_status"],
        "initialization_limitation": acceptance["initialization_limitation"],
        "fit": {
            "run": FIT_NAME,
            "source_revision": acceptance["source_revision"],
            "seed": 1004,
            "membership": "frozen F1 fit train+valid; 457 component-disjoint evaluation groups",
            "optimizer_chunk_updates": 44_760,
            "cp5_sha256": acceptance["checkpoint_sha256"]["model_ep5.pt"],
            "swa_derivation_sha256": acceptance["swa_derivation_receipt_sha256"],
            "worker_quality_fit_claim": None,
        },
        "comparison": comparison,
        "baseline_interpretation": plan["baseline"]["interpretation"],
        "exposure_and_boundaries": plan["claim_boundaries"],
        "input_hashes_before_and_after": {"before": before, "after": after},
        "resources": {
            "wall_seconds": round(time.monotonic() - started, 3),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    output = output_root / GRADE_NAME
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    """Score the frozen plan into a create-only recovery-specific result file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    parser.add_argument("--inference-source-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = grade(
        args.artifact_root,
        args.recovery_root,
        args.inference_source_root,
        args.project_root.resolve(strict=True),
    )
    print(json.dumps({"status": result["status"], "grade": str(args.recovery_root / GRADE_NAME)}))


if __name__ == "__main__":
    main()
