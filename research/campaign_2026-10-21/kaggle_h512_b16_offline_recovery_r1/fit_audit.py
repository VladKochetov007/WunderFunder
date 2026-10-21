"""Audit saved B16 fit provenance, membership, updates and checkpoint bytes."""

from __future__ import annotations

import json
from pathlib import Path

from recovery_contract import (
    DERIVED_SHA,
    EXPECTED_CHECKPOINTS,
    EXPECTED_NORMALIZATION,
    FIT_NAME,
    RUN_SOURCE_SHA,
    SOURCE_COMMIT,
    JsonObject,
    _verify_source_commit,
    digest_int64,
    load_frozen_inputs,
    read_json,
    require,
    sha256,
    validate_worker_receipts,
)


def _check_recipe(provenance: JsonObject, project_root: Path) -> None:
    """Validate cold-start arguments and the exact clean trainer source closure."""
    args = provenance.get("args", {})
    expected = {
        "name": FIT_NAME,
        "seed": 1004,
        "hidden": 512,
        "layers": 1,
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "adapter_mode": "conditional",
        "adapter_rank": 8,
        "valid_fold": 1,
        "batch": 16,
        "chunk": 2_000,
        "epochs": 6,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 5,
        "precision": "fp32",
        "loss": "wmse",
        "loss_norm": "chunk",
        "optim": "adamw",
        "sched": "onecycle",
        "lr": 0.0001,
        "wd": 0.01,
        "clip": 1.0,
        "warmup_frac": 0.05,
        "train_source": "train+valid",
        "train_seqs": 10_607,
        "full_valid": True,
        "eval_source": "vfold",
        "n_valid_folds": 4,
        "valid_repeat": 1,
        "valid_train_fraction": 1.0,
        "no_eval_during_fit": True,
        "dropout": 0.0,
        "fused_adam": True,
        "raw_only": False,
        "research_features": "none",
        "research_feature_control": "none",
        "resume_state": None,
        "init": "",
        "init_qmc": False,
        "distill": "",
        "propensity": "",
        "prop_input": "",
        "privileged_mode": "none",
        "final_distill_initializer": False,
        "final_model_adaptation": False,
        "aux": "",
        "aux_next": 0.0,
        "aux_sign": 0.0,
        "aux_quote": 0.0,
        "aux_struct": 0.0,
        "aux_tier": 0.0,
        "split_manifest_sha256": DERIVED_SHA,
    }
    require(
        all(args.get(key) == value for key, value in expected.items()),
        "Saved arguments differ from the frozen cold B16 recipe",
    )
    require(
        provenance.get("seed") == 1004 and provenance.get("selected_epoch") == 5,
        "Saved fit seed or selected epoch changed",
    )
    require(
        provenance.get("split_manifest_sha256") == DERIVED_SHA,
        "Fit provenance is not bound to the F1 execution view",
    )
    require(
        provenance.get("initial_checkpoint_sha256") is None,
        "Fit unexpectedly inherited an initializer",
    )
    require(
        "initial_parameter_sha256" not in provenance,
        "This recovery only applies to the source-schema identity gap",
    )
    require(
        provenance.get("run_identity_source_sha256") == RUN_SOURCE_SHA,
        "Trainer source identity differs from the observed run",
    )
    source_files = provenance.get("trainer_implementation", {}).get("files", {})
    require(
        isinstance(source_files, dict) and len(source_files) >= 50,
        "Trainer implementation closure is incomplete",
    )
    require(
        source_files.get("src/wnn/training/setup_model_runtime.py")
        == "8bee93409ccb00e629e8b2e9340e0a80b703e36c4ee687d853f28e375a1fe982",
        "Canonical runtime setup hash changed",
    )
    _verify_source_commit(project_root, source_files)


def _check_membership(provenance: JsonObject, derived: JsonObject) -> None:
    """Require exact source-qualified fit/evaluation sets and no component overlap."""
    fit_groups = provenance.get("fit_row_groups", {})
    train = list(map(int, fit_groups.get("train", [])))
    valid = list(map(int, fit_groups.get("valid", [])))
    evaluation = list(map(int, provenance.get("evaluation_row_groups", [])))
    require(
        len(train) == 10_533 and len(set(train)) == len(train),
        "Training membership is incomplete or duplicated",
    )
    require(
        len(valid) == 1_416 and len(set(valid)) == len(valid),
        "Validation-fit membership is incomplete or duplicated",
    )
    require(
        len(evaluation) == 457 and len(set(evaluation)) == len(evaluation),
        "Evaluation membership is incomplete or duplicated",
    )
    frozen_fit = derived.get("fit", {})
    frozen_eval = derived.get("evaluation", {}).get("valid", [])
    require(
        set(train) == set(map(int, frozen_fit.get("train", []))),
        "Training groups differ from frozen F1 membership",
    )
    require(
        set(valid) == set(map(int, frozen_fit.get("valid", []))),
        "Validation-fit groups differ from frozen F1 membership",
    )
    require(
        set(evaluation) == set(map(int, frozen_eval)),
        "Evaluation groups differ from frozen F1 membership",
    )
    identity = provenance.get("fit_membership_identity", {})
    require(
        identity.get("dataset_length") == 11_949 and identity.get("valid_repeat") == 1,
        "Fit membership identity changed",
    )
    require(
        identity.get("train")
        == {"count": len(train), "ordered_int64_le_sha256": digest_int64(train)},
        "Training membership digest is invalid",
    )
    require(
        identity.get("valid")
        == {"count": len(valid), "ordered_int64_le_sha256": digest_int64(valid)},
        "Validation-fit membership digest is invalid",
    )
    component_of = {
        (split, int(item["row_group"])): int(item["component"])
        for split, records in derived["sequences"].items()
        for item in records
    }
    fitted = {component_of[("train", group)] for group in train}
    fitted.update(component_of[("valid", group)] for group in valid)
    evaluated = {component_of[("valid", group)] for group in evaluation}
    require(
        not fitted.intersection(evaluated), "Fit rows share a connected component with evaluation"
    )


def _check_updates(provenance: JsonObject) -> list[JsonObject]:
    """Check the six completed epochs, zero skipped updates and invariant normalization."""
    normalization = provenance.get("normalization_identity")
    require(
        normalization == EXPECTED_NORMALIZATION,
        "Input normalization differs from the recorded 211-input fit",
    )
    cumulative = provenance.get("cumulative_updates", {})
    expected = {
        "completed_batches": 4_476,
        "optimizer_attempts": 44_760,
        "optimizer_completed_updates": 44_760,
        "amp_skipped_updates": 0,
    }
    require(
        all(cumulative.get(key) == value for key, value in expected.items()),
        "Cumulative B16 update counts changed",
    )
    diagnostics = provenance.get("update_diagnostics", [])
    require(len(diagnostics) == 6, "Fit provenance lacks six epoch diagnostics")
    for epoch, record in enumerate(diagnostics):
        counts = record.get("epoch", {})
        require(record.get("epoch_index") == epoch, "Per-epoch diagnostic order changed")
        require(
            record.get("normalization_identity") == normalization,
            "Normalization changed during fitting",
        )
        require(counts.get("completed_batches") == 746, "Per-epoch batch count changed")
        require(counts.get("optimizer_attempts") == 7_460, "Per-epoch attempted updates changed")
        require(
            counts.get("optimizer_completed_updates") == 7_460,
            "Per-epoch completed updates changed",
        )
        require(counts.get("amp_skipped_updates") == 0, "Per-epoch skipped updates changed")
    return diagnostics


def _check_checkpoints(fit_dir: Path, provenance: JsonObject) -> list[Path]:
    """Verify immutable CP3/4/5 sizes and digests against provenance and collection pins."""
    paths = []
    for epoch, (size, digest) in EXPECTED_CHECKPOINTS.items():
        path = fit_dir / f"model_ep{epoch}.pt"
        require(path.stat().st_size == size, f"CP{epoch} byte size differs from collected artifact")
        require(sha256(path) == digest, f"CP{epoch} bytes differ from collected artifact")
        require(
            provenance.get("checkpoint_sha256", {}).get(f"model_ep{epoch}.pt") == digest,
            f"CP{epoch} provenance hash differs",
        )
        paths.append(path)
    return paths


def validate_fit_artifacts(
    project_root: Path, artifact_root: Path
) -> tuple[JsonObject, list[JsonObject], Path]:
    """Validate the completed cold fit without reading target arrays or scoring."""
    proposal, derived, _, _ = load_frozen_inputs(project_root)
    worker = validate_worker_receipts(project_root)
    require(
        proposal.get("lineage", {}).get("execution_source_revision_requested") == SOURCE_COMMIT,
        "Proposal trainer revision changed",
    )
    fit_dir = artifact_root / "artifacts/runs" / FIT_NAME
    provenance_path = fit_dir / "provenance.json"
    history_path = fit_dir / "history.json"
    provenance = read_json(provenance_path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    require(isinstance(history, list) and len(history) == 6, "Expected six saved epoch records")
    require([item.get("epoch") for item in history] == list(range(6)), "Saved epoch order changed")
    require(
        all(item.get("evaluation") == "withheld" for item in history),
        "Fit history evaluated labels",
    )
    _check_recipe(provenance, project_root)
    _check_membership(provenance, derived)
    diagnostics = _check_updates(provenance)
    checkpoint_paths = _check_checkpoints(fit_dir, provenance)
    require(
        "initial_parameter_sha256" not in provenance,
        "Initial tensor identity must remain unavailable",
    )
    require(
        provenance_path.is_file() and history_path.is_file() and len(checkpoint_paths) == 3,
        "Fit source artifacts are incomplete",
    )
    _check_worker_sources(worker)
    return provenance, diagnostics, fit_dir


def _check_worker_sources(worker: JsonObject) -> None:
    """Confirm raw train/valid identities captured by canonical input preflight."""
    sources = worker.get("input_identity", {}).get("raw_sources", {})
    require(
        sources.get("datasets/train.parquet", {}).get("sha256")
        == "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
        "Training raw source identity changed",
    )
    require(
        sources.get("datasets/valid.parquet", {}).get("sha256")
        == "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
        "Validation raw source identity changed",
    )
    require(
        sources.get("datasets/train.parquet", {}).get("bytes") == 29_240_219_011,
        "Training raw source size changed",
    )
    require(
        sources.get("datasets/valid.parquet", {}).get("bytes") == 5_615_509_128,
        "Validation raw source size changed",
    )
