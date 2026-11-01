"""Focused fail-closed checks for the W18-only final receipt contract."""

from __future__ import annotations

import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.package_final_w18 import (
    _LINEAGE_CHECKPOINTS,
    FINAL_FIT_POLICY,
    FINAL_FIT_SCHEMA,
    FINAL_W18_TEACHERS,
    TOTAL_SEQUENCES,
    TRAIN_SEQUENCES,
    VALID_SEQUENCES,
    W18_ENDPOINT,
    W18_FIT_PROCESS_RECEIPT,
    W18_LINEAGE_SCHEMA,
    W18_PLAN_SHA256,
    W18_RECEIPT_SCHEMA,
    W18_RUN,
    _fit_members_digest,
    _load_fit_process_receipt,
    _validate_checkpoint_cache_provenance,
    canonical_sha256,
    validate_final_w18_receipt,
)
from wnn.training.final_submit_bridge import SOURCE_VALIDATION_SHA256


def _receipt() -> dict[str, object]:
    groups = {
        "train": list(range(TRAIN_SEQUENCES)),
        "valid": list(range(VALID_SEQUENCES)),
    }
    fit_members_sha = _fit_members_digest(groups)
    fit = {
        "schema": FINAL_FIT_SCHEMA,
        "policy": FINAL_FIT_POLICY,
        "manifest_sha256": "1" * 64,
        "fit_members_sha256": fit_members_sha,
        "source_sha256": {"train": "2" * 64, "valid": "3" * 64},
        "source_validation_sha256": SOURCE_VALIDATION_SHA256,
        "fit_row_groups": groups,
        "independent_evaluation_members": [],
        "independent_score_claim": "forbidden",
        "post_fit_primary_valid_scoring": "forbidden",
    }
    lineage = [
        {
            "run": run,
            "checkpoint_file": checkpoint_file,
            "checkpoint_sha256": str(index % 10) * 64,
            "final_fit_manifest_sha256": fit["manifest_sha256"],
            "final_fit_members_sha256": fit_members_sha,
            "source_validation_sha256": SOURCE_VALIDATION_SHA256,
        }
        for index, (run, checkpoint_file) in enumerate(sorted(_LINEAGE_CHECKPOINTS.items()), start=1)
    ]
    lineage_by_run = {record["run"]: record for record in lineage}
    w18_checkpoint_sha = lineage_by_run[W18_RUN]["checkpoint_sha256"]
    teacher_recipe = [
        {
            "run": run,
            "family": family,
            "coefficient": coefficient,
            "epoch": epoch,
            "checkpoint_file": _LINEAGE_CHECKPOINTS[run],
            "checkpoint_sha256": lineage_by_run[run]["checkpoint_sha256"],
        }
        for run, family, coefficient, epoch in FINAL_W18_TEACHERS
    ]
    command = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=10s",
        "21590s",
        "/runtime/python",
        "/runtime/source/scripts/train_rnn.py",
        "--name",
        W18_RUN,
        "--epochs",
        "2",
        "--checkpoint-epoch",
        "1",
        "--train-source",
        "train+valid",
        "--train-seqs",
        str(TRAIN_SEQUENCES),
        "--final-fit-source-validation-sha256",
        SOURCE_VALIDATION_SHA256,
        "--include-heldout-valid",
        "--no-eval-during-fit",
    ]
    process = {
        "schema": "wnn.multistage-owned-process.v1",
        "stage": 1,
        "trial_number": 0,
        "mode": "final_submit_finite_chain",
        "command": command,
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
        "cwd": "/runtime/source",
        "output_dir": "/runtime/data/artifacts/runs/" + W18_RUN,
        "started_at_epoch": 1_798_800_000.0,
        "timeout_seconds": 21_600,
        "status": "completed",
        "exit_code": 0,
        "timed_out": False,
        "finished_at_epoch": 1_798_800_010.0,
    }
    return {
        "schema": W18_RECEIPT_SCHEMA,
        "final_fit": fit,
        "fit_sequences": TOTAL_SEQUENCES,
        "independent_evaluation_members": [],
        "independent_evaluation_sequences": 0,
        "independent_score_claim": "forbidden",
        "fixed_endpoint": dict(W18_ENDPOINT),
        "members": {
            "w18": {
                "run": W18_RUN,
                "checkpoint_sha256": w18_checkpoint_sha,
                "configuration_sha256": "a" * 64,
                "provenance_sha256": "b" * 64,
            }
        },
        "plan_sha256": W18_PLAN_SHA256,
        "fit_process_receipt": {
            "path": W18_FIT_PROCESS_RECEIPT,
            "sha256": "d" * 64,
            "record": process,
        },
        "teacher_recipe": teacher_recipe,
        "teacher_cache": {
            "schema": "wnn.final-w18-teacher-cache-binding.v1",
            "metadata_sha256": "c" * 64,
            "verified_files": {
                name: hashlib.sha256(name.encode()).hexdigest()
                for name in sorted(
                    {
                        "families_train",
                        "families_valid",
                        "main_source_train",
                        "main_source_valid",
                    }
                )
            },
        },
        "lineage_closure": {
            "schema": W18_LINEAGE_SCHEMA,
            "roles": lineage,
            "sha256": canonical_sha256(lineage),
        },
    }


def test_single_w18_receipt_accepts_exact_member_and_thirteen_run_closure() -> None:
    receipt = _receipt()
    assert validate_final_w18_receipt(receipt) == receipt
    assert set(receipt["members"]) == {"w18"}
    assert len(receipt["lineage_closure"]["roles"]) == 13


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt.update(schema="wnn.final-model-set-receipt.v1"),
        lambda receipt: receipt["members"].update(
            tiedlead_s5={"run": "tl3final_fttall_tlead_s5"}
        ),
        lambda receipt: receipt["members"].update(
            source4_lstm={"run": "final-lstm192x1"}
        ),
    ],
)
def test_full_mixture_or_inactive_member_receipt_is_rejected(mutation) -> None:
    receipt = deepcopy(_receipt())
    mutation(receipt)
    with pytest.raises(ValueError):
        validate_final_w18_receipt(receipt)


def test_missing_ancestor_is_rejected_even_after_rehashing_closure() -> None:
    receipt = deepcopy(_receipt())
    roles = receipt["lineage_closure"]["roles"]
    roles.pop()
    receipt["lineage_closure"]["sha256"] = canonical_sha256(roles)
    with pytest.raises(ValueError, match="lineage closure is malformed"):
        validate_final_w18_receipt(receipt)


@pytest.mark.parametrize(
    "field,value",
    [
        ("independent_evaluation_sequences", 1),
        ("independent_score_claim", "reported"),
        ("fixed_endpoint", {"configured_epochs": 2, "checkpoint_epoch": 0}),
    ],
)
def test_receipt_rejects_evaluation_claims_and_endpoint_drift(field: str, value: object) -> None:
    receipt = deepcopy(_receipt())
    receipt[field] = value
    with pytest.raises(ValueError):
        validate_final_w18_receipt(receipt)


def test_receipt_rejects_incomplete_all_corpus_membership() -> None:
    receipt = deepcopy(_receipt())
    receipt["final_fit"]["fit_row_groups"]["valid"].pop()
    with pytest.raises(ValueError, match="all 12,480 whole sequences"):
        validate_final_w18_receipt(receipt)


@pytest.mark.parametrize(
    "field,value",
    [("status", "failed"), ("exit_code", 1), ("timed_out", True)],
)
def test_receipt_rejects_uncompleted_owned_fit_process(field: str, value: object) -> None:
    receipt = deepcopy(_receipt())
    receipt["fit_process_receipt"]["record"][field] = value
    with pytest.raises(ValueError, match="successful completion"):
        validate_final_w18_receipt(receipt)


def test_fit_process_loader_binds_plan_command_and_successful_child_exit() -> None:
    process = deepcopy(_receipt()["fit_process_receipt"]["record"])
    command = list(process["command"])
    planned_args = command[6:]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        output = root / "data" / "artifacts" / "runs" / W18_RUN
        process["cwd"] = str(source.resolve())
        process["output_dir"] = str(output.resolve())
        command[4] = str((root / "python").resolve())
        command[5] = str((source / "scripts" / "train_rnn.py").resolve())
        process["command"] = command
        process["command_sha256"] = hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest()
        receipt_path = root / "fit-process.json"
        receipt_path.write_text(json.dumps(process), encoding="utf-8")
        binding, exact_bytes = _load_fit_process_receipt(
            receipt_path,
            checkpoint_path=output / "model_ep1.pt",
            planned_args=planned_args,
        )
        assert binding["sha256"] == hashlib.sha256(exact_bytes).hexdigest()
        assert binding["record"] == process

        process["timed_out"] = True
        receipt_path.write_text(json.dumps(process), encoding="utf-8")
        with pytest.raises(ValueError, match="successful owned-process receipt"):
            _load_fit_process_receipt(
                receipt_path,
                checkpoint_path=output / "model_ep1.pt",
                planned_args=planned_args,
            )


def test_checkpoint_cache_provenance_joins_exact_validated_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "teacher-cache"
    cache_root.mkdir()
    metadata = {
        "source_coefficients": [
            {"run": "teacher-a", "coefficient": 0.8125},
        ],
        "coefficient_sum": 0.8125,
        "homogeneous_family": "strong",
        "family_order": ["strong", "full_finetuned", "raw_wmse", "samefold_baseline"],
        "epoch_convention": {"kind": "fixed"},
    }
    cache_result = {
        "metadata": metadata,
        "metadata_sha256": "a" * 64,
        "verified_files": {"main_source_train": "b" * 64},
    }
    checkpoint_cache = {
        "root": str(cache_root.resolve()),
        "variant": "source",
        "main_target_scale": 1.0,
        "main_target_scale_applied_before_training_clip": True,
        "metadata_sha256": cache_result["metadata_sha256"],
        "verified_files": cache_result["verified_files"],
        **metadata,
    }
    expected_args = {"distill_cache_variant": "source", "distill_cache_scale": 1.0}

    _validate_checkpoint_cache_provenance(
        checkpoint_cache,
        expected_args=expected_args,
        cache_root=cache_root,
        cache_result=cache_result,
    )

    for field, wrong_value in (
        ("root", str(tmp_path.resolve())),
        ("variant", "other"),
        ("metadata_sha256", "c" * 64),
        ("verified_files", {"main_source_train": "d" * 64}),
        ("source_coefficients", [{"run": "replacement", "coefficient": 0.8125}]),
        ("homogeneous_family", "full_finetuned"),
    ):
        altered = deepcopy(checkpoint_cache)
        altered[field] = wrong_value
        with pytest.raises(ValueError):
            _validate_checkpoint_cache_provenance(
                altered,
                expected_args=expected_args,
                cache_root=cache_root,
                cache_result=cache_result,
            )

