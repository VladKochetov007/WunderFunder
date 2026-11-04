"""Focused checks for the final TL3/LSTM tensor export boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.check_final_tl3_package import (
    EXPECTED_QUANTIZATION_CONTRACT_SHA256,
    EXPECTED_R11_NATIVE_V3_SHA256,
    EXPECTED_SERVING_HELPER_SHA256,
    LEGACY_MALFORMED_NATIVE_SHA256_DECLARATION,
    _is_positive_finite_epoch,
    _numeric_error_summary,
    _numeric_error_within_ceilings,
    _require_fields_with_optional_metadata,
)
from scripts.package_final_tl3 import (
    FINAL_FIXED_EPOCHS,
    FINAL_RECEIPT_SCHEMA,
    FINAL_RUNS,
    ROLES,
    _final_membership,
    _validate_final_checkpoint,
    canonical_sha256,
    load_final_model_receipt,
    validate_final_model_receipt,
    validate_serving_options,
    validate_tensor_layout,
)
from wnn.final_submit_manifest import FinalFitMembership
from wnn.training.final_submit_bridge import SOURCE_VALIDATION_SHA256, final_fit_identity

TRAIN_GROUPS = 10_607
VALID_GROUPS = 1_873


def test_frozen_quantization_sha256_pins_are_64_lowercase_hex() -> None:
    for value in (
        EXPECTED_QUANTIZATION_CONTRACT_SHA256,
        EXPECTED_R11_NATIVE_V3_SHA256,
        EXPECTED_SERVING_HELPER_SHA256,
    ):
        assert len(value) == 64
        assert set(value) <= set("0123456789abcdef")
    assert len(LEGACY_MALFORMED_NATIVE_SHA256_DECLARATION) == 65
    assert EXPECTED_R11_NATIVE_V3_SHA256 != LEGACY_MALFORMED_NATIVE_SHA256_DECLARATION


def test_frozen_json_epochs_accept_finite_positive_numbers_only() -> None:
    assert _is_positive_finite_epoch(1_798_876_800)
    assert _is_positive_finite_epoch(1_798_876_800.125)
    assert not _is_positive_finite_epoch(True)
    assert not _is_positive_finite_epoch(0)
    assert not _is_positive_finite_epoch(float("inf"))
    assert not _is_positive_finite_epoch(float("nan"))


def test_numeric_ceiling_schema_allows_only_whitelisted_informational_fields() -> None:
    _require_fields_with_optional_metadata(
        {"scientific": 1, "calibration_scope": "pinned"},
        {"scientific"},
        {"calibration_scope", "claim_policy"},
        "numeric ceilings",
    )
    with pytest.raises(ValueError, match="missing required or unsupported fields"):
        _require_fields_with_optional_metadata(
            {"scientific": 1, "unrecognized": 2},
            {"scientific"},
            {"calibration_scope", "claim_policy"},
            "numeric ceilings",
        )
    with pytest.raises(ValueError, match="missing required or unsupported fields"):
        _require_fields_with_optional_metadata(
            {"calibration_scope": "pinned"},
            {"scientific"},
            {"calibration_scope", "claim_policy"},
            "numeric ceilings",
        )


def test_serving_policy_precision_requirements_are_explicit() -> None:
    validate_serving_options("full_mix", "f32", False)
    validate_serving_options("full_mix", "q8vnni", True)
    validate_serving_options("w18_only", "q8vnni", True)
    with pytest.raises(ValueError, match="q8vnni and fast_activations"):
        validate_serving_options("w18_only", "f32", True)
    with pytest.raises(ValueError, match="q8vnni and fast_activations"):
        validate_serving_options("w18_only", "q8vnni", False)
    with pytest.raises(ValueError, match="serving policy"):
        validate_serving_options("silent_fallback", "q8vnni", True)


def test_final_q8_parity_keeps_allclose_diagnostic_separate_from_frozen_ceilings() -> None:
    reference = np.ones((3, 2), dtype=np.float32)
    predictions = reference + np.asarray([0.05, 0.06], dtype=np.float32)
    summary = _numeric_error_summary(predictions, reference, atol=0.02, rtol=0.02)
    ceilings = {
        "maximum_rms_by_target": [0.051, 0.061],
        "maximum_p99_abs_by_target": [0.051, 0.061],
        "maximum_abs_by_target": [0.051, 0.061],
    }
    assert summary["allclose"] is False
    assert _numeric_error_within_ceilings(summary, ceilings) is True
    ceilings["maximum_abs_by_target"] = [0.049, 0.061]
    assert _numeric_error_within_ceilings(summary, ceilings) is False


def _receipt() -> dict[str, object]:
    """Construct a structurally complete synthetic final receipt."""
    groups = {"train": list(range(TRAIN_GROUPS)), "valid": list(range(VALID_GROUPS))}
    fit_members = [[source, group] for source in ("train", "valid") for group in groups[source]]
    fit_sha = hashlib.sha256(
        json.dumps(fit_members, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    fit = {
        "schema": "wnn.final-fit-membership.v1",
        "policy": "all_official_train_and_valid_labels_fit_no_independent_score_v1",
        "manifest_sha256": "1" * 64,
        "fit_members_sha256": fit_sha,
        "source_sha256": {"train": "2" * 64, "valid": "3" * 64},
        "source_validation_sha256": SOURCE_VALIDATION_SHA256,
        "fit_row_groups": groups,
        "independent_evaluation_members": [],
        "independent_score_claim": "forbidden",
        "post_fit_primary_valid_scoring": "forbidden",
    }
    members = {
        role: {
            "run": run,
            "checkpoint_sha256": "4567"[index] * 64,
            "configuration_sha256": "5678"[index] * 64,
            "provenance_sha256": "6789"[index] * 64,
            **({"data_sha256": "789a"[index] * 64} if role == "source4_lstm" else {}),
        }
        for index, (role, run) in enumerate(zip(ROLES, FINAL_RUNS, strict=True))
    }
    return {
        "schema": FINAL_RECEIPT_SCHEMA,
        "final_fit": fit,
        "fit_sequences": TRAIN_GROUPS + VALID_GROUPS,
        "independent_evaluation_members": [],
        "independent_evaluation_sequences": 0,
        "independent_score_claim": "forbidden",
        "fixed_epochs": FINAL_FIXED_EPOCHS,
        "members": members,
    }


def _legacy_checkpoint(
    role: str, membership: FinalFitMembership
) -> tuple[dict[str, object], dict[str, str]]:
    """Build final-fit args/provenance for the frozen TL3 endpoint validator."""
    endpoint = FINAL_FIXED_EPOCHS[role]
    args: dict[str, object] = {
        "name": FINAL_RUNS[ROLES.index(role)],
        "arch": "gru",
        "cell": "gru",
        "hidden": 256,
        "layers": 2,
        "epochs": endpoint["configured_epochs"],
        "checkpoint_epoch": endpoint["checkpoint_epoch"],
        "checkpoint_policy": "fixed",
        "final_fit_manifest_sha256": membership.manifest_sha256,
        "final_fit_members_sha256": membership.fit_members_sha256,
        "final_fit_train_sha256": membership.train_sha256,
        "final_fit_valid_sha256": membership.valid_sha256,
        "final_fit_source_validation_sha256": membership.source_validation_sha256,
        "split_manifest": None,
        "split_manifest_sha256": None,
        "train_source": "train+valid",
        "include_heldout_valid": True,
        "no_eval_during_fit": True,
    }
    if role.startswith("tiedlead"):
        args["enc"] = "tiedlead"
    provenance: dict[str, object] = {
        "args": args,
        "final_fit": final_fit_identity(membership),
        "fit_row_groups": {
            "train": list(membership.train_groups),
            "valid": list(membership.valid_groups),
        },
        "evaluation_row_groups": [],
    }
    checkpoint = {"args": args, "provenance": provenance, "epoch": endpoint["checkpoint_epoch"]}
    member = {
        "run": FINAL_RUNS[ROLES.index(role)],
        "checkpoint_sha256": "4" * 64,
        "configuration_sha256": canonical_sha256(args),
        "provenance_sha256": canonical_sha256(provenance),
    }
    return checkpoint, member


def _state(role: str, *, gates: int | None = None) -> dict[str, np.ndarray]:
    """Build a small synthetic native-compatible state dictionary."""
    width = {"w18": 211, "tiedlead_s5": 263, "tiedlead_s6": 263, "source4_lstm": 112}[role]
    encoder = hidden = readout = 192 if role == "source4_lstm" else 256
    count = gates or (4 if role == "source4_lstm" else 3)
    state = {
        "mean": np.zeros(width, dtype=np.float32),
        "std": np.ones(width, dtype=np.float32),
        "inp.weight": np.zeros((encoder, width), dtype=np.float32),
        "inp.bias": np.zeros(encoder, dtype=np.float32),
        "gru.weight_ih_l0": np.zeros((count * hidden, encoder), dtype=np.float32),
        "gru.weight_hh_l0": np.zeros((count * hidden, hidden), dtype=np.float32),
        "gru.bias_ih_l0": np.zeros(count * hidden, dtype=np.float32),
        "gru.bias_hh_l0": np.zeros(count * hidden, dtype=np.float32),
        "head.0.weight": np.zeros((readout, hidden + encoder), dtype=np.float32),
        "head.0.bias": np.zeros(readout, dtype=np.float32),
        "head.2.weight": np.zeros((2, readout), dtype=np.float32),
        "head.2.bias": np.zeros(2, dtype=np.float32),
    }
    if role != "source4_lstm":
        state.update(
            {
                "gru.weight_ih_l1": np.zeros((count * hidden, hidden), dtype=np.float32),
                "gru.weight_hh_l1": np.zeros((count * hidden, hidden), dtype=np.float32),
                "gru.bias_ih_l1": np.zeros(count * hidden, dtype=np.float32),
                "gru.bias_hh_l1": np.zeros(count * hidden, dtype=np.float32),
            }
        )
    if role.startswith("tiedlead"):
        state.update(
            {
                "tied.0.weight": np.zeros((128, 104), dtype=np.float32),
                "tied.0.bias": np.zeros(128, dtype=np.float32),
                "tied.2.weight": np.zeros((64, 128), dtype=np.float32),
                "tied.2.bias": np.zeros(64, dtype=np.float32),
                "tied_out.weight": np.zeros((encoder, 128), dtype=np.float32),
                "tied_out.bias": np.zeros(encoder, dtype=np.float32),
                "lead_out.weight": np.zeros((encoder, 64), dtype=np.float32),
                "lead_out.bias": np.zeros(encoder, dtype=np.float32),
            }
        )
    return state


@pytest.mark.parametrize("role", ["w18", "tiedlead_s5", "tiedlead_s6", "source4_lstm"])
def test_fixed_member_shapes_are_admitted(role: str) -> None:
    layout = validate_tensor_layout(role, _state(role))
    assert (
        layout["input_width"]
        == {"w18": 211, "tiedlead_s5": 263, "tiedlead_s6": 263, "source4_lstm": 112}[role]
    )
    assert layout["output_count_before_trim"] == 2


def test_tiedlead_missing_nonlinear_branch_is_rejected() -> None:
    state = _state("tiedlead_s5")
    del state["lead_out.weight"]
    with pytest.raises(ValueError, match="lead_out.weight"):
        validate_tensor_layout("tiedlead_s5", state)


def test_unknown_tensor_is_not_silently_ignored() -> None:
    state = _state("w18")
    state["unsupported_encoder.weight"] = np.zeros((1,), dtype=np.float32)
    with pytest.raises(ValueError, match="unsupported model tensors"):
        validate_tensor_layout("w18", state)


def test_source4_hint_tensors_are_not_silently_ignored() -> None:
    state = _state("source4_lstm")
    state["hint_projector.linear.weight"] = np.zeros((3, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="unsupported model tensors"):
        validate_tensor_layout("source4_lstm", state)


def test_source4_requires_lstm_gates() -> None:
    with pytest.raises(ValueError, match="four-gate LSTM"):
        validate_tensor_layout("source4_lstm", _state("source4_lstm", gates=3))


def test_head_trim_shape_is_checked_after_export() -> None:
    state = _state("w18")
    state["head.2.weight"] = np.zeros((31, 256), dtype=np.float32)
    state["head.2.bias"] = np.zeros(31, dtype=np.float32)
    assert validate_tensor_layout("w18", state)["output_count_before_trim"] == 31


def test_final_receipt_binds_complete_membership_and_fixed_endpoints() -> None:
    receipt = _receipt()
    checked = validate_final_model_receipt(receipt, "a" * 64, Path("receipt.json"))
    membership = checked["membership"]
    assert len(membership.train_groups) == TRAIN_GROUPS
    assert len(membership.valid_groups) == VALID_GROUPS


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fit_sequences", 12_479, "12,480"),
        ("independent_evaluation_sequences", 1, "independent evaluation"),
        ("independent_score_claim", "available", "forbid"),
        ("fixed_epochs", {"w18": {"configured_epochs": 3}}, "fixed endpoint"),
    ],
)
def test_final_receipt_rejects_wrong_policy_or_endpoints(
    field: str, value: object, message: str
) -> None:
    receipt = _receipt()
    receipt[field] = value
    with pytest.raises(ValueError, match=message):
        validate_final_model_receipt(receipt, "a" * 64, Path("receipt.json"))


def test_final_receipt_rejects_incomplete_or_changed_membership() -> None:
    receipt = _receipt()
    receipt["final_fit"]["fit_row_groups"]["valid"].pop()
    with pytest.raises(ValueError, match="complete-fit identity"):
        validate_final_model_receipt(receipt, "a" * 64, Path("receipt.json"))

    receipt = _receipt()
    receipt["final_fit"]["fit_members_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="all 12,480 members"):
        validate_final_model_receipt(receipt, "a" * 64, Path("receipt.json"))


def test_final_receipt_rejects_unpinned_source_validation_sha() -> None:
    receipt = _receipt()
    receipt["final_fit"]["source_validation_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="pinned source-validation"):
        validate_final_model_receipt(receipt, "a" * 64, Path("receipt.json"))


def test_final_receipt_rejects_missing_or_extra_model_roles() -> None:
    receipt = _receipt()
    del receipt["members"]["tiedlead_s6"]
    with pytest.raises(ValueError, match="exactly the four"):
        validate_final_model_receipt(receipt, "a" * 64, Path("receipt.json"))


def test_final_receipt_rejects_malformed_role_digest() -> None:
    receipt = _receipt()
    receipt["members"]["w18"]["checkpoint_sha256"] = "bad"
    with pytest.raises(ValueError, match="malformed SHA256"):
        validate_final_model_receipt(receipt, "a" * 64, Path("receipt.json"))


def test_final_receipt_file_requires_exact_byte_sha256(tmp_path: Path) -> None:
    receipt_path = tmp_path / "approved.json"
    raw = json.dumps(_receipt(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt_path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    assert load_final_model_receipt(receipt_path, expected)["sha256"] == expected
    with pytest.raises(ValueError, match="bytes differ"):
        load_final_model_receipt(receipt_path, "f" * 64)


def test_final_legacy_checkpoint_requires_final_fit_args_and_provenance() -> None:
    membership = _final_membership(_receipt(), Path("receipt.json"))
    checkpoint, member = _legacy_checkpoint("w18", membership)
    _validate_final_checkpoint("w18", checkpoint, member, membership)
    checkpoint["provenance"]["evaluation_row_groups"] = [0]
    member["provenance_sha256"] = canonical_sha256(checkpoint["provenance"])
    with pytest.raises(ValueError, match="evaluation membership"):
        _validate_final_checkpoint("w18", checkpoint, member, membership)


def test_final_legacy_checkpoint_requires_receipt_bound_config() -> None:
    membership = _final_membership(_receipt(), Path("receipt.json"))
    checkpoint, member = _legacy_checkpoint("tiedlead_s5", membership)
    checkpoint["args"]["checkpoint_epoch"] = 2
    checkpoint["provenance"]["args"] = checkpoint["args"]
    member["configuration_sha256"] = canonical_sha256(checkpoint["args"])
    member["provenance_sha256"] = canonical_sha256(checkpoint["provenance"])
    with pytest.raises(ValueError, match="fixed configured final endpoint"):
        _validate_final_checkpoint("tiedlead_s5", checkpoint, member, membership)


def test_source4_final_checkpoint_requires_fitting_only_data_provenance() -> None:
    membership = _final_membership(_receipt(), Path("receipt.json"))
    fit_members = [
        [source, group]
        for source, groups in (
            ("train", membership.train_groups),
            ("valid", membership.valid_groups),
        )
        for group in groups
    ]
    config: dict[str, object] = {
        "model": {
            "architecture": "lstm",
            "hidden": 192,
            "layers": 1,
            "hint_dim": 0,
            "input_feature_mode": "raw112",
            "dropout": 0.0,
            "compile_blocks": False,
        },
        "epochs": 8,
        "batch_size": 256,
        "accumulate_chunks": 4,
        "chunk_rows": 256,
        "seed": 1302,
        "optimizer": "adamw",
        "lr": 0.0015008627097145002,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "warmup_fraction": 0.05,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.999,
        "adamw_eps": 1e-8,
        "precision_mode": "fp32",
        "schedule": "legacy_warmup_cosine",
        "min_lr_ratio": 0.0,
        "training_mask": "all_after_warmup",
        "normalization_scope": "prefix",
        "normalization_sequences": 32,
        "kd_weight": 0.0,
        "hint_weight": 0.0,
        "checkpoint_retention": "all_epochs",
    }
    data: dict[str, object] = {
        "final_fit": final_fit_identity(membership),
        "fit_row_groups": {
            "train": list(membership.train_groups),
            "valid": list(membership.valid_groups),
        },
        "evaluation_row_groups": [],
        "fit_members": fit_members,
        "sources": {
            "datasets/train.parquet": {"sha256": membership.train_sha256},
            "datasets/valid.parquet": {"sha256": membership.valid_sha256},
        },
        "evaluation_members": [],
        "evaluation_role": "none",
        "evaluation_source": "none",
        "evaluation_mask": "none",
        "monitoring_plan": None,
        "normalization": {"scope": "prefix", "sequence_count": 32, "fit_prefix_sequences": 32},
    }
    provenance = {
        "completed_epochs": 8,
        "requested_epochs": 8,
        "history": [{"epoch": epoch, "total_loss": 1.0} for epoch in range(8)],
    }
    checkpoint = {
        "config": config,
        "data": data,
        "provenance": provenance,
        "epoch": 7,
        "completed_epochs": 8,
    }
    member = {
        "run": FINAL_RUNS[3],
        "checkpoint_sha256": "7" * 64,
        "configuration_sha256": canonical_sha256(config),
        "provenance_sha256": canonical_sha256(provenance),
        "data_sha256": canonical_sha256(data),
    }
    _validate_final_checkpoint("source4_lstm", checkpoint, member, membership)
    data["evaluation_row_groups"] = [["valid", 0]]
    member["data_sha256"] = canonical_sha256(data)
    with pytest.raises(ValueError, match="evaluation membership"):
        _validate_final_checkpoint("source4_lstm", checkpoint, member, membership)
