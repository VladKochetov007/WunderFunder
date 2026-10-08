from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import types
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1] / "packet"
sys.path.insert(0, str(PACKET))

from profile_worker import (  # noqa: E402
    MAX_PROJECTED_FIT_SECONDS,
    PREDICTION_EXPORT_CAP_S,
    WORKER_WALL_CAP_S,
    ProfileAbort,
    _probe_source_input_width,
    derive_fit_argv,
    evaluate_conditional_fit_gate,
    validate_external_review,
)


def _archived_source_width_probe(tmp_path: Path) -> dict[str, object]:
    """Run the real pinned event_features code on the requested zero tensor."""
    source_root = tmp_path / "pinned_source"
    source_root.mkdir()
    archive_path = PACKET / "source_archive.tar"
    with tarfile.open(archive_path, "r:") as archive:
        for name in ("src/wnn/__init__.py", "src/wnn/data.py", "src/wnn/features.py"):
            member = archive.getmember(name)
            source = archive.extractfile(member)
            assert source is not None
            destination = source_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read())
    sys.path.insert(0, str(source_root / "src"))
    try:
        for module_name in tuple(sys.modules):
            if module_name == "wnn" or module_name.startswith("wnn."):
                del sys.modules[module_name]
        from wnn.data import N_FEATURES

        return _probe_source_input_width(types.SimpleNamespace(N_FEATURES=N_FEATURES))
    finally:
        sys.path.remove(str(source_root / "src"))
        for module_name in tuple(sys.modules):
            if module_name == "wnn" or module_name.startswith("wnn."):
                del sys.modules[module_name]


def _trace() -> dict[str, object]:
    attempts = []
    for phase, count in (("warmup", 30), ("timed", 200)):
        attempts.extend(
            {
                "phase": phase,
                "loss_finite": True,
                "unscaled_found_inf": False,
                "optimizer_update_completed": True,
                "scale_schedule_consistent": True,
            }
            for _ in range(count)
        )
    return {
        "schema": "h232-profile-update-trace-v1",
        "expected_timed_attempts": 200,
        "valid_for_throughput": True,
        "numeric_policy": {"initial_scale": 4096.0},
        "warmup": {
            "attempts": 30,
            "completed_updates": 30,
            "loss_checks": 30,
            "loss_finite": True,
            "gradient_finite": True,
            "amp_skipped_updates": 0,
        },
        "timed": {
            "attempts": 200,
            "completed_updates": 200,
            "amp_skipped_updates": 0,
            "loss_checks": 200,
            "loss_finite": True,
            "gradient_finite": True,
            "valid_for_throughput": True,
        },
        "attempts": attempts,
    }


def _gate_inputs(feature_width_probe: dict[str, object]) -> dict[str, object]:
    return {
        "trace": _trace(),
        "timing": {"end_to_end_inter_start_ms": {"p50": 1000.0, "p90": 1100.0}},
        "loader_budget": {
            "effective_workers": 2,
            "effective_prefetch_factor": 1,
            "batch_tensor_bytes": 73_120_000,
        },
        "trainer_args": {
            "device": "cuda",
            "arch": "gru",
            "cell": "gru",
            "hidden": 384,
            "layers": 2,
            "seed": 3,
            "train_source": "train+valid",
            "train_seqs": 10_607,
            "batch": 8,
            "chunk": 2000,
            "epochs": 4,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 3,
            "no_eval_during_fit": True,
            "precision": "fp16",
            "amp_init_scale": 4096.0,
            "raw_only": False,
            "heuristics": False,
        },
        "n_in": 211,
        "feature_width_probe": feature_width_probe,
        "resources": {
            "gpu_resource_query_status": "ok",
            "cuda_initialized": True,
            "gpu_selected_physical_gpu_uuid": "GPU-test",
            "gpu_bound_physical_gpu_uuid": "GPU-test",
            "rss_peak_bytes": 1_000_000_000,
            "cuda_allocated_peak_bytes": 2_000_000_000,
            "cuda_reserved_peak_bytes": 2_500_000_000,
            "gpu_free_bytes": 7_000_000_000,
            "host_available_bytes": 10_000_000_000,
            "cgroup_available_bytes": 10_000_000_000,
            "disk_free_bytes": 9_000_000_000,
        },
        "export_probe": [{"output_finite": True}, {"output_finite": True}],
        "raw_input_identity": {"all_unchanged": True},
        "profile_handoff_delta": {
            "verifier_lookups": 6,
            "digest_reuse_hits": 2,
            "digest_reuse_misses": 4,
            "fallback_digest_calls": 4,
        },
        "raw_digest_path_profile_delta": {
            "train": {
                "verifier_lookups": 1,
                "digest_reuse_hits": 1,
                "digest_reuse_misses": 0,
                "fallback_digest_calls": 0,
            },
            "valid": {
                "verifier_lookups": 1,
                "digest_reuse_hits": 1,
                "digest_reuse_misses": 0,
                "fallback_digest_calls": 0,
            },
        },
        "elapsed_seconds": 4000.0,
    }


def test_archived_event_feature_smoke_derives_211_width(tmp_path: Path) -> None:
    probe = _archived_source_width_probe(tmp_path)
    assert probe["probe_input_shape"] == [2, 3, 112]
    assert probe["probe_event_output_shape"] == [2, 3, 99]
    assert probe["derived_model_input_width"] == 211


def test_fit_argv_changes_only_name_and_profile_diagnostics() -> None:
    profile = [
        "--name",
        "profile",
        "--device",
        "cuda",
        "--amp-init-scale",
        "4096.0",
        "--fast",
        "--profile-warmup",
        "3",
        "--profile-steps",
        "20",
        "--profile-trace-output",
        "/tmp/trace.json",
    ]
    fit = derive_fit_argv(profile, "fit-v7-proactive")
    assert fit == [
        "--name",
        "fit-v7-proactive",
        "--device",
        "cuda",
        "--amp-init-scale",
        "4096.0",
        "--fast",
    ]


def test_v7_worker_budget_keeps_four_epochs_and_reserves_valid_export() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((PACKET / "H232_PROFILE_PLAN.json").read_text(encoding="utf-8"))
    recipe = json.loads((PACKET / "H232_AMP4096_SUCCESSOR_RECIPE.json").read_text(encoding="utf-8"))
    budget = plan["v7_execution_budget"]
    assert WORKER_WALL_CAP_S == 16_000
    assert MAX_PROJECTED_FIT_SECONDS == 12_000
    assert PREDICTION_EXPORT_CAP_S == 1_800
    assert budget["whole_worker_wall_cap_seconds"] == WORKER_WALL_CAP_S
    assert budget["valid_export_cap_seconds"] == PREDICTION_EXPORT_CAP_S
    assert budget["duration_cuts_training"] is False
    assert recipe["scientific_recipe"]["epochs"] == 4
    assert recipe["scientific_recipe"]["fixed_checkpoint_epoch_zero_based"] == 3
    assert recipe["runtime"]["worker_wall_cap_seconds"] == WORKER_WALL_CAP_S
    assert root.name == "h232_conditional_v7b_reviewed_20261008"


def test_v7_gate_rejects_when_fit_and_full_export_exceed_worker_budget(
    tmp_path: Path,
) -> None:
    args = _gate_inputs(_archived_source_width_probe(tmp_path))
    args["timing"]["end_to_end_inter_start_ms"]["p90"] = 1_400.0
    args["elapsed_seconds"] = 5_000.0
    result = evaluate_conditional_fit_gate(**args)
    assert result["status"] == "reject_no_fit"
    assert (
        "p90_projected_total_lifetime_including_export_exceeds_worker_wall_cap" in result["reasons"]
    )
    assert (
        result["lifetime_projection"]["projected_total_seconds_p90_including_export"]
        > WORKER_WALL_CAP_S
    )


def test_actual_profile_width_and_six_digest_lookups_pass_gate(tmp_path: Path) -> None:
    feature_probe = _archived_source_width_probe(tmp_path)
    result = evaluate_conditional_fit_gate(**_gate_inputs(feature_probe))
    assert result["status"] == "pass"
    assert result["fit_authorized_by_gate"] is True
    assert result["loader_tensor_bytes_policy"]["expected_batch_tensor_bytes"] == 73_120_000


def test_160_width_is_rejected_against_archived_source_probe(tmp_path: Path) -> None:
    args = _gate_inputs(_archived_source_width_probe(tmp_path))
    args["n_in"] = 160
    result = evaluate_conditional_fit_gate(**args)
    assert result["status"] == "reject_no_fit"
    assert "profile_input_width_differs_from_pinned_source_probe" in result["reasons"]


def test_digest_miss_without_original_verifier_fallback_rejects(tmp_path: Path) -> None:
    args = _gate_inputs(_archived_source_width_probe(tmp_path))
    args["profile_handoff_delta"]["fallback_digest_calls"] = 3
    result = evaluate_conditional_fit_gate(**args)
    assert result["status"] == "reject_no_fit"
    assert "profile_digest_lookup_accounting_or_fallback_mismatch" in result["reasons"]


def test_raw_file_path_missing_fail_closed_lookup_rejects(tmp_path: Path) -> None:
    args = _gate_inputs(_archived_source_width_probe(tmp_path))
    args["raw_digest_path_profile_delta"]["train"]["verifier_lookups"] = 0
    result = evaluate_conditional_fit_gate(**args)
    assert result["status"] == "reject_no_fit"
    assert "raw_train_valid_digest_verification_not_fail_closed" in result["reasons"]


def test_preprofile_raw_checks_cannot_substitute_for_profile_delta(tmp_path: Path) -> None:
    args = _gate_inputs(_archived_source_width_probe(tmp_path))
    args["raw_digest_path_profile_delta"] = {
        "train": {
            "verifier_lookups": 0,
            "digest_reuse_hits": 0,
            "digest_reuse_misses": 0,
            "fallback_digest_calls": 0,
        },
        "valid": {
            "verifier_lookups": 0,
            "digest_reuse_hits": 0,
            "digest_reuse_misses": 0,
            "fallback_digest_calls": 0,
        },
    }
    result = evaluate_conditional_fit_gate(**args)
    assert result["status"] == "reject_no_fit"
    assert "raw_train_valid_digest_verification_not_fail_closed" in result["reasons"]


def test_loader_tensor_bytes_must_match_measured_source_shape(tmp_path: Path) -> None:
    args = _gate_inputs(_archived_source_width_probe(tmp_path))
    args["loader_budget"]["batch_tensor_bytes"] = 73_120_001
    result = evaluate_conditional_fit_gate(**args)
    assert result["status"] == "reject_no_fit"
    assert "observed_loader_tensor_bytes_differ_from_frozen_batch_shape" in result["reasons"]


def test_v7b_review_authorizes_only_profile_gated_fit_not_provider_push() -> None:
    receipt_bytes = (PACKET / "external_review.json").read_bytes()
    digest = hashlib.sha256(receipt_bytes).hexdigest()
    receipt = json.loads(receipt_bytes)
    assert (
        receipt["experiment_id"]
        == "E529_H232_GRU384_B8_S3_AMP4096_PROACTIVE_CACHE_RECOVERY_20261008_V7A"
    )
    assert receipt["status"] == "approved"
    assert receipt["conditional_fit_authorized"] is True
    assert receipt["fit_admission"] == "same-worker-only-after-profile-gate"
    assert receipt["provider_push_authorized"] is False
    validated = validate_external_review(receipt_bytes, digest, "h232_cloud_operator")
    assert validated["root_approval"]["provider_push_authorized"] is False


def test_v7_build_receipt_binds_source_and_keeps_push_unadmitted() -> None:
    receipt_path = PACKET.parent / "V7_BUILD_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report_hash = hashlib.sha256(
        (PACKET.parent / "H232_V7_PROACTIVE_RECOVERY_REPORT_V1.md").read_bytes()
    ).hexdigest()
    assert (
        receipt["experiment_id"]
        == "E529_H232_GRU384_B8_S3_AMP4096_PROACTIVE_CACHE_RECOVERY_20261008_V7A"
    )
    assert receipt["provider_push_authorized"] is False
    assert receipt["provider_status"] == "not_queried"
    assert receipt["source_archive"]["byte_identical_to_v6"] is True
    assert receipt["source_archive"]["sha256"] == receipt["parent_v6"]["source_archive_sha256"]
    assert receipt["source_archive"]["member_count"] == 54
    assert receipt["diagnosis_report_sha256"] == report_hash
    execution_diff = receipt["execution_diff"]
    diff_bytes = (PACKET.parent / execution_diff["path"]).read_bytes()
    assert hashlib.sha256(diff_bytes).hexdigest() == execution_diff["sha256"]
    assert len(diff_bytes) == execution_diff["bytes"]


def test_pending_review_cannot_authorize_execution() -> None:
    receipt = json.loads((PACKET / "external_review.json").read_text(encoding="utf-8"))
    receipt.update(
        status="pending_independent_review",
        conditional_fit_authorized=False,
        fit_admission="not_admitted",
    )
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(receipt_bytes).hexdigest()
    try:
        validate_external_review(receipt_bytes, digest, "h232_cloud_operator")
    except ProfileAbort as error:
        assert "lacks conditional fit authorization" in str(error)
    else:
        raise AssertionError("pending review unexpectedly authorized a fit")


def test_timed_amp_skip_rejects_without_fit(tmp_path: Path) -> None:
    args = _gate_inputs(_archived_source_width_probe(tmp_path))
    trace = args["trace"]
    assert isinstance(trace, dict)
    trace["attempts"][30]["unscaled_found_inf"] = True
    trace["attempts"][30]["optimizer_update_completed"] = False
    trace["timed"].update(
        {"completed_updates": 199, "amp_skipped_updates": 1, "gradient_finite": False}
    )
    trace["valid_for_throughput"] = False
    result = evaluate_conditional_fit_gate(**args)
    assert result["status"] == "reject_no_fit"
    assert result["fit_authorized_by_gate"] is False
    assert "timed_updates_not_all_finite_and_completed" in result["reasons"]


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    test_fit_argv_changes_only_name_and_profile_diagnostics()
    test_v7_worker_budget_keeps_four_epochs_and_reserves_valid_export()
    with TemporaryDirectory() as temp:
        test_v7_gate_rejects_when_fit_and_full_export_exceed_worker_budget(Path(temp))
    with TemporaryDirectory() as temp:
        test_actual_profile_width_and_six_digest_lookups_pass_gate(Path(temp))
    with TemporaryDirectory() as temp:
        test_160_width_is_rejected_against_archived_source_probe(Path(temp))
    with TemporaryDirectory() as temp:
        test_digest_miss_without_original_verifier_fallback_rejects(Path(temp))
    with TemporaryDirectory() as temp:
        test_raw_file_path_missing_fail_closed_lookup_rejects(Path(temp))
    with TemporaryDirectory() as temp:
        test_preprofile_raw_checks_cannot_substitute_for_profile_delta(Path(temp))
    with TemporaryDirectory() as temp:
        test_loader_tensor_bytes_must_match_measured_source_shape(Path(temp))
    with TemporaryDirectory() as temp:
        test_timed_amp_skip_rejects_without_fit(Path(temp))
    test_v7b_review_authorizes_only_profile_gated_fit_not_provider_push()
    test_v7_build_receipt_binds_source_and_keeps_push_unadmitted()
    test_pending_review_cannot_authorize_execution()
    print("conditional H232 gate tests passed")
