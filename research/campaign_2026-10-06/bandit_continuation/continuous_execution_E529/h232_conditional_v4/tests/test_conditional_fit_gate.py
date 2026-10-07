from __future__ import annotations

import sys
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1] / "packet"
sys.path.insert(0, str(PACKET))

from profile_worker import derive_fit_argv, evaluate_conditional_fit_gate  # noqa: E402


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
        "warmup": {"attempts": 30, "completed_updates": 30, "loss_checks": 30, "loss_finite": True, "gradient_finite": True, "amp_skipped_updates": 0},
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


def _gate_inputs() -> dict[str, object]:
    return {
        "trace": _trace(),
        "timing": {"end_to_end_inter_start_ms": {"p50": 1000.0, "p90": 1100.0}},
        "loader_budget": {"effective_workers": 2, "effective_prefetch_factor": 1, "batch_tensor_bytes": 73_120_000},
        "trainer_args": {
            "device": "cuda", "arch": "gru", "cell": "gru", "hidden": 384, "layers": 2,
            "seed": 3, "train_source": "train+valid", "train_seqs": 10_607, "batch": 8,
            "chunk": 2000, "epochs": 4, "checkpoint_policy": "fixed", "checkpoint_epoch": 3,
            "no_eval_during_fit": True, "precision": "fp16", "amp_init_scale": 4096.0,
            "raw_only": False, "heuristics": False,
        },
        "n_in": 160,
        "resources": {
            "gpu_resource_query_status": "ok", "cuda_initialized": True,
            "gpu_selected_physical_gpu_uuid": "GPU-test", "gpu_bound_physical_gpu_uuid": "GPU-test",
            "rss_peak_bytes": 1_000_000_000, "cuda_allocated_peak_bytes": 2_000_000_000,
            "cuda_reserved_peak_bytes": 2_500_000_000, "gpu_free_bytes": 7_000_000_000,
            "host_available_bytes": 10_000_000_000, "cgroup_available_bytes": 10_000_000_000,
            "disk_free_bytes": 9_000_000_000,
        },
        "export_probe": [{"output_finite": True}, {"output_finite": True}],
        "raw_input_identity": {"all_unchanged": True},
        "profile_handoff_delta": {"verifier_lookups": 2, "digest_reuse_hits": 2, "digest_reuse_misses": 0, "fallback_digest_calls": 0},
        "elapsed_seconds": 4000.0,
    }


def test_fit_argv_changes_only_name_and_profile_diagnostics() -> None:
    profile = ["--name", "profile", "--device", "cuda", "--amp-init-scale", "4096.0", "--fast", "--profile-warmup", "3", "--profile-steps", "20", "--profile-trace-output", "/tmp/trace.json"]
    fit = derive_fit_argv(profile, "fit-v4")
    assert fit == ["--name", "fit-v4", "--device", "cuda", "--amp-init-scale", "4096.0", "--fast"]


def test_complete_profile_passes_gate() -> None:
    result = evaluate_conditional_fit_gate(**_gate_inputs())
    assert result["status"] == "pass"
    assert result["fit_authorized_by_gate"] is True


def test_timed_amp_skip_rejects_without_fit() -> None:
    args = _gate_inputs()
    trace = args["trace"]
    assert isinstance(trace, dict)
    trace["attempts"][30]["unscaled_found_inf"] = True
    trace["attempts"][30]["optimizer_update_completed"] = False
    trace["timed"].update({"completed_updates": 199, "amp_skipped_updates": 1, "gradient_finite": False})
    trace["valid_for_throughput"] = False
    result = evaluate_conditional_fit_gate(**args)
    assert result["status"] == "reject_no_fit"
    assert result["fit_authorized_by_gate"] is False
    assert "timed_updates_not_all_finite_and_completed" in result["reasons"]


if __name__ == "__main__":
    test_fit_argv_changes_only_name_and_profile_diagnostics()
    test_complete_profile_passes_gate()
    test_timed_amp_skip_rejects_without_fit()
    print("conditional H232 gate tests passed")
