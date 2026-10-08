from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import scripts.e540_kaggle_worker as e540_worker
from scripts.e540_cli_preflight import normalize_entry_argv, parse_project_cli
from scripts.e540_kaggle_worker import (
    AUTHORIZATION_PATH,
    EXPECTED_PROFILE_ATTEMPTS,
    PACKET_ID,
    RUN_NAMES,
    call_main,
    elapsed_projection_seconds,
    expected_fit_updates,
    parse_trainer_output,
    trainer_argv,
    validate_conditional_authorization,
    validate_frozen_membership,
    validate_profile,
    validate_recipe_cli,
    worker_wall_budget_seconds,
)
from scripts.kaggle_bundle import local_bundle_descriptors
from scripts.kaggle_common import verify_files
from scripts.kaggle_gpu_runner import resolve_worker_timeout_seconds
from scripts.kaggle_kernel import terminate_process_tree, worker_timeout_for_deadline


def _profile_output(*, step_wall: str = "12.5", skipped: int = 0) -> str:
    evidence = {
        "optimizer_attempts": EXPECTED_PROFILE_ATTEMPTS,
        "optimizer_completed_updates": EXPECTED_PROFILE_ATTEMPTS - skipped,
        "amp_skipped_updates": skipped,
        "loss_checks": EXPECTED_PROFILE_ATTEMPTS,
        "loss_finite": True,
        "gradient_finite": skipped == 0,
        "valid_for_throughput": skipped == 0,
    }
    return (
        f"PROFILE {{\"step_wall\": {step_wall}, \"gpu_total\": 8.0}}\n"
        f"PROFILE_UPDATE_EVIDENCE {json.dumps(evidence)}\n"
    )


def test_profile_gate_requires_three_timed_batches_finite_wall_and_zero_skips() -> None:
    parsed = parse_trainer_output(_profile_output())
    step_wall, evidence = validate_profile(parsed)
    assert step_wall == 12.5
    assert evidence["optimizer_attempts"] == 316
    assert EXPECTED_PROFILE_ATTEMPTS == 316

    with pytest.raises(ValueError, match="finite and positive"):
        validate_profile(parse_trainer_output(_profile_output(step_wall="NaN")))
    with pytest.raises(ValueError, match="316 finite optimizer updates"):
        validate_profile(parse_trainer_output(_profile_output(skipped=1)))


def test_v3_profile_and_fit_pin_the_same_amp_scale_and_no_auxiliary_paths() -> None:
    profile = trainer_argv("ple", profile=True)
    fit = trainer_argv("ple", profile=False)
    assert profile[profile.index("--amp-init-scale") + 1] == "4096"
    assert fit[fit.index("--amp-init-scale") + 1] == "4096"
    for argv in (profile, fit):
        assert argv[argv.index("--loss") + 1] == "wmse"
        assert argv[argv.index("--loss-norm") + 1] == "chunk"
        assert argv[argv.index("--aux-struct") + 1] == "0"
        assert argv[argv.index("--propensity") + 1] == ""
        assert argv[argv.index("--distill") + 1] == ""
        assert "--no-eval-during-fit" in argv
    assert "--profile-steps" not in fit


def test_actual_recipe_parsers_accept_both_profiles_fits_and_predictions_without_torch() -> None:
    import builtins

    prior_torch_loaded = "torch" in sys.modules
    original_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("E540 CLI preflight imported Torch")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = no_torch
    try:
        checks = validate_recipe_cli()
    finally:
        builtins.__import__ = original_import
    assert len(checks) == 6
    assert all(check["status"] == "parsed" for check in checks)
    assert all(check["script_token_removed"] is True for check in checks)
    assert prior_torch_loaded or "torch" not in sys.modules


def test_duplicate_script_token_would_fail_actual_trainer_parser() -> None:
    root = Path(__file__).resolve().parents[1]
    argv = trainer_argv("linear", profile=True)
    with pytest.raises(SystemExit):
        parse_project_cli(
            root / "scripts/train_rnn.py",
            argv,
            parser_name="ap",
            dynamics_path=root / "src/wnn/training_dynamics.py",
        )
    normalized = normalize_entry_argv("train_rnn.py", argv)
    parsed = parse_project_cli(
        root / "scripts/train_rnn.py",
        normalized,
        parser_name="ap",
        dynamics_path=root / "src/wnn/training_dynamics.py",
    )
    assert parsed.name == RUN_NAMES["linear"]
    assert parsed.numeric_map == "linear"


def test_in_process_entry_point_strips_exact_logged_script_token(tmp_path: Path) -> None:
    observed: list[list[str]] = []
    module = SimpleNamespace(main=lambda: observed.append(sys.argv.copy()))
    call_main(
        module,
        "train_rnn.py",
        ["scripts/train_rnn.py", "--name", "parser-smoke"],
        tmp_path / "train.log",
    )
    assert observed == [["train_rnn.py", "--name", "parser-smoke"]]
    with pytest.raises(ValueError, match="declared entry point"):
        normalize_entry_argv("train_rnn.py", ["--name", "parser-smoke"])


def test_conditional_authorization_fails_closed_and_binds_packet_hashes(tmp_path: Path) -> None:
    expected = {
        "packet_id": PACKET_ID,
        "code_manifest_sha256": "a" * 64,
        "recipe_sha256": "b" * 64,
        "subset_manifest_sha256": "c" * 64,
        "kernel_process_wall_seconds_max": 6000,
        "worker_wall_seconds_max": 5200,
    }
    receipt_path = tmp_path / AUTHORIZATION_PATH.name
    pending = {
        "schema": "wnn.e540-conditional-fit-authorization.v1",
        "owner": "root",
        "status": "pending_root_review",
        "authorized": False,
    }
    receipt_path.write_text(json.dumps(pending))
    with pytest.raises(ValueError, match="valid status"):
        validate_conditional_authorization(receipt_path, expected)

    approved = {
        **pending,
        "status": "approved",
        "authorized": True,
        "conditional_fit_after_profile_gate": True,
        "authorization_id": "root-e540-v6-review-1",
        **expected,
    }
    receipt_path.write_text(json.dumps(approved))
    assert validate_conditional_authorization(receipt_path, expected) == approved

    approved["code_manifest_sha256"] = "d" * 64
    receipt_path.write_text(json.dumps(approved))
    with pytest.raises(ValueError, match="does not bind code_manifest_sha256"):
        validate_conditional_authorization(receipt_path, expected)


def test_startup_parses_all_recipe_commands_then_rejects_pending_auth_before_data_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization_path = tmp_path / "root-auth.json"
    authorization_path.write_text(json.dumps({
        "schema": "wnn.e540-conditional-fit-authorization.v1",
        "owner": "root",
        "status": "pending_root_review",
        "authorized": False,
    }))
    runtime = tmp_path / "e540_cohort"
    monkeypatch.setattr(e540_worker, "AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(e540_worker, "RUNTIME", runtime)
    monkeypatch.setattr(
        e540_worker,
        "verify_mounted_data",
        lambda: pytest.fail("large mounted-input hashing started before the auth gate"),
    )
    monkeypatch.setenv("WNN_KERNEL_DEADLINE_MONOTONIC", str(time.monotonic() + 6_000))
    monkeypatch.setenv("WNN_E540_PACKET_ID", PACKET_ID)
    monkeypatch.setenv("WNN_E540_AUTHORIZATION_PATH", authorization_path.as_posix())
    monkeypatch.setenv("WNN_E540_CODE_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("WNN_E540_RECIPE_SHA256", "b" * 64)
    monkeypatch.setenv("WNN_E540_SUBSET_SHA256", "c" * 64)
    monkeypatch.setenv("WNN_E540_KERNEL_WALL_SECONDS", "6000")
    monkeypatch.setenv("WNN_E540_WORKER_WALL_SECONDS", "5200")

    with pytest.raises(ValueError, match="valid status"):
        e540_worker.run()

    receipt = json.loads((runtime / "startup_preflight.json").read_text())
    assert receipt["status"] == "failed_before_mounted_data_hashing"
    assert len(receipt["cli_checks"]) == 6


def test_fit_update_count_uses_drop_last_outer_batches_and_79_chunk_updates() -> None:
    assert expected_fit_updates(340, 172) == (340 + 172) // 2 * 79
    assert expected_fit_updates(341, 172) == (341 + 172) // 2 * 79
    # Frozen V3 membership selects 1,597 train + 252 valid sequences. The
    # batch-2 drop_last loader consumes 1,848 of those 1,849 sequences.
    assert expected_fit_updates(1597, 252) == 72996


def test_projection_does_not_add_profile_wall_twice_and_keeps_300_second_reserve() -> None:
    profiles = {
        "linear": {"profile_wall_seconds": 20.0, "step_wall_ms": 0.0},
        "ple": {"profile_wall_seconds": 30.0, "step_wall_ms": 0.0},
    }
    assert elapsed_projection_seconds(70.0, profiles, 0, 0, 0) == 370.0
    near_limit = elapsed_projection_seconds(4_900.0, profiles, 0, 0, 0)
    assert near_limit == 5_200.0
    assert near_limit <= 5_200


def test_worker_wall_budget_shrinks_with_bootstrap_and_runner_preflight_time() -> None:
    deadline = 6_000.0
    assert worker_wall_budget_seconds(deadline, now_monotonic=500.0) == 5_200
    assert worker_wall_budget_seconds(deadline, now_monotonic=800.0) == 4_900
    assert worker_wall_budget_seconds(deadline, now_monotonic=5_700.0) == 0


def test_kernel_timeout_is_shrunk_from_bootstrap_entry_and_leaves_finalize_reserve() -> None:
    assert worker_timeout_for_deadline(5_200, 6_000.0, now_monotonic=500.0) == 5_200
    assert worker_timeout_for_deadline(5_200, 6_000.0, now_monotonic=800.0) == 4_900
    assert worker_timeout_for_deadline(5_200, 6_000.0, now_monotonic=5_700.0) == 0


def test_runner_recomputes_timeout_at_actual_launch_after_preflight_delay() -> None:
    # Kernel writes the initial runtime spec at t=200, before runner preflight.
    spec_timeout = worker_timeout_for_deadline(5_200, 6_000.0, now_monotonic=200.0)
    assert spec_timeout == 5_200
    # Runner launches the worker at t=700 and must preserve the 300s finalization reserve.
    assert resolve_worker_timeout_seconds(
        spec_timeout, "6000.0", now_monotonic=700.0
    ) == 5_000


def test_runner_keeps_legacy_timeout_without_deadline_and_rejects_invalid_deadlines() -> None:
    assert resolve_worker_timeout_seconds(7_000, None, now_monotonic=700.0) == 7_000
    assert resolve_worker_timeout_seconds(7_000, "10000.0", now_monotonic=700.0) == 5_200
    assert resolve_worker_timeout_seconds(1_000, "10000.0", now_monotonic=700.0) == 1_000
    with pytest.raises(ValueError, match="must be finite"):
        resolve_worker_timeout_seconds(5_200, "NaN", now_monotonic=700.0)
    with pytest.raises(ValueError, match="current monotonic time must be finite"):
        resolve_worker_timeout_seconds(5_200, "6000", now_monotonic="700")
    with pytest.raises(TimeoutError, match="no worker time"):
        resolve_worker_timeout_seconds(5_200, "6000", now_monotonic=5_700.0)


def test_kernel_deadline_cleanup_stops_runner_and_separate_session_descendant() -> None:
    import psutil

    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], start_new_session=True); "
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent_code], start_new_session=True
    )
    child_pids: list[int] = []
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                child_pids = [child.pid for child in psutil.Process(process.pid).children(recursive=True)]
            except psutil.NoSuchProcess:
                break
            if child_pids:
                break
            time.sleep(0.02)
        assert child_pids, "test process did not create its separate-session child"
        terminate_process_tree(process, grace_seconds=0.2)
        assert process.poll() is not None
        for pid in child_pids:
            poll_deadline = time.monotonic() + 1
            while psutil.pid_exists(pid) and time.monotonic() < poll_deadline:
                try:
                    if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                        break
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.02)
            if psutil.pid_exists(pid):
                assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if process.poll() is None:
            terminate_process_tree(process, grace_seconds=0.2)


def test_external_bundle_inputs_are_not_hashed_in_the_local_controller_checkout() -> None:
    files = {
        "scripts/train.py": {"path": "scripts/train.py", "size": 4, "sha256": "a" * 64},
        "datasets/train.parquet": {
            "path": "datasets/train.parquet",
            "size": 29_000_000_000,
            "sha256": "b" * 64,
        },
    }
    assert local_bundle_descriptors(
        files, {"datasets/train.parquet": "train.parquet"}
    ) == [files["scripts/train.py"]]


def test_input_identity_receipt_persists_expected_and_observed_hashes(tmp_path: Path) -> None:
    value = b"sealed-small-input"
    path = tmp_path / "input.json"
    path.write_bytes(value)
    descriptor = {
        "path": "input.json",
        "size": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }
    receipt = tmp_path / "identity.json"
    observations = verify_files(tmp_path, [descriptor], identity_receipt=receipt)
    assert observations[0]["status"] == "match"
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "verified"
    assert payload["observations"][0]["observed"] == {
        "size": len(value),
        "sha256": descriptor["sha256"],
    }

    altered = {**descriptor, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_files(tmp_path, [altered], identity_receipt=receipt)
    failed = json.loads(receipt.read_text())
    assert failed["observations"][0]["status"] == "mismatch"


def _synthetic_membership() -> tuple[dict[str, object], SimpleNamespace]:
    import numpy as np

    train = list(range(512))
    valid = list(range(2_000, 2_512)) + [3_000]
    evaluation = list(range(4_000, 4_468))
    components = [
        ("train", group, group) for group in train
    ] + [
        ("valid", group, group - 2_000) for group in valid[:-1]
    ] + [
        ("valid", 3_000, 0)
    ] + [
        ("valid", group, 512 + index) for index, group in enumerate(evaluation)
    ]
    selected_components = list(range(512))
    groups = {"train": train, "valid": valid}
    members = sorted((split, group) for split, values in groups.items() for group in values)

    def digest_groups(values: list[int]) -> str:
        return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()

    subset: dict[str, object] = {
        "split_manifest_sha256": "s" * 64,
        "selection_seed": 540,
        "eligible_component_count": 3_148,
        "selected_component_count": 512,
        "selected_components": selected_components,
        "selected_component_sha256": digest_groups(selected_components),
        "selected_groups": groups,
        "selected_group_counts": {name: len(values) for name, values in groups.items()},
        "selected_group_sha256": {
            name: digest_groups(values) for name, values in groups.items()
        },
        "selected_members_sha256": hashlib.sha256(
            json.dumps(members, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).hexdigest(),
        "evaluation_groups": evaluation,
        "evaluation_group_count": 468,
        "evaluation_membership_sha256": digest_groups(evaluation),
    }
    evaluation_components = set(range(512, 980))
    component_by_member = {
        (split, group): component for split, group, component in components
    }

    def assert_fit_groups(fit_train: list[int], fit_valid: list[int]) -> None:
        assert set(fit_train).issubset(train)
        assert set(fit_valid).issubset(valid)
        fitted_components = {component_by_member[("train", group)] for group in fit_train}
        fitted_components.update(
            component_by_member[("valid", group)] for group in fit_valid
        )
        assert not fitted_components.intersection(evaluation_components)

    campaign = SimpleNamespace(
        sha256="s" * 64,
        train_groups=tuple(train),
        valid_groups=tuple(valid),
        evaluation_groups=tuple(evaluation),
        components=tuple(components),
        assert_fit_groups=assert_fit_groups,
    )
    return subset, campaign


def test_exact_membership_receipt_binds_component_and_group_sets() -> None:
    subset, campaign = _synthetic_membership()
    receipt = validate_frozen_membership(subset, subset, campaign)
    assert receipt["selected_component_count"] == 512
    assert receipt["evaluation_group_count"] == 468
    assert receipt["fit_group_counts"] == {"train": 512, "valid": 513}
    assert receipt["component_complete"] is True


def test_membership_gate_rejects_partial_selected_component() -> None:
    import numpy as np

    subset, campaign = _synthetic_membership()
    subset["selected_groups"]["valid"].remove(3_000)
    subset["selected_group_counts"]["valid"] -= 1
    subset["selected_group_sha256"]["valid"] = hashlib.sha256(
        np.asarray(subset["selected_groups"]["valid"], dtype="<i8").tobytes()
    ).hexdigest()
    members = sorted(
        (split, group)
        for split, values in subset["selected_groups"].items()
        for group in values
    )
    subset["selected_members_sha256"] = hashlib.sha256(
        json.dumps(members, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match="not component-complete"):
        validate_frozen_membership(subset, subset, campaign)


def test_ple_checkpoint_receipt_binds_exact_fit_members_and_stored_knots() -> None:
    import numpy as np

    from scripts.e540_kaggle_worker import validate_numeric_checkpoint

    subset, _ = _synthetic_membership()
    knots = np.zeros((112, 9), dtype=np.float32)
    counts = np.ones(112, dtype=np.int64)
    knot_digest = hashlib.sha256(
        knots.astype("<f4", copy=False).tobytes()
        + counts.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    numeric = {
        "mode": "ple",
        "n_features": 112,
        "n_intervals": 8,
        "mapped_input_width": 1107,
        "split_manifest_sha256": subset["split_manifest_sha256"],
        "fit_members": subset["selected_groups"],
        "fit_members_sha256": subset["selected_members_sha256"],
        "fit_member_count": 1_025,
        "samples_per_sequence": 128,
        "candidate_sample_rows": 131_200,
        "retained_sample_rows": 65_536,
        "knot_sha256": knot_digest,
    }
    model_state = {
        "numeric_feature_map.knots": knots,
        "numeric_feature_map.knot_counts": counts,
        "numeric_feature_map.knots_fitted": np.asarray(True),
    }
    identity = validate_numeric_checkpoint("ple", numeric, subset, model_state)
    assert identity["fit_members_sha256"] == subset["selected_members_sha256"]
    assert identity["knot_sha256"] == knot_digest
    changed_state = {**model_state, "numeric_feature_map.knots": knots + 1}
    with pytest.raises(ValueError, match="knot-buffer digest"):
        validate_numeric_checkpoint("ple", numeric, subset, changed_state)
