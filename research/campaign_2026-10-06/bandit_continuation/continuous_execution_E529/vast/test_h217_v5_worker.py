from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent
PACKET = BASE / "h217_packet_v5"
SPEC = importlib.util.spec_from_file_location("h217_v5_worker", PACKET / "worker.py")
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_h217_v5_packet", BASE / "build_h217_v5_packet.py"
)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)


@pytest.fixture(scope="session", autouse=True)
def _sealed_v4_project() -> None:
    BUILDER.seal_project_copy()


@pytest.mark.parametrize("value", ["1", "600", "7200", "21600"])
def test_accepts_positive_incoming_runtime_under_reviewed_ceiling(value: str) -> None:
    assert WORKER.validate_incoming_runtime(value, 21600) == int(value)


@pytest.mark.parametrize("value", ["0", "-1", "21601", "", "not-a-number"])
def test_rejects_invalid_or_over_ceiling_incoming_runtime(value: str) -> None:
    with pytest.raises(ValueError):
        WORKER.validate_incoming_runtime(value, 21600)


def test_phase_deadline_uses_actual_incoming_runtime_and_reserves() -> None:
    deadline = WORKER.cohort_phase_deadline(100.0, 5400, 450, 600)
    assert deadline == 4450.0
    assert deadline < 100.0 + 21600 - 450 - 600


def test_profile_projection_runs_full128_and_defers_unaffordable_full256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    output = tmp_path / "output"
    output.mkdir()
    fits = {
        arm: {
            "argv": [],
            "output_path": f"runs/{arm}",
            "prediction_export": {"output_name": arm},
            "run_name": f"{arm}-run",
        }
        for arm in WORKER.ARMS
    }
    recipe = {
        "fit_branches": {"batch8": [{"arm": arm, **branch} for arm, branch in fits.items()]},
        "profile_sequence": [{"arm": arm, "argv": []} for arm in WORKER.ARMS],
    }
    calls: list[str] = []

    def phase_runner(
        argv: list[str],
        name: str,
        timeout_seconds: int,
        project_root: Path,
        output_root: Path,
        deadline: float,
    ) -> dict[str, object]:
        calls.append(name)
        if name.startswith("fit_"):
            arm = name.removeprefix("fit_")
            checkpoint = project_root / fits[arm]["output_path"] / "model_ep3.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"fixed epoch 3 checkpoint")
        return {"elapsed_seconds": 0.01, "error": None, "exit_code": 0}

    def gate_runner(path: Path, fit: dict[str, object]) -> dict[str, object]:
        projection = 700 if "full128" in path.name else 5000
        return {"projected_fit_and_export_seconds": projection, "status": "passed"}

    monkeypatch.setattr(
        WORKER, "validate_prediction", lambda project_root, export, checkpoint: {"verified": True}
    )
    result: dict[str, object] = {}
    deadline = time.monotonic() + 2600
    WORKER.execute_arms(recipe, project, output, deadline, result, phase_runner, gate_runner)
    arms = result["arms"]
    assert arms["tcn_full128"]["delivery_status"] == "delivered_ungraded"
    assert arms["tcn_full256"]["fit_status"] == "deferred_cohort_runtime"
    assert "fit_tcn_full256" not in calls
    assert "gru" not in calls
    assert arms["tcn_full128"]["runtime_admission"]["later_arm_projected_delivery_seconds"] == 6800


def _profile_log(path: Path, evidence: dict[str, object]) -> Path:
    path.write_text(
        '{"precision":"fp16","amp_initial_scale":4096}\n'
        'PROFILE {"step_wall":100}\n'
        f"PROFILE_UPDATE_EVIDENCE {json.dumps(evidence)}\n"
    )
    return path


def _valid_profile_evidence() -> dict[str, object]:
    return {
        "amp_skipped_updates": 0,
        "counts_consistent": True,
        "gradient_finite": True,
        "loss_checks": 230,
        "loss_finite": True,
        "optimizer_attempts": 230,
        "optimizer_completed_updates": 230,
        "scaler_initial_scale": 4096,
        "valid_for_throughput": True,
    }


def test_profile_gate_accepts_exact_230_update_evidence(tmp_path: Path) -> None:
    log = _profile_log(tmp_path / "profile.log", _valid_profile_evidence())
    gate = WORKER.profile_gate(log, {"batch_size": 8})
    assert gate["status"] == "passed"
    assert gate["projected_fit_and_export_seconds"] < 7200


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("optimizer_attempts", None),
        ("optimizer_attempts", 0),
        ("optimizer_attempts", True),
        ("optimizer_completed_updates", 229),
        ("amp_skipped_updates", 1),
        ("loss_checks", 229),
    ],
)
def test_profile_gate_rejects_missing_or_mismatched_update_counts(
    tmp_path: Path, field: str, value: object
) -> None:
    evidence = _valid_profile_evidence()
    if value is None:
        evidence.pop(field)
    else:
        evidence[field] = value
    log = _profile_log(tmp_path / "profile.log", evidence)
    with pytest.raises(ValueError, match="profile update counter evidence"):
        WORKER.profile_gate(log, {"batch_size": 8})


def test_profile_gate_rejects_projection_above_fit_cap(tmp_path: Path) -> None:
    evidence = _valid_profile_evidence()
    path = tmp_path / "profile.log"
    path.write_text(
        '{"precision":"fp16","amp_initial_scale":4096}\n'
        'PROFILE {"step_wall":800}\n'
        f"PROFILE_UPDATE_EVIDENCE {json.dumps(evidence)}\n"
    )
    with pytest.raises(TimeoutError, match="exceeds fit budget: 7766.4s"):
        WORKER.profile_gate(path, {"batch_size": 8})


def test_profile_gate_rejects_nonfinite_gradient(tmp_path: Path) -> None:
    evidence = _valid_profile_evidence()
    evidence["gradient_finite"] = False
    log = _profile_log(tmp_path / "profile.log", evidence)
    with pytest.raises(ValueError, match="finite-update"):
        WORKER.profile_gate(log, {"batch_size": 8})


def test_short_positive_runtime_is_packaged_as_deferred_without_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = json.loads((PACKET / "runtime_resources.json").read_text())
    output = tmp_path / "output"
    monkeypatch.setenv("WNN_OUTPUT_DIR", str(output))
    monkeypatch.setenv("WNN_ATTEMPT_ID", "short-runtime-test")
    monkeypatch.setenv("WNN_PACKET_SHA256", "test-packet")
    monkeypatch.setenv("WNN_MAX_RUNTIME_SECONDS", "1")
    for environment, field in {
        "WNN_GPU_MEMORY_BYTES": "cuda_allocator_max_bytes",
        "WNN_LOADER_WORKERS": "loader_workers",
        "WNN_THREADS": "cpu_threads",
        "WNN_WORKER_RAM_BYTES": "worker_memory_max_bytes",
        "WNN_WORKER_VRAM_BYTES": "observed_total_vram_max_bytes",
    }.items():
        monkeypatch.setenv(environment, str(runtime[field]))

    assert WORKER.main() == 0
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "deferred_insufficient_runtime"
    assert result["runtime_budget"]["incoming_runtime_seconds"] == 1
    assert result["runtime_budget"]["phase_deadline_offset_seconds"] == -1049
    assert result["scientific_score"] is None
    assert result["grade_performed"] is False
    assert result["historical_raw_only_tcn_is_paired_factor"] is False
    assert result["fit_authorized_by_packet"] is False
    assert set(result["arms"]) == set(WORKER.ARMS)
    assert all(
        arm["profile_status"] == "deferred_cohort_runtime"
        and arm["fit_status"] == "deferred_cohort_runtime"
        for arm in result["arms"].values()
    )
    assert (output / "artifacts.tar").is_file()


def _recipe_and_runtime() -> tuple[dict[str, object], dict[str, object]]:
    return (
        json.loads((PACKET / "recipe.json").read_text()),
        json.loads((PACKET / "runtime_resources.json").read_text()),
    )


def test_real_recipe_is_two_scratch_full_feature_tcns() -> None:
    recipe, runtime = _recipe_and_runtime()
    WORKER.validate_recipe(recipe, runtime)
    WORKER.verify_source_provenance(PACKET / "project", recipe)
    membership = WORKER.verify_split_membership(PACKET / "project", recipe)
    assert membership == {
        "excluded_evaluation_groups": 468,
        "fit_train": 10535,
        "fit_valid": 1405,
        "fitted_groups": 11940,
    }
    assert recipe["feature_view"]["raw_only"] is False
    assert recipe["feature_view"]["input_width"] == 211
    commands = [
        row["argv"] for row in [*recipe["profile_sequence"], *recipe["fit_branches"]["batch8"]]
    ]
    assert commands
    assert all("--raw-only" not in argv and "gru" not in argv for argv in commands)
    assert [row["arm"] for row in recipe["fit_branches"]["batch8"]] == [
        "tcn_full128",
        "tcn_full256",
    ]


def test_validator_rejects_raw_only_and_gru() -> None:
    recipe, runtime = _recipe_and_runtime()
    recipe["profile_sequence"][0]["argv"] = [*recipe["profile_sequence"][0]["argv"], "--raw-only"]
    with pytest.raises(ValueError, match="stale raw-only"):
        WORKER.validate_recipe(recipe, runtime)
    recipe, runtime = _recipe_and_runtime()
    recipe["fit_branches"]["batch8"][1]["argv"] = [
        "gru" if item == "tcn" else item for item in recipe["fit_branches"]["batch8"][1]["argv"]
    ]
    with pytest.raises(ValueError, match="stale raw-only|GRU|architecture"):
        WORKER.validate_recipe(recipe, runtime)
    recipe, runtime = _recipe_and_runtime()
    hidden_at = recipe["fit_branches"]["batch8"][0]["argv"].index("--hidden")
    recipe["fit_branches"]["batch8"][0]["argv"][hidden_at + 1] = "64"
    with pytest.raises(ValueError, match="architecture dimensions"):
        WORKER.validate_recipe(recipe, runtime)


def test_authorization_stays_pending() -> None:
    review = BUILDER.pending_root_review()
    recipe, _runtime = _recipe_and_runtime()
    assert review["status"] == "pending_root_review"
    assert recipe["status"] == "PREPARED_AWAITING_ROOT_REVIEW"
    assert recipe["authorization"] == {
        "automatic_admission": False,
        "fit": False,
        "profile": False,
        "provider_create": False,
        "push": False,
        "valid_export": False,
    }
    for field in (
        "fit_authorized",
        "grade_authorized",
        "profile_authorized",
        "provider_create_authorized",
        "push_authorized",
    ):
        assert review[field] is False
        assert recipe["pending_root_review"][field] is False


def test_staged_input_hash_must_match_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"sealed-bytes"
    source = tmp_path / "input" / "datasets"
    source.mkdir(parents=True)
    (source / "train.parquet").write_bytes(payload)
    project = tmp_path / "project"
    project.mkdir()
    digest = WORKER.digest(source / "train.parquet")
    registry = {
        "sources": {
            "datasets/train.parquet": {
                "original_identity": {"bytes": len(payload)},
                "sha256": digest,
            }
        },
        "version": 1,
    }
    monkeypatch.setenv(
        "WNN_INPUT_MANIFEST",
        json.dumps([{"path": "datasets/train.parquet", "size_bytes": len(payload)}]),
    )
    monkeypatch.setenv("WNN_INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setattr(WORKER, "SEALED_INPUT_HASHES", {"datasets/train.parquet": digest})
    observed = WORKER.verify_staged_inputs(project, registry)
    assert observed[0]["sha256"] == digest
    assert (project / "datasets" / "train.parquet").is_symlink()
    (source / "train.parquet").write_bytes(b"sealed-bytex")
    project2 = tmp_path / "project2"
    project2.mkdir()
    with pytest.raises(ValueError, match="staged input hash"):
        WORKER.verify_staged_inputs(project2, registry)


def test_run_shell_keeps_terminal_contract_without_provider_calls() -> None:
    script = (PACKET / "run.sh").read_text()
    assert "delivered_ungraded" in script
    assert "partial_delivery_ungraded" in script
    assert "deferred_insufficient_runtime" in script
    assert "450s" in script and "600s" in script
    assert "/venv/main/bin/python" in script
    lowered = script.lower()
    assert "vast" not in lowered
    assert "kaggle" not in lowered
    assert "curl" not in lowered


def test_project_tree_matches_sealed_v4_source() -> None:
    hashes = BUILDER.seal_project_copy()
    BUILDER.assert_project_matches_v4_archive(hashes)
    trainer = (PACKET / "project" / "scripts" / "train_rnn.py").read_bytes()
    sealed = (BASE / "h217_packet_v4" / "project" / "scripts" / "train_rnn.py").read_bytes()
    assert trainer == sealed
    assert b"--raw-only" in trainer
