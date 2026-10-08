from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
AUTOPILOT_DIR = (
    Path(__file__).resolve().parents[1]
    / "research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008/autopilot_r1"
)
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "research_autopilot", SCRIPTS / "research_autopilot.py"
)
assert SPEC is not None and SPEC.loader is not None
autopilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(autopilot)

RUNS = {
    "seed99": "h230-faithful-student-seed99-v1",
    "seed9": "h230-faithful-student-seed9-v1",
    "z4": "z4_seed99_student_hft_alldata_v2",
    "z5": "z5_seed9_student_hft_alldata_v2",
}
PRIOR_ID = RUNS["seed99"]
PENDING_IDS = [RUNS["seed9"], RUNS["z4"], RUNS["z5"]]
REQUIRED_SOURCES = [
    "scripts/research_autopilot.py",
    "scripts/research_dispatch.py",
    "scripts/bandit_resources.py",
    "scripts/warm_queue_worker.py",
    "scripts/limited",
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_fixture(root: Path, *, contract_authorized: bool = True) -> dict[str, Any]:
    """Build a fully hash-bound local queue without invoking a real worker."""
    for relative in REQUIRED_SOURCES:
        source = SCRIPTS / Path(relative).name
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    base = root / "research/campaign/test/autopilot"
    queue_dir = root / "runtime/warm_queue_local_v1"
    state_dir = root / "runtime/bandit"
    contract_path = base / "LOCAL_V4_ROOT_CONTRACT.json"
    contract = {
        "schema_version": 1,
        "owner": "root",
        "authorized": contract_authorized,
        "attempt_id": "warm-queue-local-v4-autopilot-20261008",
        "prior_attempt_id": "warm-queue-local-v3-20261008",
        "resources": {
            "cpu_cores": 2,
            "worker_ram_bytes": 9_000_000_000,
            "observed_vram_ceiling_bytes": 5_000_000_000,
            "timeout_seconds": 21_600,
            "idle_exit_seconds": 300,
        },
        "pending_spec_hashes": {},
    }

    all_hashes: dict[str, str] = {}
    pending_hashes: dict[str, str] = {}
    for run_id in RUNS.values():
        spec_path = root / "specs" / f"{run_id}.json"
        write_json(spec_path, {"run_id": run_id, "steps": []})
        all_hashes[run_id] = digest(spec_path)
        if run_id in PENDING_IDS:
            pending_hashes[run_id] = all_hashes[run_id]
            destination = queue_dir / "pending" / spec_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(spec_path, destination)
    contract["pending_spec_hashes"] = pending_hashes
    write_json(contract_path, contract)

    dependency_model = root / "artifacts/runs/seed99/model_ep1.pt"
    dependency_provenance = root / "artifacts/runs/seed99/provenance.json"
    for path, payload in (
        (dependency_model, b"v3 checkpoint"),
        (dependency_provenance, b"v3 provenance"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (queue_dir / "done" / f"{PRIOR_ID}.json").parent.mkdir(parents=True, exist_ok=True)
    write_json(queue_dir / "done" / f"{PRIOR_ID}.json", {"run_id": PRIOR_ID})
    (queue_dir / "results.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (queue_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "run_id": PRIOR_ID,
                "spec_sha256": all_hashes[PRIOR_ID],
                "status": "succeeded",
                "expected_outputs": [
                    {
                        "path": str(dependency_model),
                        "exists": True,
                        "sha256": digest(dependency_model),
                    },
                    {
                        "path": str(dependency_provenance),
                        "exists": True,
                        "sha256": digest(dependency_provenance),
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state_path = state_dir / "attempts/warm-queue-local-v3-20261008.json"
    write_json(state_path, {"status": "finished"})

    source_hashes = {relative: digest(root / relative) for relative in REQUIRED_SOURCES}
    job = {
        "attempt_id": "warm-queue-local-v4-autopilot-20261008",
        "lane": "local_gpu",
        "prepared": True,
        "predecessors": [],
        "admission_receipt": "research/campaign/test/autopilot/runtime/admission.json",
        "completion_receipt": "research/campaign/test/autopilot/runtime/completion.json",
        "expected_artifacts": ["local-v4-results.json", "local-v4-cost.json"],
        "cost_artifact_name": "local-v4-cost.json",
        "launch_command": ["/bin/false"],
        "source_hashes": source_hashes,
        "local_queue": {
            "queue_dir": "runtime/warm_queue_local_v1",
            "bandit_state": "runtime/bandit",
            "root_contract": str(contract_path.relative_to(root)),
            "prior_attempt_id": "warm-queue-local-v3-20261008",
            "prior_unit": "warm-queue-local-v3.service",
            "pending_spec_hashes": pending_hashes,
            "all_spec_hashes": all_hashes,
            "required_prior_outputs": {
                PRIOR_ID: {
                    "spec_sha256": all_hashes[PRIOR_ID],
                    "paths": [
                        str(dependency_model.relative_to(root)),
                        str(dependency_provenance.relative_to(root)),
                    ],
                }
            },
            "results_manifest": "research/campaign/test/autopilot/runtime/local-v4-results.json",
            "results_artifact_name": "local-v4-results.json",
            "cost_artifact": "research/campaign/test/autopilot/runtime/local-v4-cost.json",
        },
    }
    queue_path = base / "LOCAL_V4_FINITE_QUEUE.json"
    write_json(queue_path, {"schema_version": 1, "jobs": [job]})
    queue_sha = digest(queue_path)
    policy_path = base / "LOCAL_V4_POLICY.json"
    policy = {
        "schema_version": 1,
        "owner": "root",
        "authorized": True,
        "queue_sha256": queue_sha,
        "expires_utc": "2099-01-01T00:00:00Z",
        "poll_seconds": 10,
        "max_runtime_seconds": 86_400,
        "min_free_disk_bytes": 1,
    }
    write_json(policy_path, policy)
    review_path = base / "LOCAL_V4_ROOT_REVIEW.json"
    write_json(
        review_path,
        {
            "schema_version": 1,
            "owner": "root",
            "authorized": True,
            "queue_sha256": queue_sha,
            "policy_sha256": digest(policy_path),
            "local_contract_sha256": digest(contract_path),
            "expires_utc": policy["expires_utc"],
            "source_hashes": source_hashes,
        },
    )
    runtime = base / "runtime"
    return {
        "root": root,
        "base": base,
        "queue_dir": queue_dir,
        "state_dir": state_dir,
        "contract_path": contract_path,
        "queue_path": queue_path,
        "policy_path": policy_path,
        "review_path": review_path,
        "review_sha": digest(review_path),
        "runtime": runtime,
        "job": job,
        "all_hashes": all_hashes,
        "pending_hashes": pending_hashes,
    }


def test_root_authorization_must_be_enabled_before_config_load(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path, contract_authorized=False)
    policy, _, jobs = autopilot.load_config(
        fixture["root"],
        fixture["queue_path"],
        fixture["policy_path"],
        fixture["review_path"],
        fixture["review_sha"],
    )
    assert policy["authorized"] is True
    assert len(jobs) == 1
    result = autopilot.preflight(fixture["root"], policy, jobs[0])
    assert result == {"state": "waiting_for_root_enable"}


def test_preflight_requires_terminal_v3_and_binds_its_output_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = create_fixture(tmp_path)
    policy, _, jobs = autopilot.load_config(
        fixture["root"],
        fixture["queue_path"],
        fixture["policy_path"],
        fixture["review_path"],
        fixture["review_sha"],
    )
    monkeypatch.setattr(autopilot, "_systemd_state", lambda _unit: ("inactive", "success"))
    monkeypatch.setattr(autopilot, "_live_queue_worker", lambda _queue: False)
    monkeypatch.setattr(autopilot.bandit_resources, "parent_properties", lambda: {})
    monkeypatch.setattr(autopilot.bandit_resources, "verify_parent", lambda _props: "campaign")
    monkeypatch.setattr(autopilot.bandit_resources, "verify_parent_membership", lambda _group: None)
    monkeypatch.setattr(autopilot.bandit_resources, "memory_current", lambda _group: 1_000_000_000)
    monkeypatch.setattr(
        autopilot.bandit_resources, "verify_capacity", lambda *a, **k: {"fits": True}
    )
    monkeypatch.setattr(autopilot.bandit_resources, "gpu_memory_bytes", lambda: 0)
    monkeypatch.setattr(
        autopilot.shutil, "disk_usage", lambda _path: SimpleNamespace(free=20_000_000_000)
    )

    result = autopilot.preflight(fixture["root"], policy, jobs[0])

    assert result["state"] == "ready"
    assert result["pending_attempts"] == sorted(PENDING_IDS)
    expected_inputs = {
        "artifacts/runs/seed99/model_ep1.pt": digest(
            fixture["root"] / "artifacts/runs/seed99/model_ep1.pt"
        ),
        "artifacts/runs/seed99/provenance.json": digest(
            fixture["root"] / "artifacts/runs/seed99/provenance.json"
        ),
        "research/campaign/test/autopilot/LOCAL_V4_ROOT_CONTRACT.json": digest(
            fixture["contract_path"]
        ),
    }
    for run_id in PENDING_IDS:
        path = fixture["queue_dir"] / "pending" / f"{run_id}.json"
        expected_inputs[str(path.relative_to(fixture["root"]))] = digest(path)
    assert result["input_hashes"] == expected_inputs


def test_preflight_waits_for_v3_service_and_does_not_probe_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = create_fixture(tmp_path)
    policy, _, jobs = autopilot.load_config(
        fixture["root"],
        fixture["queue_path"],
        fixture["policy_path"],
        fixture["review_path"],
        fixture["review_sha"],
    )
    monkeypatch.setattr(autopilot, "_systemd_state", lambda _unit: ("active", None))
    monkeypatch.setattr(
        autopilot.bandit_resources,
        "parent_properties",
        lambda: pytest.fail("resource probe before drain"),
    )

    assert autopilot.preflight(fixture["root"], policy, jobs[0]) == {
        "state": "waiting_for_v3_service_drain",
        "prior_status": "finished",
    }


def test_terminal_collection_failures_are_bounded_and_need_root_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = create_fixture(tmp_path)
    job = fixture["job"]
    state_path = autopilot.bandit_resources.attempt_path(fixture["state_dir"], job["attempt_id"])
    write_json(state_path, {"status": "finished", "elapsed_seconds": 12.0})
    monkeypatch.setattr(
        autopilot.research_dispatch, "verified_completion", lambda _root, _job: False
    )

    def broken_collection(*_args: object, **_kwargs: object) -> None:
        raise autopilot.AutopilotError("missing terminal model output")

    monkeypatch.setattr(autopilot, "collect_terminal", broken_collection)
    observed = [
        autopilot._observe_worker(fixture["root"], fixture["state_dir"], fixture["runtime"], job)
        for _ in range(autopilot.MAX_COLLECTION_ATTEMPTS)
    ]

    assert [item["state"] for item in observed] == [
        "collection_retry",
        "collection_retry",
        "collection_needs_review",
    ]
    receipt = json.loads(
        (fixture["runtime"] / "collection_attempts" / f"{job['attempt_id']}.json").read_text()
    )
    assert receipt["attempts"] == autopilot.MAX_COLLECTION_ATTEMPTS
    assert receipt["status"] == "needs_review"


def test_successful_collection_saves_result_artifact_hashes_and_cost(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    for run_id, spec_hash in fixture["all_hashes"].items():
        output = tmp_path / "artifacts" / f"{run_id}.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"checkpoint for {run_id}".encode())
        (fixture["queue_dir"] / "done" / f"{run_id}.json").parent.mkdir(parents=True, exist_ok=True)
        write_json(fixture["queue_dir"] / "done" / f"{run_id}.json", {"run_id": run_id})
        with (fixture["queue_dir"] / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "spec_sha256": spec_hash,
                        "status": "succeeded",
                        "steps": [],
                        "expected_outputs": [
                            {"path": str(output), "exists": True, "sha256": digest(output)}
                        ],
                    }
                )
                + "\n"
            )

    results_path, cost_path = autopilot.collect_local(
        fixture["root"],
        fixture["runtime"],
        fixture["job"],
        {
            "status": "finished",
            "elapsed_seconds": 123.5,
            "peak_ram_bytes": 8_000_000_000,
            "peak_parent_ram_bytes": 15_000_000_000,
            "peak_observed_vram_bytes": 2_000_000_000,
        },
    )

    manifest = json.loads(results_path.read_text())
    cost = json.loads(cost_path.read_text())
    assert {row["run_id"] for row in manifest["runs"]} == set(RUNS.values())
    assert all(row["outputs"][0]["exists"] for row in manifest["runs"])
    assert cost["provider_quota_seconds"] == 123
    assert cost["actual_cost_usd"] == "0.00"


def test_stop_closes_admission_but_waits_for_live_v3_to_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = create_fixture(tmp_path)
    monkeypatch.setattr(autopilot, "_systemd_state", lambda _unit: ("active", None))
    monkeypatch.setattr(autopilot, "_live_queue_worker", lambda _queue: True)
    monkeypatch.setattr(
        autopilot.research_dispatch.subprocess,
        "run",
        lambda *a, **k: pytest.fail("must not launch"),
    )

    result = autopilot.run_cycle(
        fixture["root"],
        fixture["queue_path"],
        fixture["policy_path"],
        fixture["review_path"],
        fixture["runtime"],
        fixture["review_sha"],
        stop_requested=lambda: True,
    )

    assert result["status"] == "draining"
    assert result["should_exit"] is False
    assert result["external_v3_active"] is True
    assert (fixture["runtime"] / "STOP_ADMISSION.json").is_file()


def authorize_job(fixture: dict[str, Any]) -> None:
    policy, review, jobs = autopilot.load_config(
        fixture["root"],
        fixture["queue_path"],
        fixture["policy_path"],
        fixture["review_path"],
        fixture["review_sha"],
    )
    del policy
    autopilot._write_admission(
        fixture["root"],
        digest(fixture["queue_path"]),
        review,
        fixture["review_sha"],
        jobs[0],
        {
            str(fixture["contract_path"].relative_to(fixture["root"])): digest(
                fixture["contract_path"]
            )
        },
    )


def test_stop_barrier_blocks_an_already_admitted_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = create_fixture(tmp_path)
    authorize_job(fixture)
    monkeypatch.setattr(
        autopilot.research_dispatch.subprocess,
        "run",
        lambda *a, **k: pytest.fail("stop barrier must prevent launch"),
    )

    autopilot.request_stop(fixture["runtime"], "root_stop")
    rows = autopilot.research_dispatch.dispatch_once(
        fixture["root"], fixture["queue_path"], fixture["runtime"], execute=True
    )

    assert rows[0]["status"] == "ready"
    assert not (fixture["runtime"] / "attempts" / f"{fixture['job']['attempt_id']}.json").exists()


def test_stop_linearizes_after_a_dispatch_already_inside_launch_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = create_fixture(tmp_path)
    authorize_job(fixture)
    launch_entered = threading.Event()
    release_launch = threading.Event()

    def blocked_submit(*_args: object, **_kwargs: object) -> SimpleNamespace:
        launch_entered.set()
        assert release_launch.wait(timeout=2)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(autopilot.research_dispatch.subprocess, "run", blocked_submit)
    dispatcher = threading.Thread(
        target=autopilot.research_dispatch.dispatch_once,
        args=(fixture["root"], fixture["queue_path"], fixture["runtime"], True),
    )
    dispatcher.start()
    assert launch_entered.wait(timeout=2)
    stopper = threading.Thread(
        target=autopilot.request_stop, args=(fixture["runtime"], "root_stop")
    )
    stopper.start()
    time.sleep(0.05)
    assert not (fixture["runtime"] / "STOP_ADMISSION.json").exists()
    release_launch.set()
    dispatcher.join(timeout=2)
    stopper.join(timeout=2)

    assert not dispatcher.is_alive()
    assert not stopper.is_alive()
    assert (fixture["runtime"] / "STOP_ADMISSION.json").is_file()
    record = json.loads(
        (fixture["runtime"] / "attempts" / f"{fixture['job']['attempt_id']}.json").read_text()
    )
    assert record["status"] == "submitted"


def test_drain_poll_cadence_clears_wakeup_instead_of_spinning() -> None:
    wakeup = threading.Event()
    wakeup.set()
    autopilot.wait_for_next_cycle(wakeup, 0)
    assert not wakeup.is_set()

    started = time.monotonic()
    autopilot.wait_for_next_cycle(wakeup, 0.03)
    assert time.monotonic() - started >= 0.02


def test_service_keeps_polling_until_live_v3_drain_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = create_fixture(tmp_path)
    states = iter(
        [
            {"status": "draining", "should_exit": False, "reason": "live_v3"},
            {"status": "draining", "should_exit": True, "reason": "v3_drained"},
        ]
    )
    waits: list[int] = []
    monkeypatch.setattr(
        autopilot,
        "run_cycle",
        lambda *_args, **_kwargs: next(states),
    )
    monkeypatch.setattr(
        autopilot, "wait_for_next_cycle", lambda _event, seconds: waits.append(seconds)
    )

    result = autopilot.run_service(
        fixture["root"],
        fixture["queue_path"],
        fixture["policy_path"],
        fixture["review_path"],
        fixture["runtime"],
        fixture["review_sha"],
        once=False,
    )

    assert result == 0
    assert waits == [10]


def test_systemd_daemon_is_bounded_without_support_wrapper_timeout() -> None:
    unit = (AUTOPILOT_DIR / "systemd/wnn-research-autopilot-local-v4.service").read_text(
        encoding="utf-8"
    )
    exec_start = next(line for line in unit.splitlines() if line.startswith("ExecStart="))

    assert "After=default.target" not in unit
    assert "RuntimeMaxSec=" not in unit
    assert "WNN_RUNTIME_MAX" not in unit
    assert "/scripts/limited " not in exec_start
    assert "/.venv/bin/python " in exec_start
    assert "Slice=wnn-campaign.slice" in unit
    assert "CPUQuota=100%" in unit
    assert "MemoryMax=1000000000" in unit
    assert "MemorySwapMax=0" in unit
