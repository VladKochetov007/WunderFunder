from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path

import pytest
import recovery_checks as checks
import recovery_supervisor as runtime
import torch


def test_model_state_ordered_dict_projection_preserves_strict_payload() -> None:
    """CPU projection preserves real state_dict containers and still refuses changed tensor bytes."""
    from transition_support_s4 import same_payload

    model = torch.nn.Linear(3, 2)
    saved = model.state_dict()
    assert type(saved) is OrderedDict
    projected = type(saved)((key, value.cpu()) for key, value in model.state_dict().items())
    assert list(projected) == list(saved)
    assert type(projected) is type(saved)
    assert same_payload(projected, saved)
    changed = OrderedDict((key, value.clone()) for key, value in projected.items())
    changed["weight"][0, 0] += 1
    assert not same_payload(changed, saved)
    assert not same_payload(dict(projected), saved)


def test_callback_failure_returns_to_native_popen_poll_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy helper never receives the exception that previously waited until deadline."""
    runner = runtime.Recovery.__new__(runtime.Recovery)
    runner.failures, runner.active, runner.lock = {}, {}, threading.RLock()
    runner.control, runner.owner, runner.plan_sha = tmp_path, {"pid": 1}, "plan"
    runner.guard = lambda: None
    runner.environment = lambda *args, **kwargs: {}
    stopped, polled = [], []
    runner.terminate_new = stopped.append

    def native(command: list[str], **kwargs: object) -> tuple:
        kwargs["progress"]()
        polled.append(True)
        return -9, False, {}

    def retention_failure() -> None:
        raise RuntimeError("archiver died")

    monkeypatch.setattr(runtime, "run_owned_command", native)
    with pytest.raises(RuntimeError, match="archiver died"):
        runner.execute(
            {
                "seed": 101001,
                "gpu_index": 0,
                "request": {"new_source_root": str(tmp_path), "new_output": str(tmp_path / "fit")},
            },
            "resume",
            ["owned"],
            progress=retention_failure,
        )
    assert polled == [True] and stopped == [101001]


def test_immediate_cleanup_excludes_other_seed_and_foreign_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3 cancellation signals only exact receipt-owned new groups, with zero grace."""
    runner = runtime.Recovery.__new__(runtime.Recovery)
    runner.owner, runner.active = {"pid": 1, "birth": "ours"}, {}
    for key, owner, pid in [
        ("101001-resume", runner.owner, 10),
        ("101001-foreign", {"pid": 99}, 11),
        ("101002-resume", runner.owner, 12),
    ]:
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps({"owner": {"controller": owner}, "child": {"pid": pid}}))
        runner.active[key] = path
    stopped = []
    monkeypatch.setattr(runtime, "is_same_process", lambda _: True)
    monkeypatch.setattr(
        runtime,
        "terminate_owned_process",
        lambda owner, *, grace_seconds: stopped.append((owner["pid"], grace_seconds)),
    )
    runner.terminate_new(101001)
    assert stopped == [(10, 0)]


@pytest.mark.parametrize("device,stubbed,real", [("cpu", True, False), ("cuda", False, True)])
def test_probe_receipts_distinguish_cpu_stub_and_actual_cuda(
    tmp_path: Path, device: str, stubbed: bool, real: bool
) -> None:
    """A CPU restoration proof can never masquerade as actual CUDA restoration."""
    path = tmp_path / "probe.json"
    value = {
        "schema": "wnn.gru-canonical-resume-probe.v3",
        "status": "passed",
        "device": device,
        "transition_proof_sha256": "proof",
        "next_epoch": 8,
        "global_step": 51680,
        "optimizer_updates": 0,
        "forward_calls": 0,
        "cuda_rng_setter_stubbed": stubbed,
        "real_cuda_resume_exercised": real,
    }
    path.write_text(json.dumps(value))
    assert checks.probe_ready(path, "proof", device)
    value["optimizer_updates"] = 1
    path.write_text(json.dumps(value))
    assert not checks.probe_ready(path, "proof", device)


def test_parallel_freshness_is_scoped_to_own_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One cell creating its owned sibling output cannot falsely fail the other cell's preflight."""
    registry = {"cleanup": {}}
    monkeypatch.setattr(checks, "pinned", lambda spec: registry[spec["path"]])
    actual_digest = checks.digest
    parents = {}
    monkeypatch.setattr(
        checks,
        "digest",
        lambda path: (
            parents.get(str(path), actual_digest(path)) if path.exists() else parents[str(path)]
        ),
    )
    cells = []
    for seed, gpu in ((101001, 0), (101002, 1)):
        old = tmp_path / f"runs/gru_s{seed}/fit"
        old.mkdir(parents=True)
        checkpoint = old / "training_state.pt"
        checkpoint.write_bytes(b"fixture")
        sha, pid, birth = checks.PARENTS[seed]
        parents[str(checkpoint)] = sha
        registry[f"clock{seed}"] = {"first_profile_epoch": checks.DEADLINE - 82800}
        registry[f"profile{seed}"] = {"approved": True, "reviewer": "ROOT", "updates": 20}
        parent = {
            "old_output": str(old),
            "old_metadata": {},
            "old_checkpoint_sha256": sha,
            "expected_next_epoch": 8,
            "expected_global_step": 51680,
            "old_source_root": "S1",
            "old_source_manifest": {},
            "first_profile_receipt": {"path": f"clock{seed}"},
            "not_after_epoch": checks.DEADLINE,
            "stopped_trainer_owner": {"pid": pid, "birth": birth},
        }
        registry[f"parent{seed}"] = parent
        request = parent | {
            "new_output": str(tmp_path / f"runs/gru_s{seed}_monitor_opt_r3/fit"),
            "approved_profile": {"path": f"profile{seed}"},
        }
        cells.append(
            {
                "seed": seed,
                "gpu_index": gpu,
                "parent_request": {"path": f"parent{seed}"},
                "request": request,
            }
        )
    tools = []
    for name in (
        "recovery_supervisor.py",
        "recovery_checks.py",
        "canonical_resume_probe.py",
        "run_probe.py",
        "migrate_monitor_checkpoint_s4.py",
        "transition_support_s4.py",
        "archive_safe_state_s4.py",
        "supervisor.py",
        "supervisor_checks.py",
        "observed_worker.py",
    ):
        path = tmp_path / name
        path.write_text(name)
        tools.append({"path": str(path), "sha256": actual_digest(path)})
    plan = {
        "schema": "wnn.gru-monitor-recovery.v3",
        "approved": True,
        "reviewer": "ROOT",
        "runtime": str(tmp_path),
        "root_cleanup_proof": {"path": "cleanup"},
        "historical_owned_receipts": [],
        "tool_pins": tools,
        "cells": cells,
    }
    Path(cells[1]["request"]["new_output"]).mkdir(parents=True)
    checks.validate_recovery(plan, lambda _: False, fresh_seed=101001)
    with pytest.raises(ValueError, match="fresh R3"):
        checks.validate_recovery(plan, lambda _: False)
