"""Terminal PR15 resource records require durable receipts and no live actors."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

CONTROL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTROL))

import resource_watch  # noqa: E402


def _terminal_inputs(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path, dict[str, object]]:
    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    (state / "COMPLETE.json").write_text(
        json.dumps({"completed_successor_cells": 104, "preserved_original_fits": 16})
    )
    (state / "RESULTS.json").write_text(
        json.dumps({f"cell-{index}": {"status": "completed", "returncode": 0} for index in range(104)})
    )
    properties = {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "Result": "success",
        "MainPID": "0",
        "ControlGroup": "",
        "MemoryMax": str(192 * 1024**3),
        "MemorySwapMax": "0",
        "TasksMax": "512",
    }
    history = {
        "lifecycle": "live",
        "control_group": "/user.slice/tabdata-pr15-execution-v2.service",
        "memory_max_bytes": 192 * 1024**3,
        "memory_swap_max_bytes": 0,
        "pids_max": 512,
        "systemd_memory_max_bytes": 192 * 1024**3,
        "systemd_memory_swap_max_bytes": 0,
        "systemd_pids_max": 512,
        "memory_events": dict.fromkeys(resource_watch.EVENT_KEYS, 0),
        "observed_unix": time.time() - 1,
    }
    return properties, cgroup_root, proc_root, state, history


def test_terminal_pr15_requires_and_records_real_completion(tmp_path: Path) -> None:
    properties, cgroups, proc, state, history = _terminal_inputs(tmp_path)
    record = resource_watch._terminal_pr15_sample(
        properties=properties,
        cgroup_root=cgroups,
        proc_root=proc,
        state_root=state,
        previous=history,
    )

    assert record is not None
    assert record["memory_current_bytes"] == 0
    assert record["memory_current_basis"] == "cgroup_absent_and_no_process_members"
    assert record["memory_events_last_observed"] == dict.fromkeys(resource_watch.EVENT_KEYS, 0)
    assert record["terminal_receipt"]["results_count"] == 104
    assert record["terminal_receipt"]["systemd_load_state"] == "loaded"
    assert "memory_events" not in record


def test_terminal_pr15_accepts_systemd_gc_only_with_historical_caps(tmp_path: Path) -> None:
    properties, cgroups, proc, state, history = _terminal_inputs(tmp_path)
    properties.update(
        LoadState="not-found",
        Result="success",
        MemoryMax="infinity",
        MemorySwapMax="infinity",
        TasksMax="3355442",
    )
    record = resource_watch._terminal_pr15_sample(
        properties=properties,
        cgroup_root=cgroups,
        proc_root=proc,
        state_root=state,
        previous=history,
    )

    assert record is not None
    terminal = record["terminal_receipt"]
    assert terminal["systemd_load_state"] == "not-found"
    assert terminal["systemd_result_observed"] is None
    assert terminal["hard_caps_basis"] == "last_observed_live_cgroup_and_systemd_properties"


def test_first_terminal_sample_rejects_stale_live_history(tmp_path: Path) -> None:
    properties, cgroups, proc, state, history = _terminal_inputs(tmp_path)
    history["observed_unix"] = time.time() - 25
    with pytest.raises(resource_watch.ObservedResourceViolation, match="more than 20 seconds"):
        resource_watch._terminal_pr15_sample(
            properties=properties,
            cgroup_root=cgroups,
            proc_root=proc,
            state_root=state,
            previous=history,
        )


def test_accepted_terminal_proof_survives_later_systemd_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    properties, cgroups, proc, state, history = _terminal_inputs(tmp_path)
    first = resource_watch._terminal_pr15_sample(
        properties=properties,
        cgroup_root=cgroups,
        proc_root=proc,
        state_root=state,
        previous=history,
    )
    history["terminal_proof"] = first["terminal_receipt"]["terminal_proof"]
    accepted = history["terminal_proof"]["first_terminal_observed_unix"]
    properties.update(
        LoadState="not-found",
        MemoryMax="infinity",
        MemorySwapMax="infinity",
        TasksMax="3355442",
    )
    monkeypatch.setattr(resource_watch.time, "time", lambda: accepted + 3600)

    later = resource_watch._terminal_pr15_sample(
        properties=properties,
        cgroup_root=cgroups,
        proc_root=proc,
        state_root=state,
        previous=history,
    )

    assert later is not None
    assert later["terminal_receipt"]["terminal_proof"]["first_terminal_observed_unix"] == accepted


def test_terminal_pr15_rejects_remaining_cgroup_or_process(tmp_path: Path) -> None:
    properties, cgroups, proc, state, history = _terminal_inputs(tmp_path)
    group = cgroups / history["control_group"].lstrip("/")
    group.mkdir(parents=True)
    with pytest.raises(ValueError, match="cgroup still exists"):
        resource_watch._terminal_pr15_sample(
            properties=properties,
            cgroup_root=cgroups,
            proc_root=proc,
            state_root=state,
            previous=history,
        )


def test_transient_invalid_sample_does_not_refresh_or_replace_last_good(tmp_path: Path) -> None:
    output = tmp_path / "live_resources.json"
    good_bytes = b'{"verified": true, "observed_unix": 10.0}\n'
    output.write_bytes(good_bytes)
    invalid = {"verified": False, "observed_unix": 20.0, "errors": ["transient proc race"]}

    written = resource_watch.publish_sample(output, invalid)

    assert output.read_bytes() == good_bytes
    assert written == tmp_path / "live_resources.json.invalid.json"
    assert json.loads(written.read_text(encoding="utf-8")) == invalid


def test_observed_hard_cap_violation_replaces_last_good_immediately(tmp_path: Path) -> None:
    output = tmp_path / "live_resources.json"
    output.write_text('{"verified": true, "observed_unix": 10.0}\n', encoding="utf-8")
    violation = {
        "verified": False,
        "observed_unix": 20.0,
        "observed_violations": ["PR15 cgroup cap changed"],
    }

    written = resource_watch.publish_sample(output, violation)

    assert written == output
    assert json.loads(output.read_text(encoding="utf-8")) == violation
