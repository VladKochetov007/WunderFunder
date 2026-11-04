from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from scripts import l4_objective_20261104 as adapter
from scripts.l4_teacher_capacity_20261104 import runner

IDENTITY = {
    "source_binding_sha256": "1" * 64,
    "data_binding_sha256": "2" * 64,
    "split_sha256": "3" * 64,
    "fit_members_sha256": "4" * 64,
}


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path, *, profile_exit: int = 0) -> tuple[Path, Path, Path, Path]:
    runtime, project = tmp_path / "runtime", tmp_path / "project"
    cells = [
        {
            "cell_id": f"cell_{slot}",
            "run_name": f"run_{slot}",
            "gpu_slot": slot,
            "gpu_uuid": f"GPU-{slot}",
            "cpu_threads": 8,
            "profile_argv": ["python", "profile.py"],
            "fit_argv": ["python", "fit.py"],
        }
        for slot in range(2, 6)
    ]
    panel = {
        "schema": "wnn.l4-objective-panel.v1",
        "identity": IDENTITY,
        "cells": cells,
        "paths": {
            "disk_mount": str(tmp_path),
            "runtime_root": str(runtime),
            "project_root": str(project),
        },
        "budget": {"max_output_bytes": 2_500_000_000},
    }
    panel_path = tmp_path / "panel.json"
    panel_sha = _write(panel_path, panel)
    authority_path = tmp_path / "authority.json"
    authority_sha = _write(
        authority_path,
        {
            "schema": "wnn.root-authority.v1",
            "max_new_scientific_fits": 6,
            "collection_reserve_seconds": 2700,
            "max_new_output_bytes": 2_500_000_000,
            "minimum_free_data2_bytes": 3_000_000_000,
            "reserved_external_output_bytes": 500_000_000,
            "max_aggregate_cpu_cores": 64,
            "new_fit_gpu_indices": [2, 3, 4, 5],
            "hard_deadline_epoch": time.time() + 7200,
        },
    )
    admission_path = tmp_path / "admission.json"
    admission_sha = _write(
        admission_path,
        {
            "schema": "wnn.l4-objective-admission.v1",
            "status": "approved",
            "authority_sha256": authority_sha,
            "panel_sha256": panel_sha,
            **IDENTITY,
        },
    )
    receipts = {}
    for cell in cells:
        receipt_path = runner.panel_path(panel, cell, "profile")["receipt"]
        receipts[cell["cell_id"]] = _write(
            receipt_path,
            {
                "status": "completed",
                "exit_code": profile_exit,
                "timed_out": False,
                "owner": {"cell_id": cell["cell_id"], "panel_sha256": panel_sha},
            },
        )
    review_path = tmp_path / "review.json"
    _write(
        review_path,
        {
            "schema": "wnn.l4-objective-profile-review.v1",
            "status": "approved",
            "panel_sha256": panel_sha,
            "admission_receipt_sha256": admission_sha,
            **IDENTITY,
            "profile_receipts": receipts,
        },
    )
    return panel_path, authority_path, admission_path, review_path


def test_profile_binds_authority_and_enforces_resource_reserve(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    panel, _, panel_sha, _, _, review_sha = adapter.validate_phase(*paths[:3], "profile", None)

    assert panel_sha == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    assert review_sha is None
    assert panel["budget"]["minimum_free_bytes"] == 3_500_000_000


def test_profile_phase_dispatches_exactly_four_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel_path, authority_path, admission_path, _ = _bundle(tmp_path)
    panel, authority, panel_sha, authority_sha, admission_sha, review_sha = adapter.validate_phase(
        panel_path, authority_path, admission_path, "profile", None
    )
    started: list[str] = []

    def run_cell(
        _panel: dict[str, object],
        cell: dict[str, object],
        _phase: str,
        _state: dict[str, object],
        _index: int,
    ) -> dict[str, object]:
        started.append(str(cell["cell_id"]))
        return {"cell_id": cell["cell_id"], "exit_code": 0}

    monkeypatch.setattr(adapter.runner, "run_cell", run_cell)
    summary = adapter.run_phase(
        panel, authority, "profile", panel_sha, authority_sha, admission_sha, review_sha
    )

    assert len(started) == 4
    assert (
        summary["expected_cells"] == summary["attempted_cells"] == summary["completed_cells"] == 4
    )
    assert summary["failed_cells"] == 0


def test_fit_requires_review_bound_to_all_four_successful_profiles(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    _, _, _, _, _, review_sha = adapter.validate_phase(*paths[:3], "fit", paths[3])
    assert review_sha == hashlib.sha256(paths[3].read_bytes()).hexdigest()


def test_fit_rejects_unsuccessful_profile_receipt(tmp_path: Path) -> None:
    paths = _bundle(tmp_path, profile_exit=1)
    with pytest.raises(ValueError, match="unsuccessful profile receipt"):
        adapter.validate_phase(*paths[:3], "fit", paths[3])


def test_admission_cannot_be_reused_for_changed_authority(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    authority = json.loads(paths[1].read_text(encoding="utf-8"))
    authority["hard_deadline_epoch"] += 1
    _write(paths[1], authority)
    with pytest.raises(ValueError, match="admission does not bind"):
        adapter.validate_phase(*paths[:3], "profile", None)
