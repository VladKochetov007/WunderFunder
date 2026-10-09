from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "h217_worker", Path(__file__).with_name("h217_worker.py")
)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def test_incomplete_updates_are_not_admitted(tmp_path: Path) -> None:
    path = tmp_path / "profile.log"
    path.write_text(
        'PROFILE {"step_wall": 20}\nPROFILE_UPDATE_EVIDENCE {"valid_for_throughput": false}\n'
    )
    with pytest.raises(ValueError, match="finite completed"):
        WORKER.profile_gate(path, {"batch_size": 8}, 10_000)


def test_profile_reserves_time_for_full_fit_and_exports(tmp_path: Path) -> None:
    path = tmp_path / "profile.log"
    path.write_text(
        'PROFILE {"step_wall": 1000}\nPROFILE_UPDATE_EVIDENCE {"valid_for_throughput": true}\n'
    )
    with pytest.raises(TimeoutError, match="allocation"):
        WORKER.profile_gate(path, {"batch_size": 8}, 1000)


def test_setup_failure_still_delivers_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "results"
    monkeypatch.setenv("WNN_OUTPUT_DIR", str(output))
    monkeypatch.setenv("WNN_MAX_RUNTIME_SECONDS", "180")
    monkeypatch.setenv("WNN_INPUT_MANIFEST", "invalid-json")
    assert WORKER.main() == 1
    assert json.loads((output / "result.json").read_text())["status"] == "failed"
    assert (output / "artifacts.tar").stat().st_size > 0


def test_stopping_completed_child_is_harmless() -> None:
    import subprocess

    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    child.wait(timeout=5)
    WORKER.stop(child)
