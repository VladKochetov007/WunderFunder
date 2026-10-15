"""Focused tests for the runner heartbeat boundary added by R3."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "collect_terminal_r3.py"
SPEC = importlib.util.spec_from_file_location("collect_terminal_r3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def heartbeat_document() -> dict[str, object]:
    """Return a bounded example emitted by the pinned resource heartbeat writer."""
    return {
        "event": "campaign_resource_heartbeat",
        "active": [
            {
                "name": "export_f3_teacher_cp2_triplet_20261015_r2",
                "gpu_index": 0,
                "gpu_uuid": "GPU-test",
                "elapsed_seconds": 30.0,
                "worker_timeout_seconds": 18000,
                "peak_ram_bytes": 1000,
                "peak_gpu_bytes": 2000,
            }
        ],
        "finished": [],
        "not_started": [],
        "scope": collector.HEARTBEAT_SCOPE,
    }


def test_runner_heartbeat_schema_accepts_expected_resource_fields() -> None:
    counts = collector.validate_heartbeat_document(heartbeat_document())

    assert counts == {
        "active_workers": 1,
        "finished_workers": 0,
        "not_started_workers": 0,
    }


def test_runner_heartbeat_schema_rejects_unexpected_output_fields() -> None:
    document = heartbeat_document()
    document["prediction_path"] = "cache/unexpected.npy"

    with pytest.raises(ValueError, match="top-level schema"):
        collector.validate_heartbeat_document(document)


def test_runner_heartbeat_schema_rejects_nonfinite_resource_values() -> None:
    document = heartbeat_document()
    active = document["active"]
    assert isinstance(active, list) and isinstance(active[0], dict)
    active[0]["elapsed_seconds"] = float("inf")

    with pytest.raises(ValueError, match="elapsed time"):
        collector.validate_heartbeat_document(document)


def test_output_inventory_allows_only_the_bounded_runner_heartbeat() -> None:
    heartbeat = SimpleNamespace(file_name=collector.HEARTBEAT_PATH, url="https://signed")

    def base_inventory(
        files: list[object],
        log_bytes: int,
        helper: object,
        *,
        complete: bool,
    ) -> tuple[list[dict[str, int]], list[str]]:
        assert files == []
        assert log_bytes == 0
        assert complete is True
        return [], []

    def relative_result_path(name: str) -> str:
        return name

    def expected_remote_paths() -> set[str]:
        return set()

    def declared_length(url: str, helper: object) -> int:
        return 502

    helper = SimpleNamespace(relative_result_path=relative_result_path)
    r2 = SimpleNamespace(
        expected_remote_paths=expected_remote_paths,
        inventory=base_inventory,
        declared_length=declared_length,
        MANIFEST_RESERVE_BYTES=4_000_000,
        MAX_BYTES=2_500_000_000,
    )

    rows, missing, selected = collector.inventory([heartbeat], 0, helper, r2, complete=True)

    assert rows == [{"path": collector.HEARTBEAT_PATH, "size_bytes": 502}]
    assert missing == []
    assert selected is heartbeat


def test_output_inventory_rejects_another_undeclared_runner_path() -> None:
    unexpected = SimpleNamespace(file_name="results/unexpected.json", url="https://signed")

    def relative_result_path(name: str) -> str:
        return name

    def expected_remote_paths() -> set[str]:
        return set()

    helper = SimpleNamespace(relative_result_path=relative_result_path)
    r2 = SimpleNamespace(expected_remote_paths=expected_remote_paths)

    with pytest.raises(ValueError, match="undeclared Kaggle result path"):
        collector.inventory([unexpected], 0, helper, r2, complete=True)
