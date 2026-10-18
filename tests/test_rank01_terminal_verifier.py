from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/campaign_2026-10-18/priority25_execution_r4"
HELPER_PATH = PACKET / "receipts/verify_e6_terminal_endpoint.py"


@pytest.fixture
def verifier() -> ModuleType:
    """Import only the endpoint verifier's schema helpers; do not load checkpoints."""
    spec = importlib.util.spec_from_file_location("rank1_endpoint_verifier", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stage1_history(arm: str) -> list[dict[str, object]]:
    """Use actual wide or deep producer history from its verified stage1 receipt."""
    filename = f"RANK01_{arm.upper()}_F0_S1001_STAGE1_VERIFICATION.json"
    receipt = json.loads((PACKET / "receipts" / filename).read_text())
    return receipt.get("history", receipt.get("epoch_summaries"))


def terminal_history(arm: str = "wide") -> list[dict[str, object]]:
    """Extend real producer rows into a producer-shaped six-epoch test fixture."""
    rows = copy.deepcopy(stage1_history(arm))
    for epoch in range(2, 6):
        row = copy.deepcopy(rows[-1])
        row["epoch"] = epoch
        completed = (epoch + 1) * 14_920
        row["update_diagnostics"]["cumulative"] = {
            "optimizer_attempts": completed,
            "amp_skipped_updates": 0,
            "optimizer_completed_updates": completed,
            "ema_updates": 0,
            "completed_batches": (epoch + 1) * 1492,
        }
        row["update_diagnostics"]["grad_stats"]["epoch"] = epoch
        row["update_diagnostics"]["grad_stats"]["steps"] = 14_920
        rows.append(row)
    return rows


def digest(path: Path) -> str:
    """Hash small fixture files for collection receipt validation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("arm", ["wide", "deep"])
def test_history_accepts_actual_producer_schema_and_six_fit_epochs(
    verifier: ModuleType, arm: str
) -> None:
    """Both actual stage1 histories remain unchanged as the terminal fixture prefix."""
    parent = stage1_history(arm)
    history = terminal_history(arm)
    result = verifier._history(history, parent)
    assert len(result) == 6
    assert result[0]["fit_summary"]["optimizer_updates"] == result[0]["update_diagnostics"]["epoch"]


def test_history_rejects_scalar_optimizer_summary(verifier: ModuleType) -> None:
    """The producer stores optimizer_updates as a counter mapping, not a scalar."""
    history = terminal_history("wide")
    history[3]["fit_summary"]["optimizer_updates"] = 14_920
    with pytest.raises(ValueError, match="optimizer summary"):
        verifier._history(history, stage1_history("wide"))


def test_attempt_requires_explicit_finished_zero_exit(verifier: ModuleType) -> None:
    """A missing or failed exit code cannot pass terminal collection verification."""
    arm = verifier.ARMS["wide"]
    plan = json.loads((PACKET / "F0_PAIR_PLAN.json").read_text())
    command = next(
        row["expected_command"] for row in plan["steps"] if row["attempt"] == arm.attempt
    )
    record = {
        "id": arm.attempt,
        "status": "finished",
        "exit_status": 0,
        "command": command,
        "unit": f"wnn-bandit-{arm.attempt}.service",
        "working_directory": "/opt/source-a266b036",
    }
    verifier._terminal(record, arm, command)
    del record["exit_status"]
    with pytest.raises(ValueError, match="attempt exit_status"):
        verifier._terminal(record, arm, command)


def test_collection_accepts_terminal_collector_seven_file_stats(
    verifier: ModuleType, tmp_path: Path
) -> None:
    """The terminal collector tracks seven stable files but exposes five artifacts."""
    arm = verifier.ARMS["wide"]
    snapshot = tmp_path / "epoch5_terminal"
    snapshot.mkdir()
    for name in (*verifier.ARTIFACTS, "attempt.json", "training.log"):
        (snapshot / name).write_bytes(name.encode())
    record = {"id": arm.attempt, "status": "finished", "exit_status": 0}
    (snapshot / "attempt.json").write_text(json.dumps(record))
    files = (*verifier.ARTIFACTS, "attempt.json", "training.log")
    stats = {
        name: {"size": (snapshot / name).stat().st_size, "mtime_ns": 1, "inode": i}
        for i, name in enumerate(files)
    }
    artifacts = {
        name: {
            "path": str(snapshot / name),
            "bytes": (snapshot / name).stat().st_size,
            "sha256": digest(snapshot / name),
        }
        for name in verifier.ARTIFACTS
    }
    attempt = {
        "path": str(snapshot / "attempt.json"),
        "bytes": (snapshot / "attempt.json").stat().st_size,
        "sha256": digest(snapshot / "attempt.json"),
        "status": "finished",
        "exit_status": 0,
    }
    log = {
        "path": str(snapshot / "training.log"),
        "bytes": (snapshot / "training.log").stat().st_size,
        "sha256": digest(snapshot / "training.log"),
    }
    receipt = {
        "run": arm.run,
        "epoch_index": 5,
        "source_directory": str(verifier.REMOTE / "data/artifacts/runs" / arm.run),
        "snapshot_directory": str(snapshot),
        "source_stats_before": stats,
        "source_stats_after": copy.deepcopy(stats),
        "artifacts": artifacts,
        "attempt_evidence": attempt,
        "log_evidence": log,
        "source_snapshot_hash_match": True,
        "producer_model_hash_matches": True,
    }
    receipt_path = tmp_path / "collection.json"
    receipt_path.write_text(json.dumps(receipt))
    fake_r4 = SimpleNamespace(sha256_file=lambda path: digest(path))
    results, _ = verifier._collection(
        arm, snapshot, receipt_path, snapshot / "attempt.json", fake_r4
    )
    assert set(results) == set(verifier.ARTIFACTS)

    receipt["source_stats_after"]["training.log"]["mtime_ns"] = 2
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="stable source/collected-file set"):
        verifier._collection(arm, snapshot, receipt_path, snapshot / "attempt.json", fake_r4)
