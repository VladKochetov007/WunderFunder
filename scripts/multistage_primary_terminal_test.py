from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_multistage_primary_terminal as terminal  # noqa: E402


def _actual_packet(monkeypatch: pytest.MonkeyPatch) -> Path:
    packet = (
        Path(__file__).resolve().parents[1]
        / "research/experiment_drafts/multistage-optuna-b300-20261031"
        / "receiving/final-evidence-r1/budget-snapshot-r1"
    )
    if not packet.is_dir():
        pytest.skip("original receiver budget packet is not available")

    def reference(path: object, expected_sha: object) -> tuple[PurePosixPath, dict[str, object]]:
        original = PurePosixPath(str(path))
        source = packet / original.name
        assert terminal._sha256(source) == expected_sha
        return original, terminal._read_json(source)

    monkeypatch.setattr(terminal, "_reference", reference)
    return packet


def test_actual_legacy_budget_qualifies_unknown_exits_without_ledger_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _actual_packet(monkeypatch)
    before = (packet / "budget.json").read_bytes()
    monkeypatch.setattr(terminal, "_process_snapshot", lambda: [])

    report = terminal.qualify_primary_fit_terminal(packet)

    assert (packet / "budget.json").read_bytes() == before
    assert report["ledger_modified"] is False
    legacy = [row for row in report["fit_attempts"] if row["qualification"] != "ledger_terminal"]
    assert len(legacy) == 10
    assert all(row["exit_code"] is None for row in legacy)
    assert all(row["process_exit_code_observed"] is False for row in legacy)
    assert sum(row["ledger_status"] == "running" for row in legacy) == 1


def test_actual_legacy_budget_refuses_live_descendant_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _actual_packet(monkeypatch)
    owner = json.loads((packet / "fit-r6-galam64.owner.json").read_bytes())
    monkeypatch.setattr(
        terminal,
        "_process_snapshot",
        lambda: [{"pid": 999, "pgid": owner["pgid"], "birth": "different", "state": "S"}],
    )

    with pytest.raises(ValueError, match="live owned process/group"):
        terminal.qualify_primary_fit_terminal(packet)


def test_actual_legacy_budget_allows_zombie_owned_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _actual_packet(monkeypatch)
    owner = json.loads((packet / "fit-r6-galam64.owner.json").read_bytes())
    monkeypatch.setattr(
        terminal,
        "_process_snapshot",
        lambda: [{key: owner[key] for key in ("pid", "pgid", "birth")} | {"state": "Z"}],
    )

    assert (
        terminal.qualify_primary_fit_terminal(packet)["status"]
        == "all_fit_processes_qualified_terminal"
    )


def test_actual_legacy_budget_refuses_orphan_trainer_in_separate_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _actual_packet(monkeypatch)
    owner = json.loads((packet / "fit-stage2-gru-polish-r1.owner.json").read_bytes())
    monkeypatch.setattr(
        terminal,
        "_process_snapshot",
        lambda: [
            {
                "pid": 999,
                "pgid": 999,
                "birth": "different",
                "state": "S",
                "command": ["python", "train_sequence.py", "--output", owner["output_dir"]],
            }
        ],
    )

    with pytest.raises(ValueError, match="process using its output"):
        terminal.qualify_primary_fit_terminal(packet)
