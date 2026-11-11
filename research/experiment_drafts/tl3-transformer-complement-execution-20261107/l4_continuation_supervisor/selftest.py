from __future__ import annotations

import json
from pathlib import Path

import pytest
from supervisor_checks import DEADLINE, archive_complete, archiver_guard, owned_trainer


def test_original_deadline_is_not_restarted() -> None:
    """Continuation inherits the first profile's original23-hour fitting maximum."""
    assert DEADLINE == 1794154066.717658


@pytest.mark.parametrize("result", [(1, False, {}), (0, True, {}), (0, False, {})])
def test_archiver_failure_or_unproven_exit_closes_training(tmp_path: Path, result: tuple) -> None:
    """Nonzero, timeout and clean-but-incomplete exits all fail retention admission."""
    with pytest.raises(RuntimeError, match="archiver exited"):
        archiver_guard(tmp_path, "proof", None, result, lambda _: True)


def test_live_archiver_and_final_clean_exit_only(tmp_path: Path) -> None:
    """A matching final12E proof permits clean retention finalization after trainer completion."""
    archiver_guard(tmp_path, "proof", {"child": {"pid": 42}}, None, lambda _: True)
    with pytest.raises(RuntimeError, match="live identity"):
        archiver_guard(tmp_path, "proof", {"child": {"pid": 42}}, None, lambda _: False)
    value = {
        "schema": "wnn.gru-safe-state-archive-complete.v1",
        "completed_epochs": 12,
        "transition_proof_sha256": "proof",
        "initial_epoch": 6,
        "archive_epochs": list(range(6, 13)),
    }
    (tmp_path / "archive_complete.json").write_text(json.dumps(value))
    assert archive_complete(tmp_path, "proof")
    archiver_guard(tmp_path, "proof", None, (0, False, {}), lambda _: False)
    with pytest.raises(RuntimeError):
        archiver_guard(tmp_path, "different", None, (0, False, {}), lambda _: False)


def test_foreign_or_reused_old_owner_is_never_stopped(tmp_path: Path) -> None:
    """Reject ownership/path mismatches before discovering or signalling any descendant."""
    leader = {"pid": 42, "pgid": 42, "birth": "old"}
    receipt = {
        "schema": "wnn.multistage-owned-process.v1",
        "mode": "fit",
        "owner": {"cell_id": "gru_s101001"},
        "output_dir": str(tmp_path),
        "gpu_key": "0",
        "child": leader,
    }
    with pytest.raises(ValueError, match="PID/birth"):
        owned_trainer(
            receipt, tmp_path, tmp_path, 101001, 0, lambda _: leader | {"birth": "reused"}
        )
    with pytest.raises(ValueError, match="intended"):
        owned_trainer(receipt, tmp_path, tmp_path, 101002, 1, lambda _: leader)


def test_failed_preflight_never_reaches_old_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed exact-contract preflight leaves the original trainer entirely untouched."""
    import supervisor

    worker = supervisor.Supervisor.__new__(supervisor.Supervisor)
    worker.control = tmp_path
    worker.plan = {"python": "python", "tools": str(tmp_path)}
    old = tmp_path / "old"
    old.mkdir()
    (old / "training_state.pt").write_bytes(b"not loaded by failed preflight")
    worker.execute = lambda *args, **kwargs: (1, False, {})
    stopped = []
    monkeypatch.setattr(
        supervisor, "terminate_owned_process", lambda *args, **kwargs: stopped.append(args)
    )
    with pytest.raises(RuntimeError, match="preflight failed"):
        worker.worker({"seed": 101001, "request": {"old_output": str(old)}})
    assert stopped == []


def test_retention_cleanup_signals_only_this_new_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continuation retention failure excludes foreign receipts and the other healthy GRU."""
    import supervisor

    worker = supervisor.Supervisor.__new__(supervisor.Supervisor)
    worker.owner = {"pid": 1, "birth": "ours", "pgid": 1}
    worker.active = {}
    for label, owner, pid in [
        ("101001-resume", worker.owner, 10),
        ("101001-foreign", {"pid": 999}, 11),
        ("101002-resume", worker.owner, 12),
    ]:
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({"owner": {"controller": owner}, "child": {"pid": pid}}))
        worker.active[label] = path
    stopped = []
    monkeypatch.setattr(supervisor, "is_same_process", lambda _: True)
    monkeypatch.setattr(
        supervisor,
        "terminate_owned_process",
        lambda identity, **kwargs: stopped.append(identity["pid"]),
    )
    worker.terminate_new(101001)
    assert stopped == [10]


def test_relay_observer_trainer_chain_selects_only_actual_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nested observer argv include trainer text but never satisfy exact argv[1] ownership."""
    from types import SimpleNamespace

    source, old = tmp_path / "source", tmp_path / "old"
    train = ["python", str(source / "scripts/train_sequence.py"), "--output", str(old)]
    observer = [
        "python",
        str(source / "scripts/observe_sequence.py"),
        "--output",
        str(old),
        "--",
        *train,
    ]
    relay = ["python", str(tmp_path / "observed_worker.py"), "--", *observer]
    raw = {
        42: (1, 42, "a"),
        43: (42, 42, "b"),
        44: (43, 42, "c"),
        55: (1, 55, "foreign"),
        56: (43, 42, "reused-old"),
    }
    argv = {42: relay, 43: observer, 44: train, 55: train, 56: train}
    identities = {
        pid: {"pid": pid, "pgid": group, "birth": birth} for pid, (_, group, birth) in raw.items()
    }
    identities[56]["birth"] = "reused-new"

    class Stat:
        """Provide one procfs row with a real parent/group/birth tuple."""

        def __init__(self, pid: int) -> None:
            self.pid, self.parent = pid, SimpleNamespace(name=str(pid))

        def read_text(self) -> str:
            parent, group, birth = raw[self.pid]
            fields = ["S", str(parent), str(group), *(["0"] * 16), birth, "0", "0"]
            return f"{self.pid} (fixture) " + " ".join(fields)

    original_glob, original_read = Path.glob, Path.read_bytes
    monkeypatch.setattr(
        Path,
        "glob",
        lambda path, pattern: (
            [Stat(pid) for pid in raw]
            if path.as_posix() == "/proc"
            else original_glob(path, pattern)
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            ("\0".join(argv[int(path.parent.name)]) + "\0").encode()
            if path.name == "cmdline"
            else original_read(path)
        ),
    )
    receipt = {
        "schema": "wnn.multistage-owned-process.v1",
        "mode": "fit",
        "gpu_key": "0",
        "owner": {"cell_id": "gru_s101001"},
        "output_dir": str(old),
        "child": identities[42],
    }
    assert owned_trainer(receipt, old, source, 101001, 0, identities.get) == identities[44]
