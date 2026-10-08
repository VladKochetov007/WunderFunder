from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("research_dispatch", SCRIPTS / "research_dispatch.py")
assert SPEC is not None and SPEC.loader is not None
dispatcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatcher)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def cohort(root: Path) -> tuple[Path, Path, dict]:
    job = {
        "attempt_id": "test-owned-attempt",
        "lane": "local_gpu",
        "prepared": True,
        "admission_receipt": "admission.json",
        "completion_receipt": "completion.json",
        "expected_artifacts": ["checkpoint"],
        "launch_command": ["guarded-provider-adapter", "test-owned-attempt"],
    }
    queue, runtime = root / "queue.json", root / "runtime"
    write(queue, {"schema_version": 1, "jobs": [job]})
    source = root / "frozen.py"
    source.write_text("# source binding\n")
    now = datetime.now(UTC)
    write(
        root / "admission.json",
        {
            "owner": "root",
            "authorized": True,
            "attempt_id": job["attempt_id"],
            "job_sha256": dispatcher.job_digest(job),
            "created_utc": now.isoformat(),
            "expires_utc": (now + timedelta(seconds=90)).isoformat(),
            "source_hashes": {"frozen.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        },
    )
    return queue, runtime, job


def test_submit_timeout_is_not_retried_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, runtime, _ = cohort(tmp_path)
    calls = []

    def timeout(*args: object, **kwargs: object) -> None:
        calls.append(args)
        raise subprocess.TimeoutExpired("provider", 60)

    monkeypatch.setattr(dispatcher.subprocess, "run", timeout)
    assert (
        dispatcher.dispatch_once(tmp_path, queue, runtime, True)[0]["status"] == "launch_uncertain"
    )
    dispatcher.dispatch_once(tmp_path, queue, runtime, True)
    assert len(calls) == 1


def test_empty_queue_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue = tmp_path / "queue.json"
    write(queue, {"schema_version": 1, "jobs": []})
    monkeypatch.setattr(dispatcher.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    assert dispatcher.dispatch_once(tmp_path, queue, tmp_path / "runtime", True) == []


@pytest.mark.parametrize("payload", [None, b"changed"])
def test_terminal_missing_or_wrong_artifact_is_not_complete(
    tmp_path: Path, payload: bytes | None
) -> None:
    queue, runtime, job = cohort(tmp_path)
    if payload is not None:
        (tmp_path / "model.bin").write_bytes(payload)
    write(
        tmp_path / "completion.json",
        {
            "status": "complete",
            "attempt_id": job["attempt_id"],
            "job_sha256": dispatcher.job_digest(job),
            "artifacts": [
                {
                    "name": "checkpoint",
                    "path": "model.bin",
                    "sha256": hashlib.sha256(b"expected").hexdigest(),
                }
            ],
        },
    )
    assert dispatcher.dispatch_once(tmp_path, queue, runtime, False)[0]["status"] == "needs_review"


def test_changed_source_never_launches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue, runtime, _ = cohort(tmp_path)
    (tmp_path / "frozen.py").write_text("changed")
    monkeypatch.setattr(dispatcher.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    assert dispatcher.dispatch_once(tmp_path, queue, runtime, True)[0]["status"] == "needs_review"


def test_only_one_launch_per_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue, runtime, job = cohort(tmp_path)
    second = {**job, "attempt_id": "later-attempt", "prepared": False}
    write(queue, {"schema_version": 1, "jobs": [job, second]})
    calls = []

    def submit(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(args)
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(dispatcher.subprocess, "run", submit)
    dispatcher.dispatch_once(tmp_path, queue, runtime, True)
    dispatcher.dispatch_once(tmp_path, queue, runtime, True)
    assert len(calls) == 1


def test_source_path_cannot_escape_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        dispatcher.project_path(tmp_path, "../unrelated-instance.json")


def test_unfinished_attempt_outside_current_queue_still_reserves_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, runtime, _ = cohort(tmp_path)
    write(
        runtime / "attempts" / "previous.json",
        {"attempt_id": "previous", "lane": "local_gpu", "status": "launch_uncertain"},
    )
    monkeypatch.setattr(dispatcher.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    dispatcher.dispatch_once(tmp_path, queue, runtime, True)


def test_failed_paid_attempt_requires_teardown_before_release(tmp_path: Path) -> None:
    _, runtime, job = cohort(tmp_path)
    write(
        runtime / "attempts" / f"{job['attempt_id']}.json",
        {
            "attempt_id": job["attempt_id"],
            "job_sha256": dispatcher.job_digest(job),
            "lane": "vast",
            "status": "needs_review",
        },
    )
    evidence = tmp_path / "terminal.json"
    evidence.write_text('{"provider_status":"error"}')
    receipt = tmp_path / "reconcile.json"
    write(
        receipt,
        {
            "attempt_id": job["attempt_id"],
            "job_sha256": dispatcher.job_digest(job),
            "owner": "root",
            "provider_terminal_confirmed": True,
            "evidence": {"terminal.json": hashlib.sha256(evidence.read_bytes()).hexdigest()},
            "teardown_verified": False,
        },
    )
    with pytest.raises(ValueError, match="teardown"):
        dispatcher.reconcile_terminal(tmp_path, runtime, receipt)


def test_diagnosed_terminal_failure_releases_lane_for_independent_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, runtime, _ = cohort(tmp_path)
    write(
        runtime / "attempts" / "prior.json",
        {
            "attempt_id": "prior",
            "job_sha256": "bound-prior-job",
            "lane": "local_gpu",
            "status": "needs_review",
        },
    )
    evidence = tmp_path / "terminal.json"
    evidence.write_text('{"exit_status":1,"worker_exited":true}')
    receipt = tmp_path / "reconcile.json"
    write(
        receipt,
        {
            "attempt_id": "prior",
            "job_sha256": "bound-prior-job",
            "owner": "root",
            "provider_terminal_confirmed": True,
            "evidence": {"terminal.json": hashlib.sha256(evidence.read_bytes()).hexdigest()},
        },
    )
    dispatcher.reconcile_terminal(tmp_path, runtime, receipt)
    calls = []

    def submit(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(args)
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(dispatcher.subprocess, "run", submit)
    dispatcher.dispatch_once(tmp_path, queue, runtime, True)
    assert len(calls) == 1


@pytest.mark.parametrize("omit_job", [False, True])
def test_lost_completion_evidence_reserves_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, omit_job: bool
) -> None:
    queue, runtime, job = cohort(tmp_path)
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"checkpoint")
    write(
        runtime / "attempts" / f"{job['attempt_id']}.json",
        {
            "attempt_id": job["attempt_id"],
            "job_sha256": dispatcher.job_digest(job),
            "lane": job["lane"],
            "status": "submitted",
        },
    )
    write(
        tmp_path / "completion.json",
        {
            "status": "complete",
            "attempt_id": job["attempt_id"],
            "job_sha256": dispatcher.job_digest(job),
            "artifacts": [
                {
                    "name": "checkpoint",
                    "path": "model.bin",
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    monkeypatch.setattr(dispatcher.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    assert dispatcher.dispatch_once(tmp_path, queue, runtime, True)[0]["status"] == "complete"
    (tmp_path / "completion.json").unlink()
    if omit_job:
        new = {**job, "attempt_id": "next-attempt"}
        admission = json.loads((tmp_path / "admission.json").read_text())
        admission.update(attempt_id=new["attempt_id"], job_sha256=dispatcher.job_digest(new))
        write(tmp_path / "admission.json", admission)
        write(queue, {"schema_version": 1, "jobs": [new]})
    result = dispatcher.dispatch_once(tmp_path, queue, runtime, True)
    if not omit_job:
        assert result[0]["status"] == "needs_review"
        assert result[0]["holds_lane"]


def test_attempt_directory_symlink_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    queue, runtime, _ = cohort(root)
    runtime.mkdir()
    (runtime / "attempts").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        dispatcher.dispatch_once(root, queue, runtime, True)


def test_terminal_receipt_itself_must_be_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, runtime, _ = cohort(root)
    with pytest.raises(ValueError, match="escapes"):
        dispatcher.reconcile_terminal(root, runtime, tmp_path / "unrelated.json")


def test_concurrent_dispatch_processes_submit_only_once(tmp_path: Path) -> None:
    queue, runtime, job = cohort(tmp_path)
    job["launch_command"] = [
        sys.executable,
        "-c",
        "from pathlib import Path; import time; "
        "Path('launches.txt').open('a').write('submitted\\n'); time.sleep(0.1)",
    ]
    write(queue, {"schema_version": 1, "jobs": [job]})
    admission = json.loads((tmp_path / "admission.json").read_text())
    admission["job_sha256"] = dispatcher.job_digest(job)
    write(tmp_path / "admission.json", admission)
    command = [
        sys.executable,
        str(SCRIPTS / "research_dispatch.py"),
        "--root",
        str(tmp_path),
        "--queue",
        str(queue),
        "--runtime",
        str(runtime),
        "--execute",
    ]
    workers = [
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)
    ]
    for worker in workers:
        _, error = worker.communicate(timeout=10)
        assert worker.returncode == 0, error.decode()
    assert (tmp_path / "launches.txt").read_text() == "submitted\n"
