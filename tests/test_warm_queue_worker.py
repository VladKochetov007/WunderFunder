"""Tests for warm_queue_worker."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

# Import the worker module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from warm_queue_worker import QueueWorker, _compute_sha256, _validate_run_id


class TestValidation:
    """Test input validation helpers."""

    def test_validate_run_id_valid(self):
        """Valid run IDs should pass."""
        assert _validate_run_id("run_1")
        assert _validate_run_id("run-2")
        assert _validate_run_id("run.3")
        assert _validate_run_id("A" * 96)

    def test_validate_run_id_invalid(self):
        """Invalid run IDs should fail."""
        assert not _validate_run_id("run@1")
        assert not _validate_run_id("run/1")
        assert not _validate_run_id("")
        assert not _validate_run_id("A" * 97)

    def test_compute_sha256(self):
        """SHA256 computation should be correct."""
        data = b"hello"
        expected = hashlib.sha256(data).hexdigest()
        assert _compute_sha256(data) == expected


class TestQueueWorker:
    """Test QueueWorker core functionality."""

    def test_initialization(self, tmp_path):
        """Worker should initialize with correct queue structure."""
        queue_dir = tmp_path / "queue"
        worker = QueueWorker(queue_dir)

        assert worker.queue_dir == queue_dir.resolve()
        assert (queue_dir / "pending").exists()
        assert (queue_dir / "running").exists()
        assert (queue_dir / "done").exists()
        assert (queue_dir / "failed").exists()
        assert (queue_dir / "logs").exists()

    def test_heartbeat_writing(self, tmp_path):
        """Heartbeat should be written as valid JSON."""
        queue_dir = tmp_path / "queue"
        worker = QueueWorker(queue_dir)
        worker._write_heartbeat()

        hb = json.loads((queue_dir / "worker_heartbeat.json").read_text())
        assert hb["pid"] == os.getpid()
        assert hb["current_run_id"] is None
        assert "last_update_utc" in hb

    def test_result_writing(self, tmp_path):
        """Results should be appended to results.jsonl as valid JSON."""
        queue_dir = tmp_path / "queue"
        worker = QueueWorker(queue_dir)

        worker._write_result(
            run_id="test_run",
            spec_sha256="abc123",
            arm="test_arm",
            hypothesis="test_hyp",
            status="succeeded",
            steps=[],
            expected_outputs=[],
        )

        results_file = queue_dir / "results.jsonl"
        assert results_file.exists()

        line = results_file.read_text().strip()
        result = json.loads(line)
        assert result["run_id"] == "test_run"
        assert result["status"] == "succeeded"
        assert result["spec_sha256"] == "abc123"

    def test_orphan_recovery(self, tmp_path):
        """Orphaned running specs should be moved to failed with correct status."""
        queue_dir = tmp_path / "queue"
        running_spec = queue_dir / "running" / "orphan_run.json"
        running_spec.parent.mkdir(parents=True)
        running_spec.write_text(
            json.dumps(
                {
                    "run_id": "orphan_run",
                    "steps": [{"name": "step1", "argv": ["echo", "test"], "timeout_seconds": 10}],
                }
            )
        )

        QueueWorker(queue_dir)

        assert not running_spec.exists()
        assert (queue_dir / "failed" / "orphan_run.json").exists()

        result_line = (queue_dir / "results.jsonl").read_text().strip()
        result = json.loads(result_line)
        assert result["status"] == "worker_restart_orphan"

    def test_priority_ordering(self, tmp_path):
        """Runs should be selected in order of (priority, filename)."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        specs = [
            ("run_c.json", {"run_id": "run_c", "priority": 50, "steps": []}),
            ("run_a.json", {"run_id": "run_a", "priority": 100, "steps": []}),
            ("run_b.json", {"run_id": "run_b", "priority": 100, "steps": []}),
        ]

        pending_dir = queue_dir / "pending"
        pending_dir.mkdir()

        for fname, spec in specs:
            (pending_dir / fname).write_text(json.dumps(spec))

        worker = QueueWorker(queue_dir)
        next_run = worker._select_next_run()

        # Should select run_c (lowest priority), then run_a (lowest filename)
        assert next_run is not None
        assert next_run.stem == "run_c"

    def test_stop_file_blocks_selection(self, tmp_path):
        """STOP file should prevent selecting new runs."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()
        (queue_dir / "pending").mkdir()
        (queue_dir / "pending" / "run_1.json").write_text(
            json.dumps({"run_id": "run_1", "steps": []})
        )
        (queue_dir / "STOP").touch()

        worker = QueueWorker(queue_dir)
        assert worker._select_next_run() is None

    def test_invalid_spec_moved_to_failed(self, tmp_path):
        """Invalid specs should be moved to failed/ with error status."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        pending_dir = queue_dir / "pending"
        pending_dir.mkdir()
        (queue_dir / "running").mkdir()
        (queue_dir / "done").mkdir()
        (queue_dir / "failed").mkdir()
        (queue_dir / "logs").mkdir()

        # Invalid spec: no steps
        spec_path = pending_dir / "invalid_run.json"
        spec_path.write_text(json.dumps({"run_id": "invalid_run", "steps": []}))

        worker = QueueWorker(queue_dir)
        run_id, success = worker._run_spec(spec_path)

        assert not success
        assert run_id == "invalid_run"

        result_line = (queue_dir / "results.jsonl").read_text().strip()
        result = json.loads(result_line)
        assert result["status"] == "invalid_spec"

    def test_successful_command_chain(self, tmp_path):
        """Multiple steps should execute in order and record results."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        for d in ["pending", "running", "done", "failed", "logs"]:
            (queue_dir / d).mkdir()

        spec_path = queue_dir / "pending" / "chain_run.json"
        spec = {
            "run_id": "chain_run",
            "steps": [
                {
                    "name": "step1",
                    "argv": [sys.executable, "-c", "print('step 1')"],
                    "timeout_seconds": 10,
                },
                {
                    "name": "step2",
                    "argv": [sys.executable, "-c", "print('step 2')"],
                    "timeout_seconds": 10,
                },
            ],
        }
        spec_path.write_text(json.dumps(spec))

        worker = QueueWorker(queue_dir)
        run_id, success = worker._run_spec(spec_path)

        assert success
        assert run_id == "chain_run"

        result_line = (queue_dir / "results.jsonl").read_text().strip()
        result = json.loads(result_line)
        assert result["status"] == "succeeded"
        assert len(result["steps"]) == 2
        assert result["steps"][0]["status"] == "succeeded"
        assert result["steps"][1]["status"] == "succeeded"

    def test_failing_step_stops_chain(self, tmp_path):
        """Chain should stop at first failing step."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        for d in ["pending", "running", "done", "failed", "logs"]:
            (queue_dir / d).mkdir()

        spec_path = queue_dir / "pending" / "fail_run.json"
        spec = {
            "run_id": "fail_run",
            "steps": [
                {"name": "step1", "argv": [sys.executable, "-c", "exit(1)"], "timeout_seconds": 10},
                {
                    "name": "step2",
                    "argv": [sys.executable, "-c", "print('should not run')"],
                    "timeout_seconds": 10,
                },
            ],
        }
        spec_path.write_text(json.dumps(spec))

        worker = QueueWorker(queue_dir)
        run_id, success = worker._run_spec(spec_path)

        assert not success
        result_line = (queue_dir / "results.jsonl").read_text().strip()
        result = json.loads(result_line)
        assert result["status"] == "failed"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["status"] == "failed"

    def test_timeout_kills_child(self, tmp_path):
        """Step timeout should kill process and record timeout status."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        for d in ["pending", "running", "done", "failed", "logs"]:
            (queue_dir / d).mkdir()

        spec_path = queue_dir / "pending" / "timeout_run.json"
        spec = {
            "run_id": "timeout_run",
            "steps": [
                {
                    "name": "slow_step",
                    "argv": [sys.executable, "-c", "import time; time.sleep(100)"],
                    "timeout_seconds": 1,
                }
            ],
        }
        spec_path.write_text(json.dumps(spec))

        worker = QueueWorker(queue_dir)
        run_id, success = worker._run_spec(spec_path)

        assert not success
        result_line = (queue_dir / "results.jsonl").read_text().strip()
        result = json.loads(result_line)
        assert result["status"] == "failed"
        assert result["steps"][0]["status"] == "timeout"

    def test_env_overlay(self, tmp_path):
        """Step env vars should overlay on worker environment."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        for d in ["pending", "running", "done", "failed", "logs"]:
            (queue_dir / d).mkdir()

        spec_path = queue_dir / "pending" / "env_run.json"
        spec = {
            "run_id": "env_run",
            "steps": [
                {
                    "name": "env_step",
                    "argv": [sys.executable, "-c", "import os; print(os.getenv('TEST_VAR'))"],
                    "timeout_seconds": 10,
                    "env": {"TEST_VAR": "test_value"},
                }
            ],
        }
        spec_path.write_text(json.dumps(spec))

        worker = QueueWorker(queue_dir)
        worker._run_spec(spec_path)

        log_file = queue_dir / "logs" / "env_run.env_step.log"
        assert log_file.exists()

    def test_expected_outputs_recorded(self, tmp_path):
        """Expected outputs should be recorded with size/hash."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        for d in ["pending", "running", "done", "failed", "logs"]:
            (queue_dir / d).mkdir()

        output_file = tmp_path / "output.txt"
        spec_path = queue_dir / "pending" / "output_run.json"
        spec = {
            "run_id": "output_run",
            "steps": [
                {
                    "name": "create_output",
                    "argv": [
                        sys.executable,
                        "-c",
                        f"open('{output_file}', 'w').write('test data')",
                    ],
                    "timeout_seconds": 10,
                }
            ],
            "expected_outputs": [str(output_file)],
        }
        spec_path.write_text(json.dumps(spec))

        worker = QueueWorker(queue_dir)
        worker._run_spec(spec_path)

        result_line = (queue_dir / "results.jsonl").read_text().strip()
        result = json.loads(result_line)
        assert len(result["expected_outputs"]) == 1
        assert result["expected_outputs"][0]["exists"] is True
        assert "size" in result["expected_outputs"][0]
        assert "sha256" in result["expected_outputs"][0]

    def test_once_mode_exits_after_one_run(self, tmp_path):
        """With --once, worker should process one run then exit."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        for d in ["pending", "running", "done", "failed", "logs"]:
            (queue_dir / d).mkdir()

        spec_path = queue_dir / "pending" / "once_run.json"
        spec = {
            "run_id": "once_run",
            "steps": [
                {
                    "name": "step1",
                    "argv": [sys.executable, "-c", "print('done')"],
                    "timeout_seconds": 10,
                }
            ],
        }
        spec_path.write_text(json.dumps(spec))

        worker = QueueWorker(queue_dir, once=True)
        exit_code = worker.run()

        assert exit_code == 0
        assert worker.runs_completed == 1

    def test_max_runs_limit(self, tmp_path):
        """Worker should exit after max_runs."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        for d in ["pending", "running", "done", "failed", "logs"]:
            (queue_dir / d).mkdir()

        for i in range(3):
            spec_path = queue_dir / "pending" / f"run_{i}.json"
            spec = {
                "run_id": f"run_{i}",
                "steps": [
                    {"name": "step", "argv": [sys.executable, "-c", "pass"], "timeout_seconds": 10}
                ],
            }
            spec_path.write_text(json.dumps(spec))

        worker = QueueWorker(queue_dir, max_runs=2)
        exit_code = worker.run()

        assert exit_code == 0
        assert worker.runs_completed == 2

    def test_idle_exit(self, tmp_path):
        """Worker should exit after idle_exit_seconds with empty queue."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        for d in ["pending", "running", "done", "failed", "logs"]:
            (queue_dir / d).mkdir()

        worker = QueueWorker(queue_dir, idle_exit_seconds=1, poll_seconds=1)
        exit_code = worker.run()

        assert exit_code == 0

    def test_results_jsonl_valid_json(self, tmp_path):
        """All lines in results.jsonl should be valid JSON with required keys."""
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()

        for d in ["pending", "running", "done", "failed", "logs"]:
            (queue_dir / d).mkdir()

        specs = [
            ("run_1.json", {"run_id": "run_1", "steps": []}),
            ("run_2.json", {"run_id": "run_2", "arm": "test", "steps": []}),
        ]

        for fname, spec in specs:
            (queue_dir / "pending" / fname).write_text(json.dumps(spec))

        worker = QueueWorker(queue_dir)
        for _ in range(2):
            spec_path = worker._select_next_run()
            if spec_path:
                running_path = queue_dir / "running" / spec_path.name
                spec_path.rename(running_path)
                worker._run_spec(running_path)

        results_file = queue_dir / "results.jsonl"
        for line in results_file.read_text().strip().split("\n"):
            result = json.loads(line)
            assert "run_id" in result
            assert "status" in result
            assert "steps" in result
            assert "expected_outputs" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _spec(queue: Path, run_id: str, code: str, timeout: int = 30) -> None:
    (queue / "pending").mkdir(parents=True, exist_ok=True)
    spec = {
        "run_id": run_id,
        "steps": [{"name": "s", "argv": [sys.executable, "-c", code], "timeout_seconds": timeout}],
    }
    (queue / "pending" / f"{run_id}.json").write_text(json.dumps(spec))


def _results(queue: Path) -> list[dict]:
    return [json.loads(line) for line in (queue / "results.jsonl").read_text().splitlines()]


def test_malformed_json_does_not_kill_worker(tmp_path: Path) -> None:
    (tmp_path / "pending").mkdir(parents=True)
    (tmp_path / "pending" / "a_bad.json").write_text("{not json")
    _spec(tmp_path, "b_good", "print('ok')")
    assert QueueWorker(tmp_path, poll_seconds=0, idle_exit_seconds=0).run() == 0
    statuses = {row["run_id"]: row["status"] for row in _results(tmp_path)}
    assert statuses == {"a_bad": "invalid_spec", "b_good": "succeeded"}
    assert (tmp_path / "failed" / "a_bad.json").exists()


def test_stop_file_exits_zero(tmp_path: Path) -> None:
    _spec(tmp_path, "r1", "print('x')")
    (tmp_path / "STOP").write_text("")
    assert QueueWorker(tmp_path, poll_seconds=0).run() == 0
    assert (tmp_path / "pending" / "r1.json").exists()


def test_sigterm_kills_running_child_group(tmp_path: Path) -> None:
    import signal
    import subprocess
    import time

    marker = tmp_path / "child.pid"
    _spec(
        tmp_path,
        "slow",
        f"import os,time; open({str(marker)!r},'w').write(str(os.getpid())); time.sleep(60)",
        timeout=120,
    )
    script = Path(__file__).parent.parent / "scripts" / "warm_queue_worker.py"
    worker = subprocess.Popen(
        [sys.executable, str(script), "--queue", str(tmp_path), "--poll-seconds", "0"]
    )
    deadline = time.time() + 20
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert marker.exists()
    child = int(marker.read_text())
    worker.send_signal(signal.SIGTERM)
    assert worker.wait(timeout=30) == 130
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)
    rows = _results(tmp_path)
    assert rows[-1]["run_id"] == "slow" and rows[-1]["status"] == "worker_terminated"
    assert (tmp_path / "failed" / "slow.json").exists()
