from __future__ import annotations

from pathlib import Path

import pytest

from wnn import multistage_search_controller as controller
from wnn.multistage_search import atomic_write_json, read_json


def test_teacher_export_launch_accepts_only_completed_campaign_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch an adopted negative-number result by exact path with owned process evidence."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    workers = run_dir / "workers"
    workers.mkdir()
    repo_root = tmp_path / "operator"
    admission_path = run_dir / "admission.json"
    atomic_write_json(
        admission_path,
        {
            "profile_policy": {"devices": {"0": {"cuda_visible_devices": "GPU-uuid-0"}}},
            "reserved_gpu_keys": [],
        },
        replace=False,
    )
    atomic_write_json(
        run_dir / "run.json",
        {"repo_root": str(repo_root), "admission": {"path": str(admission_path)}},
        replace=False,
    )
    result_path = run_dir / "stage1" / "preexisting" / "attempt-gru" / "trial_result.json"
    result_path.parent.mkdir(parents=True)
    atomic_write_json(
        result_path,
        {"status": "COMPLETE", "stage": 1, "trial_number": -1001},
        replace=False,
    )
    launched: dict[str, object] = {}

    def launch_detached(argv, *, cwd, env, receipt_path, metadata):
        launched["argv"] = argv
        launched["cwd"] = cwd
        launched["env"] = env
        record = {
            "schema": "wnn.multistage-worker-owner.v1",
            "worker_id": metadata["worker_id"],
            "mode": metadata["mode"],
            "stage": metadata["stage"],
            "gpu_key": metadata["gpu_key"],
            "pid": 93412,
            "birth": "birth-token",
            "pgid": 93412,
            "status": "running",
        } | dict(metadata)
        atomic_write_json(receipt_path, record, replace=False)
        return record

    monkeypatch.setattr(controller, "launch_detached_worker", launch_detached)
    monkeypatch.setattr(controller, "process_identity", lambda _pid: None)

    result = controller.launch_worker(
        run_dir,
        "teacher-export",
        "teacher-adopted-gru",
        "0",
        stage=1,
        result_path=result_path,
    )

    argv = launched["argv"]
    assert result["status"] == "running"
    assert argv[argv.index("--trial-number") + 1] == "-1001"
    assert argv[argv.index("--result-path") + 1] == str(result_path.resolve())
    assert read_json(workers / "teacher-adopted-gru.json")["trial_number"] == -1001
    outside_result = tmp_path / "outside-result.json"
    outside_result.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the frozen campaign"):
        controller.launch_worker(
            run_dir,
            "teacher-export",
            "teacher-outside",
            "0",
            stage=1,
            result_path=outside_result,
        )
