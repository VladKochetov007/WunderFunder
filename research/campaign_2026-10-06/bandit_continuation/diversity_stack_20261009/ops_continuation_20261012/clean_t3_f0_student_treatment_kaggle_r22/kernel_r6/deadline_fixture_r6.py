from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKET = HERE.parent
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
RUNNER_PATH = MAIN / "scripts/kaggle_gpu_runner.py"
EXPECTED_RUNNER_SHA256 = "4576c460764ec947e9d61867a3f418e1b6a7a4665e1ca4b67a523b5f8c892e8f"
EXPECTED_R5_KERNEL_SHA256 = "eee975e22a12939ec0d82465d3c4612491fb8354a00b6e42a176db07928243df"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    r6 = load_module("treatment_kernel_r6_fixture", HERE / "kernel.py")
    if sha256(RUNNER_PATH) != EXPECTED_RUNNER_SHA256:
        raise AssertionError("actual Kaggle GPU runner source hash changed")
    if r6.WALL_SECONDS != 10_000 or r6.FINALIZATION_SECONDS != 300:
        raise AssertionError("kernel wall or worker finalization budget changed")

    sys.path.insert(0, str(RUNNER_PATH.parent))
    runner = load_module("kaggle_gpu_runner_r6_fixture", RUNNER_PATH)
    if runner.WORKER_TIMEOUT_FINALIZATION_RESERVE_SECONDS != 300:
        raise AssertionError("actual runner cleanup reserve changed")

    deadline = 2_000.0
    monotonic_values = iter((1000.0, 1000.1, 1000.2, 1000.3, 1000.4, 1700.0))
    observed: dict[str, object] = {}
    stopped: list[int] = []

    class FakeProcess:
        pid = 987654
        returncode = None

        def poll(self) -> None:
            return None

    with tempfile.TemporaryDirectory(prefix="wnn-r6-deadline-") as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        spec_path = root / "spec.json"
        spec_path.write_text("{}\n")
        artifact = root / "artifacts" / "partial_checkpoint.bin"
        artifact.parent.mkdir()
        artifact.write_bytes(b"partial-checkpoint-fixture")
        output = Path(temporary) / "runner-output"
        spec_record = {
            "name": "deadline_fixture_job",
            "environment": {},
            "inputs": [],
            "code": [],
            "argv": [sys.executable, "-c", "pass"],
            "limits": {
                "cpus": 1,
                "ram_bytes": 1_000_000_000,
                "gpu_memory_bytes": 1_000_000_000,
                "gpu_stop_bytes": 5_000_000_000,
                "timeout_seconds": 5_200,
            },
            "artifacts": ["artifacts/partial_checkpoint.bin"],
        }
        host = {"capacity_bytes": 2_000_000_000, "affinity": [0]}
        devices = [
            {
                "index": 0,
                "uuid": "GPU-fixture",
                "name": "Tesla T4 fixture",
                "total_bytes": 16 * 1024**3,
                "used_bytes": 0,
            }
        ]

        def execute_actual_runner(
            command: list[str],
            cwd: Path,
            env: dict[str, str],
            timeout: float,
            check: bool,
        ) -> subprocess.CompletedProcess:
            del cwd, check
            observed.update(
                command=command,
                timeout=timeout,
                env_deadline=env["WNN_KERNEL_DEADLINE_MONOTONIC"],
            )
            with patch.dict(os.environ, env, clear=False):
                try:
                    runner.run(root, [spec_path], output)
                except RuntimeError as error:
                    if str(error) != "Timeout for deadline_fixture_job":
                        raise
                else:
                    raise AssertionError("actual runner watchdog did not expire the fixture worker")
            return subprocess.CompletedProcess(command, 1)

        with (
            patch.object(r6.time, "monotonic", side_effect=lambda: next(monotonic_values)),
            patch.object(r6.subprocess, "run", side_effect=execute_actual_runner),
            patch.object(runner, "read_spec", return_value=spec_record),
            patch.object(runner, "host_inventory", return_value=host),
            patch.object(runner, "gpu_inventory", return_value=devices),
            patch.object(runner, "admit_specs", return_value=1),
            patch.object(runner, "verify_files", return_value=None),
            patch.object(runner, "process_tree_rss", return_value=1),
            patch.object(runner, "stop_group", side_effect=lambda proc: stopped.append(proc.pid)),
            patch.object(runner.subprocess, "Popen", return_value=FakeProcess()),
            patch.object(runner.time, "sleep", return_value=None),
        ):
            try:
                r6.run_one(root, spec_path, output, deadline)
            except RuntimeError as error:
                if "worker stage failed" not in str(error):
                    raise
            else:
                raise AssertionError(
                    "R6 parent accepted the watchdog timeout as a successful stage"
                )

        timeout = observed["timeout"]
        if not isinstance(timeout, float):
            raise AssertionError("outer runner wait is not the full remaining float")
        if observed["env_deadline"] != str(deadline):
            raise AssertionError("R6 changed the runner's inherited monotonic kernel deadline")
        worker_launch = 1000.4
        worker_timeout = runner.resolve_worker_timeout_seconds(
            5_200, str(deadline), now_monotonic=worker_launch
        )
        outer_expiry = 1000.2 + timeout
        worker_cutoff = worker_launch + worker_timeout
        cleanup_window = outer_expiry - worker_cutoff
        if cleanup_window < 300.0:
            raise AssertionError(
                f"outer wait truncates the worker cleanup window: {cleanup_window}"
            )

        failure_path = output / "deadline_fixture_job" / "failure.json"
        summary_path = output / "summary.json"
        if stopped != [FakeProcess.pid] or not failure_path.is_file() or not summary_path.is_file():
            raise AssertionError(
                "runner watchdog cleanup did not stop and persist its terminal receipts"
            )
        failure = json.loads(failure_path.read_text())
        summary = json.loads(summary_path.read_text())
        if (
            failure["worker_timeout_seconds"] != worker_timeout
            or summary["incident"] != "RuntimeError: Timeout for deadline_fixture_job"
        ):
            raise AssertionError("runner failure/summary disagree with the resolved worker timeout")
        files = failure["artifacts"]["files"]
        expected_artifact_sha = hashlib.sha256(b"partial-checkpoint-fixture").hexdigest()
        if len(files) != 1 or files[0]["sha256"] != expected_artifact_sha:
            raise AssertionError("timeout terminal receipt did not preserve the partial checkpoint")
        if failure["artifacts"]["missing"]:
            raise AssertionError("fixture unexpectedly reports a missing declared partial artifact")

        result = {
            "schema": "wnn.clean_f0_treatment_deadline_cleanup_fixture.r6",
            "status": "pass",
            "kernel_source_sha256": sha256(HERE / "kernel.py"),
            "r5_kernel_parent_sha256": EXPECTED_R5_KERNEL_SHA256,
            "actual_runner_source_sha256": sha256(RUNNER_PATH),
            "runner_worker_reserve_seconds": 300,
            "outer_timeout_type": type(timeout).__name__,
            "outer_timeout_seconds": timeout,
            "runner_worker_timeout_seconds": worker_timeout,
            "cleanup_window_seconds": cleanup_window,
            "worker_cutoff_precedes_outer_expiry": cleanup_window >= 300.0,
            "watchdog_stop_group_called": stopped == [FakeProcess.pid],
            "failure_json_written": failure_path.is_file(),
            "summary_json_written": summary_path.is_file(),
            "partial_artifact_sha256": files[0]["sha256"],
            "fixture_boundaries": {
                "actual_runner_python_code_exercised": True,
                "process_and_gpu_inventory_faked": True,
                "gpu_fit_or_provider_used": False,
                "timeout_clock_accelerated": True,
            },
        }
        (HERE / "DEADLINE_FIXTURE_OUTPUT_R6.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
