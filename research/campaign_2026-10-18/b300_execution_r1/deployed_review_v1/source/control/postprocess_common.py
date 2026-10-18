"""Export completed fit cells and independently grade available paired folds."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from .policy import ACTOR_SHA256, DISK_FREE_MIN
except ImportError:  # Supports direct execution with control/ on sys.path.
    from policy import ACTOR_SHA256, DISK_FREE_MIN

GRADER_SHA256 = "56ffc693da52668726bab1f7008e31753a76aed3ff5d79089167785079d9eb31"
SOURCE_TREE_SHA256 = "f4ef9ca8450a3e76d6b6f8295259e70aec5024d3a4f3296558bb0323add74130"
SOURCE_MANIFEST_SHA256 = "2accd0c001c0e05ff2aac8e45cee93f25dc25805a10f5a038e94adaf256c8a42"
GPU_BUDGET_BYTES = 6 * 1024**3
EVAL_SHAPE_TAIL = [20_000, 2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _remaining(deadline: float) -> float:
    value = deadline - time.time()
    if not math.isfinite(value) or value <= 0:
        raise TimeoutError("postprocess wall deadline has expired")
    return value


def _check_disk(root: Path) -> None:
    if shutil.disk_usage(root).free < DISK_FREE_MIN:
        raise OSError("owned physical disk is below the 8 GiB postprocess reserve")


def _runtime_paths(root: Path, jobs: list[dict[str, Any]]) -> dict[str, Path]:
    runtime = root / "runtime"
    actor_paths = []
    for job in jobs:
        argv = job.get("argv", [])
        if isinstance(argv, list) and len(argv) > 1:
            actor_paths.append(Path(str(argv[1])))
    if actor_paths:
        resolved_actors = {path.resolve(strict=True) for path in actor_paths}
        if len(resolved_actors) != 1:
            raise ValueError("wave jobs point to different frozen GPU actor bundles")
        actor_path = resolved_actors.pop()
        code = actor_path.parent
        packet = code.parent
        if actor_path.name != "gpu_entrypoint.py" or packet.parent.resolve(
            strict=True
        ) != runtime.resolve(strict=True):
            raise ValueError("wave actor command is outside the deployed runtime packet")
    else:
        packet = runtime / "packet-r2"
        code = packet / "code"
    source = runtime / "source-a266-lf-subset-v1"
    python = root / ".venv" / "bin" / "python"
    paths = {
        "packet": packet,
        "code": code,
        "source": source,
        "python": python,
        "grader": packet / "grader" / "grade_endpoint.py",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"postprocess runtime is incomplete: {missing}")
    if _sha256(paths["grader"]) != GRADER_SHA256:
        raise ValueError("deployed R4 grader bytes differ from the frozen grader pin")
    builder = _read_json(code / "command_builder.json")
    manifest_path = code / str(builder.get("source_tree_manifest"))
    if (
        _sha256(manifest_path) != SOURCE_MANIFEST_SHA256
        or builder.get("source_tree_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or builder.get("expected_source_tree_identity_sha256") != SOURCE_TREE_SHA256
    ):
        raise ValueError("deployed source-tree manifest differs from its frozen pin")
    files = {name: _sha256(code / name) for name in sorted(builder.get("adapter_bundle_files", []))}
    bundle_sha = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if bundle_sha != ACTOR_SHA256:
        raise ValueError("deployed actor bundle differs from the controller pin")
    paths["builder"] = code / "command_builder.json"
    paths["manifest"] = manifest_path
    return paths


def _child_environment(root: Path, gpu_uuid: str, output: Path, *, grading: bool) -> dict[str, str]:
    keys = ("PATH", "HOME", "LD_LIBRARY_PATH", "CUDA_HOME")
    env = {key: os.environ[key] for key in keys if key in os.environ}
    env["PATH"] = str(root / ".venv" / "bin") + os.pathsep + env.get("PATH", "")
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "" if grading else gpu_uuid,
            "WNN_RUN_DIR": str(output),
            "WNN_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "WNN_LOADER_WORKERS": "0",
            "WNN_LOADER_RAM_GB": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(output / "tmp"),
        }
    )
    return env


def _terminate_owned_process(process: subprocess.Popen[Any]) -> None:
    """Terminate and reap only this child's process group (or process on Windows)."""
    if os.name != "posix":
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return

    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    grace_end = time.monotonic() + 5
    while time.monotonic() < grace_end:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0.05)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _execute(
    argv: list[str],
    env: dict[str, str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    deadline: float,
    cpus: list[int],
) -> dict[str, Any]:
    """Run one isolated actor with durable logs, a hard deadline and Wave CPU affinity."""
    start = time.time()
    _remaining(deadline)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        remaining = _remaining(deadline)
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=(os.name == "posix"),
                preexec_fn=(lambda: os.sched_setaffinity(0, set(cpus)))
                if os.name == "posix"
                else None,
            )
        except (OSError, subprocess.SubprocessError) as error:
            status, exit_code, launch_error = "launch_failed", None, str(error)
        else:
            launch_error = None
            try:
                try:
                    exit_code = process.wait(timeout=remaining)
                    status = "exited"
                except subprocess.TimeoutExpired:
                    _terminate_owned_process(process)
                    exit_code, status = None, "deadline_exceeded"
                stdout.flush()
                stderr.flush()
                os.fsync(stdout.fileno())
                os.fsync(stderr.fileno())
            except BaseException:
                _terminate_owned_process(process)
                raise
        if launch_error is not None:
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    if launch_error is not None:
        return {"status": status, "exit_code": exit_code, "error": launch_error}
    return {
        "status": status,
        "exit_code": exit_code,
        "elapsed_seconds": time.time() - start,
        "stdout": {
            "path": str(stdout_path),
            "bytes": stdout_path.stat().st_size,
            "sha256": _sha256(stdout_path),
        },
        "stderr": {
            "path": str(stderr_path),
            "bytes": stderr_path.stat().st_size,
            "sha256": _sha256(stderr_path),
        },
    }
