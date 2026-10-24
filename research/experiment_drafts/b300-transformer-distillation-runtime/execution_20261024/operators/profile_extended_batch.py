"""Measure small B128/B96 and large B64 on disjoint W2 GPU pairs."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
SOURCE = ROOT / "runtime/pr32-source-f1beeb4c"
PACKET = (
    SOURCE
    / "research/experiment_drafts/b300-transformer-distillation-runtime/large_teacher_r3_muon30"
)
PYTHON = ROOT / ".venv/bin/python"


def write(path: Path, value: dict) -> None:
    """Atomically record commands and the last completed operator phase."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def allocations(layout: str) -> list[tuple[str, list[int]]]:
    """Return the bounded panel, preserving 16 independent sequence slots per fit."""
    if layout == "w2":
        return [
            ("t1280_f0", [0, 1]),
            ("t1792_f0", [2, 3]),
            ("t1280_f2", [4, 5]),
            ("t1792_f2", [6, 7]),
        ]
    if layout == "w4":
        return [("t1280_f0", list(range(4))), ("t1792_f0", list(range(4, 8)))]
    return [("t1280_f0" if layout == "w8-small" else "t1792_f0", list(range(8)))]


def main() -> None:
    """Profile small B128/B96 on F0 and large B64 on actual F0/F2; small f2 directory is an engineering alias."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", choices=("w2", "w4", "w8-small", "w8-large"))
    args = parser.parse_args()
    run = ROOT / "runs/pr32-blackwell-20261024" / ("extended-batch-r2-" + args.layout)
    manifest = json.loads((SOURCE / "TRANSPORT_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected, name
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = [[part.strip() for part in row.split(",")] for row in raw.splitlines()]
    assert len(gpus) == 8 and len({row[1] for row in gpus}) == 8
    assert all("B300" in row[2] and int(row[5]) < 1024 for row in gpus)
    run.mkdir(parents=True, exist_ok=False)
    write(
        run / "HOST.json",
        {
            "gpus": gpus,
            "source_commit": manifest["commit"],
            "meminfo": (Path("/proc/meminfo")).read_text(),
            "filesystem": subprocess.check_output(["df", "-B1", str(run)], text=True),
            "engineering_only": True,
        },
    )
    workers = []

    def cleanup() -> None:
        """Terminate only live children created by this finite profile operator."""
        for process, log, _directory, _receipt in workers:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            log.close()

    def stop(_signum: int, _frame: object) -> None:
        """Leave through registered cleanup on cancellation."""
        raise SystemExit(130)

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for cell, devices in allocations(args.layout):
        directory = run / cell
        directory.mkdir()
        config = json.loads(
            (
                PACKET
                / "configs"
                / ((cell.replace("_f2", "_f0") if cell.startswith("t1280") else cell) + ".json")
            ).read_text()
        )
        config.update(
            required_world_size=len(devices),
            batch_size=(128 if cell.endswith("f0") else 96) if cell.startswith("t1280") else 64,
            precision_mode="bf16_bf16x9",
        )
        config["model"].update(
            checkpoint_blocks=False, attention_backend="efficient", compile_blocks=True
        )
        config_path = directory / "config.json"
        write(config_path, config)
        cpus = []
        for device in devices:
            cpus += list(
                range(
                    32 + 8 * (device % 4) + int(device >= 4),
                    40 + 8 * (device % 4) + int(device >= 4),
                    2,
                )
            )
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=",".join(gpus[i][1] for i in devices),
            PYTHONPATH=str(SOURCE / "src"),
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            CUBLAS_EMULATE_SINGLE_PRECISION="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            TMPDIR=str(ROOT / "tmp/pr32-r3-20261024-r2"),
        )
        environment.update(TORCH_LOGS="graph_breaks,recompiles", TORCHINDUCTOR_COMPILE_THREADS="1")
        fold = config["expected_fold"]
        command = [
            "taskset",
            "-c",
            ",".join(map(str, cpus)),
            str(PYTHON),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={len(devices)}",
            str(SOURCE / "scripts/train_sequence.py"),
            "--config",
            str(config_path),
            "--data-root",
            str(ROOT / "data"),
            "--split-manifest",
            str(ROOT / "splits" / f"fold{fold}.json"),
            "--output",
            str(directory / "profile"),
            "--profile-updates",
            "12",
        ]
        receipt = {
            "cell": cell,
            "gpus": [gpus[i] for i in devices],
            "cpus": cpus,
            "source_commit": manifest["commit"],
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "argv": command,
            "environment": {
                key: environment[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "CUBLAS_EMULATE_SINGLE_PRECISION",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "OMP_NUM_THREADS",
                )
            },
            "started_unix": time.time(),
            "limit_seconds": 900,
            "global_batch_sequences": 2 * config["batch_size"],
            "engineering_only": True,
        }
        log = (directory / "console.log").open("x")
        process = subprocess.Popen(
            command,
            cwd=SOURCE,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        receipt["pid"] = process.pid
        write(directory / "operator.json", receipt)
        workers.append((process, log, directory, receipt))
    failures = []
    while workers:
        for entry in workers[:]:
            process, log, directory, receipt = entry
            expired = time.time() - receipt["started_unix"] > 900
            if expired and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            if process.poll() is not None:
                receipt.update(
                    exit_code=process.returncode,
                    finished_unix=time.time(),
                    expired=expired,
                )
                write(directory / "operator.json", receipt)
                log.close()
                workers.remove(entry)
                if process.returncode:
                    failures.append(directory.name)
        if workers:
            time.sleep(3)
    write(run / "COMPLETE.json", {"finished_unix": time.time(), "failures": failures})
    if failures:
        raise RuntimeError(f"Failed profiles retained: {failures}")


if __name__ == "__main__":
    main()
