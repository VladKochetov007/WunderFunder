from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import os
import signal
import subprocess
import sys
from pathlib import Path

from impact81.bindings import load_cell, write_new
from impact81.runtime_family import (
    environment_for_card,
    family_for_card,
    runtime_commit_for_card,
    world_size_for_card,
)


def build_worker_argv(python: str, worker: Path, mode: str, world_size: int) -> list[str]:
    """Build the per-cell process command for the selected placement."""
    argv = [python]
    if world_size > 1:
        if mode not in ("profile", "fit", "export"):
            raise ValueError("only model execution modes may use a distributed process group")
        argv += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={world_size}",
        ]
    return argv + [str(worker), mode]


def main() -> None:
    """Launch exactly one owned Linux stage with a process-group timeout; no queue or retries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check", "profile", "fit", "export", "score"))
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument(
        "--devices",
        help="Free GPU IDs: one per fit/profile/export or one for receiver checks",
    )
    parser.add_argument("--preprocessing", type=Path)
    parser.add_argument("--profile-receipt", type=Path)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--checks-receipt", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("execution is restricted to the receiving Linux server; never this Mac")
    if not args.owner.strip() or not 1 <= args.timeout_seconds <= 172800:
        parser.error("a named owner and timeout1..172800 seconds are required")
    cell, _ = load_cell(args.cell)
    family = family_for_card(cell["card"])
    runtime_commit = runtime_commit_for_card(cell["card"])
    world_size = world_size_for_card(cell["card"])
    distributed = world_size > 1 and args.mode in ("profile", "fit", "export")
    if args.mode not in ("prepare", "score"):
        devices = (args.devices or "").split(",")
        required = world_size if args.mode in ("profile", "fit", "export") else 1
        if (
            len(devices) != required
            or len(set(devices)) != required
            or not all(item.isdigit() for item in devices)
        ):
            parser.error(f"--devices must name exactly {required} distinct free GPU indices")
    worker = Path(__file__).resolve().with_name("worker.py")
    argv = build_worker_argv(sys.executable, worker, args.mode, world_size if distributed else 1)
    for name in (
        "runtime_root",
        "cell",
        "data_root",
        "split_manifest",
        "output",
        "owner",
        "timeout_seconds",
        "preprocessing",
        "profile_receipt",
        "resume",
        "checkpoint",
        "predictions",
        "checks_receipt",
    ):
        value = getattr(args, name)
        if value is not None:
            argv += ["--" + name.replace("_", "-"), str(value)]
    argv += ["--profile-updates", str(args.profile_updates)]
    env = os.environ.copy()
    for name, value in environment_for_card(cell["card"]).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices
    now = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    receipt_root = (
        args.output.resolve().parent / "launch-receipts" / f"{cell['id']}-{args.mode}-{now}"
    )
    receipt_root.mkdir(parents=True, exist_ok=False)
    write_new(
        receipt_root / "launch.json",
        {
            "schema": "impact81.launch.v1",
            "cell_id": cell["id"],
            "family": family,
            "runtime_commit": runtime_commit,
            "world_size": world_size,
            "environment_overrides": environment_for_card(cell["card"]),
            "mode": args.mode,
            "owner": args.owner,
            "started_at": now,
            "timeout_seconds": args.timeout_seconds,
            "argv": argv,
            "devices": args.devices,
            "automatic_retries": 0,
        },
    )
    status = "failed"
    returncode = None
    with (receipt_root / "stdout.log").open("xb") as log:
        process = subprocess.Popen(
            argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=args.timeout_seconds)
            status = "completed" if returncode == 0 else "failed"
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            status = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode = process.returncode
        finally:
            write_new(
                receipt_root / "terminal.json",
                {
                    "status": status,
                    "returncode": returncode,
                    "owned_pid": process.pid,
                    "finished_at": dt.datetime.now(dt.UTC).isoformat(),
                    "scientific_completion": "inspect canonical terminal_status.json; process exit alone is insufficient",
                },
            )
    if status != "completed":
        raise SystemExit(f"{status}; evidence retained in {receipt_root}")


if __name__ == "__main__":
    main()
