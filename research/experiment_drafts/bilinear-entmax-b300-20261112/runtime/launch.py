"""Run one explicitly owned Bilinear Entmax stage on the receiving Linux host."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import os
import signal
import subprocess
import sys
from pathlib import Path

from bilinear_entmax.bindings import (
    RUNTIME_COMMIT,
    RUNTIME_MANIFEST_SHA256,
    VARIANTS,
    load_cell,
    write_new,
)

WORLD_SIZE = 2
TIMEOUT_LIMITS = {
    "prepare": 14_400,
    "check": 3_600,
    "profile": 3_600,
    "fit": 28_800,
    "export": 7_200,
    "score": 7_200,
}
PRECISION_ENV = {
    "CUBLAS_EMULATE_SINGLE_PRECISION": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def main() -> None:
    """Launch exactly one stage with bounded process-group cleanup and no retries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check", "profile", "fit", "export", "score"))
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--devices", help="Free GPU IDs: two for fit/profile/export, one for check")
    parser.add_argument("--preprocessing", type=Path)
    parser.add_argument("--checks-receipt", type=Path)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--profile-receipt", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("model and data stages are restricted to the receiving Linux host")
    if not args.owner.strip() or not 1 <= args.timeout_seconds <= TIMEOUT_LIMITS[args.mode]:
        parser.error(
            f"a named owner and timeout between 1 and {TIMEOUT_LIMITS[args.mode]} "
            f"seconds are required for {args.mode}"
        )
    cell, _ = load_cell(args.cell)
    if cell["variant"] not in VARIANTS:
        parser.error("cell variant is outside the frozen panel")
    distributed = args.mode in ("profile", "fit", "export")
    required_devices = WORLD_SIZE if distributed else (1 if args.mode == "check" else 0)
    if required_devices:
        devices = (args.devices or "").split(",")
        if (
            len(devices) != required_devices
            or len(set(devices)) != required_devices
            or not all(item.isdigit() for item in devices)
        ):
            parser.error(f"--devices must name exactly {required_devices} distinct free GPUs")
    if args.mode in ("check", "profile", "fit") and args.preprocessing is None:
        parser.error(f"{args.mode} requires --preprocessing")
    if args.mode in ("profile", "fit") and args.checks_receipt is None:
        parser.error("profile and fit require the all-variant server --checks-receipt")
    if args.mode == "profile" and args.profile_updates != 50:
        parser.error("Bilinear Entmax requires an exact 50-update profile")
    if args.mode != "profile" and args.profile_updates != 0:
        parser.error("--profile-updates is permitted only for the profile stage")
    if args.mode == "fit" and args.profile_receipt is None:
        parser.error("fit requires a human-approved --profile-receipt")
    if args.mode != "fit" and args.resume is not None:
        parser.error("resume is permitted only for a fit stage")
    if args.mode == "export" and args.checkpoint is None:
        parser.error("export requires --checkpoint")
    if args.mode == "score" and args.predictions is None:
        parser.error("score requires --predictions")

    worker = Path(__file__).resolve().with_name("worker.py")
    argv = [sys.executable]
    if distributed:
        argv += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2"]
    argv += [str(worker), args.mode]
    for name in (
        "runtime_root",
        "cell",
        "data_root",
        "split_manifest",
        "output",
        "owner",
        "timeout_seconds",
        "preprocessing",
        "checks_receipt",
        "profile_receipt",
        "resume",
        "checkpoint",
        "predictions",
    ):
        value = getattr(args, name)
        if value is not None:
            argv += ["--" + name.replace("_", "-"), str(value)]
    argv += ["--profile-updates", str(args.profile_updates)]
    environment = os.environ.copy() | PRECISION_ENV
    if args.devices:
        environment["CUDA_VISIBLE_DEVICES"] = args.devices
    now = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    receipt_root = (
        args.output.resolve().parent / "launch-receipts" / f"{cell['id']}-{args.mode}-{now}"
    )
    receipt_root.mkdir(parents=True, exist_ok=False)
    write_new(
        receipt_root / "launch.json",
        {
            "schema": "bilinear-entmax.launch.v1",
            "cell_id": cell["id"],
            "variant": cell["variant"],
            "seed": cell["seed"],
            "runtime_commit": RUNTIME_COMMIT,
            "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
            "world_size": WORLD_SIZE if distributed else required_devices or 0,
            "resources": {
                "gpu_model": "B300" if required_devices else None,
                "gpu_count": required_devices,
                "device_ids": args.devices,
                "world_size": WORLD_SIZE if distributed else required_devices or 0,
            },
            "environment_overrides": PRECISION_ENV,
            "mode": args.mode,
            "owner": args.owner,
            "started_at": now,
            "timeout_seconds": args.timeout_seconds,
            "argv": argv,
            "devices": args.devices,
            "automatic_retries": 0,
            "optimization_changes": False,
        },
    )
    status, returncode = "failed", None
    with (receipt_root / "stdout.log").open("xb") as log:
        process = subprocess.Popen(
            argv, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
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
                    "scientific_completion": (
                        "inspect the stage-specific receipt; process exit alone is insufficient"
                    ),
                },
            )
    if status != "completed":
        raise SystemExit(f"{status}; evidence retained in {receipt_root}")


if __name__ == "__main__":
    main()
