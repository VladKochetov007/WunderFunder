"""Durably schedule the preregistered clean since/since control fits.

Every GPU step is a separate monitored campaign job. The queue stops on the
first failure and never replaces an existing job or checkpoint.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

from campaign_job import STOP_NEW_JOBS, request_stop

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "research" / "campaign_2026-09-29" / "jobs"
LIMITED = ["env", "WNN_THREADS=10", "WNN_CPU_QUOTA=1000%", "scripts/limited"]


class QueueStopped(RuntimeError):
    """The persistent campaign barrier has stopped this queue."""


def check_stop() -> None:
    """Check the durable barrier before each launch and every wait poll."""
    if STOP_NEW_JOBS.exists():
        raise QueueStopped(f"control queue stopped by {STOP_NEW_JOBS.relative_to(ROOT)}")


def on_signal(signum: int, frame: FrameType | None) -> None:
    """Persist a stop request even when the queue is run without its monitor."""
    request_stop(f"control queue received {signal.Signals(signum).name}")


def parent(fold: int, seed: int) -> list[str]:
    name = f"c20260929_since_s{seed}_f{fold}"
    return [
        *LIMITED,
        ".venv/bin/python",
        "scripts/train_rnn.py",
        "--name",
        name,
        "--train-source",
        "train+valid",
        "--valid-fold",
        str(fold),
        "--n-valid-folds",
        "4",
        "--eval-source",
        "vfold",
        "--full-valid",
        "--loss",
        "wmse",
        "--aux-struct",
        "0.3",
        "--aux-next",
        "0.3",
        "--propensity",
        f"c20260929_prop_f{fold}",
        "--prop-power",
        "2",
        "--prop-eps",
        "0.03",
        "--valid-true-mask",
        "0.03",
        "--hidden",
        "128",
        "--layers",
        "2",
        "--batch",
        "32",
        "--chunk",
        "2000",
        "--epochs",
        "4",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--seed",
        str(seed),
        "--ema-teacher",
        "1.0",
        "--ema-decay",
        "0.999",
        "--eval-ema",
        "--since-weight",
        "32",
        "--since-floor",
        "0.25",
        "--fast",
        "--split-manifest",
        f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "3",
        "--no-eval-during-fit",
    ]


def fine_tune(fold: int, seed: int) -> list[str]:
    parent_name = f"c20260929_since_s{seed}_f{fold}"
    child_name = f"c20260929_ft_since_s{seed}_f{fold}"
    return [
        "env",
        "WNN_IN_CAMPAIGN_JOB=1",
        "WNN_THREADS=10",
        "WNN_CPU_QUOTA=1000%",
        "scripts/limited",
        "bash",
        "scripts/ft_gclp.sh",
        f"{parent_name}@ep3",
        child_name,
        str(fold),
        "3",
        "--split-manifest",
        f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "0",
        "--no-eval-during-fit",
    ]


def steps() -> list[tuple[str, int, list[str], Path]]:
    plan: list[tuple[str, int, list[str], Path]] = []
    for fold in (1, 2, 3):
        if fold > 1:
            propensity = f"c20260929_prop_f{fold}"
            plan.append(
                (
                    propensity,
                    900,
                    [
                        *LIMITED,
                        ".venv/bin/python",
                        "scripts/train_propensity.py",
                        "--name",
                        propensity,
                        "--hidden",
                        "64",
                        "--epochs",
                        "4",
                        "--seed",
                        "0",
                        "--heldout-fold",
                        str(fold),
                        "--split-manifest",
                        f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json",
                        "--checkpoint-policy",
                        "fixed",
                        "--checkpoint-epoch",
                        "3",
                        "--no-eval-during-fit",
                    ],
                    ROOT / "artifacts" / "runs" / propensity / "model_ep3.pt",
                )
            )
            plan.append(
                (
                    f"c20260929_cache_prop_f{fold}",
                    1200,
                    [
                        *LIMITED,
                        ".venv/bin/python",
                        "scripts/cache_propensity.py",
                        propensity,
                        "--check",
                        "16",
                        "--batch",
                        "16",
                    ],
                    ROOT / "cache" / f"propcache_{propensity}_valid.npy",
                )
            )
        for seed in (3, 4):
            parent_name = f"c20260929_since_s{seed}_f{fold}"
            child_name = f"c20260929_ft_since_s{seed}_f{fold}"
            plan.append(
                (
                    parent_name,
                    2400,
                    parent(fold, seed),
                    ROOT / "artifacts" / "runs" / parent_name / "model_ep3.pt",
                )
            )
            plan.append(
                (
                    child_name,
                    1200,
                    fine_tune(fold, seed),
                    ROOT / "cache" / f"pred_{child_name}_valid.npy",
                )
            )
    return plan


def wait_success(name: str, expected: Path, timeout: int) -> None:
    record_path = JOBS / f"{name}.json"
    started = time.monotonic()
    while not record_path.exists():
        check_stop()
        if time.monotonic() - started > 30:
            raise RuntimeError(f"{name}: no durable job record appeared")
        time.sleep(2)
    while True:
        check_stop()
        if time.monotonic() - started > timeout + 60:
            raise TimeoutError(f"{name}: durable job did not finish within its time bound")
        record = json.loads(record_path.read_text())
        if record["status"] != "running":
            if record["status"] != "finished" or record.get("exit_code") != 0:
                raise RuntimeError(f"{name}: {record['status']} exit={record.get('exit_code')}")
            if not expected.is_file() or expected.stat().st_size == 0:
                raise RuntimeError(f"{name}: expected artifact missing or empty: {expected}")
            print(
                json.dumps(
                    {"step": name, "status": "finished", "seconds": record["elapsed_seconds"]}
                ),
                flush=True,
            )
            return
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    schedule = steps()
    if args.plan:
        print(
            json.dumps(
                [
                    {
                        "name": name,
                        "timeout": timeout,
                        "command": command,
                        "expected": str(expected.relative_to(ROOT)),
                    }
                    for name, timeout, command, expected in schedule
                ],
                indent=2,
            )
        )
        return
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    check_stop()
    # The already launched fold-1 seed-3 parent is the frozen first dependency.
    wait_success(
        "c20260929_since_s3_f1", ROOT / "artifacts/runs/c20260929_since_s3_f1/model_ep3.pt", 2400
    )
    for name, timeout, command, expected in schedule:
        check_stop()
        if (JOBS / f"{name}.json").exists():
            wait_success(name, expected, timeout)
            continue
        launch = [
            sys.executable,
            str(ROOT / "scripts" / "campaign_job.py"),
            "launch",
            "--name",
            name,
            "--kind",
            "gpu",
            "--phase",
            "controls",
            "--timeout",
            str(timeout),
            "--threads",
            "10",
            "--",
            *command,
        ]
        check_stop()
        subprocess.run(launch, cwd=ROOT, check=True)
        wait_success(name, expected, timeout)
    print(json.dumps({"control_queue": "complete", "steps": len(schedule)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except QueueStopped as exc:
        print(json.dumps({"control_queue": "stopped", "reason": str(exc)}), flush=True)
        raise SystemExit(1) from exc
