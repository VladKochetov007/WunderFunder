"""Run a frozen sequence of already authorized, individually guarded jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from bandit_resources import atomic_json, read_json

LOG = logging.getLogger(__name__)


def run_sequence(plan_path: Path, receipt: Path) -> None:
    """Wait for each terminal success before admitting the next frozen command.

    A failed or stale observer never triggers a retry. A stop-admission marker
    lets the current healthy worker finish and prevents the next launch.
    """
    plan_bytes = plan_path.read_bytes()
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    runtime = Path(plan["runtime"])
    steps = plan["steps"]
    identifiers = [step["attempt"] for step in steps]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sequence attempt IDs must be unique")
    state = {"plan": str(plan_path), "plan_sha256": plan_hash, "status": "running"}

    def publish(**updates: object) -> None:
        state.update(updates, heartbeat_utc=datetime.now(UTC).isoformat())
        atomic_json(receipt, state)

    try:
        for index, step in enumerate(steps):
            if plan_path.read_bytes() != plan_bytes:
                raise RuntimeError("sequence plan changed after admission")
            attempt = str(step["attempt"])
            record_path = runtime / "attempts" / f"{attempt}.json"
            publish(index=index, attempt=attempt)
            if not record_path.exists():
                if (runtime / "STOP_ADMISSION.json").exists():
                    publish(status="drained", reason="new admission is closed")
                    return
                if "disk_path" in plan:
                    free = shutil.disk_usage(plan["disk_path"]).free
                    required = int(plan["min_free_bytes"]) + int(step.get("output_bytes", 0))
                    if free < required:
                        raise RuntimeError(f"insufficient disk headroom: {free} < {required}")
                for output in step.get("must_not_exist", []):
                    if Path(output).exists():
                        raise FileExistsError(f"refusing to overwrite existing output: {output}")
                command = step.get("launch_command")
                if not command:
                    raise RuntimeError(f"required predecessor record is missing: {attempt}")
                LOG.info("Admitting %s", attempt)
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                publish(launch_returncode=result.returncode, launch_stderr=result.stderr[-4000:])
                if result.returncode:
                    raise RuntimeError(f"guard refused {attempt}; inspect its retained record")
            while True:
                record = read_json(record_path)
                if "expected_command" in step and record.get("command") != step["expected_command"]:
                    raise RuntimeError(f"recorded command differs from frozen sequence: {attempt}")
                publish(
                    worker_status=record["status"], worker_heartbeat=record.get("heartbeat_utc")
                )
                if record["status"] == "finished" and record.get("exit_status") == 0:
                    LOG.info("Completed %s", attempt)
                    break
                if record["status"] not in {"reserved", "waiting", "running"}:
                    raise RuntimeError(f"{attempt} needs review: {record['status']}")
                heartbeat = datetime.fromisoformat(str(record["heartbeat_utc"]))
                if (datetime.now(UTC) - heartbeat).total_seconds() > 180:
                    raise RuntimeError(f"stale worker observation for {attempt}; no next admission")
                time.sleep(15)
        publish(status="finished")
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        publish(status="needs_review", error=f"{type(error).__name__}: {error}")
        raise


def main() -> None:
    """Read a preregistered sequence; work remains bounded by each job guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_sequence(args.plan.resolve(strict=True), args.receipt.resolve())


if __name__ == "__main__":
    main()
