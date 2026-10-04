from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from kaggle_common import relative_path, write_json


def run(plan_path: Path) -> None:
    """Execute one sealed sequence of Python steps in the admitted worker process group."""
    plan = json.loads(plan_path.read_text())
    if set(plan) != {"version", "steps", "report"} or plan["version"] != 1:
        raise ValueError("Unsupported sequential worker plan")
    report_path = Path(relative_path(plan["report"]))
    if report_path.exists() or not plan["steps"]:
        raise ValueError("Plan must have steps and a new report path")
    for step in plan["steps"]:
        if (
            set(step) != {"name", "argv"}
            or not isinstance(step["name"], str)
            or not isinstance(step["argv"], list)
            or not step["argv"]
            or step["argv"][0] != "{python}"
            or any(not isinstance(value, str) or "\0" in value for value in step["argv"])
        ):
            raise ValueError("Plan steps require explicit literal Python argv")
    report = {"status": "running", "steps": []}
    write_json(report_path, report)
    for step in plan["steps"]:
        started = time.monotonic()
        print(json.dumps({"step": step["name"], "event": "starting"}), flush=True)
        completed = subprocess.run([sys.executable, *step["argv"][1:]], check=False)
        report["steps"].append(
            {
                "name": step["name"],
                "returncode": completed.returncode,
                "wall_seconds": time.monotonic() - started,
            }
        )
        report["status"] = "failed" if completed.returncode else "running"
        write_json(report_path, report)
        if completed.returncode:
            raise RuntimeError(f"Sealed step failed: {step['name']}")
    report["status"] = "complete"
    write_json(report_path, report)


def main() -> None:
    """Run preflight, fit, fixed prediction export and score without shell chaining."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan)


if __name__ == "__main__":
    main()
