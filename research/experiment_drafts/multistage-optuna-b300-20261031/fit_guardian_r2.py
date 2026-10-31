"""Record exact child exits and budget outcomes for independently reviewed fits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wnn.multistage_search_budget_fit import finish_budget_attempt
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import current_process_owner


def main() -> None:
    """Execute an immutable owned request and retain its observed terminal outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    environment = os.environ.copy()
    environment.update(request.get("child_environment", {}))
    code, timed_out, _ = run_owned_command(
        request["command"],
        cwd=Path(request["source_root"]),
        env=environment,
        log_path=Path(request["log_path"]),
        receipt_path=Path(request["child_receipt_path"]),
        owner=current_process_owner(),
        stage=request["stage"],
        trial_number=request["trial_number"],
        mode="fit",
        gpu_key=request["gpu_key"],
        output_dir=Path(request["output_dir"]),
        timeout_seconds=request["timeout_seconds"],
    )
    status = "censored_timeout" if timed_out else "completed" if code == 0 else "failed"
    finish_budget_attempt(Path(request["budget_run_dir"]), request["fit_attempt_id"], status, code)
    if timed_out or code != 0:
        raise SystemExit(124 if timed_out else code or 1)


if __name__ == "__main__":
    main()
