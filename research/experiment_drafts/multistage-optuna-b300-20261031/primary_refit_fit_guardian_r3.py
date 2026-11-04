"""Run one owned primary refit inside both individual and campaign deadlines."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from primary_refit_actor_support import DEADLINE_SHA256, FIT_WALL_SECONDS, read_json, sha256

from wnn.multistage_search_budget_fit import finish_budget_attempt
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import current_process_owner


def main() -> None:
    """Recheck the absolute reserve immediately before the trainer starts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = read_json(args.request, "owned primary fit request")
    run_dir = Path(str(request["budget_run_dir"]))
    attempt_id = str(request["fit_attempt_id"])
    try:
        deadline_path = run_dir / "campaign_deadline.json"
        deadline = read_json(deadline_path, "campaign deadline")
        reserve_start = deadline.get("reserve_start_epoch")
        if (
            sha256(deadline_path) != DEADLINE_SHA256
            or type(reserve_start) not in (int, float)
            or request.get("reserve_start_epoch") != reserve_start
            or type(request.get("timeout_seconds")) is not int
            or not 0 < request["timeout_seconds"] <= FIT_WALL_SECONDS
            or request.get("stage") != 4
        ):
            raise ValueError("owned fit request lacks its frozen campaign and fit bounds")
        wall = min(request["timeout_seconds"], int(reserve_start - time.time()))
        if wall <= 0:
            raise RuntimeError("collection reserve has started; trainer launch refused")
        environment = os.environ.copy()
        environment.update(request["child_environment"])
        code, timed_out, _ = run_owned_command(
            request["command"],
            cwd=Path(str(request["source_root"])),
            env=environment,
            log_path=Path(str(request["log_path"])),
            receipt_path=Path(str(request["child_receipt_path"])),
            owner=current_process_owner(),
            stage=4,
            trial_number=request["trial_number"],
            mode="fit",
            gpu_key=str(request["gpu_key"]),
            output_dir=Path(str(request["output_dir"])),
            timeout_seconds=wall,
        )
    except Exception:
        finish_budget_attempt(run_dir, attempt_id, "guardian_failed", None)
        raise
    status = "censored_timeout" if timed_out else "completed" if code == 0 else "failed"
    finish_budget_attempt(run_dir, attempt_id, status, code)
    if timed_out or code != 0:
        raise SystemExit(124 if timed_out else code or 1)


if __name__ == "__main__":
    main()
