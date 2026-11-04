"""Persist bounded child-process exits for a frozen, independently admitted fit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wnn.multistage_search_runtime import current_process_owner, run_owned_command


def main() -> None:
    """Run one owned request without opening an SSH connection or changing its recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    code, timed_out, _ = run_owned_command(
        request["command"], cwd=Path(request["source_root"]), env=os.environ.copy(),
        log_path=Path(request["log_path"]), receipt_path=Path(request["child_receipt_path"]),
        owner=current_process_owner(), stage=request["stage"],
        trial_number=request["trial_number"], mode="fit", gpu_key=request["gpu_key"],
        output_dir=Path(request["output_dir"]), timeout_seconds=request["timeout_seconds"],
    )
    if timed_out or code != 0:
        raise SystemExit(124 if timed_out else code or 1)


if __name__ == "__main__":
    main()
