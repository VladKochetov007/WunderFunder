"""Validate a completed H377 hardware profile without grading or exporting weights."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _json_objects(text: str) -> list[dict[str, Any]]:
    objects = []
    for line in text.splitlines():
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def validate(args: argparse.Namespace) -> dict[str, Any]:
    preflight_text = args.preflight_log.read_text(errors="replace")
    if "H377_SOURCE_INPUT_PREFLIGHT_OK " not in preflight_text:
        raise ValueError("successful source/input preflight marker is absent")

    log_text = args.profile_log.read_text(errors="replace")
    run_args = next(
        (row for row in _json_objects(log_text) if row.get("name") == args.run_name),
        None,
    )
    if run_args is None:
        raise ValueError("trainer startup arguments for the declared profile are absent")
    if run_args.get("profile_steps") != args.profile_steps:
        raise ValueError("trainer profile step count differs")
    if run_args.get("profile_warmup") != args.profile_warmup:
        raise ValueError("trainer profile warmup differs")

    profile_rows = [
        line[len("PROFILE ") :] for line in log_text.splitlines() if line.startswith("PROFILE ")
    ]
    if len(profile_rows) != 1:
        raise ValueError(f"expected one profile summary, found {len(profile_rows)}")
    profile = json.loads(profile_rows[0])
    if not isinstance(profile, dict) or not profile:
        raise ValueError("profile summary is empty or malformed")
    for name, value in profile.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"profile metric is not numeric: {name}")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"profile metric is invalid: {name}")
    for name in ("gpu_total", "step_wall", "seq_per_s"):
        if name not in profile or profile[name] <= 0:
            raise ValueError(f"profile summary lacks a positive {name}")

    run_dir = args.run_dir
    forbidden = [
        run_dir / "model.pt",
        run_dir / "training_state.pt",
        *run_dir.glob("model_ep*.pt"),
        args.project_root / "cache" / f"pred_{args.run_name}_valid.npy",
        args.project_root / "cache" / f"pred_{args.run_name}_valid.json",
    ]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise ValueError(f"profile-only packet unexpectedly produced fit artifacts: {present}")
    return {
        "format": "wnn.h377-profile-validation-result",
        "run_name": args.run_name,
        "profile_steps": args.profile_steps,
        "profile_warmup": args.profile_warmup,
        "metrics": profile,
        "fit_checkpoint_or_prediction": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-log", type=Path, required=True)
    parser.add_argument("--profile-log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--profile-steps", type=int, required=True)
    parser.add_argument("--profile-warmup", type=int, required=True)
    args = parser.parse_args()
    try:
        result = validate(args)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"H377_PROFILE_VALIDATION_FAIL: {exc}", file=sys.stderr)
        return 1
    print("H377_PROFILE_VALIDATION_OK", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
