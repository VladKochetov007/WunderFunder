"""Bounded explicit B300 stages for the TL3-architecture student KD comparison."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from audit import audit_fit, require_fit, teacher_targets
from cache_targets import build_targets, export_prediction
from commands import run_name
from common import (
    PACKET,
    bind_assets,
    command_environment,
    digest,
    load_assets,
    output_root,
    preflight,
    read_json,
    write_new,
)

# Runtime receipt records retain canonical heterogeneous fields.


def require_profile(assets: dict[str, Any], cell: str, role: str) -> dict[str, Any]:
    """Require actual finite B300 profile updates and measured memory margin at this source."""
    name = run_name(cell, "teacher_a" if role == "teacher_b" else role, True)
    bound = preflight(assets)
    start = read_json(output_root(assets) / "start" / f"{name}.json")
    terminal = read_json(output_root(assets) / "terminal" / f"{name}.json")
    if any(start[key] != value or terminal[key] != value for key, value in bound.items()):
        raise ValueError("profile source/data/recipe binding changed")
    log = output_root(assets) / "logs" / f"{name}.log"
    evidence = []
    with log.open() as stream:
        for line in stream:
            if line.startswith("PROFILE_UPDATE_EVIDENCE "):
                evidence.append(json.loads(line.partition(" ")[2]))
    if (
        terminal["status"] != "completed"
        or not terminal["profile"]
        or len(evidence) != 1
        or evidence[0].get("valid_for_throughput") is not True
        or terminal["free_bytes_after"] < 0.1 * terminal["total_bytes"]
        or terminal["peak_reserved_bytes"] > 0.9 * terminal["total_bytes"]
    ):
        raise ValueError("missing finite, complete, memory-admitted profile")
    return {
        "runtime_sha256": start["runtime_sha256"],
        "terminal_sha256": digest(output_root(assets) / "terminal" / f"{name}.json"),
    }


def fit_one(
    assets: dict[str, Any], cell: str, role: str, profile: bool, resume: bool = False
) -> None:
    """Dispatch one explicit model/profile, never search or expand the fixed panel."""
    if role in ("labels", "kd"):
        teacher_targets(assets, cell)
    if not profile:
        require_profile(assets, cell, role)
    name = run_name(cell, role, profile)
    if (output_root(assets) / "audit" / f"{name}.json").exists() and not profile:
        require_fit(assets, cell, role)
        return
    if (output_root(assets) / "terminal" / f"{name}.json").exists():
        if profile:
            require_profile(assets, cell, role)
        else:
            audit_fit(assets, cell, role)
        return
    log = output_root(assets) / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PACKET / "worker.py"),
        "--assets",
        assets["_path"],
        "--cell",
        cell,
        "--role",
        role,
    ]
    if profile:
        command += ["--profile"]
    elif resume:
        state = Path(assets["project_root"]) / "artifacts/runs" / name / "training_state.pt"
        if state.exists():
            attempt = uuid.uuid4().hex
            command += ["--resume-state", str(state), "--attempt", attempt]
            log = log.with_name(f"{name}_resume_{digest(state)}_{attempt}.log")
    with log.open("x") as stream:
        subprocess.run(
            command,
            env=command_environment(assets),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if profile:
        require_profile(assets, cell, role)
    else:
        audit_fit(assets, cell, role)


def main() -> None:
    """Perform only the named preparation/profile/fit/cache/predict/grade stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "bind",
            "preflight",
            "profile-teachers",
            "fit-teachers",
            "cache",
            "profile-students",
            "fit-students",
            "predict",
            "score",
            "bundle",
        ),
    )
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cell", choices=("F0", "F2"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="explicitly continue incomplete scientific fits from full epoch state",
    )
    args = parser.parse_args()
    if args.resume and args.command not in ("fit-teachers", "fit-students"):
        parser.error("--resume is only valid for scientific fitting stages")
    if args.command == "bind":
        if args.output is None:
            parser.error("bind requires --output for a new sealed assets file")
        bind_assets(args.assets, args.output)
        return
    assets = load_assets(args.assets)
    preflight(assets)
    if args.command == "preflight":
        print(
            json.dumps({"status": "source_data_preflight_passed", "gpu_execution": "not_started"})
        )
        return
    if args.command in ("score", "bundle"):
        script = "grade.py" if args.command == "score" else "bundle.py"
        subprocess.run(
            [sys.executable, str(PACKET / script), "--assets", assets["_path"]],
            env=command_environment(assets),
            check=True,
        )
        return
    if args.cell is None:
        parser.error("this phase requires an explicit --cell F0 or F2")
    directory = output_root(assets)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{args.cell}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "cache":
            build_targets(assets, args.cell)
        elif args.command == "predict":
            for role in ("teacher_a", "teacher_b", "labels", "kd"):
                require_fit(assets, args.cell, role)
                export_prediction(assets, args.cell, role, "valid", "eval")
        else:
            profile = args.command.startswith("profile")
            roles = (
                ("teacher_a",)
                if args.command == "profile-teachers"
                else (
                    ("teacher_a", "teacher_b")
                    if args.command == "fit-teachers"
                    else ("labels", "kd")
                )
            )
            for role in roles:
                fit_one(assets, args.cell, role, profile, args.resume)
    stage_path = directory / "stages" / f"{args.cell}_{args.command}.json"
    receipt = {
        "status": "completed",
        "cell": args.cell,
        "stage": args.command,
        "assets_sha256": assets["_sha256"],
    }
    if stage_path.exists():
        if read_json(stage_path) != receipt:
            raise ValueError("existing stage receipt differs")
    else:
        write_new(stage_path, receipt)


if __name__ == "__main__":
    main()
