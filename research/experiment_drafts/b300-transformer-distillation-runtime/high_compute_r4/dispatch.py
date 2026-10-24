"""Execute one explicit admitted action for a materialized R4 campaign cell."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from campaign_io import load_campaign
from materialize import get_cell, load_stage
from runner_checks import (
    PACKET,
    accepted_checks,
    execute,
    merged_cache_guard,
    parent_cache_guards,
    run_checks,
    torchrun,
    verify_admission_binding,
)

# Materialized cell records are heterogeneous JSON with separately checked hashes.
ACTIONS = (
    "list",
    "check",
    "profile-one",
    "profile-eight",
    "profile-peak",
    "admit",
    "fit",
    "resume",
    "evaluate",
    "cache",
    "cache-parity",
    "merge-cache",
    "bundle",
)


def _needed_environment(name: str) -> str:
    """Require the receiver's actual resource/operator binding."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"set {name} on the receiving host")
    return value


def execute_cell(
    action: str, root: Path, campaign: dict[str, Any], cell: dict[str, Any], checks: Path
) -> None:
    """Dispatch one action, using the existing trainer/exporter/scorer unchanged."""
    if cell["reused"]:
        raise ValueError("this is an exact reused endpoint; act on its original new cell")
    run = root / cell["run_dir"]
    if (run / "failure.json").exists():
        raise ValueError("this cell has a frozen failure; declare a new attempt allocation")
    config, split = root / cell["config"], Path(cell["split_manifest"])
    common = ["--data-root", campaign["data_root"], "--split-manifest", str(split)]
    teacher = cell.get("teacher_cache") if cell["kind"] == "student" else None
    cache_args: list[str] = []
    if teacher is not None:
        parent_cache_guards(root, cell, campaign["runtime_source_sha256"])
        if action != "merge-cache":
            merged_cache_guard(root, cell)
        cache_args = ["--teacher-cache", str(root / teacher)]
    train = ["scripts/train_sequence.py", "--config", str(config), *common, *cache_args]
    predict = ["scripts/predict_sequence.py", "--checkpoint", str(run / "fit/model.pt"), *common]
    if action in {"profile-one", "profile-eight", "profile-peak"}:
        predecessor = {"profile-eight": "profile-one", "profile-peak": "profile-eight"}.get(action)
        if predecessor and not (run / predecessor / "profile_receipt.json").is_file():
            raise ValueError(f"complete {predecessor} first")
        key = {
            "profile-one": "profile_one",
            "profile-eight": "config",
            "profile-peak": "profile_peak",
        }[action]
        execute(
            torchrun(1 if action == "profile-one" else 8)
            + [
                "scripts/train_sequence.py",
                "--config",
                str(root / cell[key]),
                *common,
                *cache_args,
                "--output",
                str(run / action),
                "--profile-updates",
                "12",
            ]
        )
    elif action == "admit":
        execute(
            [
                sys.executable,
                str(PACKET / "admit.py"),
                "profile",
                "--directory",
                str(run),
                "--config",
                str(config),
                "--checks",
                str(checks),
                "--reviewer",
                _needed_environment("WNN_REVIEWER"),
                "--host-ram-gib",
                _needed_environment("WNN_HOST_RAM_BUDGET_GIB"),
            ]
        )
    elif action in {"fit", "resume"}:
        verify_admission_binding(run, config)
        extra = ["--resume", str(run / "fit/training_state.pt")] if action == "resume" else []
        execute(
            torchrun(8)
            + train
            + [
                "--output",
                str(run / "fit"),
                "--profile-receipt",
                str(run / "approved_profile.json"),
                *extra,
            ]
        )
    elif action == "evaluate":
        execute(
            torchrun(8)
            + predict
            + ["--membership", "evaluation", "--output", str(run / "evaluation")]
        )
        execute(
            [
                sys.executable,
                "scripts/score_sequence.py",
                *common,
                "--predictions",
                str(run / "evaluation"),
                "--output",
                str(run / "fit/score.json"),
            ]
        )
    elif action == "cache":
        if cell["kind"] != "teacher":
            raise ValueError("R4 student runs are not allocated as downstream teachers")
        execute(
            torchrun(8) + predict + ["--membership", "fit", "--output", str(run / "teacher-cache")]
        )
    elif action == "cache-parity":
        if cell["kind"] != "teacher" or not (run / "teacher-cache/manifest.json").is_file():
            raise ValueError("export the eight-rank teacher cache first")
        execute(
            torchrun(1)
            + predict
            + ["--membership", "fit", "--output", str(run / "teacher-cache-one")]
        )
        execute(
            [
                sys.executable,
                str(PACKET / "cache_parity.py"),
                *common,
                "--one",
                str(run / "teacher-cache-one"),
                "--eight",
                str(run / "teacher-cache"),
                "--output",
                str(run / "cache-parity.json"),
            ]
        )
    elif action == "merge-cache":
        inputs = cell.get("cache_inputs", [])
        if teacher is None or len(inputs) != 3:
            raise ValueError("merge-cache requires the H mean-teacher KD cell")
        (root / teacher).parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "scripts/merge_sequence_teacher_caches.py",
            *common,
            "--output",
            str(root / teacher),
        ]
        for path in inputs:
            command += ["--input", str(root / path)]
        execute(command)
    elif action == "bundle":
        command = [
            sys.executable,
            "scripts/research_result_bundle.py",
            "--root",
            str(run / "fit"),
            "--output-dir",
            str(run / "bundle"),
        ]
        for name in ("run_metadata.json", "metrics.jsonl", "terminal_status.json", "score.json"):
            command += ["--evidence", name]
        for rank in range(8):
            command += ["--evidence", f"rank{rank}.jsonl"]
        execute(command + ["--weight", "model.pt", "--weight", "training_state.pt"])


def main() -> None:
    """Read immutable campaign/cell bindings and execute exactly the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple("BCDEFGH"))
    parser.add_argument("--cell")
    parser.add_argument("--check-id", default="attempt1")
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    campaign = load_campaign(root)
    os.environ["PYTEST_ADDOPTS"] = ""
    if args.action == "check":
        run_checks(root, args.check_id)
        return
    if args.stage is None:
        parser.error("--stage is required for this action")
    if args.action == "list":
        print(json.dumps(load_stage(root, args.stage)["cells"], indent=2))
        return
    if args.cell is None:
        parser.error("--cell is required for a per-cell action")
    cell = get_cell(root, args.stage, args.cell)
    checks = accepted_checks(root, campaign["runtime_source_sha256"])
    execute_cell(args.action, root, campaign, cell, checks)


if __name__ == "__main__":
    main()
