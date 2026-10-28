from __future__ import annotations

import argparse
import fcntl
import logging
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from packet_acceptance import TEST_MODULES, require_acceptance, seal_acceptance
from packet_admission import admit, verify_approval
from packet_contract import (
    CELLS,
    PACKET,
    REPO,
    binding,
    file_hash,
    guard,
    members,
    packet_files,
    require,
    source_files,
    write_new,
)
from packet_lineage import validate_cache, validate_evaluation, validate_terminal
from packet_study import DEFAULT_STUDY, STUDY, STUDY_ENV


def execute(command: list[str], receipt: Path, timeout: int) -> None:
    """Run one bounded receiver stage and stop its whole process group on interruption."""
    write_new(receipt, {"argv": command, "cwd": str(REPO), "timeout_seconds": timeout})
    logging.info("Starting %s", " ".join(command))
    environment = os.environ | {"PYTHONPATH": str(REPO / "src"), "PYTHONNOUSERSITE": "1"}
    process = subprocess.Popen(command, cwd=REPO, env=environment, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def prepare(args: argparse.Namespace) -> None:
    """Create one fresh source holdout and bind it to this exact immutable packet."""
    guard(args.root, args.expected_commit)
    require(args.data_root.is_dir(), "Data root must already exist")
    require(
        not args.data_root.is_relative_to(args.root), "Data cannot live inside the new run root"
    )
    require(
        not args.root.is_relative_to(args.data_root), "Keep run root separate from the source data"
    )
    args.root.mkdir(parents=False, exist_ok=False)
    execute(
        [
            sys.executable,
            str(REPO / "scripts/prepare_sequence_source_split.py"),
            "--data-root",
            str(args.data_root),
            "--output",
            str(args.root / "split.json"),
            "--workers",
            "2",
        ],
        args.root / "prepare_command.json",
        21_600,
    )
    fit, evaluation = members(args.root)
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv"],
        text=True,
        timeout=30,
    )
    (args.root / "nvidia-smi.csv").write_text(inventory)
    write_new(
        args.root / "packet.json",
        {
            "schema": "wnn.source-kd-packet.v1",
            "study": STUDY.slug,
            "git_commit": args.expected_commit,
            "data_root": str(args.data_root),
            "packet_files": packet_files(),
            "source_files": source_files(),
            "split_sha256": file_hash(args.root / "split.json"),
            "fit_members": fit,
            "evaluation_members": evaluation,
            "python_executable": sys.executable,
            "cells": list(CELLS),
            "status": "prepared_no_training_started",
        },
    )


def command_for(args: argparse.Namespace) -> tuple[list[str], int]:
    """Build exactly one explicit fit, profile, cache, or fixed-final evaluation command."""
    require(args.cell in CELLS, "Cell does not belong to the selected study")
    require(args.action != "cache" or STUDY.allow_cache, "This study has no KD cache stage")
    cell_root = args.root / args.cell
    common = ["--data-root", str(args.data_root), "--split-manifest", str(args.root / "split.json")]
    torchrun = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--max-restarts=0",
    ]
    if args.action in ("profile", "fit", "resume"):
        command = (
            torchrun
            + [
                str(REPO / "scripts/train_sequence.py"),
                "--config",
                str(PACKET / "configs" / f"{args.cell}.json"),
            ]
            + common
        )
        if args.cell.startswith("kd_"):
            validate_cache(args.root, args.cell)
            command += [
                "--teacher-cache",
                str(args.root / ("teacher_" + args.cell.split("_", 1)[1]) / "cache"),
            ]
        if args.action == "profile":
            require(
                not (cell_root / "profile").exists(),
                "Profile evidence already exists; do not overwrite",
            )
            return command + [
                "--output",
                str(cell_root / "profile"),
                "--profile-updates",
                "12",
            ], 3_600
        verify_approval(args.root, args.cell)
        command += [
            "--output",
            str(cell_root / "fit"),
            "--profile-receipt",
            str(cell_root / "approved_profile.json"),
        ]
        if args.action == "resume":
            require(
                not (cell_root / "fit/terminal_status.json").exists(), "Fit is already terminal"
            )
            require(
                (cell_root / "fit/training_state.pt").is_file(), "No safe epoch boundary to resume"
            )
            command += ["--resume", str(cell_root / "fit/training_state.pt")]
        else:
            require(
                not (cell_root / "fit").exists(),
                "Existing fit needs explicit resume, never overwrite",
            )
        return command, 259_200
    validate_terminal(args.root, args.cell)
    if args.action == "cache":
        require(args.cell.startswith("teacher_"), "Only the fresh teacher can produce KD targets")
        target, membership = "cache", "fit"
    else:
        target, membership = "evaluation", "evaluation"
    require(not (cell_root / target).exists(), "Export already exists; do not replace evidence")
    command = (
        torchrun
        + [
            str(REPO / "scripts/predict_sequence.py"),
            "--checkpoint",
            str(cell_root / "fit/model.pt"),
        ]
        + common
    )
    return command + ["--output", str(cell_root / target), "--membership", membership], 21_600


def score(args: argparse.Namespace) -> None:
    """Score a complete canonical final export without choosing an epoch on valid."""
    directory = args.root / args.cell
    validate_terminal(args.root, args.cell)
    require(not (directory / "score.json").exists(), "Score exists; do not replace evidence")
    execute(
        [
            sys.executable,
            str(REPO / "scripts/score_sequence.py"),
            "--data-root",
            str(args.data_root),
            "--split-manifest",
            str(args.root / "split.json"),
            "--predictions",
            str(directory / "evaluation"),
            "--output",
            str(directory / "score.json"),
        ],
        directory / f"command-score-{uuid.uuid4().hex}.json",
        21_600,
    )
    write_new(directory / "evaluation_receipt.json", validate_evaluation(args.root, args.cell))


def main() -> None:
    """Dispatch explicit receiver actions; never start another cell automatically."""
    parser = argparse.ArgumentParser(
        description="Source-holdout six-fit B300 packet; receiver only"
    )
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "accept",
            "profile",
            "admit",
            "fit",
            "resume",
            "cache",
            "evaluate",
            "score",
            "seal",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--cell", choices=CELLS)
    parser.add_argument("--reviewer")
    parser.add_argument("--host-ram-gib", type=float)
    parser.add_argument("--max-vram-fraction", type=float)
    args = parser.parse_args()
    args.root, args.data_root = args.root.resolve(), args.data_root.resolve()
    logging.basicConfig(level=logging.INFO)
    if args.action == "prepare":
        require(args.cell is None, "Preparation is shared by all six cells")
        prepare(args)
        return
    prepared = binding(args.root, args.data_root, args.expected_commit)
    require(prepared["python_executable"] == sys.executable, "Python executable changed")
    if args.action == "accept":
        require(args.cell is None, "Acceptance is shared by all six cells")
        directory = args.root / "acceptance"
        directory.mkdir(exist_ok=False)
        command = [
            "env",
            "CUBLAS_EMULATE_SINGLE_PRECISION=0",
            f"{STUDY_ENV}={DEFAULT_STUDY}",
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ]
        command += [str(REPO / "tests" / f"{name}.py") for name in TEST_MODULES]
        command += [f"--junitxml={directory / 'pytest.xml'}"]
        execute(command, directory / "command.json", 3_600)
        seal_acceptance(args.root)
        return
    require(args.cell is not None, "Select exactly one --cell")
    require_acceptance(args.root)
    directory = args.root / args.cell
    directory.mkdir(exist_ok=True)
    with (directory / ".operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "admit":
            require(
                args.reviewer is not None
                and args.host_ram_gib is not None
                and args.max_vram_fraction is not None,
                "Admission requires reviewer and explicit RAM/VRAM budgets",
            )
            admit(args.root, args.cell, args.reviewer, args.host_ram_gib, args.max_vram_fraction)
        elif args.action == "score":
            score(args)
        elif args.action == "seal":
            write_new(
                directory / "evaluation_receipt.json", validate_evaluation(args.root, args.cell)
            )
        else:
            command, timeout = command_for(args)
            execute(command, directory / f"command-{args.action}-{uuid.uuid4().hex}.json", timeout)
            if args.action == "cache":
                validate_cache(args.root, args.cell)
            elif args.action == "evaluate":
                score(args)
            elif args.action in ("fit", "resume"):
                validate_terminal(args.root, args.cell)


if __name__ == "__main__":
    main()
