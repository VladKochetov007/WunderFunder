"""Thin bounded receiving runner over canonical ownership, budgets and observation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from receiving_checks import (
    approved_profile,
    check_native_profile,
    evaluations,
    oracle_ready,
    source_files,
    verify_cell,
    verify_source,
)
from receiving_resources import ResourceGuard, optional_json, owned_records

from wnn.multistage_search import atomic_write_json, file_lock, file_sha256, read_json
from wnn.multistage_search_budget import (
    fail_reserved_profile,
    finish_profile_attempt,
    reserve_profile,
)
from wnn.multistage_search_budget_fit import finish_budget_attempt, reserve_fit
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import (
    current_process_owner,
    hold_gpu_slot,
    is_same_process,
    launch_detached_worker,
)


def worker_environment(panel: dict[str, object], cell: dict[str, object]) -> dict[str, str]:
    """Match ROOT's initial strict-FP32 profile environment for every profile and fit."""
    environment = dict(os.environ)
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "CUBLAS_EMULATE_SINGLE_PRECISION",
        "CUBLAS_WORKSPACE_CONFIG",
    ):
        environment.pop(key, None)
    environment.update(
        CUDA_VISIBLE_DEVICES=cell["gpu_uuid"],
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        PYTHONPATH=str(Path(panel["source"]) / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        NVIDIA_TF32_OVERRIDE="0",
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
    )
    return environment


class Cohort:
    """Run each declared cell independently; never retry or approve scientific work."""

    def __init__(self, panel: dict[str, object], panel_sha: str, phase: str) -> None:
        self.panel, self.panel_sha, self.phase = panel, panel_sha, phase
        self.control = Path(panel["control"])
        self.owner = current_process_owner()
        self.guard = ResourceGuard(panel)
        self.lock = threading.RLock()
        self.status = {
            cell["name"]: {
                "cell": cell["name"],
                "family": cell["family"],
                "seed": cell["seed"],
                "gpu_index": cell["gpu_index"],
                "gpu_uuid": cell["gpu_uuid"],
                "state": "pending",
                "requested_epochs": 12,
                "primary_endpoint_complete": False,
            }
            for cell in panel["cells"]
        }
        self.last_progress: dict[str, float] = {}

    def update(self, cell: dict[str, object], state: str | None = None, **fields: object) -> None:
        """Publish concise per-cell public status separately from immutable raw journals."""
        with self.lock:
            if state is not None:
                self.status[cell["name"]]["state"] = state
            self.status[cell["name"]].update(fields)
            atomic_write_json(
                self.control / "public_status.json",
                {
                    "schema": "wnn.compact-l4-status.v1",
                    "panel_sha256": self.panel_sha,
                    "updated_at_epoch": time.time(),
                    "authority": self.panel["authority"],
                    "resources": self.guard.snapshot,
                    "cells": self.status,
                    "graphs": "artifact callback pending ROOT; raw all-step journals retained",
                },
            )

    def progress(self, cell: dict[str, object], *, force: bool = False) -> None:
        """Check shared limits and summarize only actual worker progress every ten seconds."""
        self.guard.check(force=force)
        if not force and time.monotonic() - self.last_progress.get(cell["name"], 0) < 10:
            return
        self.last_progress[cell["name"]] = time.monotonic()
        output = Path(cell["fit_output"])
        events = evaluations(output)
        progress = optional_json(output / "monitor_progress.json")
        if not progress:
            progress = optional_json(Path(cell["profile_output"]) / "monitor_progress.json")
        metrics = events[-1]["metrics"] if events else {}
        self.update(
            cell,
            worker_phase=progress.get("phase"),
            last_successful_update=progress.get("last_successful_update"),
            heartbeat_at=progress.get("heartbeat_at"),
            completed_evaluation_epochs=len(events),
            valid_wp=metrics.get("valid/wp"),
            valid_t0=metrics.get("valid/wp/t0"),
            valid_t1=metrics.get("valid/wp/t1"),
        )

    def command(self, cell: dict[str, object], mode: str, seconds: int) -> list[str]:
        """Wrap the unchanged trainer in its existing independent observer."""
        python, source = self.panel["python"], Path(self.panel["source"])
        output = cell[f"{mode}_output"]
        train = [
            python,
            str(source / "scripts/train_sequence.py"),
            "--config",
            cell["config"],
            "--data-root",
            self.panel["data_root"],
            "--split-manifest",
            self.panel["split_manifest"],
            "--output",
            output,
        ]
        train += (
            ["--profile-updates", "20"]
            if mode == "profile"
            else ["--profile-receipt", cell["approved_profile"]]
        )
        return [
            python,
            str(Path(__file__).parent / "observed_worker.py"),
            "--",
            python,
            str(source / "scripts/observe_sequence.py"),
            "--output",
            output,
            "--interval",
            "10",
            "--max-seconds",
            str(seconds),
            "--",
            *train,
        ]

    def execute(self, cell: dict[str, object], mode: str, seconds: int) -> None:
        """Use the canonical exact PID/birth/PGID owner and bounded process-group lifecycle."""
        environment = worker_environment(self.panel, cell)
        self.update(
            cell, mode, owned_receipt=cell[f"{mode}_owned_receipt"], output=cell[f"{mode}_output"]
        )
        code, timed_out, _ = run_owned_command(
            self.command(cell, mode, seconds),
            cwd=Path(self.panel["source"]),
            env=environment,
            log_path=Path(self.panel["runtime"]) / "logs" / f"{mode}-{cell['name']}.log",
            receipt_path=Path(cell[f"{mode}_owned_receipt"]),
            owner={
                "controller": self.owner,
                "panel_sha256": self.panel_sha,
                "cell_id": cell["name"],
            },
            stage=1,
            trial_number=cell["gpu_index"],
            mode=mode,
            gpu_key=str(cell["gpu_index"]),
            output_dir=Path(cell[f"{mode}_output"]),
            timeout_seconds=seconds,
            progress=lambda: self.progress(cell),
        )
        if code != 0 or timed_out:
            raise RuntimeError(
                f"{mode} failed: exit={code}, timed_out={timed_out}; no automatic retry"
            )

    def profile(self, cell: dict[str, object]) -> None:
        """Adopt ROOT's initial owned profiles or reserve one new twenty-update profile."""
        path = Path(cell["profile_owned_receipt"])
        if path.exists():
            self.update(cell, "adopting_root_profile")
            while True:
                owned_records(self.panel)
                record = read_json(path)
                self.progress(cell)
                if record["status"] != "running":
                    if record.get("status") != "completed" or record.get("exit_code") != 0:
                        raise RuntimeError(
                            "existing ROOT profile failed; preserving it without retry"
                        )
                    break
                if not is_same_process(record["child"]):
                    raise RuntimeError(
                        "ROOT profile process exited without a terminal ownership receipt"
                    )
                time.sleep(1)
        else:
            seconds = min(
                7200, int(self.panel["authority"]["fit_deadline_epoch"] - time.time()) - 1
            )
            plan = Path(cell["profile_output"]).parent / "plan.json"
            plan.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(plan, {"panel_sha256": self.panel_sha, "cell": cell}, replace=False)
            reserve_profile(self.control, cell["gpu_index"], seconds)
            try:
                self.execute(cell, "profile", seconds)
                check_native_profile(
                    self.panel,
                    cell,
                    read_json(Path(cell["profile_output"]) / "profile_receipt.json"),
                )
                finish_profile_attempt(
                    self.control,
                    cell["gpu_index"],
                    "completed",
                    0,
                    plan_path=plan,
                    worker_receipt_path=path,
                    native_profile_path=Path(cell["profile_output"]) / "profile_receipt.json",
                    nvml_snapshot={"gpu_uuid": cell["gpu_uuid"], "gpu_index": cell["gpu_index"]},
                )
            except Exception as error:
                fail_reserved_profile(self.control, cell["gpu_index"], str(error))
                raise
        receipt = read_json(Path(cell["profile_output"]) / "profile_receipt.json")
        check_native_profile(self.panel, cell, receipt)
        self.update(
            cell,
            "profile_complete",
            profile_updates=receipt["updates"],
            profile_seconds=receipt["elapsed_seconds"],
            profile_peak_memory_bytes=receipt["peak_memory_bytes"],
            profile_forwarded_rows=receipt["forwarded_rows"],
            profile_host_peak_rss_bytes=receipt["host_peak_rss_bytes"],
            profile_sha256=file_sha256(Path(cell["profile_output"]) / "profile_receipt.json"),
        )

    def collect(self, cell: dict[str, object]) -> None:
        """Require complete fixed-final evidence and index every raw fit artifact by hash."""
        output = Path(cell["fit_output"])
        terminal = read_json(output / "terminal_status.json")
        native = read_json(Path(cell["profile_output"]) / "profile_receipt.json")
        if (
            terminal.get("status") != "completed"
            or terminal.get("completed_epochs") != 12
            or terminal.get("identity") != native["identity"]
            or file_sha256(output / "model.pt") != terminal.get("model_sha256")
        ):
            raise ValueError(
                "fit terminal/checkpoint identity is not a complete fixed-final twelve epochs"
            )
        events = evaluations(output, complete=True)
        artifacts = {
            str(path): file_sha256(path)
            for path in output.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        atomic_write_json(
            output.parent / "collection.json",
            {
                "schema": "wnn.compact-l4-collection.v1",
                "panel_sha256": self.panel_sha,
                "cell": cell["name"],
                "checkpoint_sha256": terminal["model_sha256"],
                "full_valid_epochs": len(events),
                "all_step_journals_retained": True,
                "files": artifacts,
                "graphs_callback_status": "pending ROOT artifact callback",
            },
            replace=False,
        )
        self.update(
            cell,
            "completed",
            primary_endpoint_complete=True,
            completed_evaluation_epochs=12,
            valid_wp=events[-1]["metrics"]["valid/wp"],
            valid_t0=events[-1]["metrics"]["valid/wp/t0"],
            valid_t1=events[-1]["metrics"]["valid/wp/t1"],
            last_successful_update=terminal["global_updates"],
            checkpoint_sha256=terminal["model_sha256"],
            collection=str(output.parent / "collection.json"),
        )

    def worker(self, cell: dict[str, object]) -> None:
        """Advance this one cell while unrelated admitted families progress independently."""
        fit_reserved = False
        try:
            with hold_gpu_slot(self.control, str(cell["gpu_index"])):
                verify_cell(self.panel, cell)
                self.progress(cell, force=True)
                while not oracle_ready(self.panel, cell):
                    self.update(cell, "waiting_mamba_cuda_oracle")
                    self.progress(cell)
                    time.sleep(10)
                self.profile(cell)
                if self.phase == "profiles":
                    return
                self.update(cell, "waiting_ROOT_profile_approval")
                while optional_json(Path(cell["approved_profile"])).get("approved") is not True:
                    self.progress(cell)
                    time.sleep(10)
                approval_sha = approved_profile(self.panel, cell)
                verify_cell(self.panel, cell)
                self.progress(cell, force=True)
                _, seconds = reserve_fit(self.control, 1, cell["name"], cell["gpu_index"], 82800)
                fit_reserved = True
                self.update(cell, approved_profile_sha256=approval_sha)
                self.execute(cell, "fit", seconds)
                self.collect(cell)
                finish_budget_attempt(self.control, cell["name"], "completed", 0)
        except Exception as error:
            if fit_reserved:
                finish_budget_attempt(self.control, cell["name"], "failed", None)
            self.update(
                cell,
                "failed",
                error=f"{type(error).__name__}: {error}",
                primary_endpoint_complete=False,
            )


def main() -> int:
    """Launch one finite detached controller, or execute its independently progressing cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--phase", choices=("all", "profiles"), default="all")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if file_sha256(args.panel) != args.panel_sha256:
        raise ValueError("ROOT-supplied immutable panel digest differs")
    panel = read_json(args.panel)
    for key in ("coverage_receipt", "first_profile_receipt"):
        reference = panel[key]
        if file_sha256(Path(reference["path"])) != reference["sha256"]:
            raise ValueError(f"frozen {key} bytes changed")
    for path, digest in panel["receiver_files"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError("receiving runner source changed after panel freeze")
    if (
        verify_source(Path(panel["source"]), source_files(Path(panel["source_inventory"])))
        != panel["trainer_source_sha256"]
    ):
        raise ValueError("staged trainer source changed")
    if args.detach:
        owner = launch_detached_worker(
            [
                panel["python"],
                str(Path(__file__).resolve()),
                "--panel",
                str(args.panel.resolve()),
                "--panel-sha256",
                args.panel_sha256,
                "--phase",
                args.phase,
            ],
            cwd=Path(panel["source"]),
            env=dict(
                os.environ,
                PYTHONPATH=str(Path(panel["source"]) / "src"),
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            ),
            receipt_path=Path(panel["control"]) / "controller-owner.json",
            metadata={"panel_sha256": args.panel_sha256},
        )
        print(json.dumps(owner))
        return 0
    with file_lock(Path(panel["control"]) / "controller.lock", blocking=False):
        cohort = Cohort(panel, args.panel_sha256, args.phase)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda _signal, _frame: cohort.guard.stop.set())
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(cohort.worker, panel["cells"]))
        return int(any(cell["state"] == "failed" for cell in cohort.status.values()))


if __name__ == "__main__":
    raise SystemExit(main())
