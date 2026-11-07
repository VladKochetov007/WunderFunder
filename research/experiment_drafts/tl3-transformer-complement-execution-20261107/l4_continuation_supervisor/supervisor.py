from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch
from supervisor_checks import (
    DEADLINE,
    archive_complete,
    archiver_guard,
    digest,
    marker,
    owned_trainer,
    pinned,
    read,
)
from wnn.multistage_search import atomic_write_json, file_lock
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import (
    current_process_owner,
    is_same_process,
    launch_detached_worker,
    process_identity,
    terminate_owned_process,
)

from wnn.training.sequence_provenance import canonical_sha256


class Supervisor:
    """Continue only the two existing GRU trajectories through their original bounded12E endpoint."""

    def __init__(self, plan: dict, plan_sha: str) -> None:
        self.plan, self.plan_sha = plan, plan_sha
        self.control = Path(plan["control"])
        self.control.mkdir(parents=True, exist_ok=True)
        self.owner, self.stop = current_process_owner(), threading.Event()
        self.lock = threading.RLock()
        self.active = {}
        if (
            plan.get("approved") is not True
            or plan.get("reviewer") != "ROOT"
            or plan.get("schema") != "wnn.gru-monitor-supervisor-plan.v1"
        ):
            raise ValueError("ROOT-approved exact supervisor plan required")
        if [(cell["seed"], cell["gpu_index"]) for cell in plan["cells"]] != [
            (101001, 0),
            (101002, 1),
        ]:
            raise ValueError("only original GRU101001/GPU0 and101002/GPU1 are permitted")
        if (self.control / "started.json").exists():
            raise FileExistsError("one-shot continuation supervisor cannot automatically restart")
        for cell in plan["cells"]:
            request = cell["request"]
            if request.get("approved") is not True or request.get("reviewer") != "ROOT":
                raise ValueError("each transition requires explicit ROOT authorization")
            clock = pinned(request["first_profile_receipt"])
            if (
                float(clock["first_profile_epoch"]) + 82800 != DEADLINE
                or request.get("not_after_epoch") != DEADLINE
            ):
                raise ValueError("original L4 source clock/deadline differs; no new clock")
            runtime = Path(plan["runtime"])
            if (
                Path(request["old_output"]) != runtime / f"runs/gru_s{cell['seed']}/fit"
                or Path(request["new_output"])
                != runtime / f"runs/gru_s{cell['seed']}_monitor_opt/fit"
                or Path(cell["old_owned_receipt"])
                != runtime / f"receipts/owned-fit-gru_s{cell['seed']}.json"
            ):
                raise ValueError("transition paths differ from the two exact admitted sibling runs")
            if Path(request["new_output"]).exists():
                raise FileExistsError("continuation output must be fresh")
        required_tools = {
            "migrate_monitor_checkpoint.py",
            "transition_support.py",
            "archive_safe_state.py",
            "supervisor.py",
            "supervisor_checks.py",
            "preflight.py",
            "observed_worker.py",
        }
        if {Path(spec["path"]).name for spec in plan["tool_pins"]} != required_tools:
            raise ValueError(
                "plan must pin every migration, archive, supervisor and observed-worker helper"
            )
        for spec in plan["tool_pins"]:
            if digest(Path(spec["path"])) != spec["sha256"]:
                raise ValueError("ROOT-reviewed migration/archive/supervisor helper changed")
        atomic_write_json(
            self.control / "started.json",
            {
                "plan_sha256": plan_sha,
                "owner": self.owner,
                "deadline_epoch": DEADLINE,
                "scientific_fit_count_increment": 0,
            },
            replace=False,
        )

    def guard(self) -> None:
        """Close all new owned work on ROOT stop or the unchanged original fit deadline."""
        if self.stop.is_set() or time.time() >= DEADLINE:
            raise RuntimeError("ROOT stop or original L4 fit deadline reached")

    def environment(self, cell: dict, *, archive: bool = False) -> dict[str, str]:
        """Match the accepted strict-FP32 S3 profile, using two threads only for CPU archival."""
        env = dict(os.environ)
        for key in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "CUBLAS_WORKSPACE_CONFIG",
            "CUBLAS_EMULATE_SINGLE_PRECISION",
        ):
            env.pop(key, None)
        env.update(
            PYTHONPATH=str(Path(cell["request"]["new_source_root"]) / "src"),
            CUDA_VISIBLE_DEVICES=str(cell["gpu_index"]),
            OMP_NUM_THREADS="2" if archive else "8",
            MKL_NUM_THREADS="2" if archive else "8",
            OPENBLAS_NUM_THREADS="2" if archive else "8",
            NVIDIA_TF32_OVERRIDE="0",
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
            PYTHONDONTWRITEBYTECODE="1",
            WNN_MONITOR_HEARTBEAT_SECONDS="5",
        )
        # Migration tools import their sibling transition_support; preflight is staged alongside them.
        env["PYTHONPATH"] += os.pathsep + str(Path(self.plan["tools"]))
        return env

    def execute(
        self, cell: dict, phase: str, command: list[str], *, progress: object = None
    ) -> tuple:
        """Reuse S1's native exact PID/birth/PGID owned-command lifecycle for every subprocess."""
        self.guard()
        path = self.control / f"owned-{phase}-gru_s{cell['seed']}.json"
        with self.lock:
            if path.exists():
                raise FileExistsError("owned continuation phase already attempted; no retry")
            self.active[f"{cell['seed']}-{phase}"] = path
        return run_owned_command(
            command,
            cwd=Path(cell["request"]["new_source_root"]),
            env=self.environment(cell, archive=phase != "resume"),
            log_path=self.control / f"{phase}-{cell['seed']}.log",
            receipt_path=path,
            owner={
                "controller": self.owner,
                "cell_id": f"gru_s{cell['seed']}",
                "plan_sha256": self.plan_sha,
            },
            stage=1,
            trial_number=cell["gpu_index"],
            mode=phase,
            gpu_key=str(cell["gpu_index"]),
            output_dir=Path(cell["request"]["new_output"]),
            timeout_seconds=min(
                1800 if phase in ("preflight", "migrate") else 86400,
                max(1, int(DEADLINE - time.time())),
            ),
            progress=progress or self.guard,
        )

    def checked(self, result: tuple, phase: str) -> None:
        """Keep native nonzero/timeout failures terminal rather than silently continuing."""
        if result[0] != 0 or result[1]:
            raise RuntimeError(f"{phase} failed: exit={result[0]}, timed_out={result[1]}")

    def terminate_new(self, seed: int | None = None) -> None:
        """Stop only this supervisor's exact still-live command identities, never original/foreign fits."""
        for key, path in list(self.active.items()):
            if seed is not None and not key.startswith(f"{seed}-"):
                continue
            if path.exists():
                record = read(path)
                if record.get("owner", {}).get("controller") == self.owner and is_same_process(
                    record.get("child", {})
                ):
                    terminate_owned_process(record["child"], grace_seconds=1)

    def worker(self, cell: dict) -> None:
        """Preflight, stop at a later safe epoch, migrate, arm retention and resume one exact trajectory."""
        seed, request = cell["seed"], dict(cell["request"])
        python, tools = self.plan["python"], Path(self.plan["tools"])
        template = self.control / f"{seed}-template.json"
        atomic_write_json(template, request, replace=False)
        preflight = self.control / f"{seed}-preflight.json"
        old = Path(request["old_output"])
        state_path = old / "training_state.pt"
        initial_marker = marker(state_path)
        self.checked(
            self.execute(
                cell,
                "preflight",
                [
                    python,
                    str(Path(__file__).with_name("preflight.py")),
                    "--template",
                    str(template),
                    "--template-sha256",
                    digest(template),
                    "--output",
                    str(preflight),
                ],
            ),
            "preflight",
        )
        initial = read(preflight)
        old, state_path = (
            Path(request["old_output"]),
            Path(request["old_output"]) / "training_state.pt",
        )
        owned = read(Path(cell["old_owned_receipt"]))
        trainer = owned_trainer(
            owned, old, Path(request["old_source_root"]), seed, cell["gpu_index"], process_identity
        )
        wait_end = min(time.monotonic() + 1800, time.monotonic() + DEADLINE - time.time())
        while True:
            self.guard()
            if not is_same_process(trainer) or not is_same_process(owned["child"]):
                raise RuntimeError(
                    "original owned trainer exited before supervised safe-boundary stop"
                )
            current = marker(state_path)
            if current != initial_marker:
                with state_path.open("rb") as stream:
                    state = torch.load(stream, map_location="cpu", weights_only=False)
                if (
                    state["next_epoch"] > initial["initial_next_epoch"]
                    and state["next_epoch"] < 12
                    and canonical_sha256(state["contract"]) == initial["contract_sha256"]
                ):
                    break
            if time.monotonic() >= wait_end:
                raise TimeoutError(
                    "new safe whole-epoch wait exhausted; original trainer preserved"
                )
            self.stop.wait(1)
        # Reprove leader ownership and trainer ancestry immediately before the authorized stop.
        for key in (
            "old_metadata",
            "new_config",
            "approved_profile",
            "old_source_manifest",
            "new_source_manifest",
            "source_patch_receipt",
            "first_profile_receipt",
        ):
            pinned(request[key])
        if (
            owned_trainer(
                read(Path(cell["old_owned_receipt"])),
                old,
                Path(request["old_source_root"]),
                seed,
                cell["gpu_index"],
                process_identity,
            )
            != trainer
        ):
            raise RuntimeError("original trainer ownership changed before stop")
        terminate_owned_process(owned["child"], grace_seconds=2)
        exit_end = time.monotonic() + 30
        while is_same_process(trainer) or is_same_process(owned["child"]):
            if time.monotonic() >= exit_end:
                raise RuntimeError("original actual trainer or owned leader did not exit")
            time.sleep(0.1)
        stable_marker, checkpoint_sha = marker(state_path), digest(state_path)
        with state_path.open("rb") as stream:
            stopped = torch.load(stream, map_location="cpu", weights_only=False)
        if (
            marker(state_path) != stable_marker
            or digest(state_path) != checkpoint_sha
            or not initial["initial_next_epoch"] < stopped["next_epoch"] < 12
            or stopped["global_step"] != stopped["next_epoch"] * 6460
            or canonical_sha256(stopped["contract"]) != initial["contract_sha256"]
        ):
            raise ValueError(
                "stopped safe state is not stable or differs from the preflight contract"
            )
        request.update(
            old_checkpoint_sha256=checkpoint_sha,
            expected_next_epoch=stopped["next_epoch"],
            expected_global_step=stopped["global_step"],
            stopped_trainer_owner=trainer,
        )
        final_request = self.control / f"{seed}-transition-request.json"
        atomic_write_json(final_request, request, replace=False)
        self.checked(
            self.execute(
                cell,
                "migrate",
                [
                    python,
                    str(tools / "migrate_monitor_checkpoint.py"),
                    "--request",
                    str(final_request),
                    "--request-sha256",
                    digest(final_request),
                ],
            ),
            "migrate",
        )
        output = Path(request["new_output"])
        proof_sha = digest(output / "transition_proof.json")
        ready = read(output / "transition_ready.json")
        if ready["proof_sha256"] != proof_sha or ready["not_after_epoch"] != DEADLINE:
            raise ValueError("migrated continuation lacks exact ready/proof/deadline binding")
        with ThreadPoolExecutor(max_workers=1) as pool:
            archive = pool.submit(
                self.execute,
                cell,
                "archive",
                [
                    python,
                    str(tools / "archive_safe_state.py"),
                    "--output",
                    str(output),
                    "--transition-proof-sha256",
                    proof_sha,
                    "--poll-seconds",
                    "1",
                ],
            )
            try:
                self.resume(cell, archive, proof_sha)
            except BaseException:
                self.terminate_new(seed)
                raise
        terminal = read(output / "terminal_status.json")
        if (
            terminal.get("status") != "completed"
            or terminal.get("completed_epochs") != 12
            or terminal.get("global_updates") != 77520
            or not archive_complete(output, proof_sha)
        ):
            raise ValueError("continuation lacks exact12E/77520 and complete safe-state archive")
        atomic_write_json(
            self.control / f"{seed}-COMPLETE.json",
            {
                "completed_epochs": 12,
                "global_updates": 77520,
                "transition_proof_sha256": proof_sha,
                "archive_complete_sha256": digest(output / "archive_complete.json"),
                "scientific_fit_count_increment": 0,
            },
            replace=False,
        )

    def resume(self, cell: dict, archive: Future, proof_sha: str) -> None:
        """Require live bound archiver readiness, then guard retention throughout canonical resume."""
        output = Path(cell["request"]["new_output"])
        receipt = self.control / f"owned-archive-gru_s{cell['seed']}.json"
        wait_end = time.monotonic() + 120
        while not (output / "archive_ready.json").exists():
            self.guard()
            if archive.done():
                raise RuntimeError("external archiver exited before readiness")
            if time.monotonic() >= wait_end:
                raise TimeoutError("external archiver readiness timeout")
            self.stop.wait(0.2)
        ready, native = read(output / "archive_ready.json"), read(receipt)
        if (
            ready.get("transition_proof_sha256") != proof_sha
            or ready.get("pid") != native["child"]["pid"]
            or ready.get("not_after_epoch") != DEADLINE
            or not is_same_process(native["child"])
        ):
            raise ValueError("archive readiness lacks matching proof/hash and exact live PID/birth")

        def guard() -> None:
            self.guard()
            archiver_guard(
                output,
                proof_sha,
                read(receipt),
                archive.result() if archive.done() else None,
                is_same_process,
            )

        request = cell["request"]
        source = Path(request["new_source_root"])
        python = self.plan["python"]
        seconds = max(1, int(DEADLINE - time.time()))
        train = [
            python,
            str(source / "scripts/train_sequence.py"),
            "--config",
            request["new_config"]["path"],
            "--data-root",
            str(Path(self.plan["runtime"]) / "project"),
            "--split-manifest",
            str(Path(self.plan["runtime"]) / "project/chronological_primary.json"),
            "--output",
            str(output),
            "--profile-receipt",
            request["approved_profile"]["path"],
            "--resume",
            str(output / "training_state.pt"),
        ]
        command = [
            python,
            self.plan["observed_worker"],
            "--",
            python,
            str(source / "scripts/observe_sequence.py"),
            "--output",
            str(output),
            "--interval",
            "10",
            "--max-seconds",
            str(seconds),
            "--",
            *train,
        ]
        self.checked(self.execute(cell, "resume", command, progress=guard), "resume")
        self.checked(archive.result(timeout=max(1, DEADLINE - time.time())), "archive")


def main() -> int:
    """Detach one finite two-worker ROOT-owned transition, preserving all old artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    plan = pinned({"path": str(args.plan), "sha256": args.plan_sha256})
    control = Path(plan["control"])
    if args.detach:
        command = [plan["python"], str(Path(__file__).resolve()), *sys.argv[1:]]
        command.remove("--detach")
        owner = launch_detached_worker(
            command,
            cwd=Path(plan["runtime"]),
            env=dict(os.environ),
            receipt_path=control / "supervisor-owner.json",
            metadata={"plan_sha256": args.plan_sha256},
        )
        print(json.dumps(owner))
        return 0
    with file_lock(control / "supervisor.lock", blocking=False):
        supervisor = Supervisor(plan, args.plan_sha256)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda _signal, _frame: supervisor.stop.set())
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(supervisor.worker, cell) for cell in plan["cells"]]
                for future in futures:
                    future.result()
            return 0
        finally:
            supervisor.terminate_new()


if __name__ == "__main__":
    raise SystemExit(main())
