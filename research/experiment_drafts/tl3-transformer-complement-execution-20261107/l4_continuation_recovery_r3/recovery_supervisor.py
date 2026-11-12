from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from recovery_checks import DEADLINE, digest, pinned, probe_ready, read, validate_recovery
from supervisor import Supervisor
from wnn.multistage_search import atomic_write_json, file_lock
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import (
    current_process_owner,
    is_same_process,
    launch_detached_worker,
    terminate_owned_process,
)


class Recovery(Supervisor):
    """Recover the two stopped S1E8 trajectories without ever signalling a historical process."""

    def __init__(self, plan: dict, plan_sha: str) -> None:
        validate_recovery(plan, self.live)
        self.plan, self.plan_sha = plan, plan_sha
        self.control = Path(plan["control"])
        self.control.mkdir(parents=True, exist_ok=True)
        self.owner, self.stop, self.lock = (
            current_process_owner(),
            threading.Event(),
            threading.RLock(),
        )
        self.active, self.failures = {}, {}
        atomic_write_json(
            self.control / "started.json",
            {
                "plan_sha256": plan_sha,
                "owner": self.owner,
                "deadline_epoch": DEADLINE,
                "original_frontier": {"next_epoch": 8, "global_step": 51680},
                "scientific_fit_count_increment": 0,
            },
            replace=False,
        )

    @staticmethod
    def live(owner: dict) -> bool:
        """Zombies have exited; native child owners are still reaped by their original Popen caller."""
        if not is_same_process(owner):
            return False
        raw = Path(f"/proc/{owner['pid']}/stat").read_text()
        return raw[raw.rfind(")") + 2 :].split()[0] != "Z"

    def terminate_new(self, seed: int | None = None) -> None:
        """Immediately stop only this R3 controller's exact new command groups."""
        for key, path in list(self.active.items()):
            if seed is not None and not key.startswith(f"{seed}-"):
                continue
            if path.exists():
                value = read(path)
                if value.get("owner", {}).get("controller") == self.owner and is_same_process(
                    value.get("child", {})
                ):
                    with contextlib.suppress(ProcessLookupError):
                        terminate_owned_process(value["child"], grace_seconds=0)

    def execute(
        self, cell: dict, phase: str, command: list[str], *, progress: object = None
    ) -> tuple:
        """Latch callback failures and return normally so the legacy Popen loop promptly reaps."""
        seed = cell["seed"]
        if seed in self.failures:
            raise RuntimeError(self.failures[seed])

        def safe_progress() -> None:
            try:
                if seed in self.failures:
                    raise RuntimeError(self.failures[seed])
                (progress or self.guard)()
            except Exception as error:
                with self.lock:
                    self.failures.setdefault(seed, f"{type(error).__name__}: {error}")
                self.terminate_new(seed)
                # Do not raise into S1 run_owned_command: its exception branch waits
                # the whole remaining deadline on TERM-ignoring observed_worker.

        self.guard()
        receipt = self.control / f"owned-{phase}-gru_s{seed}.json"
        with self.lock:
            if receipt.exists():
                raise FileExistsError("R3 phase already attempted; no automatic retry")
            self.active[f"{seed}-{phase}"] = receipt
        seconds = min(
            86400 if phase in ("archive", "resume") else 1800, max(1, int(DEADLINE - time.time()))
        )
        result = run_owned_command(
            command,
            cwd=Path(cell["request"]["new_source_root"]),
            env=self.environment(cell, archive=phase != "resume"),
            log_path=self.control / f"{phase}-{seed}.log",
            receipt_path=receipt,
            owner={
                "controller": self.owner,
                "cell_id": f"gru_s{seed}",
                "plan_sha256": self.plan_sha,
            },
            stage=1,
            trial_number=cell["gpu_index"],
            mode=phase,
            gpu_key=str(cell["gpu_index"]),
            output_dir=Path(cell["request"]["new_output"]),
            timeout_seconds=seconds,
            progress=safe_progress,
        )
        if seed in self.failures:
            raise RuntimeError(self.failures[seed])
        return result

    def probe(self, cell: dict, proof_sha: str, device: str) -> None:
        """Execute the actual canonical restore on scratch copies, with no model update."""
        seed = cell["seed"]
        spec = self.control / f"{seed}-{device}-probe-input.json"
        receipt = self.control / f"{seed}-{device}-probe.json"
        atomic_write_json(
            spec,
            {
                "output": cell["request"]["new_output"],
                "proof_sha256": proof_sha,
                "scratch": str(self.control / f"{seed}-{device}-probe-scratch"),
                "device": device,
                "receipt": str(receipt),
            },
            replace=False,
        )
        # A tiny launcher keeps per-command environment isolated across the two threads.
        command = [
            self.plan["python"],
            str(Path(__file__).with_name("run_probe.py")),
            str(spec),
            digest(spec),
            str(Path(__file__).with_name("canonical_resume_probe.py")),
        ]
        self.checked(self.execute(cell, f"{device}-probe", command), f"{device}-probe")
        if not probe_ready(receipt, proof_sha, device):
            raise ValueError(
                "canonical restore probe did not produce exact zero-update frontier proof"
            )

    def worker(self, cell: dict) -> None:
        """Migrate once, preflight actual restore on CPU/GPU, arm archive and continue to12E."""
        seed, request = cell["seed"], cell["request"]
        python, tools = self.plan["python"], Path(self.plan["tools"])
        pool, archive = None, None
        try:
            # All historical groups and unchanged original hashes are checked again here.
            validate_recovery(self.plan, self.live, fresh_seed=seed)
            path = self.control / f"{seed}-transition-request.json"
            atomic_write_json(path, request, replace=False)
            self.checked(
                self.execute(
                    cell,
                    "migrate",
                    [
                        python,
                        str(tools / "migrate_monitor_checkpoint_s4.py"),
                        "--request",
                        str(path),
                        "--request-sha256",
                        digest(path),
                    ],
                ),
                "migrate",
            )
            output = Path(request["new_output"])
            proof_sha = digest(output / "transition_proof.json")
            ready = read(output / "transition_ready.json")
            if ready["proof_sha256"] != proof_sha or ready["not_after_epoch"] != DEADLINE:
                raise ValueError("migration proof/readiness deadline differs")
            self.probe(cell, proof_sha, "cpu")
            self.probe(cell, proof_sha, "cuda")
            pool = ThreadPoolExecutor(max_workers=1)
            archive = pool.submit(
                self.execute,
                cell,
                "archive",
                [
                    python,
                    str(tools / "archive_safe_state_s4.py"),
                    "--output",
                    str(output),
                    "--transition-proof-sha256",
                    proof_sha,
                    "--poll-seconds",
                    "1",
                ],
            )
            self.resume(cell, archive, proof_sha)
            terminal = read(output / "terminal_status.json")
            from supervisor_checks import archive_complete

            if (
                terminal.get("status") != "completed"
                or terminal.get("completed_epochs") != 12
                or terminal.get("global_updates") != 77520
                or not archive_complete(output, proof_sha)
            ):
                raise ValueError("recovery lacks exact12E77520 and complete every-epoch archive")
            atomic_write_json(
                self.control / f"{seed}-COMPLETE.json",
                {
                    "transition_proof_sha256": proof_sha,
                    "completed_epochs": 12,
                    "global_updates": 77520,
                    "scientific_fit_count_increment": 0,
                },
                replace=False,
            )
        except BaseException as error:
            with self.lock:
                self.failures.setdefault(seed, f"{type(error).__name__}: {error}")
            raise
        finally:
            # Cleanup precedes any executor wait. Safe callbacks never enter the
            # legacy helper's long-grace exception path, so its Popen caller reaps.
            self.terminate_new(seed)
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)


def main() -> int:
    """Run only the explicitly ROOT-pinned fresh R3 recovery; never retry an old control folder."""
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
        print(
            json.dumps(
                launch_detached_worker(
                    command,
                    cwd=Path(plan["runtime"]),
                    env=dict(os.environ),
                    receipt_path=control / "supervisor-owner.json",
                    metadata={"plan_sha256": args.plan_sha256},
                )
            )
        )
        return 0
    with file_lock(control / "supervisor.lock", blocking=False):
        controller = Recovery(plan, args.plan_sha256)

        def stop(_signal: int, _frame: object) -> None:
            """Root cancellation immediately closes only the exact new owned groups."""
            controller.stop.set()
            controller.terminate_new()

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, stop)
        pool = ThreadPoolExecutor(max_workers=2)
        errors = []
        try:
            for future in as_completed(
                [pool.submit(controller.worker, cell) for cell in plan["cells"]]
            ):
                try:
                    future.result()
                except Exception as error:
                    errors.append(f"{type(error).__name__}: {error}")
            atomic_write_json(
                control / "terminal.json",
                {"errors": errors, "status": "failed" if errors else "completed"},
            )
            return int(bool(errors))
        finally:
            controller.stop.set()
            controller.terminate_new()
            pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    raise SystemExit(main())
