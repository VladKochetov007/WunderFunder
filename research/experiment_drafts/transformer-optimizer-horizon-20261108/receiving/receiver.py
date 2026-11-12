from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from admission import approved, authority, digest, read, software
from controller import Controller, load_helpers
from lifecycle import accepted_helpers, progress, wait_for_plots


class Receiver(Controller):
    """Reuse accepted resources/ownership helpers for exactly six fresh fixed4E cells."""

    def __init__(self, args: argparse.Namespace, helpers: object) -> None:
        self.args, self.h = args, helpers
        accepted_helpers(args.accepted_receiver, Controller)
        self.plan = read(args.plan)
        if digest(args.plan) != args.plan_sha256:
            raise ValueError("receiver plan changed")
        self.runtime, self.source = Path(self.plan["runtime"]), Path(self.plan["source_root"])
        self.run_root, self.control = self.runtime / "runs", self.runtime / "control/receiving-r1"
        self.control.mkdir(parents=True, exist_ok=True)
        self.lock, self.stop, self.owner = threading.RLock(), threading.Event(), helpers.owner()
        self.error, self.last_resource_check = None, 0.0
        self.records, self.resource_identities, self.failures = {}, {}, {}
        self.state = {
            "cells": {
                cell["cell"]: {"status": "pending", "pair": cell["pair"]}
                for cell in self.plan["cells"]
            },
            "fit_attempts": [],
            "profile_attempts": [],
        }
        expected = [
            ("C0_muon256_a4_s203101", "C0", 203101, "0,1"),
            ("C1_adamw256_a4_s203101", "C1", 203101, "2,3"),
            ("C2_muon512_a2_s203101", "C2", 203101, "4,5"),
            ("C0_muon256_a4_s203102", "C0", 203102, "6,7"),
            ("C1_adamw256_a4_s203102", "C1", 203102, "2,3"),
            ("C2_muon512_a2_s203102", "C2", 203102, "4,5"),
        ]
        if [
            (item["cell"], item["arm"], item["seed"], item["pair"]) for item in self.plan["cells"]
        ] != expected:
            raise ValueError("receiver must preserve exactly the frozen optimizer/seed/pair panel")
        if (self.control / "attempts.json").exists() or len(self.plan["cells"]) != 6:
            raise FileExistsError(
                "one-shot receiver already admitted work or panel is not six cells"
            )
        if digest(Path(self.plan["source_manifest"])) != self.plan["source_manifest_sha256"]:
            raise ValueError("source capsule manifest differs")
        for cell in self.plan["cells"]:
            if digest(Path(cell["config_path"])) != cell["config_sha256"]:
                raise ValueError("fixed4E recipe bytes differ")
        self.authority()
        self.h.write(
            self.control / "implementation.json",
            {
                "plan_sha256": args.plan_sha256,
                "receiver_sha256": {str(p): digest(p) for p in Path(__file__).parent.glob("*.py")},
                "source_manifest_sha256": self.plan["source_manifest_sha256"],
            },
            replace=False,
        )

    def authority(self, *, start: bool = False) -> dict:
        """Consume ROOT's existing earliest qualification clock without creating/extending it."""
        path = Path(self.plan["authority"])
        if digest(path) != self.plan["authority_sha256"]:
            raise ValueError("fixed four-hour authority changed")
        value = read(path)
        authority(value)
        return value

    def terminate(self, cell: str | None = None) -> None:
        """Signal only new exact owned relays, allowing their proven separate-session rank cleanup."""
        for key, path in list(self.records.items()):
            if cell is not None and not key.startswith(cell + "-"):
                continue
            if path.exists():
                record = read(path)
                if record.get("owner", {}).get("controller") == self.owner and self.h.same(
                    record.get("child", {})
                ):
                    with contextlib.suppress(ProcessLookupError):
                        self.h.terminate(record["child"], grace_seconds=1.0)

    def environment(self, pair: str) -> dict[str, str]:
        """Match exact profile precision, compiler caches and short physical-disk scratch."""
        env = dict(os.environ)
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            env.pop(key, None)
        env.update(self.plan["environment"])
        env["CUDA_VISIBLE_DEVICES"] = pair
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        return env

    def execute(self, action: str, cell: str = "", pair: str = "0,1") -> None:
        """Run fixed canonical commands; latch callback failures until the native loop reaps."""
        self.guard(admission=action == "fit")
        bounds = self.authority()
        seconds = int(
            bounds["fit_deadline_epoch" if action == "fit" else "deadline_epoch"] - time.time()
        )
        seconds = seconds if action in ("fit", "plots") else min(3600, seconds)
        if seconds <= 0:
            raise RuntimeError("phase budget closed")
        label = f"{cell or 'panel'}-{action}"
        receipt = self.control / "owned" / f"{label}.json"
        with self.lock:
            if receipt.exists():
                raise FileExistsError("phase attempted; no automatic retry/resume")
            self.records[label] = receipt
            if action == "fit":
                if len(self.state["fit_attempts"]) >= 6:
                    raise RuntimeError("six-fit cap exhausted")
                self.state["fit_attempts"].append({"cell": cell, "at_epoch": time.time()})
            self.h.write(self.control / "attempts.json", self.state)
        command = [
            str(self.args.python),
            str(Path(__file__).with_name("phase.py")),
            "--plan",
            str(self.args.plan),
            "--action",
            action,
            "--cell",
            cell,
        ]
        command = [
            str(self.args.python),
            str(self.args.accepted_receiver / "owned_phase.py"),
            "--support-src",
            str(self.args.support_src),
            "--",
            *command,
        ]
        env = self.environment(pair)
        if action == "plots":
            env["CUDA_VISIBLE_DEVICES"] = ""
        env["WNN_CONTROLLER_DESCENDANTS_RECEIPT"] = str(
            self.control / "descendants" / f"{label}.json"
        )

        def report_progress() -> None:
            """Never raise into the legacy helper's remaining-deadline TERM grace branch."""
            progress(self, label)

        code, timed_out, _ = self.h.run(
            command,
            cwd=self.source,
            env=env,
            log_path=self.control / "logs" / f"{label}.log",
            receipt_path=receipt,
            owner={"controller": self.owner, "cell_id": cell},
            stage=1,
            trial_number=int(pair.split(",")[0]),
            mode=action,
            gpu_key=pair,
            output_dir=self.run_root / cell / action,
            timeout_seconds=seconds,
            progress=report_progress,
        )
        if code != 0 or timed_out or label in self.failures:
            raise RuntimeError(f"{label} failed exit={code}, timed_out={timed_out}; no retry")

    def worker(self, cell: dict) -> None:
        """Advance only after ROOT approves this exact seed/config profile receipt."""
        name = cell["cell"]
        try:
            self.update(name, status="waiting_ROOT_profile_approval")
            while not approved(self.runtime, cell, self.plan["expected_training_source_sha256"]):
                self.guard(admission=True)
                self.stop.wait(5)
            self.execute("fit", name, cell["pair"])
            terminal = read(self.run_root / name / "fit/terminal_status.json")
            if (
                terminal.get("status") != "completed"
                or terminal.get("completed_epochs") != 4
                or terminal.get("global_updates") != 3280
            ):
                raise ValueError("fixed4E3280 endpoint is incomplete")
            self.execute("evaluate", name, cell["pair"])
            wait_for_plots(self, name)
            self.execute("bundle", name, cell["pair"])
            self.update(name, status="completed", primary_endpoint_complete=True)
        except Exception as error:
            self.update(name, status="failed", error=f"{type(error).__name__}: {error}")

    def run(self) -> int:
        """Qualify source/software first; no old process is ever signalled by this receiver."""
        self.execute("source-check")
        while not software(self.runtime, self.plan):
            self.guard(admission=True)
            self.stop.wait(5)
        plot_pool = ThreadPoolExecutor(max_workers=1)
        plot_future = plot_pool.submit(self.execute, "plots")
        try:
            while not (self.control / "plots-ready.json").exists():
                if plot_future.done():
                    plot_future.result()
                    raise RuntimeError("plot backend exited before readiness; no fits admitted")
                self.guard(admission=True)
                self.stop.wait(0.2)
            cells = {cell["cell"]: cell for cell in self.plan["cells"]}
            for wave in self.plan["profile_wave_order"]:
                with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                    list(pool.map(self.worker, (cells[name] for name in wave)))
        finally:
            self.h.write(
                self.control / "plots-stop.json", {"stop_after_final_journal_render": True}
            )
            if self.stop.is_set() or self.error is not None:
                self.terminate()
            plot_pool.shutdown(wait=True)
        if all(cell["status"] == "completed" for cell in self.state["cells"].values()):
            self.execute("compare")
            return 0
        return 1


def main() -> int:
    """Launch a finite receiver with ROOT's known support and accepted birth-bound subreaper relay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--accepted-receiver", type=Path, required=True)
    parser.add_argument("--support-src", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    helpers = load_helpers(args.support_src)
    plan = read(args.plan)
    control = Path(plan["runtime"]) / "control/receiving-r1"
    if args.detach:
        command = [str(args.python), str(Path(__file__).resolve()), *sys.argv[1:]]
        command.remove("--detach")
        print(
            json.dumps(
                helpers.detach(
                    command,
                    cwd=Path(plan["runtime"]),
                    env=dict(os.environ),
                    receipt_path=control / "controller-owner.json",
                    metadata={"plan_sha256": args.plan_sha256},
                )
            )
        )
        return 0
    with helpers.lock(control / "controller.lock", blocking=False):
        receiver = Receiver(args, helpers)

        def stop(_signal: int, _frame: object) -> None:
            """Close admission and stop only this new controller's owned command groups."""
            receiver.stop.set()
            receiver.terminate()

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, stop)
        try:
            return receiver.run()
        finally:
            receiver.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
