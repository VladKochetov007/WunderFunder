from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from receiving_state import (
    CELLS,
    COMMIT,
    DEFAULT_RUNTIME,
    DEFAULT_SUPPORT,
    PACKET,
    SHORT_RUN_ROOT,
    SOURCE_MANIFEST_SHA,
    SPLIT,
    SPLIT_SHA,
    approval_ready,
    digest,
    directory_bytes,
    process_family_rss,
    read_object,
    reference_ready,
    verify_acceptance,
)


def load_helpers(source: Path) -> SimpleNamespace:
    """Load only ROOT's known support package in the controller, never in model children."""
    sys.path.insert(0, str(source))
    from wnn.multistage_search import atomic_write_json, file_lock
    from wnn.multistage_search_owned_command import run_owned_command
    from wnn.multistage_search_runtime import (
        current_process_owner,
        is_same_process,
        launch_detached_worker,
        process_identity,
        terminate_owned_process,
    )

    return SimpleNamespace(
        write=atomic_write_json,
        lock=file_lock,
        run=run_owned_command,
        owner=current_process_owner,
        same=is_same_process,
        detach=launch_detached_worker,
        terminate=terminate_owned_process,
        identity=process_identity,
    )


class Controller:
    """Execute only four frozen cells through the existing canonical wrapper and owner helper."""

    def __init__(self, args: argparse.Namespace, helpers: SimpleNamespace) -> None:
        self.args, self.h = args, helpers
        self.runtime = args.runtime.resolve()
        self.source = self.runtime / "source-r3"
        self.run_root = self.runtime / "b300"
        self.control = self.runtime / "control/b300-receiving-r1"
        self.control.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.owner = self.h.owner()
        self.error: str | None = None
        self.last_resource_check = 0.0
        self.records: dict[str, Path] = {}
        self.resource_identities: dict[int, int] = {}
        self.state: dict[str, object] = {
            "schema": "wnn.pr67-b300-receiving.v1",
            "source_commit": COMMIT,
            "cells": {cell: {"status": "pending", "pair": pair} for cell, pair in CELLS},
            "fit_attempts": [],
            "profile_attempts": [],
        }
        self.amendment_sha: str | None = None
        if (self.control / "attempts.json").exists():
            raise FileExistsError("controller already admitted phases; no automatic restart")
        if SHORT_RUN_ROOT.resolve() != self.run_root.resolve() or not SHORT_RUN_ROOT.is_dir():
            raise ValueError("mandatory short run alias must resolve to R/b300")
        manifest = self.runtime / "manifests-r3/b300-source-r3-manifest.json"
        if digest(manifest) != SOURCE_MANIFEST_SHA or digest(SPLIT) != SPLIT_SHA:
            raise ValueError("frozen source manifest or official split differs")
        data_check = read_object(self.run_root / "data-check.json")
        if any(
            data_check.get(key) != value
            for key, value in {
                "split_sha256": SPLIT_SHA,
                "fit_train": 10325,
                "fit_valid": 0,
                "evaluation_valid": 1873,
                "component_disjoint": True,
            }.items()
        ):
            raise ValueError("successful source-bound data check is required")
        implementation = {
            "receiver": {
                str(path.resolve()): digest(path) for path in Path(__file__).parent.glob("*.py")
            },
            "support": {
                str(args.support_src / relative): digest(args.support_src / relative)
                for relative in (
                    "wnn/multistage_search.py",
                    "wnn/multistage_search_runtime.py",
                    "wnn/multistage_search_owned_command.py",
                    "wnn/multistage_search_budget.py",
                )
            },
            "model_source_commit": COMMIT,
            "model_source_manifest_sha256": SOURCE_MANIFEST_SHA,
            "python": str(args.python),
            "short_run_root": str(SHORT_RUN_ROOT),
        }
        self.h.write(self.control / "implementation.json", implementation, replace=False)

    def update(self, cell: str, **fields: object) -> None:
        """Publish operational status without rewriting scientific logs or receipts."""
        with self.lock:
            self.state["cells"][cell].update(fields)
            self.h.write(
                self.control / "status.json", self.state | {"updated_at_epoch": time.time()}
            )

    def authority(self, *, start: bool = False) -> dict[str, object] | None:
        """Start the clock immediately before first profile; bind explicit ROOT-only amendments."""
        with self.lock:
            path = self.control / "authority.json"
            if not path.exists():
                if not start:
                    return None
                now = time.time()
                self.h.write(
                    path,
                    {
                        "schema": "wnn.pr67-b300-authority.v1",
                        "first_profile_epoch": now,
                        "fit_deadline_epoch": now + 82800,
                        "deadline_epoch": now + 86400,
                        "collection_reserve_seconds": 3600,
                        "fit_cap": 4,
                        "profile_cap": 12,
                        "source_commit": COMMIT,
                    },
                    replace=False,
                )
            original = read_object(path)
            amendment = self.control / "ROOT-authority-amendment.json"
            if not amendment.exists():
                return original
            value = read_object(amendment)
            amendment_sha = digest(amendment)
            if self.amendment_sha is None and self.state["fit_attempts"]:
                raise ValueError("authority amendment must be bound before any fit reservation")
            if self.amendment_sha not in (None, amendment_sha):
                raise ValueError("bound authority amendment changed")
            if (
                value.get("approved_by") != "ROOT"
                or not value.get("reason")
                or value.get("previous_authority_sha256") != digest(path)
                or value.get("first_profile_epoch") != original["first_profile_epoch"]
                or value.get("fit_cap") != 4
                or value.get("profile_cap") != 12
                or value.get("collection_reserve_seconds") != 3600
                or value.get("source_commit") != COMMIT
                or value["deadline_epoch"] - value["fit_deadline_epoch"] != 3600
                or value["deadline_epoch"] < original["deadline_epoch"]
            ):
                raise ValueError("ROOT authority amendment is inconsistent")
            self.amendment_sha = amendment_sha
            return original | value

    def terminate(self) -> None:
        """Terminate only this controller's command receipts with exact live identities."""
        for path in list(self.records.values()):
            if path.exists():
                record = read_object(path)
                child = record.get("child")
                if (
                    record.get("owner", {}).get("controller") == self.owner
                    and isinstance(child, dict)
                    and self.h.same(child)
                ):
                    self.h.terminate(child, grace_seconds=0.5)

    def guard(self, *, admission: bool = False) -> None:
        """Check global limits every ten seconds and retain the full collection reserve."""
        with self.lock:
            authority = self.authority()
            if self.stop.is_set():
                self.error = "ROOT stop signal received"
            if authority is not None and time.time() >= authority["deadline_epoch"]:
                self.error = "campaign collection deadline reached"
            if self.error is not None:
                self.terminate()
                raise RuntimeError(self.error)
            if (
                admission
                and authority is not None
                and time.time() >= authority["fit_deadline_epoch"]
            ):
                raise RuntimeError("fit/profile admission closed for collection reserve")
            if time.monotonic() - self.last_resource_check < 10:
                return
            self.last_resource_check = time.monotonic()
            groups: set[int] = set()
            paths = set(self.records.values()) | set((self.runtime / "receipts").rglob("*.json"))
            for path in paths:
                try:
                    record = read_object(path)
                    child = record.get("child")
                    if (
                        record.get("schema") == "wnn.multistage-owned-process.v1"
                        and isinstance(child, dict)
                        and self.h.same(child)
                    ):
                        groups.add(int(child["pgid"]))
                        self.resource_identities[int(child["pid"])] = int(child["birth"])
                except (OSError, ValueError):
                    continue
            rss = process_family_rss(self.resource_identities)
            outputs = sum(
                directory_bytes(path)
                for path in (
                    self.run_root,
                    self.runtime / "reference",
                    self.runtime / "project/cache",
                )
            )
            free = shutil.disk_usage("/data").free
            self.h.write(
                self.control / "resources.json",
                {
                    "at_epoch": time.time(),
                    "receiving_rss_bytes": rss,
                    "output_bytes": outputs,
                    "data_free_bytes": free,
                    "receipt_owned_process_groups": sorted(groups),
                },
            )
            if rss > 1024 * 1024**3 or outputs > 200_000_000_000 or free < 200_000_000_000:
                self.error = "receiving RSS/output/free-space limit reached"
                self.terminate()
                raise RuntimeError(self.error)

    def environment(self, pair: str) -> dict[str, str]:
        """Bind the short scratch alias and actual frozen model source for every child."""
        env = dict(os.environ)
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            env.pop(key, None)
        env.update(
            WNN_PYTHON=str(self.args.python),
            WNN_EXPECTED_COMMIT=COMMIT,
            WNN_SOURCE_MANIFEST=str(self.runtime / "manifests-r3/b300-source-r3-manifest.json"),
            WNN_SOURCE_MANIFEST_SHA256=SOURCE_MANIFEST_SHA,
            WNN_DATA_ROOT=str(self.runtime / "project"),
            WNN_SPLIT=str(SPLIT),
            WNN_RUN_ROOT=str(SHORT_RUN_ROOT),
            WNN_REFERENCE_ROOT=str(self.runtime / "reference"),
            CUDA_VISIBLE_DEVICES=pair,
            PYTHONPATH=str(self.source / "src"),
            CC="/usr/bin/gcc",
            CXX="/usr/bin/g++",
            TORCHINDUCTOR_COMPILE_THREADS="4",
        )
        return env

    def execute(self, action: str, cell: str = "", pair: str = "6,7") -> None:
        """Run one finite wrapper phase without duplicating its independent observer."""
        self.guard(admission=action.startswith("profile") or action == "fit")
        authority = self.authority(start=action.startswith("profile"))
        seconds = 1800
        if authority is not None:
            deadline = (
                authority["fit_deadline_epoch"]
                if action == "fit" or action.startswith("profile")
                else authority["deadline_epoch"]
            )
            seconds = (
                int(deadline - time.time())
                if action == "fit"
                else min(1800, int(deadline - time.time()))
            )
        if seconds <= 0:
            raise RuntimeError("phase deadline has closed")
        label = f"{cell or 'panel'}-{action}"
        receipt = self.control / "owned" / f"{label}.json"
        with self.lock:
            if receipt.exists():
                raise FileExistsError("phase already attempted; no automatic retry")
            self.records[label] = receipt
            if action == "fit" or action.startswith("profile"):
                key, cap = ("fit_attempts", 4) if action == "fit" else ("profile_attempts", 12)
                if len(self.state[key]) >= cap:
                    raise RuntimeError("fixed scientific/profile attempt cap exhausted")
                self.state[key].append({"cell": cell, "phase": action, "at_epoch": time.time()})
            self.h.write(self.control / "attempts.json", self.state)
        command = ["bash", str(self.source / PACKET / "run.sh"), action]
        if cell:
            command.append(cell)
            self.update(cell, status=action, owned_receipt=str(receipt))
        command = [
            str(self.args.python),
            str(Path(__file__).with_name("owned_phase.py")),
            "--support-src",
            str(self.args.support_src),
            "--",
            *command,
        ]
        phase_output = self.run_root / cell / action if cell else self.run_root / action
        if action == "fit":
            phase_output = self.run_root / cell / "fit"
        elif action == "evaluate":
            phase_output = self.run_root / cell / "evaluation"
        elif action == "check":
            phase_output = self.run_root / "acceptance"
        environment = self.environment(pair)
        environment["WNN_CONTROLLER_DESCENDANTS_RECEIPT"] = str(
            self.control / "descendants" / f"{label}.json"
        )
        code, timed_out, _ = self.h.run(
            command,
            cwd=self.source,
            env=environment,
            log_path=self.control / "logs" / f"{label}.log",
            receipt_path=receipt,
            owner={"controller": self.owner, "cell_id": cell, "source_commit": COMMIT},
            stage=1,
            trial_number=int(pair.split(",")[0]),
            mode=action,
            gpu_key=pair,
            output_dir=phase_output,
            timeout_seconds=seconds,
            progress=self.guard,
        )
        if code != 0 or timed_out:
            raise RuntimeError(f"{label} failed: exit={code}, timeout={timed_out}; no retry")

    def worker(self, item: tuple[str, str]) -> None:
        """Advance one cell independently through three profiles, ROOT approval and final artifacts."""
        cell, pair = item
        try:
            for action in ("profile-one", "profile-two", "profile-peak"):
                self.execute(action, cell, pair)
            self.update(cell, status="waiting_ROOT_profile_approval")
            while not approval_ready(self.run_root, cell):
                self.guard(admission=True)
                self.stop.wait(10)
            for action in ("fit", "evaluate", "bundle"):
                self.execute(action, cell, pair)
            terminal = read_object(self.run_root / cell / "fit/terminal_status.json")
            if (
                terminal.get("status") != "completed"
                or terminal.get("completed_epochs") != 30
                or terminal.get("global_updates") != 774600
            ):
                raise ValueError("cell lacks its fixed-final complete thirty-epoch endpoint")
            self.update(cell, status="completed", primary_endpoint_complete=True)
        except Exception as error:
            self.update(cell, status="failed", error=f"{type(error).__name__}: {error}")

    def run(self) -> int:
        """Run checks in parallel with external reference preparation, then the fixed panel."""
        if not verify_acceptance(self.run_root):
            self.execute("check")
            if not verify_acceptance(self.run_root):
                raise RuntimeError("native check returned without complete acceptance")
        wait_deadline = time.monotonic() + self.args.reference_wait_seconds
        while not reference_ready(self.runtime):
            self.guard()
            if time.monotonic() >= wait_deadline:
                raise TimeoutError("reference preparation wait exhausted; no profile/fit admitted")
            self.stop.wait(10)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self.worker, CELLS))
        if all(value["status"] == "completed" for value in self.state["cells"].values()):
            self.execute("compare")
            return 0
        return 1


def main() -> int:
    """Launch this finite receiver using the known support source, with no host Git dependency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--support-src", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--reference-wait-seconds", type=int, default=21600)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if not args.python.is_absolute() or not 1 <= args.reference_wait_seconds <= 86400:
        parser.error("absolute Python and a reference wait in [1,86400] seconds are required")
    helpers = load_helpers(args.support_src)
    control = args.runtime / "control/b300-receiving-r1"
    if args.detach:
        command = [str(args.python), str(Path(__file__).resolve()), *sys.argv[1:]]
        command.remove("--detach")
        owner = helpers.detach(
            command,
            cwd=args.runtime,
            env=dict(os.environ),
            receipt_path=control / "controller-owner.json",
            metadata={"source_commit": COMMIT},
        )
        sys.stdout.write(json.dumps(owner) + "\n")
        return 0
    with helpers.lock(control / "controller.lock", blocking=False):
        controller = Controller(args, helpers)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda _signal, _frame: controller.stop.set())
        try:
            return controller.run()
        finally:
            if controller.stop.is_set() or controller.error is not None:
                controller.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
