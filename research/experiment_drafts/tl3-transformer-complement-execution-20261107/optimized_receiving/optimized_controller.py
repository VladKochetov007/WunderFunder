from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

# Launch with PYTHONPATH pointing to the existing accepted receiver directory.
from controller import Controller, load_helpers
from optimized_plan import COMMIT, approval_ready, commands, read_panel, sha
from receiving_state import (
    DEFAULT_RUNTIME,
    DEFAULT_SUPPORT,
    PACKET,
    SOURCE_MANIFEST_SHA,
    SPLIT,
    SPLIT_SHA,
    read_object,
    reference_ready,
    verify_acceptance,
)


class OptimizedController(Controller):
    """Reuse the accepted owner/resource lifecycle with separately frozen larger-batch recipes."""

    def __init__(self, args: argparse.Namespace, helpers: SimpleNamespace) -> None:
        self.args, self.h = args, helpers
        self.runtime = args.runtime.resolve()
        self.source = self.runtime / "source-r3"
        self.run_root = self.runtime / "throughput-engineering-r1/runs"
        self.control = self.runtime / "control/b300-optimized-r1"
        self.control.mkdir(parents=True, exist_ok=True)
        accepted = {
            "controller.py": "cdb319a45fce3c2084825feb515bcf128e6369f4c4623fbd9472df57129c9946",
            "receiving_state.py": "af03c2f095977402ab02c2a664d0f578c8e4028e7e5c4c76deae796033f0b7f3",
            "owned_phase.py": "68884ce6c2f277c3b8bf04ea52e7545929454198f2582be2a9a7d47a88c5a0df",
        }
        if (
            Path(sys.modules["controller"].__file__).resolve()
            != (args.receiver / "controller.py").resolve()
        ):
            raise ValueError("imported accepted receiver differs from --receiver")
        if any(sha(args.receiver / name) != expected for name, expected in accepted.items()):
            raise ValueError("accepted receiver core bytes differ")
        self.panel = read_panel(args.panel, args.panel_sha256)
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.owner, self.error = self.h.owner(), None
        self.last_resource_check = 0.0
        self.records, self.resource_identities = {}, {}
        self.state = {
            "schema": "wnn.pr67-b300-optimized-receiving.v1",
            "source_commit": COMMIT,
            "panel_sha256": args.panel_sha256,
            "cells": {
                item["cell"]: {"status": "pending", "pair": item["pair"]}
                for item in self.panel["cells"]
            },
            "fit_attempts": [],
            "profile_attempts": [],
        }
        if (self.control / "attempts.json").exists():
            raise FileExistsError(
                "optimized receiver already admitted phases; no automatic restart"
            )
        if (
            sha(self.runtime / "manifests-r3/b300-source-r3-manifest.json") != SOURCE_MANIFEST_SHA
            or sha(SPLIT) != SPLIT_SHA
        ):
            raise ValueError("source manifest or split differs")
        if not verify_acceptance(self.runtime / "b300") or not reference_ready(self.runtime):
            raise ValueError("existing native acceptance/reference baseline must pass first")
        check = read_object(self.runtime / "b300/data-check.json")
        if any(
            check.get(key) != value
            for key, value in {
                "split_sha256": SPLIT_SHA,
                "fit_train": 10325,
                "fit_valid": 0,
                "evaluation_valid": 1873,
                "component_disjoint": True,
            }.items()
        ):
            raise ValueError("original successful data check missing")
        for item in self.panel["cells"]:
            original = json.loads(
                (self.source / PACKET / "configs" / f"{item['cell']}.json").read_text()
            )
            original["batch_size"] = self.panel["batch_per_rank"]
            if json.loads(Path(item["config"]).read_text()) != original:
                raise ValueError("optimized production config changed more than the selected batch")
        self.run_root.mkdir(parents=True, exist_ok=True)
        baseline = self.runtime / "b300/baseline-score.json"
        target = self.run_root / "baseline-score.json"
        if target.exists() and sha(target) != sha(baseline):
            raise ValueError("new panel baseline differs")
        if not target.exists():
            target.write_bytes(baseline.read_bytes())
        self.h.write(
            self.control / "implementation.json",
            {
                "source_commit": COMMIT,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA,
                "panel_sha256": args.panel_sha256,
                "receiver_sha256": {
                    str(path): sha(path) for path in Path(__file__).parent.glob("*.py")
                },
                "accepted_receiver_sha256": {
                    str(path): sha(path) for path in args.receiver.glob("*.py")
                },
                "python": str(args.python),
                "old_partial_fits": "retired; never resumed or counted complete",
            },
            replace=False,
        )

    def authority(self, *, start: bool = False) -> dict | None:
        """Admit only four new fits/eight profiles inside24h with a final one-hour reserve."""
        with self.lock:
            path = self.control / "authority.json"
            if not path.exists():
                if not start:
                    return None
                now = time.time()
                self.h.write(
                    path,
                    {
                        "schema": "wnn.pr67-optimized-authority.v1",
                        "first_profile_epoch": now,
                        "fit_deadline_epoch": now + 82800,
                        "deadline_epoch": now + 86400,
                        "collection_reserve_seconds": 3600,
                        "fit_cap": 4,
                        "profile_cap": 8,
                        "source_commit": COMMIT,
                        "panel_sha256": self.args.panel_sha256,
                    },
                    replace=False,
                )
            value = read_object(path)
            if (
                value["fit_cap"] != 4
                or value["profile_cap"] != 8
                or value["collection_reserve_seconds"] != 3600
                or value["source_commit"] != COMMIT
                or value["panel_sha256"] != self.args.panel_sha256
                or value["deadline_epoch"] - value["first_profile_epoch"] != 86400
                or value["deadline_epoch"] - value["fit_deadline_epoch"] != 3600
            ):
                raise ValueError("fixed optimized authority changed")
            return value

    def environment(self, pair: str) -> dict[str, str]:
        """Use unchanged model source/precision and the short accepted scratch/compiler paths."""
        env = super().environment(pair)
        env.update(
            CUBLAS_EMULATE_SINGLE_PRECISION="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            WNN_MONITOR_HEARTBEAT_SECONDS="5",
            TMPDIR="/data/epanov_s/tdn7run/optimized/scratch",
            TORCHINDUCTOR_CACHE_DIR="/data/epanov_s/tdn7run/compiler/inductor",
            TRITON_CACHE_DIR="/data/epanov_s/tdn7run/compiler/triton",
        )
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        return env

    def execute(self, action: str, cell: str = "", pair: str = "6,7") -> None:
        """Bound a direct canonical command with the unchanged accepted descendant-cleanup relay."""
        self.guard(admission=action in ("profile", "fit"))
        authority = self.authority(start=action == "profile")
        deadline = (
            None
            if authority is None
            else authority[
                "fit_deadline_epoch" if action in ("profile", "fit") else "deadline_epoch"
            ]
        )
        seconds = 900 if action == "profile" else 1800
        if deadline is not None:
            seconds = (
                min(seconds, int(deadline - time.time()))
                if action != "fit"
                else int(deadline - time.time())
            )
        if seconds <= 0:
            raise RuntimeError("optimized phase deadline closed")
        label = f"{cell or 'panel'}-{action}"
        receipt = self.control / "owned" / f"{label}.json"
        with self.lock:
            if receipt.exists():
                raise FileExistsError("phase attempted; no automatic retry")
            read_panel(self.args.panel, self.args.panel_sha256)
            self.records[label] = receipt
            if action in ("profile", "fit"):
                key, cap = ("fit_attempts", 4) if action == "fit" else ("profile_attempts", 8)
                if len(self.state[key]) >= cap:
                    raise RuntimeError("optimized allocation exhausted")
                self.state[key].append({"cell": cell, "phase": action, "at_epoch": time.time()})
            self.h.write(self.control / "attempts.json", self.state)
        if action == "source-check":
            argv = [
                [
                    str(self.args.python),
                    "-m",
                    "wnn.training.sequence_source_capsule",
                    "--root",
                    str(self.source),
                    "--manifest",
                    str(self.runtime / "manifests-r3/b300-source-r3-manifest.json"),
                    "--manifest-sha256",
                    SOURCE_MANIFEST_SHA,
                    "--commit",
                    COMMIT,
                ]
            ]
        elif action == "compare":
            argv = [
                [
                    str(self.args.python),
                    str(
                        self.source
                        / "research/experiment_drafts/tl3-transformer-complement-20261106/compare_panel.py"
                    ),
                    "--run-root",
                    str(self.run_root),
                    "--config-root",
                    str(self.args.panel.parent / "configs"),
                    "--allow-homogeneous-devices",
                    "--output",
                    str(self.run_root / "comparison.json"),
                ]
            ]
        else:
            item = next(item for item in self.panel["cells"] if item["cell"] == cell)
            argv = commands(action, item, self.runtime, self.args.python, self.run_root)
        plan = self.control / "commands" / f"{label}.json"
        self.h.write(plan, argv, replace=False)
        environment = self.environment(pair)
        environment["WNN_CONTROLLER_DESCENDANTS_RECEIPT"] = str(
            self.control / "descendants" / f"{label}.json"
        )
        command = [
            str(self.args.python),
            str(self.args.receiver / "owned_phase.py"),
            "--support-src",
            str(self.args.support_src),
            "--",
            str(self.args.python),
            str(Path(__file__).with_name("phase.py")),
            str(plan),
        ]
        if cell:
            self.update(cell, status=action, owned_receipt=str(receipt))
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
            output_dir=self.run_root / cell / action,
            timeout_seconds=seconds,
            progress=self.guard,
        )
        if code != 0 or timed_out:
            raise RuntimeError(f"{label} failed exit={code}, timeout={timed_out}; no retry")

    def worker(self, item: dict) -> None:
        """Let every independently approved cell progress without waiting for the other cells."""
        cell, pair = item["cell"], item["pair"]
        try:
            self.execute("profile", cell, pair)
            self.update(cell, status="waiting_ROOT_profile_approval")
            while not approval_ready(self.run_root / cell):
                self.guard(admission=True)
                self.stop.wait(10)
            read_panel(self.args.panel, self.args.panel_sha256)
            self.execute("fit", cell, pair)
            terminal = read_object(self.run_root / cell / "fit/terminal_status.json")
            if (
                terminal.get("status") != "completed"
                or terminal.get("completed_epochs") != 30
                or terminal.get("global_updates") != item["expected_updates"]
            ):
                raise ValueError("optimized cell lacks exact final30E/derived-update endpoint")
            for action in ("evaluate", "bundle"):
                self.execute(action, cell, pair)
            self.update(cell, status="completed", primary_endpoint_complete=True)
        except Exception as error:
            self.update(cell, status="failed", error=f"{type(error).__name__}: {error}")

    def run(self) -> int:
        """Verify frozen S3 once, then run only the four declared matched optimized cells."""
        self.execute("source-check")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self.worker, self.panel["cells"]))
        if all(value["status"] == "completed" for value in self.state["cells"].values()):
            self.execute("compare")
            return 0
        return 1


def main() -> int:
    """Launch an explicitly identified optimized panel through the accepted owner helper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--support-src", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--receiver", type=Path, default=DEFAULT_RUNTIME / "receiver-b300-r1")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.python, args.panel, args.receiver)):
        parser.error("absolute Python/panel/accepted receiver paths required")
    helpers = load_helpers(args.support_src)
    control = args.runtime / "control/b300-optimized-r1"
    if args.detach:
        command = [str(args.python), str(Path(__file__).resolve()), *sys.argv[1:]]
        command.remove("--detach")
        owner = helpers.detach(
            command,
            cwd=args.runtime,
            env=dict(os.environ),
            receipt_path=control / "controller-owner.json",
            metadata={"source_commit": COMMIT, "panel_sha256": args.panel_sha256},
        )
        sys.stdout.write(json.dumps(owner) + "\n")
        return 0
    with helpers.lock(control / "controller.lock", blocking=False):
        controller = OptimizedController(args, helpers)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda _signal, _frame: controller.stop.set())
        try:
            return controller.run()
        finally:
            if controller.stop.is_set() or controller.error is not None:
                controller.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
