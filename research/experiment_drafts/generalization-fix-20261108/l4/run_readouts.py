from __future__ import annotations

import argparse
import queue
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wave_checks import CONFIG_HASHES, Guard, full_valid, wave_limits
from wave_execution import execute_owned

from wnn.multistage_search import atomic_write_json, file_sha256, read_json
from wnn.multistage_search_runtime import current_process_owner, hold_gpu_slot
from wnn.training.sequence_provenance import canonical_sha256, source_identity

REFERENCE_SHA = "1863e5e6f844c8f172f3fd54879ff908e0e415183766c354482b1ac9c5bfd985"


class Readouts:
    """Run only the two fixed post-fit endpoints, without modifying the live fit controller."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args, self.runtime = args, args.runtime.resolve(strict=True)
        self.control = self.runtime / "control/l4-readouts-r1"
        self.control.mkdir(parents=True, exist_ok=True)
        (self.control / "owned").mkdir(exist_ok=True)
        self.admission = read_json(args.admission)
        self.source = Path(self.admission["source"])
        if (
            self.admission.get("approved") is not True
            or self.admission.get("reviewer") != "ROOT"
            or self.admission.get("reference_manifest_sha256") != REFERENCE_SHA
            or source_identity(self.source, self.source / "scripts/train_sequence.py")["sha256"]
            != self.admission["source_sha256"]
        ):
            raise ValueError("separate fixed-readout source/reference admission differs")
        if file_sha256(Path(self.admission["reference_root"]) / "manifest.json") != REFERENCE_SHA:
            raise ValueError("qualified clean reference bytes differ")
        wave_limits(self.runtime)
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        devices = {
            int(parts[0]): parts[1].strip()
            for line in raw.splitlines()
            if len(parts := line.split(",")) == 3 and "L4" in parts[2]
        }
        self.lanes: queue.Queue[tuple[int, str]] = queue.Queue()
        for gpu in (6, 7):
            self.lanes.put((gpu, devices[gpu]))
        plan = {
            "schema": "wnn.generalization-fixed-readouts.v1",
            "cells": sorted(CONFIG_HASHES),
            "source_sha256": self.admission["source_sha256"],
            "reference_sha256": REFERENCE_SHA,
            "readouts": ["swa3456", "epoch6"],
            "mixture": [0.9, 0.1],
            "new_fits": 0,
        }
        self.plan_sha = canonical_sha256(plan)
        path = self.control / "plan.json"
        if path.exists() and read_json(path) != plan:
            raise ValueError("different fixed-readout plan already owns this namespace")
        if not path.exists():
            atomic_write_json(path, plan, replace=False)
        self.owner = current_process_owner()
        self.cancelled = threading.Event()
        self.guard = Guard(self.runtime, self.control)

    def status(self, cell: dict[str, object], stage: str, **fields: object) -> None:
        atomic_write_json(
            self.control / f"{cell['name']}-status.json",
            {
                "cell": cell["name"],
                "stage": stage,
                "at_epoch": time.time(),
                "plan_sha256": self.plan_sha,
                **fields,
            },
        )

    def run_cell(self, name: str) -> None:
        if self.cancelled.is_set():
            raise RuntimeError("ROOT cancelled readouts before admission")
        gpu, uuid = self.lanes.get()
        cell = {"name": name, "gpu": gpu, "gpu_uuid": uuid}
        try:
            with hold_gpu_slot(self.control, str(gpu)):
                fit = self.runtime / "runs" / name / "fit"
                full_valid(fit, CONFIG_HASHES[name]["normalized"])
                atomic_write_json(
                    self.control / f"claim-{name}.json",
                    {
                        "cell": name,
                        "plan_sha256": self.plan_sha,
                        "terminal_sha256": file_sha256(fit / "terminal_status.json"),
                    },
                    replace=False,
                )
                root = self.runtime / "runs" / name / "readouts"
                root.mkdir(parents=True, exist_ok=True)
                index = read_json(fit / "epoch_index.json")
                swa = root / "swa3456.pt"
                execute_owned(
                    self,
                    cell,
                    "swa",
                    [
                        self.args.python,
                        "-m",
                        "wnn.training.sequence_swa",
                        "--index",
                        str(fit / "epoch_index.json"),
                        "--index-sha256",
                        index["index_sha256"],
                        "--output",
                        str(swa),
                    ],
                    root,
                    self.source,
                )
                for readout, checkpoint in (("swa3456", swa), ("epoch6", fit / "model.pt")):
                    output = root / readout
                    execute_owned(
                        self,
                        cell,
                        f"readout_{readout}",
                        [
                            self.args.python,
                            "-m",
                            "wnn.training.sequence_fixed_readout",
                            "--checkpoint",
                            str(checkpoint),
                            "--data-root",
                            str(self.runtime / "project"),
                            "--split-manifest",
                            str(self.runtime / "project/chronological_primary.json"),
                            "--reference-root",
                            self.admission["reference_root"],
                            "--reference-manifest-sha256",
                            REFERENCE_SHA,
                            "--output",
                            str(output),
                            "--readout",
                            readout,
                        ],
                        output,
                        self.source,
                    )
                    receipt = read_json(output / "readout.json")
                    if (
                        receipt.get("sequence_count") != 1873
                        or receipt.get("selected_rows") != 3528273
                    ):
                        raise ValueError("fixed readout has incomplete native validation coverage")
                self.status(cell, "complete", fixed_readouts=["swa3456", "epoch6"])
        finally:
            self.lanes.put((gpu, uuid))

    def run(self) -> None:
        """Submit whichever fixed6E endpoints finish first, using only spare GPUs6/7."""
        pending, running, failed = set(CONFIG_HASHES), {}, []
        deadline = wave_limits(self.runtime)["wave_deadline_epoch"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            while pending or running:
                if self.cancelled.is_set() or time.time() >= deadline:
                    self.cancelled.set()
                    raise TimeoutError("ROOT cancellation or original four-hour readout deadline")
                for name in sorted(pending):
                    terminal = self.runtime / "runs" / name / "fit/terminal_status.json"
                    if terminal.exists():
                        running[pool.submit(self.run_cell, name)] = name
                        pending.remove(name)
                for future in list(running):
                    if future.done():
                        name = running.pop(future)
                        try:
                            future.result()
                        except Exception as error:
                            self.status(
                                {"name": name}, "failed", error=f"{type(error).__name__}: {error}"
                            )
                            failed.append(name)
                time.sleep(2)
        atomic_write_json(
            self.control / "terminal.json",
            {
                "status": "failed" if failed else "complete",
                "failed": failed,
                "fixed_grades": 24 if not failed else None,
                "new_fits": 0,
            },
            replace=False,
        )
        if failed:
            raise RuntimeError(f"fixed readouts failed without automatic retry: {failed}")


def main() -> None:
    """Execute the admitted post-fit DAG; never touch the immutable running fit scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    worker = Readouts(args)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda _signal, _frame: worker.cancelled.set())
    worker.run()


if __name__ == "__main__":
    main()
