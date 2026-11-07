from __future__ import annotations

import argparse
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from wave_checks import (
    ARMS,
    CONFIG_HASHES,
    SOURCE_SHA,
    SPLIT_SHA,
    Guard,
    full_valid,
    profile_identity,
)
from wave_execution import execute_owned
from wave_postfit import run_postfit

from wnn.multistage_search import atomic_write_json, file_sha256, read_json
from wnn.multistage_search_runtime import current_process_owner, hold_gpu_slot
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import (
    canonical_sha256,
    source_identity,
    validate_profile_receipt,
)

OLD = Path("/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1")
SOURCE = OLD / "source-gru-monitor-resume-r1"


class Wave:
    """Thin six-cell orchestration using accepted ownership/observer helpers and canonical CLI."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args, self.runtime = args, args.runtime.resolve(strict=True)
        self.control = self.runtime / "control/l4-wave-r1"
        self.control.mkdir(parents=True, exist_ok=True)
        (self.control / "owned").mkdir(exist_ok=True)
        authority = read_json(self.runtime / "ROOT-preparation-authority-r1.json")
        if (
            authority["uid"] != os.getuid()
            or authority["max_fresh_fits"] != 6
            or Path(authority["runtime"]).resolve() != self.runtime
            or authority["output_bytes_cap"] != 4_000_000_000
            or authority["disk_free_floor"] != 3_000_000_000
        ):
            raise ValueError("runtime authority differs from the ROOT-prepared six-fit allocation")
        if source_identity(SOURCE, SOURCE / "scripts/train_sequence.py")["sha256"] != SOURCE_SHA:
            raise ValueError("training source is not the unchanged qualified native S4 capsule")
        if file_sha256(self.runtime / "project/chronological_primary.json") != SPLIT_SHA:
            raise ValueError("official chronological split bytes differ")
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
        self.cells = []
        for gpu, (arm, mode, dropout, seed) in enumerate(
            (arm, mode, dropout, seed) for arm, mode, dropout in ARMS for seed in (101101, 101102)
        ):
            name = f"{arm}_s{seed}"
            path = self.runtime / "configs" / f"{name}.json"
            config = load_config(path)
            if (
                file_sha256(path) != CONFIG_HASHES[name]["raw"]
                or canonical_sha256(config) != CONFIG_HASHES[name]["normalized"]
            ):
                raise ValueError("raw or normalized recipe differs from its frozen six-cell hash")
            if (
                config["seed"] != seed
                or config["epochs"] != 6
                or config["batch_size"] != 32
                or config["chunk_rows"] != 1000
                or config["model"]["input_feature_mode"] != mode
                or config["model"]["dropout"] != dropout
                or config["training_mask"] != "all_after_warmup"
                or config["normalization_scope"] != "all_fit"
                or config["checkpoint_retention"] != "all_epochs"
            ):
                raise ValueError("recipe differs from the declared matched six-cell panel")
            self.cells.append(
                {
                    "name": name,
                    "arm": arm,
                    "mode": mode,
                    "seed": seed,
                    "gpu": gpu,
                    "gpu_uuid": devices[gpu],
                    "config": str(path),
                    "raw_config_sha256": file_sha256(path),
                    "config_sha256": canonical_sha256(config),
                }
            )
        plan = {
            "schema": "wnn.generalization-six-cell-wave.v1",
            "source_sha256": SOURCE_SHA,
            "cells": self.cells,
            "runtime": str(self.runtime),
            "profiles": 6,
            "max_fits": 6,
            "profile_updates": 30,
            "wave_seconds": 14400,
            "collection_reserve_seconds": 3600,
        }
        self.plan_sha = canonical_sha256(plan)
        path = self.control / "panel.json"
        if path.exists() and read_json(path) != plan:
            raise ValueError("a different frozen panel already owns this runtime")
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
                "panel_sha256": self.plan_sha,
                **fields,
            },
        )

    def execute(
        self,
        cell: dict[str, object],
        stage: str,
        argv: list[str],
        output: Path,
        source: Path = SOURCE,
        cap_seconds: int | None = None,
    ) -> None:
        execute_owned(self, cell, stage, argv, output, source, cap_seconds)

    def train_command(self, cell: dict[str, object], output: Path) -> list[str]:
        return [
            self.args.python,
            str(SOURCE / "scripts/train_sequence.py"),
            "--config",
            cell["config"],
            "--data-root",
            str(self.runtime / "project"),
            "--split-manifest",
            str(self.runtime / "project/chronological_primary.json"),
            "--output",
            str(output),
        ]

    def run_cell(self, cell: dict[str, object]) -> None:
        if self.cancelled.is_set():
            raise RuntimeError("ROOT cancellation received before cell admission")
        with hold_gpu_slot(self.control, str(cell["gpu"])):
            root = self.runtime / "runs" / cell["name"]
            profile = root / "profile"
            if self.args.phase == "profile":
                if (profile / "profile_receipt.json").exists():
                    profile_identity(read_json(profile / "profile_receipt.json"), cell)
                    self.status(
                        cell,
                        "profile_adopted",
                        native_receipt=str(profile / "profile_receipt.json"),
                    )
                    return
                if profile.exists():
                    raise FileExistsError(
                        "previous incomplete profile preserved; ROOT diagnosis required"
                    )
                if cell["mode"] == "event211":
                    receipt = read_json(self.args.event_normalization_receipt)
                    if (
                        receipt.get("status") != "passed"
                        or receipt.get("feature_count") != 211
                        or receipt.get("fit_train") != 10325
                        or receipt.get("fit_valid") != 0
                        or receipt.get("finite") is not True
                    ):
                        raise ValueError(
                            "the single canonical211 normalization job is not qualified"
                        )
                self.execute(
                    cell,
                    "profile",
                    self.train_command(cell, profile) + ["--profile-updates", "30"],
                    profile,
                    cap_seconds=300,
                )
                profile_identity(read_json(profile / "profile_receipt.json"), cell)
                self.status(
                    cell, "profile_complete", native_receipt=str(profile / "profile_receipt.json")
                )
                return
            identity = profile_identity(read_json(profile / "profile_receipt.json"), cell)
            approved = self.control / "approved" / f"{cell['name']}.json"
            receipt = read_json(approved)
            if receipt.get("reviewer") != "ROOT":
                raise ValueError("separate ROOT profile approval is missing")
            validate_profile_receipt(approved, identity)
            fit = root / "fit"
            if self.args.phase == "fit":
                claim = self.control / f"fit-claim-{cell['name']}.json"
                atomic_write_json(
                    claim,
                    {
                        "cell": cell["name"],
                        "config_sha256": cell["config_sha256"],
                        "approved_profile_sha256": file_sha256(approved),
                    },
                    replace=False,
                )
                self.execute(
                    cell,
                    "fit",
                    self.train_command(cell, fit) + ["--profile-receipt", str(approved)],
                    fit,
                )
                full_valid(fit, cell["config_sha256"])
                self.status(
                    cell,
                    "fit_complete",
                    output=str(fit),
                    postfit="fixed SWA3456 + E6 pending readout admission",
                )
            else:
                self.postfit(cell, fit)

    def postfit(self, cell: dict[str, object], fit: Path) -> None:
        run_postfit(self, cell, fit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--phase", choices=("profile", "fit", "readout"), required=True)
    parser.add_argument("--arms", default="raw112_d0,raw112_d02,event211_d02")
    parser.add_argument("--event-normalization-receipt", type=Path)
    args = parser.parse_args()
    wave = Wave(args)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda _signal, _frame: wave.cancelled.set())
    selected = [cell for cell in wave.cells if cell["arm"] in args.arms.split(",")]
    if not selected or set(args.arms.split(",")) - {arm[0] for arm in ARMS}:
        parser.error("select only declared nonempty arms")
    failures = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(wave.run_cell, cell): cell for cell in selected}
        for future in as_completed(futures):
            cell = futures[future]
            try:
                future.result()
            except Exception as error:
                wave.status(cell, "failed", error=f"{type(error).__name__}: {error}")
                failures.append(cell["name"])
    if failures:
        raise RuntimeError(f"failed cells preserved without retry: {failures}")


if __name__ == "__main__":
    main()
