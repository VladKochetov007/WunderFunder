"""Run owned compact-GRU profiles or admitted fits with the chronological contract."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/data/lc-epanov_s/tabdata-final-quality-20261112-r1")
SOURCE = Path("/data2/lc-epanov_s/tabdata-all-followups-20261112-r1/source-r2")
DATA = Path("/data/lc-epanov_s/tabdata-stem-successors-20261111-r1/project")
CONTROL_SOURCE = Path(
    "/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1/src"
)
OBSERVER = "/data2/lc-epanov_s/tabdata/runtime/generalization-fix-20261108-r1/control/l4-wave-r1/observed_worker.py"
PYTHON = "/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python"
PACKET = ROOT / "age-intake-r1/since-scale-followup-r1"
TEACHER = Path("/data2/lc-epanov_s/tabdata-final-quality-20261112-r1/teacher-cache")
SOURCE_SHA = "29fb45164667e69a71d4be2c696731d9824884d2d51674f37d2a633bd17d9511"
SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"


def digest(path: Path) -> str:
    """Hash bounded source, config, or profile metadata."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Select frozen cells and run them with one GPU each and no retries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("profile-age", "fit-age"))
    args = parser.parse_args()
    sys.path.insert(0, str(CONTROL_SOURCE))
    from wnn.multistage_search import atomic_write_json
    from wnn.multistage_search_owned_command import run_owned_command
    from wnn.multistage_search_runtime import (
        current_process_owner,
        is_same_process,
        terminate_owned_process,
    )
    from wnn.training.sequence_observer_resources import (
        process_table,
        tracked_processes,
    )

    clock = json.loads((ROOT / "control/CLOCK_R1.json").read_bytes())
    if time.time() >= clock["fit_deadline_epoch"]:
        raise TimeoutError("New fitting window has closed")
    if digest(DATA / "chronological_primary.json") != SPLIT_SHA:
        raise ValueError("Original chronological split identity differs")
    manifest = PACKET / "PANEL_MANIFEST_R1.json"
    if digest(manifest) != "37f959c0a3d8bcdfc8b7db31fc47e19135ff11ac5d4d13e5c5079078b4a4ed5e":
        raise ValueError("Frozen panel manifest changed")
    verify_source = runpy.run_path(str(SOURCE / "src/wnn/training/sequence_source_capsule.py"))[
        "verify_source_capsule"
    ]
    verify_source(
        SOURCE,
        SOURCE / "SOURCE_MANIFEST.json",
        "00782a64e1557a8af759e6f987038cabb6ec09f098c43ab61d6cfa6bb07e9e5f",
        "390c846cbdacd8e0e06e6079dd3b04312fa6bcc5",
    )
    panel = json.loads(manifest.read_bytes())
    kd = args.phase.endswith("-kd")
    cells = [entry for entry in panel["cells"] if bool(entry["teacher_cache_required"]) == kd]
    cells.sort(
        key=lambda entry: (entry["role"] == "targeted_controls", entry["seed"], entry["cell"])
    )
    if len(cells) != 4:
        raise ValueError("The frozen phase must contain exactly four cells")
    if kd and not (TEACHER / "manifest.json").is_file():
        raise FileNotFoundError("KD requires the independently admitted fitting-only cache")
    control = ROOT / "control" / args.phase
    control.mkdir(parents=True, exist_ok=False)
    (control / "owned").mkdir()
    (ROOT / "scratch").mkdir(exist_ok=True)
    owner = current_process_owner()
    atomic_write_json(control / "owner.json", owner, replace=False)
    profile = args.phase.startswith("profile-")

    def run_cell(index: int, entry: dict[str, object]) -> dict[str, object]:
        cell = str(entry["cell"])
        gpu = index + 4
        config = PACKET / str(entry["config"])
        if digest(config) != entry["config_sha256"]:
            raise ValueError("Frozen config bytes changed")
        raw = json.loads(config.read_bytes())
        if raw["seed"] != entry["seed"] or raw["kd_weight"] != entry["kd_weight"]:
            raise ValueError("Frozen cell/config mismatch")
        mode = "profile" if profile else "fit"
        output = ROOT / "runs" / cell / mode
        if output.exists():
            raise FileExistsError(f"Preserve previous {cell} attempt")
        env = dict(
            os.environ,
            PYTHONPATH=str(SOURCE / "src"),
            CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS="8",
            MKL_NUM_THREADS="8",
            OPENBLAS_NUM_THREADS="8",
            NVIDIA_TF32_OVERRIDE="0",
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
            PYTHONDONTWRITEBYTECODE="1",
            WNN_MONITOR_HEARTBEAT_SECONDS="10",
            TMPDIR=str(ROOT / "scratch"),
        )
        for key in (
            "CUBLAS_WORKSPACE_CONFIG",
            "CUBLAS_EMULATE_SINGLE_PRECISION",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            env.pop(key, None)
        command = [
            PYTHON,
            str(SOURCE / "scripts/train_sequence.py"),
            "--config",
            str(config),
            "--data-root",
            str(DATA),
            "--split-manifest",
            str(DATA / "chronological_primary.json"),
            "--output",
            str(output),
        ]
        if kd:
            command += ["--teacher-cache", str(TEACHER)]
        if profile:
            command += ["--profile-updates", "30"]
        else:
            admission = ROOT / "approved-profiles" / (cell + ".json")
            if (
                not admission.is_file()
                or json.loads(admission.read_bytes()).get("approved") is not True
            ):
                raise ValueError(f"No explicit reviewed profile for {cell}")
            command += ["--profile-receipt", str(admission)]
        remaining = int(clock["fit_deadline_epoch"] - time.time())
        timeout = min(600 if profile else 25_200, remaining)
        if timeout <= 0:
            raise TimeoutError("Fitting admission window has closed")
        bootstrap = (
            "import sys,runpy;sys.path.insert(0,"
            + repr(str(CONTROL_SOURCE))
            + ");runpy.run_path("
            + repr(OBSERVER)
            + ',run_name="__main__")'
        )
        observed = [
            PYTHON,
            "-c",
            bootstrap,
            "--",
            PYTHON,
            str(SOURCE / "scripts/observe_sequence.py"),
            "--output",
            str(output),
            "--interval",
            "10",
            "--max-seconds",
            str(timeout),
            "--",
            *command,
        ]
        receipt_path = control / "owned" / (cell + ".json")
        failures: list[str] = []

        def progress() -> None:
            try:
                if time.time() >= clock["fit_deadline_epoch"]:
                    raise TimeoutError("Fitting deadline reached")
                if shutil.disk_usage(ROOT).free < 300_000_000:
                    raise RuntimeError("Physical output disk reserve below300MB")
                table = process_table()
                identities = {int(owner["pid"]): int(owner["birth"])}
                own = tracked_processes(table, identities)
                rss = sum(row.resident_bytes for row in own.values())
                if rss > 256 * 1024**3:
                    raise RuntimeError("Owned aggregate RAM exceeds256GiB")
            except (RuntimeError, TimeoutError, OSError, ValueError) as error:
                if failures:
                    return
                failures.append(str(error))
                if receipt_path.exists():
                    record = json.loads(receipt_path.read_bytes())
                    child = record.get("child")
                    if (
                        record.get("command") == observed
                        and isinstance(child, dict)
                        and is_same_process(child)
                    ):
                        terminate_owned_process(child, grace_seconds=5.0)

        code, timed_out, _ = run_owned_command(
            observed,
            cwd=SOURCE,
            env=env,
            log_path=control / (cell + ".log"),
            receipt_path=receipt_path,
            owner={"controller": owner, "cell_id": cell},
            stage=1,
            trial_number=gpu,
            mode="engineering_profile" if profile else "scientific_fit",
            gpu_key=str(gpu),
            output_dir=output,
            timeout_seconds=timeout,
            progress=progress,
        )
        result = {
            "cell": cell,
            "gpu": gpu,
            "exit_code": code,
            "timed_out": timed_out,
            "resource_errors": failures,
            "mode": mode,
            "at_epoch": time.time(),
        }
        if code == 0 and not timed_out and not failures and profile:
            receipt = json.loads((output / "profile_receipt.json").read_bytes())
            identity = receipt["identity"]
            if (
                receipt["status"] != "profile_complete"
                or receipt["updates"] != 30
                or receipt["errors"]
                or identity["source_sha256"] != SOURCE_SHA
                or identity["data"]["split_sha256"] != SPLIT_SHA
                or identity["world_size"] != 1
                or receipt["engineering_reload"]["status"] != "passed"
            ):
                raise ValueError(f"Failed exact source/split/reload profile gate for {cell}")
            result["profile_receipt_sha256"] = digest(output / "profile_receipt.json")
        atomic_write_json(control / (cell + "-result.json"), result, replace=False)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_cell, index, entry) for index, entry in enumerate(cells)]
        results = [future.result() for future in futures]
    atomic_write_json(
        control / "complete.json", {"phase": args.phase, "results": results}, replace=False
    )


if __name__ == "__main__":
    main()
