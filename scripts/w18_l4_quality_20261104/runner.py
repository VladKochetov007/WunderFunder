"""Finite W18 L4 profile, fit, and full-valid export entrypoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import signal
import sys
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]

from scripts.w18_l4_quality_20261104.contract import (  # noqa: E402
    CohortPanel,
    ContractError,
    _object,
    _sha256,
    load_panel,
    read_object,
    sha256_file,
)
from scripts.w18_l4_quality_20261104.runtime import (  # noqa: E402
    cell_paths as _cell_paths,
    check_budget as _check_budget,
    command_prefix as _command_prefix,
    environment as _environment,
    gpu_snapshot as _gpu_snapshot,
    load_or_start_state as _load_or_start_state,
    owned_rss_bytes as _owned_rss_bytes,
    remaining_seconds as _remaining_seconds,
    state_path as _state_path,
)
from wnn.multistage_search import atomic_write_json  # noqa: E402
from wnn.multistage_search_owned_command import run_owned_command  # noqa: E402
from wnn.multistage_search_runtime import (  # noqa: E402
    current_process_owner,
    hold_gpu_slot,
    terminate_owned_process,
)

STOP = threading.Event()
PROFILE_SUFFIX = "_profile_r1"


def _stop_owned_child(receipt_path: Path) -> None:
    """Use the helper's recorded PID/start-ticks/PGID with a short owned stop grace."""
    if receipt_path.is_symlink() or not receipt_path.is_file():
        return
    child = read_object(receipt_path).get("child")
    if isinstance(child, dict):
        terminate_owned_process(child, grace_seconds=10.0)


def _run_cells(panel: CohortPanel, state: dict[str, object], phase: str) -> list[dict[str, object]]:
    paths = _object(panel.value["paths"], "paths")
    python = str(paths["python"])
    source = Path(str(paths["source_root"]))
    cells = cast(list[dict[str, object]], panel.value["cells"])
    if phase == "profile":
        if int(state["profile_attempts"]) > int(_object(panel.value["budget"], "budget")["max_profiles"]):
            raise RuntimeError("frozen engineering profile cap would be exceeded")
    elif phase == "fit":
        if int(state["scientific_fit_attempts"]) != len(cells):
            raise RuntimeError("scientific fit attempt count differs from the frozen eight-cell cohort")
        review_path = Path(str(panel.value["admission_receipt"]))
        if not review_path.is_file() or review_path.is_symlink():
            raise RuntimeError("source/data/parent admission receipt is missing")
    results: list[dict[str, object]] = []

    def run_cell(index: int, cell: dict[str, object]) -> dict[str, object]:
        if STOP.is_set():
            raise RuntimeError("a cohort cell failed; refusing another launch")
        cell_id, run_name = str(cell["cell_id"]), str(cell["run_name"])
        gpu_uuid = str(cell["gpu_uuid"])
        cell_paths = _cell_paths(panel, cell)
        env = _environment(panel, cell)
        args = list(cast(list[str], cell["train_args"]))
        request: dict[str, object] | None = None
        if phase == "profile":
            if any(cell_paths[key].exists() for key in ("profile_log", "profile_receipt", "profile_request")) or cell_paths["profile_run_dir"].exists():
                raise FileExistsError(f"{cell_id}: profile output exists; no automatic retry")
            if "--save-training-state" in args:
                args.remove("--save-training-state")
            name_pos = args.index("--name") + 1
            args[name_pos] = str(args[name_pos]) + PROFILE_SUFFIX
            policy = _object(panel.value["profile_policy"], "profile_policy")
            args.extend(
                ["--profile-warmup", str(policy["warmup_batches"]), "--profile-steps", str(policy["timed_batches"])]
            )
            output_dir = cell_paths["profile_run_dir"]
            command = [python, str(source / "scripts" / "train_rnn.py"), *args]
            log_path, receipt_path = cell_paths["profile_log"], cell_paths["profile_receipt"]
            timeout = min(int(policy["timeout_seconds"]), _remaining_seconds(state))
            request_path = cell_paths["profile_request"]
            request = {
                "schema": "wnn.w18-l4-profile-request.v1",
                "cell_id": cell_id,
                "configuration_id": cell["configuration_id"],
                "recipe_sha256": hashlib.sha256(json.dumps(cell["train_args"], separators=(",", ":")).encode()).hexdigest(),
                "argv": command,
                "argv_sha256": hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest(),
                "source_binding_sha256": panel.value["identity"]["source_binding_sha256"],
                "data_binding_sha256": panel.value["identity"]["data_binding_sha256"],
                "split_sha256": panel.value["identity"]["split_sha256"],
                "fit_members_sha256": panel.value["identity"]["fit_members_sha256"],
                "baseline_ancestry_sha256": panel.value["identity"]["baseline_ancestry_sha256"],
                "seed": cell["seed"],
                "gpu_uuid": gpu_uuid,
                "cpu_threads": cell["cpu_threads"],
                "timed_batches": policy["timed_batches"],
                "warmup_batches": policy["warmup_batches"],
                "no_full_valid_evaluation": True,
            }
        elif phase == "fit":
            if cell_paths["profile_receipt"].is_file():
                profile_receipt = read_object(cell_paths["profile_receipt"])
                if profile_receipt.get("status") != "completed" or profile_receipt.get("exit_code") != 0:
                    raise RuntimeError(f"{cell_id}: profile did not complete successfully")
            else:
                raise RuntimeError(f"{cell_id}: profile receipt is missing")
            if cell_paths["fit_receipt"].exists() or cell_paths["run_dir"].exists():
                raise FileExistsError(f"{cell_id}: fit output exists; no automatic retry")
            output_dir = cell_paths["run_dir"]
            command = [python, str(source / "scripts" / "train_rnn.py"), *args]
            log_path, receipt_path = cell_paths["fit_log"], cell_paths["fit_receipt"]
            timeout = min(int(cell["fit_timeout_seconds"]), _remaining_seconds(state))
        else:
            fit_receipt = read_object(cell_paths["fit_receipt"])
            if fit_receipt.get("status") != "completed" or fit_receipt.get("exit_code") != 0:
                raise RuntimeError(f"{cell_id}: successful fit receipt is required before export")
            checkpoint = cell_paths["run_dir"] / "model_ep1.pt"
            if not checkpoint.is_file():
                raise RuntimeError(f"{cell_id}: fixed endpoint model_ep1.pt is missing")
            partial = cell_paths["prediction"].with_name(cell_paths["prediction"].name + ".partial")
            sidecar = cell_paths["prediction"].with_suffix(".json")
            stale_metrics = cell_paths["prediction"].parent / f"seqsums_{run_name}.npz"
            if any(path.exists() for path in (cell_paths["prediction"], partial, sidecar, stale_metrics)):
                raise FileExistsError(f"{cell_id}: prediction exists; no automatic export retry")
            split = str(paths["split_manifest"])
            command = [
                python,
                str(source / "scripts" / "predict_run.py"),
                run_name,
                "valid",
                "--epoch",
                "1",
                "--campaign-split-manifest",
                split,
                "--campaign-full-valid-feature-only",
                "--device",
                "cuda",
                "--inference-precision",
                "fp32",
            ]
            output_dir = cell_paths["prediction"].parent
            log_path, receipt_path = cell_paths["export_log"], cell_paths["export_receipt"]
            timeout = min(3600, _remaining_seconds(state))
        command = _command_prefix(cell, command)
        if STOP.is_set():
            raise RuntimeError("a cohort cell failed; refusing another launch")
        if request is not None:
            request["argv"] = command
            request["argv_sha256"] = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
            request["environment"] = {
                key: env[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "WNN_PROJECT_ROOT",
                    "WNN_THREADS",
                    "WNN_LOADER_WORKERS",
                    "WNN_LOADER_RAM_GB",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                )
            }
            request["created_utc"] = datetime.now(UTC).isoformat()
            request_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(request_path, request, replace=False)
        try:
            _check_budget(panel)
            _gpu_snapshot(gpu_uuid)
        except (OSError, RuntimeError, subprocess.SubprocessError):
            STOP.set()
            raise
        owner = current_process_owner()
        owner["cohort_panel_sha256"] = panel.sha256
        owner["cell_id"] = cell_id
        owner["run_name"] = run_name
        owner["source_binding_sha256"] = str(panel.value["identity"]["source_binding_sha256"])
        owner["data_binding_sha256"] = str(panel.value["identity"]["data_binding_sha256"])
        owner["split_sha256"] = str(panel.value["identity"]["split_sha256"])
        owner["recipe_sha256"] = hashlib.sha256(json.dumps(cell["train_args"], separators=(",", ":")).encode()).hexdigest()
        if phase == "fit":
            owner["profile_review_sha256"] = state["profile_review_sha256"]
        last_sample = 0.0

        def progress() -> None:
            nonlocal last_sample
            if STOP.is_set():
                _stop_owned_child(receipt_path)
                raise RuntimeError("cohort stop requested; owned child will be stopped")
            try:
                _remaining_seconds(state)
                now = time.monotonic()
                if now - last_sample < 30:
                    return
                last_sample = now
                owned_rss = _owned_rss_bytes(panel, phase)
                if owned_rss > int(_object(panel.value["budget"], "budget")["aggregate_memory_bytes"]):
                    raise RuntimeError("cohort-owned process RSS crossed the aggregate memory cap")
                sample = {
                    "phase": phase,
                    "cell_id": cell_id,
                    **_check_budget(panel),
                    "cohort_owned_rss_bytes": owned_rss,
                    "gpu": _gpu_snapshot(gpu_uuid),
                }
                resource_path = Path(str(paths["runtime_root"])) / "receipts" / "resources.jsonl"
                resource_path.parent.mkdir(parents=True, exist_ok=True)
                with resource_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")
            except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError):
                STOP.set()
                _stop_owned_child(receipt_path)
                raise

        with hold_gpu_slot(Path(str(paths["control_root"])), gpu_uuid):
            code, timed_out, _ = run_owned_command(
                command,
                cwd=source,
                env=env,
                log_path=log_path,
                receipt_path=receipt_path,
                owner=owner,
                stage={"profile": 0, "fit": 1, "export": 2}[phase],
                trial_number=index,
                mode=f"w18_l4_{phase}",
                gpu_key=gpu_uuid,
                output_dir=output_dir,
                timeout_seconds=timeout,
                progress=progress,
            )
        result = {"cell_id": cell_id, "exit_code": code, "timed_out": timed_out, "receipt": str(receipt_path)}
        if code != 0 or timed_out:
            raise RuntimeError(f"{cell_id}: {phase} failed; preserve its receipt and do not retry")
        return result

    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cells)) as pool:
        futures = [pool.submit(run_cell, index, cell) for index, cell in enumerate(cells, start=1)]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                STOP.set()
    if errors:
        raise RuntimeError("; ".join(errors))
    return sorted(results, key=lambda item: str(item["cell_id"]))


def run_phase(panel: CohortPanel, phase: str, review_path: Path | None, review_sha: str | None) -> None:
    state = _load_or_start_state(panel, phase)
    phase_started = time.time()
    _remaining_seconds(state)
    admission_path = Path(str(panel.value["admission_receipt"]))
    if not admission_path.is_file() or admission_path.is_symlink():
        raise RuntimeError("frozen source/data/parent admission receipt is unavailable")
    if phase == "fit":
        if review_path is None or review_sha is None:
            raise RuntimeError("fit requires the independent profile-review receipt and exact SHA256")
        if review_path.is_symlink() or not review_path.is_file() or sha256_file(review_path) != review_sha:
            raise RuntimeError("profile-review receipt is missing or changed")
        for cell in cast(list[dict[str, object]], panel.value["cells"]):
            receipt = read_object(_cell_paths(panel, cell)["profile_receipt"])
            if receipt.get("status") != "completed" or receipt.get("exit_code") != 0:
                raise RuntimeError(f"{cell['cell_id']}: all engineering profiles must exit zero")
        state["profile_review_receipt"] = str(review_path.resolve())
        state["profile_review_sha256"] = review_sha
        state["scientific_fit_attempts"] = 8
    if phase == "profile":
        state["profile_attempts"] = int(state["profile_attempts"]) + len(cast(list[object], panel.value["cells"]))
    state["phase"] = phase + "_running"
    atomic_write_json(_state_path(panel), state)
    results = _run_cells(panel, state, phase)
    receipt_path = Path(str(_object(panel.value["paths"], "paths")["runtime_root"])) / "receipts" / f"{phase.upper()}_R1.json"
    atomic_write_json(
        receipt_path,
        {
            "schema": f"wnn.w18-l4-{phase}.v1",
            "status": "complete",
            "panel_sha256": panel.sha256,
            "admission_receipt_sha256": sha256_file(admission_path),
            "profile_review_sha256": review_sha or state.get("profile_review_sha256"),
            "started_at_epoch": phase_started,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "results": results,
            "resources": _check_budget(panel),
        },
        replace=False,
    )
    state["phase"] = phase + "_complete"
    atomic_write_json(_state_path(panel), state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("profile", "fit", "export"))
    parser.add_argument("--profile-review-receipt", type=Path)
    parser.add_argument("--profile-review-sha256")
    args = parser.parse_args()
    panel = load_panel(args.panel)
    review_sha = _sha256(args.profile_review_sha256, "profile review SHA-256") if args.profile_review_sha256 else None
    signal.signal(signal.SIGTERM, lambda signum, frame: STOP.set())
    run_phase(panel, args.phase, args.profile_review_receipt, review_sha)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, FileNotFoundError, RuntimeError, TimeoutError, ValueError) as error:
        sys.stderr.write(f"W18 L4 cohort stopped: {type(error).__name__}: {error}\n")
        raise SystemExit(2) from error
