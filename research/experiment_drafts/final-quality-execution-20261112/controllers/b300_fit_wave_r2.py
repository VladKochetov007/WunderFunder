"""Run approved PR91 input-panel fits and fixed-endpoint readouts on B300."""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path("/data/epanov_s/tabdata-final-quality-20261112-r1")
PREVIOUS = Path("/data/epanov_s/tabdata-all-20261112-r1")
WARM = Path("/data/epanov_s/tabdata-qknorm-b300-20261111-r1")
RUNTIME = PREVIOUS / "runtimes/pr75"
PACKET = (
    ROOT / "intake/pr91-source/research/experiment_drafts/bilinear-entmax-b300-20261112"
)
PYTHON = "/data/epanov_s/tabdata/.venv/bin/python"
OWNER_NAME = "SemyonEpanov-root-final-quality-20261112"
RUNS = ROOT / "runs/pr91-input"
CONTROL_ROOT = ROOT / "control/fit-wave-r2"
CLOCK_PATH = ROOT / "control/CLOCK_R1.json"
SPLIT_PATH = WARM / "project/chronological_primary.json"
PREPROCESSING = RUNS / "preprocessing/preprocessing.json"
CHECKS = RUNS / "acceptance/checks.json"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
RUNTIME_COMMIT = "8626000a4b965763a70011a0291a891aa6535d7e"
CELLS = (
    ("be_input_linear_s206201", (0, 1)),
    ("be_input_bilinear_s206201", (2, 3)),
    ("be_input_linear_s206202", (4, 5)),
    ("be_input_bilinear_s206202", (6, 7)),
)
TIMEOUT_LIMITS = {"fit": 28_800, "export": 7_200, "score": 7_200}
EXPECTED_VARIANTS = [
    "attention_entmax",
    "attention_softmax",
    "input_bilinear",
    "input_linear",
]
EXPECTED_SCORE_ROWS = 3_528_273
EXPECTED_SCORE_SEQUENCES = 1_873


def _load_helpers() -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    """Load the previously reviewed ownership runner and packet bindings."""
    sys.path.insert(0, str(WARM / "control/profile-receiver-r1"))
    sys.path.insert(0, str(PACKET / "runtime"))
    import b300_driver_core_r1 as core
    import b300_driver_support_r1 as support
    import b300_owned_runner_r2 as runner
    from bilinear_entmax import bindings

    return core, support, runner, bindings


def _sha256(path: Path) -> str:
    """Hash a regular, non-symlink file in bounded chunks."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _clock(core: ModuleType) -> dict[str, Any]:
    """Read the original fresh-wave clock without resetting either deadline."""
    value = core.read_json(CLOCK_PATH, 16_000)
    started = value.get("started_epoch")
    fit_deadline = value.get("fit_deadline_epoch")
    wave_deadline = value.get("wave_deadline_epoch")
    if (
        value.get("schema") != "wnn.final-quality-wave-clock.r1"
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            for item in (started, fit_deadline, wave_deadline)
        )
        or abs(float(fit_deadline) - float(started) - 7 * 3600) > 1
        or abs(float(wave_deadline) - float(started) - 8 * 3600) > 1
        or float(wave_deadline) <= float(fit_deadline)
    ):
        raise ValueError("CLOCK_R1.json is not the authorized fresh 7h/8h wave clock")
    return value


def _gpu_map(claimed_indices: tuple[int, int]) -> dict[int, str]:
    """Map all eight host GPUs and require only this cell's pair to be free."""
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    devices: dict[int, str] = {}
    for line in query.splitlines():
        index_text, uuid, used_text = [part.strip() for part in line.split(",")]
        index, used = int(index_text), int(used_text)
        if index in devices:
            raise RuntimeError(f"duplicate host GPU index {index}")
        if index in claimed_indices and used > 256:
            raise RuntimeError(f"claimed B300 GPU {index} is occupied")
        devices[index] = uuid
    if sorted(devices) != list(range(8)):
        raise RuntimeError(f"expected host B300 indices 0..7, found {sorted(devices)}")
    return devices


def _validate_prepared(core: ModuleType, bindings: ModuleType) -> None:
    """Require the completed PR91 preparation and checks for this exact source."""
    complete = core.read_json(ROOT / "control/prepare-check/complete.json", 32_000)
    preprocessing = bindings.read_object(PREPROCESSING)
    checks = bindings.read_object(CHECKS)
    if (
        complete.get("phase") != "prepare-check"
        or preprocessing.get("schema") != "bilinear-entmax.preprocessing.v1"
        or preprocessing.get("runtime_commit") != RUNTIME_COMMIT
        or preprocessing.get("fit_identity", {}).get("split_sha256") != SPLIT_SHA256
        or _sha256(SPLIT_PATH) != SPLIT_SHA256
        or checks.get("schema") != "bilinear-entmax.receiver-checks.v1"
        or checks.get("status") != "passed"
        or checks.get("checked_variants") != EXPECTED_VARIANTS
        or checks.get("preprocessing_sha256") != _sha256(PREPROCESSING)
        or checks.get("precision_mode") != "bf16_bf16x9"
    ):
        raise ValueError(
            "PR91 preparation/check receipts do not match the frozen panel"
        )
    for cell_id, _ in CELLS:
        cell, native_path = bindings.load_cell(PACKET / "configs" / f"{cell_id}.json")
        native = bindings.read_object(native_path)
        if (
            cell.get("id") != cell_id
            or native.get("epochs") != 4
            or native.get("required_world_size") != 2
            or native.get("required_device") != "B300"
            or native.get("precision_mode") != "bf16_bf16x9"
        ):
            raise ValueError(f"{cell_id} does not retain the fixed-E4 PR91 recipe")


def _approved_profile(
    bindings: ModuleType, cell_id: str
) -> tuple[Path | None, str | None]:
    """Admit only the immutable, root-reviewed approval at this cell's fixed path."""
    path = RUNS / cell_id / "approved_profile.json"
    if path.is_symlink() or not path.is_file():
        return None, "approved profile is absent"
    receipt = bindings.read_object(path)
    cell = bindings.read_object(PACKET / "configs" / f"{cell_id}.json")
    acceptance = receipt.get("full_shape_acceptance")
    approval = receipt.get("approval")
    identity = receipt.get("identity")
    if (
        receipt.get("approved") is not True
        or receipt.get("status") != "profile_complete"
        or receipt.get("updates") != 50
        or not isinstance(acceptance, dict)
        or acceptance.get("status") != "passed"
        or acceptance.get("human_approved") is not True
        or not isinstance(approval, dict)
        or approval.get("schema") != "bilinear-entmax.profile-approval.v1"
        or not isinstance(approval.get("reviewer"), str)
        or not approval["reviewer"].strip()
        or approval.get("cell") != cell
        or not isinstance(identity, dict)
        or identity.get("bilinear_entmax_cell") != cell
    ):
        return None, "approved profile is malformed or does not bind this cell"
    return path, None


def _throttle(
    callback: Callable[[], None], interval: float = 10.0
) -> Callable[[], None]:
    """Limit state snapshots while retaining the canonical monitor heartbeat."""
    last_call = [0.0]

    def update() -> None:
        now = time.monotonic()
        if now - last_call[0] >= interval:
            last_call[0] = now
            callback()

    return update


def _monitor_callback(
    core: ModuleType,
    state_path: Path,
    state: dict[str, Any],
    cell_id: str,
    stage: str,
    output: Path,
    state_lock: threading.Lock,
) -> Callable[[], None]:
    """Persist the canonical monitor snapshot and controller heartbeat."""

    def update() -> None:
        progress: dict[str, Any] = {}
        monitor_path = output / "monitor_progress.json"
        if monitor_path.is_file() and not monitor_path.is_symlink():
            try:
                progress = core.read_json(monitor_path, 32_000)
            except (OSError, ValueError):
                progress = {"progress_read_error": True}
        with state_lock:
            record = state["cells"][cell_id]
            record["phase"] = stage
            record["last_progress"] = progress
            record["heartbeat_epoch"] = time.time()
            _save_state(core, state_path, state)

    return update


def _run_stage(
    *,
    mode: str,
    cell_id: str,
    cell_path: Path,
    output: Path,
    extra: list[str],
    device_ids: list[int],
    devices: dict[int, str],
    clock_deadline: float,
    core: ModuleType,
    support: ModuleType,
    runner: ModuleType,
    owner: dict[str, int | str],
    guard: Any,
    control: Path,
    state_path: Path,
    state: dict[str, Any],
    state_lock: threading.Lock,
) -> dict[str, object]:
    """Run one canonical stage under the original deadline and owned-child guard."""
    timeout = min(TIMEOUT_LIMITS[mode], int(clock_deadline - time.time()))
    if timeout <= 0:
        raise TimeoutError(f"{mode} deadline has passed")
    uuids = [devices[index] for index in device_ids]
    command = [
        PYTHON,
        str(PACKET / "runtime/launch.py"),
        mode,
        "--runtime-root",
        str(RUNTIME),
        "--cell",
        str(cell_path),
        "--data-root",
        str(WARM / "project"),
        "--split-manifest",
        str(SPLIT_PATH),
        "--owner",
        OWNER_NAME,
        "--timeout-seconds",
        str(timeout),
        "--output",
        str(output),
        *extra,
    ]
    if mode in ("fit", "export"):
        command.extend(("--devices", ",".join(str(index) for index in device_ids)))
    environment = support._environment(RUNTIME, WARM, uuids)
    environment["PATH"] = (
        str(PREVIOUS / "tools/git-r2/usr/bin") + ":" + environment.get("PATH", "")
    )
    environment["WNN_MONITOR_HEARTBEAT_SECONDS"] = "10"
    progress = _throttle(
        _monitor_callback(core, state_path, state, cell_id, mode, output, state_lock)
    )
    result = runner.run_owned(
        command,
        root=ROOT,
        control=control,
        cell=cell_id,
        stage=mode,
        gpu_uuids=uuids,
        owner=owner,
        environment=environment,
        deadline=clock_deadline,
        stage_timeout=timeout,
        guard=guard,
        progress=progress,
    )
    if result.get("status") != "completed":
        raise runner.OwnedRunError(f"{cell_id} {mode} did not complete")
    return result


def _score_receipt(bindings: ModuleType, path: Path) -> dict[str, Any]:
    """Require the exact official metric and complete exposed-valid coverage."""
    value = bindings.read_object(path)
    score = value.get("score")
    if not isinstance(score, dict):
        raise TypeError("score output has no canonical metric receipt")
    if (
        value.get("split_sha256") != SPLIT_SHA256
        or value.get("sequences") != EXPECTED_SCORE_SEQUENCES
        or score.get("metric_id") != "global_wp_clip2_v1"
        or score.get("selected_rows") != EXPECTED_SCORE_ROWS
    ):
        raise ValueError("score receipt does not cover the pinned official valid split")
    for key in ("wp", "t0", "t1"):
        metric = score.get(key)
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(metric)
        ):
            raise ValueError(f"score field {key} is not finite")
    return {
        "metric_id": score["metric_id"],
        "wp": score["wp"],
        "t0": score["t0"],
        "t1": score["t1"],
        "selected_rows": score["selected_rows"],
        "sequences": value["sequences"],
        "split_sha256": value["split_sha256"],
        "score_sha256": _sha256(path),
    }


def _save_state(core: ModuleType, state_path: Path, state: dict[str, Any]) -> None:
    """Atomically persist controller and per-cell outcomes."""
    state["updated_epoch"] = time.time()
    core.atomic_json(state_path, state, replace=True)


def main() -> None:
    """Dispatch one root-approved input cell once, without retries or auto-approval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell", required=True, choices=[cell_id for cell_id, _ in CELLS]
    )
    args = parser.parse_args()
    core, support, runner, bindings = _load_helpers()
    if sys.platform != "linux":
        raise RuntimeError("the B300 fit wave runs only on the receiving Linux host")
    selected = next(item for item in CELLS if item[0] == args.cell)
    cell_id, device_ids = selected
    control = CONTROL_ROOT / cell_id
    control.mkdir(parents=True, exist_ok=False)
    owner = support._identity(ROOT)
    state_path = control / "state.json"
    state: dict[str, Any] = {
        "schema": "wnn.final-quality-fit-wave.r2",
        "status": "running",
        "owner": owner,
        "clock_sha256": _sha256(CLOCK_PATH),
        "runtime_commit": RUNTIME_COMMIT,
        "automatic_retries": 0,
        "automatic_profile_approval": False,
        "cells": {},
    }
    state_lock = threading.Lock()
    _save_state(core, state_path, state)
    try:
        clock = _clock(core)
        _validate_prepared(core, bindings)
        devices = _gpu_map(device_ids)
        guard = runner.ResourceGuard(
            ROOT,
            control,
            {
                "owned_rss_bytes": 128 * 1024**3,
                "output_bytes": 128 * 1024**3,
                "minimum_free_bytes": 128 * 1024**3,
            },
        )
        core.atomic_json(
            control / "admission.json",
            {
                "owner": owner,
                "clock": clock,
                "gpu_indices_to_uuids": devices,
                "cell": cell_id,
                "gpu_pair": list(device_ids),
                "fixed_endpoint": "configured_final_epoch_E4",
                "resource_caps": guard.caps,
                "training_changes": False,
                "approval_source": "per-cell approved_profile.json reviewed by ROOT",
            },
        )
        _save_state(core, state_path, state)
        for cell_id, device_ids in (selected,):
            record: dict[str, Any] = {
                "status": "queued",
                "gpu_indices": list(device_ids),
                "gpu_uuids": [devices[index] for index in device_ids],
                "stages": {},
            }
            state["cells"][cell_id] = record
            _save_state(core, state_path, state)
            if time.time() >= float(clock["fit_deadline_epoch"]):
                record["status"] = "not_started_fit_deadline"
                _save_state(core, state_path, state)
                continue
            if guard.failure:
                record["status"] = "not_started_resource_guard"
                record["error"] = guard.failure
                _save_state(core, state_path, state)
                continue

            profile_path, profile_error = _approved_profile(bindings, cell_id)
            if profile_path is None:
                record["status"] = "not_started_profile_unapproved"
                record["error"] = profile_error
                _save_state(core, state_path, state)
                continue
            record["approved_profile"] = {
                "path": str(profile_path),
                "sha256": _sha256(profile_path),
            }
            cell_path = PACKET / "configs" / f"{cell_id}.json"
            fit_output = RUNS / cell_id / "fit"
            predictions = RUNS / cell_id / "predictions"
            score_path = RUNS / cell_id / "score.json"
            if any(
                path.exists() or path.is_symlink()
                for path in (fit_output, predictions, score_path)
            ):
                record["status"] = "not_started_existing_output_preserved"
                _save_state(core, state_path, state)
                continue

            try:
                record["status"] = "fitting"
                _save_state(core, state_path, state)
                fit_result = _run_stage(
                    mode="fit",
                    cell_id=cell_id,
                    cell_path=cell_path,
                    output=fit_output,
                    extra=[
                        "--preprocessing",
                        str(PREPROCESSING),
                        "--checks-receipt",
                        str(CHECKS),
                        "--profile-receipt",
                        str(profile_path),
                    ],
                    device_ids=list(device_ids),
                    devices=devices,
                    clock_deadline=float(clock["fit_deadline_epoch"]),
                    core=core,
                    support=support,
                    runner=runner,
                    owner=owner,
                    guard=guard,
                    control=control,
                    state_path=state_path,
                    state=state,
                    state_lock=state_lock,
                )
                record["stages"]["fit"] = fit_result
            except (runner.OwnedRunError, OSError, ValueError) as error:
                record["status"] = "fit_failed_or_partial"
                record["error"] = f"{type(error).__name__}: {error}"
                _save_state(core, state_path, state)
                continue

            checkpoint = fit_output / "model.pt"
            if (
                checkpoint.is_symlink()
                or not checkpoint.is_file()
                or checkpoint.stat().st_size == 0
            ):
                record["status"] = "fit_exit0_missing_endpoint_checkpoint"
                _save_state(core, state_path, state)
                continue
            record["checkpoint"] = {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "bytes": checkpoint.stat().st_size,
            }
            if time.time() >= float(clock["wave_deadline_epoch"]):
                record["status"] = "fit_complete_export_not_started_wave_deadline"
                _save_state(core, state_path, state)
                continue

            try:
                record["status"] = "exporting"
                _save_state(core, state_path, state)
                export_result = _run_stage(
                    mode="export",
                    cell_id=cell_id,
                    cell_path=cell_path,
                    output=predictions,
                    extra=["--checkpoint", str(checkpoint)],
                    device_ids=list(device_ids),
                    devices=devices,
                    clock_deadline=float(clock["wave_deadline_epoch"]),
                    core=core,
                    support=support,
                    runner=runner,
                    owner=owner,
                    guard=guard,
                    control=control,
                    state_path=state_path,
                    state=state,
                    state_lock=state_lock,
                )
                record["stages"]["export"] = export_result
                manifest = bindings.read_object(predictions / "manifest.json")
                if (
                    manifest.get("complete") is not True
                    or manifest.get("split_sha256") != SPLIT_SHA256
                ):
                    raise ValueError(
                        "export manifest is incomplete or uses another split"
                    )
            except (runner.OwnedRunError, OSError, ValueError) as error:
                record["status"] = "fit_complete_export_failed_or_partial"
                record["error"] = f"{type(error).__name__}: {error}"
                _save_state(core, state_path, state)
                continue
            if time.time() >= float(clock["wave_deadline_epoch"]):
                record["status"] = "export_complete_score_not_started_wave_deadline"
                _save_state(core, state_path, state)
                continue

            try:
                record["status"] = "scoring"
                _save_state(core, state_path, state)
                score_result = _run_stage(
                    mode="score",
                    cell_id=cell_id,
                    cell_path=cell_path,
                    output=score_path,
                    extra=["--predictions", str(predictions)],
                    device_ids=[],
                    devices=devices,
                    clock_deadline=float(clock["wave_deadline_epoch"]),
                    core=core,
                    support=support,
                    runner=runner,
                    owner=owner,
                    guard=guard,
                    control=control,
                    state_path=state_path,
                    state=state,
                    state_lock=state_lock,
                )
                record["stages"]["score"] = score_result
                record["score"] = _score_receipt(bindings, score_path)
                record["status"] = "scored"
            except (runner.OwnedRunError, OSError, ValueError) as error:
                record["status"] = "fit_export_complete_score_failed_or_partial"
                record["error"] = f"{type(error).__name__}: {error}"
            _save_state(core, state_path, state)

        if guard.failure:
            state["status"] = "stopped_resource_guard"
            state["error"] = guard.failure
        elif time.time() >= float(clock["wave_deadline_epoch"]):
            state["status"] = "wave_deadline_reached"
        else:
            state["status"] = "completed_or_partial"
    except Exception as error:
        state["status"] = "controller_failed"
        state["error"] = f"{type(error).__name__}: {error}"
        _save_state(core, state_path, state)
        raise
    state["finished_epoch"] = time.time()
    _save_state(core, state_path, state)
    core.atomic_json(
        control / "complete.json",
        {
            "schema": "wnn.final-quality-fit-wave.r2",
            "status": state["status"],
            "state": str(state_path),
        },
    )


if __name__ == "__main__":
    main()
