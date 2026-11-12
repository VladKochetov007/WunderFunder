"""Finite, one-pass dispatcher for the frozen 40-cell B300 Impact81 wave.

This module is deliberately standard-library-only until ``main`` imports the
reviewed Linux ownership runner. Every cell is dispatched at most once. The
controller preserves per-stage receipts and continues other lanes after a cell
failure; it never retries or resumes a fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Any is required at JSON boundaries: receipts are heterogeneous, untrusted mappings.
from typing import Any

CLOCK_SHA256 = "fc1d4b8b3d554a0d9b131152288f7af2d3c355d22c29b0279592d3e62f716398"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
TRANSFORMER_COMMIT = "8626000a4b965763a70011a0291a891aa6535d7e"
GRU_COMMIT = "67c3bd818ea243177eeb1dc789cd3d145eba0e81"
EXPECTED_CELLS = 40
PROFILE_UPDATES = 50
MAX_VRAM_GIB_PER_RANK = 256.0
MAX_HOST_RAM_GIB = 128.0
RESOURCE_CAPS = {
    "owned_rss_bytes": 512 * 1024**3,
    "output_bytes": 512 * 1024**3,
    "minimum_free_bytes": 64 * 1024**3,
}


class DispatchError(RuntimeError):
    """Frozen roster, authority, or receiving evidence failed a hard boundary."""


def sha256(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Read bounded JSON object metadata without following symlinks."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32_000_000:
        raise DispatchError(f"missing, unsafe, or oversized JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DispatchError(f"expected JSON object: {path}")
    return value


def validate_order(order: list[str], canonical: list[str]) -> None:
    """Require an exact permutation of the frozen 40-cell roster."""
    if len(canonical) != EXPECTED_CELLS or len(set(canonical)) != EXPECTED_CELLS:
        raise DispatchError("canonical wave must contain 40 unique cells")
    if len(order) != EXPECTED_CELLS or len(set(order)) != EXPECTED_CELLS:
        raise DispatchError("dispatch order must contain 40 unique cells")
    missing = sorted(set(canonical) - set(order))
    extra = sorted(set(order) - set(canonical))
    if missing or extra:
        raise DispatchError(
            f"dispatch order mismatch; missing={missing}, extra={extra}"
        )


def lane_orders(order: list[str], lanes: int) -> list[list[str]]:
    """Assign stable round-robin queue positions to independent GPU workers."""
    if lanes <= 0:
        raise ValueError("lanes must be positive")
    return [order[index::lanes] for index in range(lanes)]


def remaining_timeout(deadline: float, now: float, maximum: int) -> int:
    """Return a positive bounded launcher timeout, or stop before a late launch."""
    seconds = min(maximum, int(deadline - now))
    if seconds < 1:
        raise DispatchError("shared campaign cutoff reached; no late stage launch")
    return seconds


def observed_fit_command(
    python: str, runtime: Path, output: Path, timeout: int, command: list[str]
) -> list[str]:
    """Wrap one fit in the pinned read-only GPU/CPU observer."""
    if timeout < 1:
        raise DispatchError("fit observer requires a positive remaining timeout")
    return [
        python,
        str(runtime / "scripts/observe_sequence.py"),
        "--output",
        str(output),
        "--interval",
        "10",
        "--max-seconds",
        str(timeout),
        "--",
        *command,
    ]


def family_for(card: str) -> str:
    """Map frozen card IDs to the packet runtime family."""
    if card in {f"{index:02}" for index in range(1, 8)}:
        return "transformer"
    if card in {"08", "09", "10"}:
        return "gru"
    raise DispatchError(f"unsupported card {card!r}")


def filter_family_order(
    order: list[str], roster: list[dict[str, Any]], families: set[str]
) -> list[str]:
    """Select an active family queue without dropping cells from the inventory."""
    if not families or not families.issubset({"transformer", "gru"}):
        raise DispatchError("selected families must be transformer, gru, or both")
    by_id = {entry["cell_id"]: entry for entry in roster}
    return [
        cell_id for cell_id in order if family_for(by_id[cell_id]["card"]) in families
    ]


def check_family_receipt(
    path: Path,
    preprocessing: Path,
    family: str,
    expected_runtime: str,
) -> dict[str, Any]:
    """Gate profiles on actual R4 check output and its exact preprocessing bytes."""
    check = read_object(path)
    if (
        check.get("schema") != "impact81.receiver-checks.v1"
        or check.get("status") != "passed"
    ):
        raise DispatchError(f"{family} receiver check is not passed: {path}")
    if check.get("runtime_commit") != expected_runtime:
        raise DispatchError(f"{family} receiver check runtime pin mismatch")
    if check.get("preprocessing_sha256") != sha256(preprocessing):
        raise DispatchError(
            f"{family} receiver check does not bind current preprocessing"
        )
    if not isinstance(check.get("source_identity"), dict):
        raise DispatchError(f"{family} receiver check lacks canonical source identity")
    required_extensions = (
        {"04", "05", "06", "07"} if family == "transformer" else {"08", "09", "10"}
    )
    if not required_extensions.issubset(set(check.get("extension_cards", []))):
        raise DispatchError(
            f"{family} receiver check lacks an implemented card-family gate"
        )
    if family == "gru" and not isinstance(check.get("card10_feasible"), bool):
        raise DispatchError(
            "GRU receiver check lacks the frozen card10 feasibility result"
        )
    return check


def card10_allowed(check: dict[str, Any]) -> bool:
    """Use only the receiver's recorded train-only fixed-tail gate."""
    value = check.get("card10_feasible")
    if not isinstance(value, bool):
        raise DispatchError("card10 feasibility is absent from the passed GRU receipt")
    return value


def _atomic_json(path: Path, value: object, *, replace: bool = True) -> None:
    """Atomically update dispatcher state without replacing stage evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    if replace:
        temporary.replace(path)
    else:
        temporary.rename(path)


class State:
    """Thread-safe manifest for all 40 planned cells and their stage evidence."""

    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path
        self.payload = payload
        self.lock = threading.RLock()
        _atomic_json(path, payload)

    def update(self, cell_id: str, **values: Any) -> None:
        with self.lock:
            cell = self.payload["cells"][cell_id]
            cell.update(values)
            self.payload["updated_epoch"] = time.time()
            _atomic_json(self.path, self.payload)

    def event(self, cell_id: str, event: dict[str, Any]) -> None:
        with self.lock:
            self.payload["cells"][cell_id].setdefault("events", []).append(event)
            self.payload["updated_epoch"] = time.time()
            _atomic_json(self.path, self.payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        required=True,
        help="nested R2 packet root ending in b300-all-20261112",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="persistent native campaign root, containing runs/control",
    )
    parser.add_argument("--runtime-transformer", type=Path, required=True)
    parser.add_argument("--runtime-gru", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--clock", type=Path, required=True)
    parser.add_argument(
        "--control",
        type=Path,
        required=True,
        help="new one-pass control directory; must not already exist",
    )
    parser.add_argument(
        "--transformer-family-dir",
        type=Path,
        required=True,
        help="R4 directory containing preprocessing/ and acceptance/",
    )
    parser.add_argument("--gru-family-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu-uuids",
        required=True,
        help="eight physical B300 UUIDs in the same order as --gpu-indices",
    )
    parser.add_argument("--gpu-indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--path-prefix",
        default="",
        help="optional PATH prefix, including the provisioned portable git bin",
    )
    parser.add_argument(
        "--order-file",
        type=Path,
        default=Path(__file__).with_name("dispatch_order.json"),
    )
    parser.add_argument(
        "--order",
        default="wave_stages",
        choices=("wave_stages", "breadth_first_one_seed"),
    )
    parser.add_argument(
        "--families",
        default="transformer,gru",
        help="comma-separated subset; use gru to queue its 12 cells and defer Transformer cells",
    )
    parser.add_argument("--reviewer", default="SemyonEpanov-root-20261112")
    parser.add_argument("--heartbeat-seconds", type=int, default=10)
    return parser.parse_args()


def _load_order(path: Path, name: str, roster: list[dict[str, Any]]) -> list[str]:
    value = read_object(path)
    orders = value.get("orders")
    if not isinstance(orders, dict) or not isinstance(orders.get(name), list):
        raise DispatchError(f"dispatch order {name!r} is missing from {path}")
    canonical = [entry["cell_id"] for entry in roster]
    order = orders[name]
    if any(not isinstance(item, str) for item in order):
        raise DispatchError("dispatch order contains a non-string cell ID")
    validate_order(order, canonical)
    return order


def _load_wave(packet: Path) -> dict[str, Any]:
    wave = read_object(packet / "WAVE_20261112.json")
    roster = wave.get("fit_accounting", {}).get("complete_roster")
    if not isinstance(roster, list) or len(roster) != EXPECTED_CELLS:
        raise DispatchError("packet WAVE does not contain the frozen 40-cell roster")
    canonical_ids = [entry.get("cell_id") for entry in roster]
    if (
        any(not isinstance(value, str) for value in canonical_ids)
        or len(set(canonical_ids)) != EXPECTED_CELLS
    ):
        raise DispatchError("packet roster IDs are invalid or duplicated")
    validate_order(canonical_ids, canonical_ids)
    return wave


def _expected_clock(path: Path) -> dict[str, Any]:
    if sha256(path) != CLOCK_SHA256:
        raise DispatchError(
            "native campaign CLOCK does not match fixed authority SHA-256"
        )
    clock = read_object(path)
    if clock.get("fit_seconds") != 16500 or clock.get("wave_seconds") != 18000:
        raise DispatchError("fixed B300 fit/wave limits differ from approved clock")
    return clock


def main() -> None:
    """Run profiles and fits in eight persistent lanes, then export and score."""
    args = _parse_args()
    packet = args.packet.resolve(strict=True)
    experiment_root = args.experiment_root.resolve(strict=True)
    for required in (
        packet / "runtime/launch.py",
        packet / "runtime/approve_profile.py",
        packet / "WAVE_20261112.json",
        args.data_root / "datasets/train.parquet",
        args.data_root / "datasets/valid.parquet",
    ):
        if not required.is_file():
            raise DispatchError(
                f"required frozen campaign input is missing: {required}"
            )
    for runtime in (args.runtime_transformer, args.runtime_gru):
        if not (runtime / "scripts/observe_sequence.py").is_file():
            raise DispatchError(f"pinned sequence observer is missing: {runtime}")
    if sha256(args.split_manifest) != SPLIT_SHA256:
        raise DispatchError(
            "official chronological split hash differs from frozen packet"
        )
    wave = _load_wave(packet)
    roster = wave["fit_accounting"]["complete_roster"]
    order = _load_order(args.order_file, args.order, roster)
    requested_families = {
        family.strip() for family in args.families.split(",") if family.strip()
    }
    selected_order = filter_family_order(order, roster, requested_families)
    indices = [int(value) for value in args.gpu_indices.split(",")]
    uuids = [value.strip().removeprefix("GPU-") for value in args.gpu_uuids.split(",")]
    if (
        len(indices) != 8
        or len(set(indices)) != 8
        or any(value < 0 for value in indices)
    ):
        raise DispatchError(
            "exactly eight distinct nonnegative GPU indices are required"
        )
    if len(uuids) != 8 or len(set(uuids)) != 8 or any(not value for value in uuids):
        raise DispatchError("exactly eight distinct physical GPU UUIDs are required")
    clock = _expected_clock(args.clock)
    transformer_pre = args.transformer_family_dir / "preprocessing/preprocessing.json"
    transformer_checks = args.transformer_family_dir / "acceptance/checks.json"
    gru_pre = args.gru_family_dir / "preprocessing/preprocessing.json"
    gru_checks = args.gru_family_dir / "acceptance/checks.json"
    checked: dict[str, dict[str, Any]] = {}
    if "transformer" in requested_families:
        checked["transformer"] = check_family_receipt(
            transformer_checks, transformer_pre, "transformer", TRANSFORMER_COMMIT
        )
    if "gru" in requested_families:
        checked["gru"] = check_family_receipt(gru_checks, gru_pre, "gru", GRU_COMMIT)
    by_id = {entry["cell_id"]: entry for entry in roster}
    selected_positions = {
        cell_id: index for index, cell_id in enumerate(selected_order)
    }
    run_root = experiment_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    args.control.mkdir(parents=True, exist_ok=False)

    # Imports are deferred so CPU boundary tests remain independent of Linux /proc.
    old_support = Path(
        "/data/epanov_s/tabdata-qknorm-b300-20261111-r1/control/profile-receiver-r1"
    )
    if not old_support.is_dir():
        raise DispatchError(f"reviewed R2 runner support is missing: {old_support}")
    sys.path.insert(0, str(old_support))
    import b300_driver_core_r1 as core
    import b300_driver_support_r1 as support
    import b300_owned_runner_r2 as owned

    owner = support._identity(experiment_root)
    core.atomic_json(args.control / "owner.json", owner)
    guard = owned.ResourceGuard(experiment_root, args.control, RESOURCE_CAPS)
    fit_deadline = float(clock["fit_deadline_epoch"])
    wave_deadline = float(clock["wave_deadline_epoch"])
    state_cells: dict[str, Any] = {}
    for position, cell_id in enumerate(order):
        entry = by_id[cell_id]
        card = entry["card"]
        family = family_for(card)
        queue_position = selected_positions.get(cell_id)
        state_cells[cell_id] = {
            "priority": position,
            "wave_stage": next(
                (
                    stage["stage"]
                    for stage in wave["fit_accounting"]["stages"]
                    if cell_id in stage["cells"]
                ),
                None,
            ),
            "card": card,
            "arm": entry["arm"],
            "seed": entry["seed"],
            "family": family,
            "queue_priority": queue_position,
            "gpu_index": indices[queue_position % len(indices)]
            if queue_position is not None
            else None,
            "gpu_uuid": uuids[queue_position % len(uuids)]
            if queue_position is not None
            else None,
            "status": "pending" if queue_position is not None else "deferred",
            "deferred_reason": None
            if queue_position is not None
            else "outside current family scope",
            "events": [],
        }
    state = State(
        args.control / "status.json",
        {
            "schema": "wnn.b300-impact81-dispatch.r1",
            "status": "running",
            "started_epoch": time.time(),
            "clock_sha256": CLOCK_SHA256,
            "fit_deadline_epoch": fit_deadline,
            "wave_deadline_epoch": wave_deadline,
            "packet_root": str(packet),
            "split_sha256": SPLIT_SHA256,
            "order_name": args.order,
            "selected_families": sorted(requested_families),
            "selected_cell_count": len(selected_order),
            "ordered_cells": order,
            "automatic_retries": 0,
            "resource_caps": RESOURCE_CAPS,
            "profile_admission_caps": {
                "max_vram_gib_per_rank": MAX_VRAM_GIB_PER_RANK,
                "max_host_ram_gib": MAX_HOST_RAM_GIB,
            },
            "family_checks": {
                name: {
                    "path": str(
                        transformer_checks if name == "transformer" else gru_checks
                    ),
                    "status": receipt["status"],
                    "runtime_commit": receipt["runtime_commit"],
                    "preprocessing_sha256": receipt["preprocessing_sha256"],
                    "card10_feasible": receipt.get("card10_feasible"),
                }
                for name, receipt in checked.items()
            },
            "cells": state_cells,
        },
    )

    def env_for(runtime: Path, gpu: list[str]) -> dict[str, str]:
        env = support._environment(runtime, experiment_root, gpu)
        env.update(
            {
                "WNN_MONITOR_HEARTBEAT_SECONDS": str(args.heartbeat_seconds),
            }
        )
        if args.path_prefix:
            env["PATH"] = args.path_prefix + os.pathsep + env.get("PATH", "")
        return env

    def progress_callback(
        cell_id: str, stage_name: str, output: Path
    ) -> Callable[[], None]:
        def progress() -> None:
            path = output / "monitor_progress.json"
            report: dict[str, Any] = {}
            if path.is_file() and not path.is_symlink():
                try:
                    report = read_object(path)
                except (OSError, ValueError, json.JSONDecodeError):
                    report = {"progress_read_error": True}
            state.update(
                cell_id,
                active_stage=stage_name,
                last_progress=report,
                heartbeat_epoch=time.time(),
            )

        return progress

    def run_stage(
        cell_id: str,
        stage_name: str,
        command: list[str],
        runtime: Path,
        gpu: list[str],
        stage_deadline: float,
        output: Path,
        maximum_timeout: int,
        pass_timeout: bool = True,
    ) -> dict[str, Any]:
        timeout = remaining_timeout(stage_deadline, time.time(), maximum_timeout)
        if pass_timeout:
            command = [*command, "--timeout-seconds", str(timeout)]
        if stage_name == "fit":
            command = observed_fit_command(
                args.python, runtime, output, timeout, command
            )
        started = time.time()
        state.update(
            cell_id,
            status=stage_name,
            active_stage=stage_name,
            stage_started_epoch=started,
            gpu_uuids=gpu,
        )
        try:
            receipt = owned.run_owned(
                command,
                root=runtime,
                control=args.control,
                cell=cell_id,
                stage=stage_name,
                gpu_uuids=gpu,
                owner=owner,
                environment=env_for(runtime, gpu),
                deadline=stage_deadline,
                stage_timeout=timeout,
                guard=guard,
                progress=progress_callback(cell_id, stage_name, output),
            )
            state.event(
                cell_id,
                {
                    "stage": stage_name,
                    "status": "completed",
                    "started_epoch": started,
                    "finished_epoch": time.time(),
                    "owned_receipt": str(
                        args.control / "owned" / f"{cell_id}-{stage_name}-r2.json"
                    ),
                    "command": command,
                },
            )
            return receipt
        except Exception as error:
            state.event(
                cell_id,
                {
                    "stage": stage_name,
                    "status": "failed",
                    "started_epoch": started,
                    "finished_epoch": time.time(),
                    "error": f"{type(error).__name__}: {error}",
                    "owned_receipt": str(
                        args.control / "owned" / f"{cell_id}-{stage_name}-r2.json"
                    ),
                    "command": command,
                },
            )
            raise

    def cell_fit(cell_id: str, position: int) -> None:
        entry = by_id[cell_id]
        family = family_for(entry["card"])
        if entry["card"] == "10" and not card10_allowed(checked["gru"]):
            state.update(
                cell_id,
                status="skipped",
                reason="frozen train-only card10 feasibility gate failed",
                feasibility_receipt=str(gru_checks),
                finished_epoch=time.time(),
            )
            return
        if time.time() >= fit_deadline:
            state.update(
                cell_id,
                status="not_started",
                reason="fixed fit cutoff reached before this lane's next cell",
                finished_epoch=time.time(),
            )
            return
        runtime = (
            args.runtime_transformer if family == "transformer" else args.runtime_gru
        )
        prep = transformer_pre if family == "transformer" else gru_pre
        checks = transformer_checks if family == "transformer" else gru_checks
        cell_path = packet / entry["cell_config"]
        run_dir = run_root / cell_id
        run_dir.mkdir(parents=True, exist_ok=False)
        gpu_index = indices[position % len(indices)]
        gpu_uuid = uuids[position % len(uuids)]
        gpu = [gpu_uuid]
        common = [
            args.python,
            str(packet / "runtime/launch.py"),
            "",  # replaced per mode below
            "--runtime-root",
            str(runtime),
            "--cell",
            str(cell_path),
            "--data-root",
            str(args.data_root),
            "--split-manifest",
            str(args.split_manifest),
            "--owner",
            args.owner,
        ]

        def launch(
            mode: str,
            output: Path,
            extra: list[str],
            deadline: float,
            timeout: int,
            stage_name: str,
            stage_gpu: list[str] | None = None,
        ) -> None:
            if output.exists() or output.is_symlink():
                raise DispatchError(
                    f"refusing to overwrite existing {cell_id}/{stage_name}: {output}"
                )
            command = list(common)
            command[2] = mode
            command += ["--output", str(output), *extra]
            run_stage(
                cell_id,
                stage_name,
                command,
                runtime,
                gpu if stage_gpu is None else stage_gpu,
                deadline,
                output,
                timeout,
            )

        model_inputs = [
            "--preprocessing",
            str(prep),
            "--checks-receipt",
            str(checks),
        ]
        try:
            profile_dir = run_dir / "profile"
            launch(
                "profile",
                profile_dir,
                [
                    *model_inputs,
                    "--devices",
                    str(gpu_index),
                    "--profile-updates",
                    str(PROFILE_UPDATES),
                ],
                fit_deadline,
                3600,
                "profile",
            )
            profile_receipt = profile_dir / "profile_receipt.json"
            if not profile_receipt.is_file():
                raise DispatchError(
                    "profile command completed without profile_receipt.json"
                )
            approved = run_dir / "approved_profile.json"
            if approved.exists():
                raise DispatchError(
                    "refusing to overwrite an existing approval receipt"
                )
            approve_command = [
                args.python,
                str(packet / "runtime/approve_profile.py"),
                "--runtime-root",
                str(runtime),
                "--cell",
                str(cell_path),
                "--profile",
                str(profile_receipt),
                "--output",
                str(approved),
                "--reviewer",
                args.reviewer,
                "--note",
                "Reviewed exact-cell 50-update B300 profile and fixed resource ceilings",
                "--max-host-ram-gib",
                str(MAX_HOST_RAM_GIB),
                "--max-vram-gib-per-rank",
                str(MAX_VRAM_GIB_PER_RANK),
                "--approve",
            ]
            run_stage(
                cell_id,
                "approve",
                approve_command,
                runtime,
                [],
                fit_deadline,
                approved.parent,
                900,
                pass_timeout=False,
            )
            if not approved.is_file():
                raise DispatchError(
                    "approval process completed without approved_profile.json"
                )
            fit_dir = run_dir / "fit"
            launch(
                "fit",
                fit_dir,
                [
                    *model_inputs,
                    "--profile-receipt",
                    str(approved),
                    "--devices",
                    str(gpu_index),
                ],
                fit_deadline,
                14400,
                "fit",
            )
            terminal = read_object(fit_dir / "terminal_status.json")
            expected_epochs = 4 if family == "transformer" else 6
            expected_updates = 3280 if family == "transformer" else 38760
            model_path = fit_dir / "model.pt"
            if (
                terminal.get("status") != "completed"
                or terminal.get("completed_epochs") != expected_epochs
                or terminal.get("global_updates") != expected_updates
                or not model_path.is_file()
            ):
                raise DispatchError(
                    "fit process exited without fixed-endpoint scientific completion"
                )
            if terminal.get("model_sha256") != sha256(model_path):
                raise DispatchError("terminal model digest does not bind model.pt")
            state.update(
                cell_id,
                status="fit_completed_pending_export",
                completed_epochs=terminal["completed_epochs"],
                global_updates=terminal.get("global_updates"),
                terminal_status=str(fit_dir / "terminal_status.json"),
                fit_completed_epoch=time.time(),
                gpu_index=gpu_index,
            )
        except Exception as error:  # noqa: BLE001 - preserve any cell failure and continue lanes.
            state.update(
                cell_id,
                status="failed",
                error=f"{type(error).__name__}: {error}",
                failed_epoch=time.time(),
                gpu_index=gpu_index,
            )

    lane_work = lane_orders(selected_order, len(indices))

    def run_lane(lane_index: int, cells: list[str]) -> None:
        for cell_id in cells:
            try:
                cell_fit(cell_id, lane_index)
            except Exception as error:  # noqa: BLE001 - isolate one cell and preserve its failure.
                state.update(
                    cell_id,
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    failed_epoch=time.time(),
                )

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="b300-lane") as pool:
        futures = [
            pool.submit(run_lane, lane_index, lane)
            for lane_index, lane in enumerate(lane_work)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - retain controller lane failure evidence.
                with state.lock:
                    state.payload["controller_lane_errors"] = state.payload.get(
                        "controller_lane_errors", []
                    )
                    state.payload["controller_lane_errors"].append(
                        f"{type(error).__name__}: {error}"
                    )
                    _atomic_json(state.path, state.payload)

    # Fit cutoff and wave cutoff have separate authority. Once every fit worker
    # has stopped, exports can use a fresh shared guard through the original wave cutoff.
    export_guard = owned.ResourceGuard(experiment_root, args.control, RESOURCE_CAPS)
    score_candidates = [
        cell_id
        for cell_id in order
        if state.payload["cells"][cell_id].get("status")
        == "fit_completed_pending_export"
    ]

    def export_score(cell_id: str) -> None:
        entry = by_id[cell_id]
        family = family_for(entry["card"])
        runtime = (
            args.runtime_transformer if family == "transformer" else args.runtime_gru
        )
        cell_path = packet / entry["cell_config"]
        run_dir = run_root / cell_id
        gpu_index = state.payload["cells"][cell_id]["gpu_index"]
        gpu = [uuids[indices.index(gpu_index)]]
        common = [
            args.python,
            str(packet / "runtime/launch.py"),
            "",
            "--runtime-root",
            str(runtime),
            "--cell",
            str(cell_path),
            "--data-root",
            str(args.data_root),
            "--split-manifest",
            str(args.split_manifest),
            "--owner",
            args.owner,
        ]
        prediction_dir = run_dir / "predictions"
        fit_dir = run_dir / "fit"
        try:
            if prediction_dir.exists():
                raise DispatchError("refusing to overwrite existing prediction export")
            command = list(common)
            command[2] = "export"
            command += [
                "--checkpoint",
                str(fit_dir / "model.pt"),
                "--output",
                str(prediction_dir),
                "--devices",
                str(gpu_index),
            ]
            timeout = remaining_timeout(wave_deadline, time.time(), 7200)
            state.update(cell_id, status="export")
            owned.run_owned(
                [*command, "--timeout-seconds", str(timeout)],
                root=runtime,
                control=args.control,
                cell=cell_id,
                stage="export",
                gpu_uuids=gpu,
                owner=owner,
                environment=env_for(runtime, gpu),
                deadline=wave_deadline,
                stage_timeout=timeout,
                guard=export_guard,
                progress=progress_callback(cell_id, "export", prediction_dir),
            )
            manifest = read_object(prediction_dir / "manifest.json")
            if (
                manifest.get("schema") != "sequence-predictions-v1"
                or manifest.get("complete") is not True
                or manifest.get("split_sha256") != SPLIT_SHA256
                or len(manifest.get("sequences", [])) != 1873
            ):
                raise DispatchError(
                    "export manifest is incomplete or has wrong validation coverage"
                )
            state.event(
                cell_id,
                {
                    "stage": "export",
                    "status": "completed",
                    "manifest": str(prediction_dir / "manifest.json"),
                    "finished_epoch": time.time(),
                },
            )
            score_path = run_dir / "score.json"
            if score_path.exists():
                raise DispatchError("refusing to overwrite existing score receipt")
            command = list(common)
            command[2] = "score"
            command += [
                "--predictions",
                str(prediction_dir),
                "--output",
                str(score_path),
            ]
            timeout = remaining_timeout(wave_deadline, time.time(), 7200)
            state.update(cell_id, status="score")
            owned.run_owned(
                [*command, "--timeout-seconds", str(timeout)],
                root=runtime,
                control=args.control,
                cell=cell_id,
                stage="score",
                gpu_uuids=[],
                owner=owner,
                environment=env_for(runtime, []),
                deadline=wave_deadline,
                stage_timeout=timeout,
                guard=export_guard,
                progress=progress_callback(cell_id, "score", score_path.parent),
            )
            score = read_object(score_path)
            if (
                score.get("sequences") != 1873
                or score.get("split_sha256") != SPLIT_SHA256
                or not isinstance(score.get("score"), dict)
            ):
                raise DispatchError(
                    "score receipt does not cover frozen official validation"
                )
            state.update(
                cell_id,
                status="scored",
                score_path=str(score_path),
                score=score["score"],
                scored_epoch=time.time(),
            )
        except Exception as error:  # noqa: BLE001 - preserve export/score failure and continue.
            state.update(
                cell_id,
                status="failed_after_fit",
                error=f"{type(error).__name__}: {error}",
                failed_epoch=time.time(),
            )

    # Independent exports can use all eight GPUs. CPU score runs after each export.
    export_lanes = (
        lane_orders(score_candidates, len(indices)) if score_candidates else []
    )

    def export_lane(cells: list[str]) -> None:
        for cell_id in cells:
            export_score(cell_id)

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="b300-export") as pool:
        futures = [pool.submit(export_lane, lane) for lane in export_lanes]
        for future in as_completed(futures):
            future.result()

    with state.lock:
        now = time.time()
        for cell_id, record in state.payload["cells"].items():
            if record.get("status") == "pending":
                record.update(
                    status="not_started",
                    reason="fit cutoff reached before lane dispatch",
                )
            elif record.get("status") in {"profile", "approve", "fit"}:
                record.update(
                    status="failed",
                    reason="controller ended with an unfinished owned stage",
                )
        state.payload.update(
            status="completed_with_cell_outcomes",
            finished_epoch=now,
            fit_deadline_reached=now >= fit_deadline,
            wave_deadline_reached=now >= wave_deadline,
        )
        _atomic_json(state.path, state.payload)


if __name__ == "__main__":
    main()
