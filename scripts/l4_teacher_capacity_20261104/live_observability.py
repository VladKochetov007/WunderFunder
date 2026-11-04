"""Bridge the frozen L4 teacher panel into the existing dashboard and enforce its live resource cap."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

SOURCE_ROOT = Path(
    os.environ.get("WNN_TEACHER_SOURCE_ROOT", Path(__file__).resolve().parents[2])
).resolve()
sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]

from scripts import tl3_dashboard_bridge as bridge  # noqa: E402
from scripts import w18_l4_quality_dashboard_20261104 as w18_bridge  # noqa: E402
from wnn.multistage_search_runtime import (  # noqa: E402
    is_same_process,
    terminate_owned_process,
)

MAX_PANEL_BYTES = 4_000_000
MAX_LOG_BYTES = 64 << 20
WATCH_FILENAME = "l4_live_watch.jsonl"


def _read_panel(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PANEL_BYTES:
        raise ValueError(f"unsafe or oversized panel: {path}")
    raw = path.read_bytes()
    panel = json.loads(raw)
    if not isinstance(panel, dict) or panel.get("schema") != schema:
        raise ValueError(f"unexpected panel schema: {path.name}")
    if not isinstance(panel.get("cells"), list):
        raise ValueError(f"panel has no cell list: {path.name}")
    import hashlib

    return cast(dict[str, Any], panel), hashlib.sha256(raw).hexdigest()


def _safe_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1 << 20:
        raise ValueError(f"unsafe receipt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"receipt is not an object: {path}")
    return cast(dict[str, Any], value)


def _sha256_file(path: Path) -> str:
    import hashlib

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular identity file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _fit_location(panel: dict[str, Any], cell: dict[str, Any], family: str) -> dict[str, Any]:
    if family == "gru":
        runtime = Path(panel["paths"]["runtime_root"])
        project = Path(panel["paths"]["project_root"])
        cell_id, run_name = str(cell["cell_id"]), str(cell["run_name"])
        return {
            "log": runtime / "logs/fits" / f"{cell_id}.log",
            "receipt": runtime / "receipts/fits" / f"{cell_id}.json",
            "output": project / "artifacts/runs" / run_name,
            "argv": cell["fit_argv"],
        }
    return {
        "log": Path(cell["fit_log"]),
        "receipt": Path(cell["fit_process_receipt"]),
        "output": Path(cell["fit_output"]),
        "argv": cell["fit_argv"],
    }


def _run_name(cell: dict[str, Any]) -> str:
    explicit = cell.get("run_name")
    if isinstance(explicit, str) and bridge.RUN_ID.fullmatch(explicit):
        return explicit
    output = Path(str(cell.get("fit_output", "")))
    leaf = output.name
    if leaf not in {"", "fit", "profile"} and bridge.RUN_ID.fullmatch(leaf):
        return leaf
    config_id = str(cell.get("configuration_id", "teacher"))
    seed = cell.get("seed", "unknown")
    candidate = f"{config_id}_s{seed}"
    if not bridge.RUN_ID.fullmatch(candidate):
        raise ValueError(f"cannot derive a dashboard run ID for {cell.get('cell_id')}")
    return candidate


def _status(
    receipt_path: Path,
    log_path: Path,
    output_path: Path,
    cell: dict[str, Any],
    panel_sha: str,
) -> tuple[str, dict[str, Any]]:
    receipt = _safe_object(receipt_path)
    if receipt is None:
        return ("running" if log_path.is_file() and log_path.stat().st_size else "not_started", {})
    owner = receipt.get("owner")
    if not isinstance(owner, dict):
        raise ValueError(f"receipt has no owner identity: {receipt_path.name}")
    if owner.get("cell_id") != cell["cell_id"]:
        raise ValueError(f"receipt cell identity changed: {receipt_path.name}")
    recorded_panel = owner.get("panel_sha256", owner.get("transformer_panel_sha256"))
    if recorded_panel != panel_sha:
        raise ValueError(f"receipt panel identity changed: {receipt_path.name}")
    if receipt.get("command") != cell["fit_argv"]:
        raise ValueError(f"receipt argv differs from frozen panel cell: {receipt_path.name}")
    if os.path.normpath(str(receipt.get("output_dir", ""))) != os.path.normpath(str(output_path)):
        raise ValueError(f"receipt output path differs from frozen panel cell: {receipt_path.name}")
    status = receipt.get("status")
    if not isinstance(status, str):
        raise ValueError(f"receipt status missing: {receipt_path.name}")
    return status, receipt


def _metadata(
    panel: dict[str, Any],
    panel_sha: str,
    cell: dict[str, Any],
    family: str,
    status: str,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    identity = panel.get("identity", {})
    state_path = Path(str(panel.get("paths", {}).get("runtime_root", ""))) / "GRU_STATE_R1.json"
    state = _safe_object(state_path) or {} if family == "gru" else {}
    args = cell["fit_argv"]
    epochs = int(args[args.index("--epochs") + 1]) if "--epochs" in args else 4
    cfg: dict[str, Any] = {
        "run": _run_name(cell),
        "cell_id": cell["cell_id"],
        "configuration_id": cell.get("configuration_id"),
        "role": cell.get("role", "capacity"),
        "seed": cell["seed"],
        "status": status,
        "checkpoint_policy": "fixed",
        "fixed_endpoint_epoch_index": epochs - 1,
        "fixed_endpoint_completed_epoch": epochs,
        "precision": "fp32",
        "effective_args": args,
        "gpu_uuid": cell["gpu_uuid"],
    }
    if family == "tf":
        config_path = Path(str(cell["config_path"]))
        config_sha = _sha256_file(config_path)
        expected_config_sha = cell.get("config_sha256")
        if not isinstance(expected_config_sha, str) or expected_config_sha != config_sha:
            raise ValueError(f"Transformer config digest changed for {cell['cell_id']}")
        cfg["config_sha256"] = config_sha
        cfg["sequence_config"] = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "source": f"L4 teacher capacity frozen {family.upper()} panel; fit-only training, full official validation",
        "config": cfg,
        "provenance": {
            "dashboard_adapter": "live_observability.py",
            "panel_schema": panel.get("schema"),
            "panel_sha256": panel_sha,
            "cell_id": cell["cell_id"],
            "configuration_id": cell.get("configuration_id"),
            "config_sha256": cfg.get("config_sha256"),
            "source_binding_sha256": (
                identity.get("source_binding_sha256")
                if family == "tf"
                else state.get("source_binding_sha256", identity.get("source_binding_sha256"))
            ),
            "data_binding_sha256": (
                identity.get("data_binding_sha256")
                if family == "tf"
                else state.get("data_binding_sha256", identity.get("data_binding_sha256"))
            ),
            "fit_members_sha256": (
                identity.get("fit_members_sha256")
                if family == "tf"
                else state.get("fit_members_sha256", identity.get("fit_members_sha256"))
            ),
            "split_sha256": identity.get("split_sha256"),
            "fit_receipt_status": receipt.get("status"),
            "fit_child": receipt.get("child", {}),
            "validation_policy": {
                "primary_valid_row_mask": "is_scored AND rows>=99",
                "primary_valid_sequences": 1873,
                "primary_valid_scored_rows": 3_528_273,
                "train_eval_source": "train",
                "train_eval_row_mask": "all_after_warmup",
                "train_eval_sequences": 32,
                "train_eval_comparable_to_primary_valid": False,
            },
        },
        "synthetic": False,
    }


def _publish_native_metadata(
    panel: dict[str, Any],
    panel_sha: str,
    cell: dict[str, Any],
    output: Path,
    dashboard_root: Path,
    status: str,
    receipt: dict[str, Any],
) -> None:
    """Expose native Transformer journals through hardlinked, local dashboard run views."""
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"Transformer fit output is not a regular directory: {output}")
    run_name = _run_name(cell)
    view = dashboard_root / run_name
    if view != output:
        if view.parent != dashboard_root or view.is_symlink():
            raise ValueError(f"unsafe Transformer dashboard run view: {view}")
        view.mkdir(mode=0o700, exist_ok=True)
        if view.is_symlink() or not view.is_dir():
            raise ValueError(f"Transformer dashboard view is not a regular directory: {view}")
    metadata = _metadata(panel, panel_sha, cell, "tf", status, receipt)
    metadata_path = view / "run_metadata.json"
    prior = _safe_object(metadata_path)
    if prior is not None:
        provenance = prior.get("provenance")
        if not isinstance(provenance, dict) or any(
            provenance.get(key) != metadata["provenance"].get(key)
            for key in (
                "panel_sha256",
                "cell_id",
                "config_sha256",
                "source_binding_sha256",
                "data_binding_sha256",
                "fit_members_sha256",
                "split_sha256",
            )
        ):
            raise ValueError(
                f"native run metadata belongs to another panel cell: {cell['cell_id']}"
            )
    encoded = json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if prior is None or metadata_path.read_text(encoding="utf-8") != encoded:
        bridge._atomic_json(metadata_path, metadata)
    if view != output:
        for filename in ("monitor_events.jsonl", "monitor_evaluations.jsonl"):
            source, target = output / filename, view / filename
            if source.is_symlink() or not source.is_file():
                continue
            if target.exists():
                if target.is_symlink() or not target.is_file():
                    raise ValueError(f"unsafe linked Transformer stream: {target}")
                source_stat, target_stat = source.stat(), target.stat()
                if (source_stat.st_dev, source_stat.st_ino) != (
                    target_stat.st_dev,
                    target_stat.st_ino,
                ):
                    raise ValueError(
                        f"Transformer dashboard stream no longer links to native: {filename}"
                    )
            else:
                os.link(source, target, follow_symlinks=False)


def _append_watch_events(
    view: Path, cell_id: str, run_name: str, panel_sha: str, records: list[dict[str, Any]]
) -> int:
    event_path = view / "monitor_events.jsonl"
    known = bridge._read_events(event_path)
    added = 0
    for record in records:
        cell_metrics = record.get("cells", {}).get(cell_id)
        if not isinstance(cell_metrics, dict):
            continue
        metrics: dict[str, int | float] = {}
        for source, target in (
            ("group_rss_bytes", "cpu/group_rss_bytes"),
            ("group_threads", "cpu/group_threads"),
            ("group_affinity_cores", "cpu/group_affinity_cores"),
            ("wave_rss_bytes", "cpu/wave_rss_bytes"),
            ("wave_threads", "cpu/wave_threads"),
            ("wave_affinity_cores", "cpu/wave_affinity_cores"),
            ("disk_free_bytes", "disk/free_bytes"),
            ("output_bytes", "disk/output_bytes"),
            ("gpu_free_mib", "gpu/memory_free_mib"),
            ("gpu_total_mib", "gpu/memory_total_mib"),
            ("gpu_utilization_percent", "gpu/utilization_percent"),
        ):
            value = cell_metrics.get(source)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[target] = value
        if not metrics:
            continue
        sample = int(record["sample_id"])
        event = bridge._event(
            "resources",
            sample,
            float(record.get("latest_epoch", 0.0)),
            metrics,
            {
                "run": run_name,
                "cell_id": cell_id,
                "panel_sha256": panel_sha,
                "gpu_uuid": cell_metrics.get("gpu_uuid"),
                "observed_at_utc": record.get("observed_at_utc"),
                "scope": "receipt-owned process group plus wave-wide totals",
            },
        )
        added += int(bridge._append_event(event_path, event, known))
    return added


def _read_watch(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_LOG_BYTES:
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw[: raw.rfind(b"\n") + 1] if b"\n" in raw else b""
    records = []
    for line in raw.splitlines():
        value = json.loads(line)
        if isinstance(value, dict) and isinstance(value.get("sample_id"), int):
            records.append(value)
    return records


def bridge_once(gru_path: Path, tf_path: Path, dashboard_root: Path, selected: set[str]) -> int:
    gru, gru_sha = _read_panel(gru_path, "wnn.l4-teacher-capacity-gru.v1")
    tf, tf_sha = _read_panel(tf_path, "wnn.l4-teacher-capacity-transformer-panel.v1")
    dashboard_root = dashboard_root.expanduser().absolute()
    dashboard_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if dashboard_root.is_symlink() or dashboard_root.resolve(strict=True) != dashboard_root:
        raise ValueError("dashboard runs root must be a resolved directory")
    runtime = Path(str(gru["paths"]["runtime_root"]))
    watch = _read_watch(runtime / "receipts" / WATCH_FILENAME)
    appended = 0
    for family, panel, panel_sha in (("gru", gru, gru_sha), ("tf", tf, tf_sha)):
        for raw_cell in panel["cells"]:
            cell = cast(dict[str, Any], raw_cell)
            run_name, cell_id = _run_name(cell), str(cell["cell_id"])
            if selected and run_name not in selected and cell_id not in selected:
                continue
            location = _fit_location(panel, cell, family)
            status, receipt = _status(
                location["receipt"], location["log"], location["output"], cell, panel_sha
            )
            if family == "tf":
                if location["output"].is_dir() and not location["output"].is_symlink():
                    _publish_native_metadata(
                        panel,
                        panel_sha,
                        cell,
                        location["output"],
                        dashboard_root,
                        status,
                        receipt,
                    )
                continue
            view = w18_bridge._ensure_view(dashboard_root, run_name, panel_sha, cell_id)
            metadata = _metadata(panel, panel_sha, cell, family, status, receipt)
            encoded = json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n"
            metadata_path = view / "run_metadata.json"
            if not metadata_path.exists() or metadata_path.read_text(encoding="utf-8") != encoded:
                bridge._atomic_json(metadata_path, metadata)
            lines = bridge._read_log(location["log"])
            parsed = bridge.parse_legacy_log(lines)
            bridge_cell = dict(cell)
            bridge_cell["train_args"] = cell["fit_argv"]
            appended += w18_bridge._append_training_events(view, parsed, bridge_cell, panel_sha)
            appended += _append_watch_events(view, cell_id, run_name, panel_sha, watch)
            if status in {"completed", "failed", "timed_out", "watchdog_terminated"}:
                known = bridge._read_events(view / "monitor_events.jsonl")
                epochs = int(cell["fit_argv"][cell["fit_argv"].index("--epochs") + 1])
                valid = parsed["valid"]
                best_epoch = (
                    max(valid, key=lambda e: valid[e].get("weighted_pearson", float("-inf")))
                    if valid
                    else None
                )
                terminal = bridge._event(
                    "status",
                    1,
                    0,
                    {
                        "run/completed": int(status == "completed"),
                        "run/failed": int(status != "completed"),
                        "selection/fixed_endpoint_epoch": epochs,
                        "selection/best_observed_epoch": best_epoch + 1
                        if best_epoch is not None
                        else 0,
                    },
                    {
                        "run": run_name,
                        "cell_id": cell_id,
                        "panel_sha256": panel_sha,
                        "lifecycle_status": status,
                        "selected_fixed_epoch_index_from_log": parsed["selected_fixed"],
                        "best_observed_epoch_index": best_epoch,
                        "valid_records_seen": len(valid),
                    },
                )
                appended += int(
                    bridge._append_event(view / "monitor_events.jsonl", terminal, known)
                )
    return appended


def _proc_group_stats(pgid: int) -> dict[str, Any]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    rss_bytes = threads = 0
    affinity_cpus: set[int] = set()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        process = Path("/proc") / name
        try:
            stat_fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat_fields[2]) != pgid:
                continue
            status = (process / "status").read_text()
            thread_line = next(line for line in status.splitlines() if line.startswith("Threads:"))
            rss_pages = int((process / "statm").read_text().split()[1])
            threads += int(thread_line.split()[1])
            rss_bytes += rss_pages * page_size
            for task_id in os.listdir(process / "task"):
                if task_id.isdigit():
                    try:
                        affinity_cpus.update(os.sched_getaffinity(int(task_id)))
                    except (AttributeError, OSError, ProcessLookupError):
                        continue
        except (
            FileNotFoundError,
            ProcessLookupError,
            PermissionError,
            StopIteration,
            ValueError,
            IndexError,
        ):
            continue
    return {
        "rss_bytes": rss_bytes,
        "threads": threads,
        "affinity_cpus": sorted(affinity_cpus),
    }


def _gpu_stats() -> dict[str, dict[str, float]]:
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
    except (OSError, subprocess.SubprocessError):
        return {}
    devices: dict[str, dict[str, float]] = {}
    for line in result.stdout.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if len(columns) != 4:
            continue
        try:
            devices[columns[0]] = {
                "free_mib": float(columns[1]),
                "total_mib": float(columns[2]),
                "utilization_percent": float(columns[3]),
            }
        except ValueError:
            continue
    return devices


def _latest_native_epoch(event_path: Path) -> float:
    if event_path.is_symlink() or not event_path.is_file():
        return 0.0
    with event_path.open("rb") as stream:
        size = event_path.stat().st_size
        stream.seek(max(0, size - (2 << 20)))
        raw = stream.read(2 << 20)
    for line in reversed(raw.splitlines()):
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(event, dict)
            and event.get("schema") == bridge.EVENT_SCHEMA
            and isinstance(event.get("epoch"), (int, float))
        ):
            return float(event["epoch"])
    return 0.0


def _append_native_tf_resource(
    output: Path,
    panel: dict[str, Any],
    panel_sha: str,
    dashboard_root: Path,
    cell: dict[str, Any],
    cell_data: dict[str, Any],
    sample_id: int,
) -> bool:
    """Append one low-rate resource event to the Transformer-native monitor stream."""
    event_path = output / "monitor_events.jsonl"
    if event_path.is_symlink() or not event_path.is_file():
        return False
    view = dashboard_root / _run_name(cell)
    metadata = _safe_object(view / "run_metadata.json")
    if metadata is None or not isinstance(metadata.get("provenance"), dict):
        return False
    provenance = metadata["provenance"]
    identity = panel.get("identity", {})
    expected_provenance = {
        "source_binding_sha256": identity.get("source_binding_sha256"),
        "data_binding_sha256": identity.get("data_binding_sha256"),
        "fit_members_sha256": identity.get("fit_members_sha256"),
        "split_sha256": identity.get("split_sha256"),
    }
    if (
        provenance.get("panel_sha256") != panel_sha
        or provenance.get("cell_id") != cell["cell_id"]
        or provenance.get("config_sha256") != cell.get("config_sha256")
        or any(provenance.get(key) != value for key, value in expected_provenance.items())
    ):
        raise ValueError(f"Transformer dashboard metadata identity differs for {cell['cell_id']}")
    view_events = view / "monitor_events.jsonl"
    if view_events.is_symlink() or not view_events.is_file():
        return False
    native_stat, view_stat = event_path.stat(), view_events.stat()
    if (native_stat.st_dev, native_stat.st_ino) != (view_stat.st_dev, view_stat.st_ino):
        raise ValueError(
            f"Transformer dashboard event stream is not the native hardlink: {cell['cell_id']}"
        )
    metrics: dict[str, int | float] = {}
    for source, target in (
        ("group_rss_bytes", "cpu/cohort_group_rss_bytes"),
        ("group_threads", "cpu/cohort_group_threads"),
        ("group_affinity_cores", "cpu/cohort_group_affinity_cores"),
        ("wave_rss_bytes", "cpu/wave_rss_bytes"),
        ("wave_threads", "cpu/wave_threads"),
        ("wave_affinity_cores", "cpu/wave_affinity_cores"),
        ("disk_free_bytes", "disk/free_bytes"),
        ("output_bytes", "disk/output_bytes"),
        ("gpu_free_mib", "gpu/device_memory_free_mib"),
        ("gpu_total_mib", "gpu/device_memory_total_mib"),
        ("gpu_utilization_percent", "gpu/utilization_pct"),
    ):
        value = cell_data.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[target] = value
    if not metrics:
        return False
    event = bridge._event(
        "resources",
        1_000_000_000 + sample_id,
        _latest_native_epoch(event_path),
        metrics,
        {
            "run": _run_name(cell),
            "cell_id": cell["cell_id"],
            "panel_sha256": panel_sha,
            "gpu_uuid": cell_data.get("gpu_uuid"),
            "observed_at_utc": cell_data.get("observed_at_utc"),
            "resource_scope": "receipt-owned process group and wave-wide totals",
            "config_sha256": cell.get("config_sha256"),
            "split_sha256": identity.get("split_sha256"),
            "source_binding_sha256": identity.get("source_binding_sha256"),
            "data_binding_sha256": identity.get("data_binding_sha256"),
            "fit_members_sha256": identity.get("fit_members_sha256"),
        },
    )
    encoded = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(
        event_path,
        os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size + len(encoded) > (1 << 30):
            raise ValueError("Transformer native monitor stream is unsafe or exceeds 1 GiB")
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("short append to Transformer native monitor stream")
    finally:
        os.close(descriptor)
    return True


def _owned_receipts(
    gru: dict[str, Any], gru_sha: str, tf: dict[str, Any], tf_sha: str
) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for family, panel, panel_sha in (("gru", gru, gru_sha), ("tf", tf, tf_sha)):
        for cell in panel["cells"]:
            cell = cast(dict[str, Any], cell)
            phases: list[tuple[str, Path, list[str], Path]] = []
            if family == "gru":
                runtime = Path(str(panel["paths"]["runtime_root"]))
                project = Path(str(panel["paths"]["project_root"]))
                cell_id, run_name = str(cell["cell_id"]), str(cell["run_name"])
                for phase, suffix in (("profile", "_profile_r1"), ("fit", "")):
                    phases.append(
                        (
                            phase,
                            runtime
                            / "receipts"
                            / ("profiles" if phase == "profile" else "fits")
                            / f"{cell_id}.json",
                            cell[f"{phase}_argv"],
                            project / "artifacts/runs" / f"{run_name}{suffix}",
                        )
                    )
            else:
                for phase in ("profile", "fit"):
                    phases.append(
                        (
                            phase,
                            Path(cell[f"{phase}_process_receipt"]),
                            cell[f"{phase}_argv"],
                            Path(cell[f"{phase}_output"]),
                        )
                    )
            for phase, receipt_path, expected_command, expected_output in phases:
                receipt = _safe_object(receipt_path)
                if receipt is None or receipt.get("status") != "running":
                    continue
                owner = receipt.get("owner")
                child = receipt.get("child")
                recorded_panel = (
                    owner.get("panel_sha256", owner.get("transformer_panel_sha256"))
                    if isinstance(owner, dict)
                    else None
                )
                if (
                    not isinstance(owner, dict)
                    or owner.get("cell_id") != cell["cell_id"]
                    or recorded_panel != panel_sha
                    or receipt.get("command") != expected_command
                    or os.path.normpath(str(receipt.get("output_dir", "")))
                    != os.path.normpath(str(expected_output))
                    or not isinstance(child, dict)
                    or type(child.get("pid")) is not int
                    or type(child.get("pgid")) is not int
                    or not isinstance(child.get("birth"), (int, str))
                ):
                    continue
                identity = {"pid": child["pid"], "pgid": child["pgid"], "birth": child["birth"]}
                if identity["pgid"] != identity["pid"] or not is_same_process(identity):
                    continue
                stats = _proc_group_stats(int(identity["pgid"]))
                active.append(
                    {
                        "family": family,
                        "phase": phase,
                        "cell_id": cell["cell_id"],
                        "run_name": _run_name(cell),
                        "gpu_uuid": cell["gpu_uuid"],
                        "identity": identity,
                        "receipt_path": receipt_path,
                        "rss_bytes": stats["rss_bytes"],
                        "threads": stats["threads"],
                        "affinity_cpus": stats["affinity_cpus"],
                    }
                )
    return active


def watch_once(
    gru_path: Path,
    tf_path: Path,
    report_only: bool,
    dashboard_root: Path | None = None,
) -> dict[str, Any]:
    gru, gru_sha = _read_panel(gru_path, "wnn.l4-teacher-capacity-gru.v1")
    tf, tf_sha = _read_panel(tf_path, "wnn.l4-teacher-capacity-transformer-panel.v1")
    runtime = Path(str(gru["paths"]["runtime_root"]))
    dashboard_root = dashboard_root or runtime / "dashboard/runs"
    state_path = runtime / "GRU_STATE_R1.json"
    state = _safe_object(state_path)
    if state is None or state.get("panel_sha256") != gru_sha:
        raise ValueError(
            "shared GRU_STATE deadline is missing or belongs to a different frozen panel"
        )
    active = _owned_receipts(gru, gru_sha, tf, tf_sha)
    rss_bytes = sum(item["rss_bytes"] for item in active)
    thread_count = sum(item["threads"] for item in active)
    affinity_cpus = {cpu for item in active for cpu in item["affinity_cpus"]}
    affinity_unverified = [item["cell_id"] for item in active if not item["affinity_cpus"]]
    disk_mount = Path(str(gru["paths"]["disk_mount"]))
    free_bytes = shutil.disk_usage(disk_mount).free
    initial_free = int(state.get("initial_data2_free_bytes", 10_172_567_552))
    output_bytes = max(0, initial_free - free_bytes)
    gpu = _gpu_stats()
    now = time.time()
    deadline = float(state["hard_deadline_epoch"])
    reserve = int(state.get("collection_reserve_seconds", 2700))
    budget = cast(dict[str, Any], gru["budget"])
    violations: list[str] = []
    unauthorized_cpus = sorted(affinity_cpus - set(range(64)))
    if unauthorized_cpus:
        violations.append(
            "receipt-owned process affinity includes CPUs outside authorized IDs 0-63"
        )
    if rss_bytes > 400 * (1 << 30):
        violations.append("receipt-owned process-group RSS exceeds 400 GiB")
    if free_bytes < int(budget["minimum_free_bytes"]):
        violations.append("data2 free space fell below 3 GB")
    if output_bytes > int(budget["max_output_bytes"]):
        violations.append("new runtime output exceeded 6.5 GB")
    if now >= deadline - reserve and active:
        violations.append("shared hard deadline collection reserve has begun")
    cell_totals = {item["cell_id"]: item for item in active}
    for cell in [*gru["cells"], *tf["cells"]]:
        cell = cast(dict[str, Any], cell)
        cell_data = cell_totals.get(str(cell["cell_id"]))
        if cell_data is None:
            continue
        gpu_data = gpu.get(str(cell["gpu_uuid"]), {})
        cell_data.update(
            {
                "gpu_free_mib": gpu_data.get("free_mib"),
                "gpu_total_mib": gpu_data.get("total_mib"),
                "gpu_utilization_percent": gpu_data.get("utilization_percent"),
            }
        )
    sample_id = 1
    watch_path = runtime / "receipts" / WATCH_FILENAME
    if watch_path.exists() and not watch_path.is_symlink():
        with watch_path.open("rb") as stream:
            sample_id = sum(1 for _ in stream) + 1
    record = {
        "sample_id": sample_id,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "remaining_to_deadline_seconds": max(0, int(deadline - now)),
        "wave_rss_bytes": rss_bytes,
        "wave_threads": thread_count,
        "wave_affinity_cores": len(affinity_cpus),
        "affinity_unverified_cells": affinity_unverified,
        "disk_free_bytes": free_bytes,
        "output_bytes": output_bytes,
        "output_accounting": "conservative physical free-space delta from the frozen initial baseline; may include unrelated writers",
        "limits": {
            "authorized_cpu_ids": list(range(64)),
            "threads_observed_only": thread_count,
            "rss_bytes": 400 * (1 << 30),
            "minimum_free_bytes": int(budget["minimum_free_bytes"]),
            "max_output_bytes": int(budget["max_output_bytes"]),
            "collection_reserve_seconds": reserve,
        },
        "active_groups": [
            {key: value for key, value in item.items() if key not in {"identity", "receipt_path"}}
            for item in active
        ],
        "cells": {
            item["cell_id"]: {
                "group_rss_bytes": item["rss_bytes"],
                "group_threads": item["threads"],
                "group_affinity_cores": len(item["affinity_cpus"]),
                "wave_rss_bytes": rss_bytes,
                "wave_threads": thread_count,
                "wave_affinity_cores": len(affinity_cpus),
                "disk_free_bytes": free_bytes,
                "output_bytes": output_bytes,
                "gpu_uuid": item["gpu_uuid"],
                "gpu_free_mib": item.get("gpu_free_mib"),
                "gpu_total_mib": item.get("gpu_total_mib"),
                "gpu_utilization_percent": item.get("gpu_utilization_percent"),
            }
            for item in active
            if item["phase"] == "fit"
        },
        "violations": violations,
        "enforcement": "report_only" if report_only else "exact_receipt_owned_groups_only",
    }
    tf_by_id = {str(cell["cell_id"]): cast(dict[str, Any], cell) for cell in tf["cells"]}
    for item in active:
        if item["family"] != "tf" or item["phase"] != "fit":
            continue
        cell = tf_by_id[str(item["cell_id"])]
        cell_data = cast(dict[str, Any], record["cells"][str(item["cell_id"])])
        cell_data["observed_at_utc"] = record["observed_at_utc"]
        _append_native_tf_resource(
            Path(str(cell["fit_output"])),
            tf,
            tf_sha,
            dashboard_root,
            cell,
            cell_data,
            sample_id,
        )
    killed = []
    if violations and not report_only:
        for item in active:
            if is_same_process(item["identity"]) and terminate_owned_process(
                item["identity"], grace_seconds=0.0
            ):
                killed.append(
                    {
                        "cell_id": item["cell_id"],
                        "identity": item["identity"],
                        "receipt_path": str(item["receipt_path"]),
                    }
                )
        record["killed_owned_groups"] = killed
    watch_path.parent.mkdir(parents=True, exist_ok=True)
    if watch_path.is_symlink():
        raise ValueError("watch receipt cannot be a symlink")
    with watch_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bridge_parser = sub.add_parser(
        "bridge", help="import GRU logs and hardlink native Transformer monitoring streams"
    )
    bridge_parser.add_argument("--gru-panel", type=Path, required=True)
    bridge_parser.add_argument("--transformer-panel", type=Path, required=True)
    bridge_parser.add_argument("--dashboard-root", type=Path)
    bridge_parser.add_argument("--run", action="append", default=[])
    bridge_parser.add_argument("--poll-seconds", type=int, default=5)
    bridge_parser.add_argument("--once", action="store_true")
    watch_parser = sub.add_parser(
        "watch", help="sample caps and terminate only exact receipt-owned live groups on violation"
    )
    watch_parser.add_argument("--gru-panel", type=Path, required=True)
    watch_parser.add_argument("--transformer-panel", type=Path, required=True)
    watch_parser.add_argument("--dashboard-root", type=Path)
    watch_parser.add_argument("--poll-seconds", type=int, default=5)
    watch_parser.add_argument("--once", action="store_true")
    watch_parser.add_argument(
        "--report-only",
        action="store_true",
        help="record violations without signaling owned groups (default)",
    )
    watch_parser.add_argument(
        "--enforce",
        action="store_true",
        help="enable exact-receipt termination after root verifies the live bindings",
    )
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least one")
    if args.command == "watch" and args.report_only and args.enforce:
        parser.error("--report-only and --enforce cannot be combined")
    if args.command == "bridge":
        gru, _ = _read_panel(args.gru_panel, "wnn.l4-teacher-capacity-gru.v1")
        default_root = Path(str(gru["paths"]["runtime_root"])) / "dashboard/runs"
        while True:
            count = bridge_once(
                args.gru_panel,
                args.transformer_panel,
                args.dashboard_root or default_root,
                set(args.run),
            )
            print(
                f"bridged {count} new events into {args.dashboard_root or default_root}", flush=True
            )
            if args.once:
                return 0
            time.sleep(args.poll_seconds)
    while True:
        record = watch_once(
            args.gru_panel,
            args.transformer_panel,
            not args.enforce,
            args.dashboard_root,
        )
        print(
            json.dumps(
                {
                    key: record[key]
                    for key in (
                        "observed_at_utc",
                        "wave_threads",
                        "wave_rss_bytes",
                        "disk_free_bytes",
                        "output_bytes",
                        "violations",
                        "killed_owned_groups",
                    )
                    if key in record
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.once or record["violations"]:
            return 2 if record["violations"] else 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
