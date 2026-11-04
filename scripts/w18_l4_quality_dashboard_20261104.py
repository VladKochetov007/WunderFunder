"""Bridge one frozen W18 L4 cohort into the existing local run dashboard."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]

from scripts import tl3_dashboard_bridge as bridge  # noqa: E402
from scripts.w18_l4_quality_20261104.contract import (  # noqa: E402
    CohortPanel,
    load_panel,
    read_object,
)
from scripts.w18_l4_quality_20261104.runtime import cell_paths  # noqa: E402

MAX_RESOURCES_BYTES = 16 << 20
# Legacy log parser values are heterogeneous JSON without a narrower declared contract.


def _number(record: Mapping[str, object], key: str) -> int | float | None:
    value = record.get(key)
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (isinstance(value, int) or math.isfinite(value))
    ):
        return value
    return None


def _read_resources(path: Path, cell_id: str) -> list[dict[str, object]]:
    if path.is_symlink() or not path.exists():
        return []
    if not path.is_file() or path.stat().st_size > MAX_RESOURCES_BYTES:
        raise bridge.BridgeError("cohort resource receipt is unsafe or exceeds 16 MiB")
    raw = path.read_bytes()
    raw = raw[: raw.rfind(b"\n") + 1] if raw and not raw.endswith(b"\n") else raw
    records: list[dict[str, object]] = []
    for line in raw.splitlines():
        try:
            value: object = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise bridge.BridgeError("cohort resource receipt contains invalid JSON") from error
        if not isinstance(value, dict):
            raise bridge.BridgeError("cohort resource receipt record is not an object")
        if value.get("phase") == "fit" and value.get("cell_id") == cell_id:
            records.append(cast(dict[str, object], value))
    return records


def _latest_epoch(parsed: dict[str, Any]) -> float:
    progress = parsed["progress"]
    if not progress:
        return 0.0
    point = progress[-1]
    total = max(1, int(point["total"]))
    return int(point["epoch"]) + (int(point["batch"]) + 1) / total


def _ensure_view(root: Path, run_name: str, panel_sha: str, cell_id: str) -> Path:
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise bridge.BridgeError("dashboard runs root must be an existing resolved directory")
    view = root / run_name
    if view.parent != root or view.is_symlink():
        raise bridge.BridgeError("dashboard view must be a direct run-name child")
    if not view.exists():
        view.mkdir(mode=0o700)
    elif not view.is_dir():
        raise bridge.BridgeError("dashboard view path is not a directory")
    metadata_path = view / "run_metadata.json"
    if metadata_path.exists():
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise bridge.BridgeError("dashboard metadata path is unsafe")
        prior = read_object(metadata_path)
        provenance = prior.get("provenance")
        if not isinstance(provenance, dict) or (
            provenance.get("panel_sha256") != panel_sha or provenance.get("cell_id") != cell_id
        ):
            raise bridge.BridgeError("dashboard view belongs to a different frozen panel cell")
    for filename in ("monitor_events.jsonl", "monitor_evaluations.jsonl"):
        path = view / filename
        if path.is_symlink():
            raise bridge.BridgeError(f"dashboard stream is a symlink: {filename}")
        if not path.exists():
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
    return view


def _run_status(
    receipt_path: Path, log_path: Path, panel_sha: str, cell_id: str
) -> tuple[str, dict[str, object]]:
    if not receipt_path.exists():
        return ("running" if log_path.exists() and log_path.stat().st_size else "not_started", {})
    receipt = read_object(receipt_path)
    owner = receipt.get("owner")
    if (
        not isinstance(owner, dict)
        or owner.get("cohort_panel_sha256") != panel_sha
        or owner.get("cell_id") != cell_id
    ):
        raise bridge.BridgeError("fit receipt is not owned by this frozen panel cell")
    status = receipt.get("status")
    if not isinstance(status, str):
        raise bridge.BridgeError("fit receipt has no status")
    return status, receipt


def _metadata(
    panel: CohortPanel, cell: dict[str, object], status: str, receipt: dict[str, object]
) -> dict[str, object]:
    identity = cast(dict[str, object], panel.value["identity"])
    args = cast(list[str], cell["train_args"])
    epochs = int(args[args.index("--epochs") + 1])
    return {
        "source": "W18 L4 matched quality screen; canonical modular trainer; full official validation",
        "config": {
            "run": cell["run_name"],
            "cell_id": cell["cell_id"],
            "configuration_id": cell["configuration_id"],
            "role": cell["role"],
            "hypothesis_id": cell.get("hypothesis_id"),
            "seed": cell["seed"],
            "status": status,
            "checkpoint_policy": "fixed",
            "fixed_endpoint_epoch_index": epochs - 1,
            "fixed_endpoint_completed_epoch": epochs,
            "precision": "fp32",
            "effective_args": args,
            "gpu_uuid": cell["gpu_uuid"],
        },
        "provenance": {
            "dashboard_adapter": "w18_l4_quality_dashboard_20261104.py",
            "panel_sha256": panel.sha256,
            "cell_id": cell["cell_id"],
            "source_binding_sha256": identity["source_binding_sha256"],
            "data_binding_sha256": identity["data_binding_sha256"],
            "split_sha256": identity["split_sha256"],
            "fit_members_sha256": identity["fit_members_sha256"],
            "valid_members_sha256": identity["valid_members_sha256"],
            "baseline_ancestry_sha256": identity["baseline_ancestry_sha256"],
            "fit_receipt_status": receipt.get("status"),
            "fit_child": receipt.get("child", {}),
            "validation_policy": {
                "primary_valid_row_mask": "is_scored",
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


def _append_training_events(
    view: Path, parsed: dict[str, Any], cell: dict[str, object], panel_sha: str
) -> int:
    events_path = view / "monitor_events.jsonl"
    known = bridge._read_events(events_path)
    added = 0
    for point in parsed["progress"]:
        total = max(1, int(point["total"]))
        epoch_index, batch_index = int(point["epoch"]), int(point["batch"])
        if batch_index >= total:
            continue
        step = epoch_index * total + batch_index + 1
        event = bridge._event(
            "train_step",
            step,
            epoch_index + (batch_index + 1) / total,
            {
                "train/loss": point["loss"],
                "train/objective_batch": point["loss"],
                "optimizer/lr": point["lr"],
                "throughput/sequences_per_second": point["rate"],
                "train/epoch_fraction": (batch_index + 1) / total,
            },
            {
                "run": cell["run_name"],
                "cell_id": cell["cell_id"],
                "panel_sha256": panel_sha,
                "objective_definition": "legacy combined training batch scalar; not validation WMSE",
                "step_unit": "logged training batch ordinal",
                "source_epoch_index": epoch_index,
            },
        )
        added += int(bridge._append_event(events_path, event, known))

    valid_records = parsed["valid"]
    args = cast(list[str], cell["train_args"])
    fixed_epoch = int(args[args.index("--epochs") + 1]) - 1
    for epoch_index, valid in sorted(valid_records.items()):
        prior_valid = [epoch for epoch in valid_records if epoch <= epoch_index]
        best_epoch = max(
            prior_valid,
            key=lambda epoch: valid_records[epoch].get("weighted_pearson", float("-inf")),
        )
        completed_epoch = epoch_index + 1
        update = valid.get("update_diagnostics", {})
        cumulative = update.get("cumulative", {}) if isinstance(update, dict) else {}
        step = cumulative.get("optimizer_attempts") if isinstance(cumulative, dict) else None
        if type(step) is not int:
            step = completed_epoch
        metrics: dict[str, int | float] = {}
        for source, target in (
            ("weighted_pearson", "valid/score"),
            ("weighted_pearson", "valid/wp"),
            ("global_t0", "valid/score/t0"),
            ("global_t1", "valid/score/t1"),
            ("valid_wmse", "valid/loss"),
            ("valid_wmse", "valid/wmse"),
            ("train_eval_wp", "train_eval/score"),
            ("train_eval_wp", "train_eval/wp"),
            ("train_eval_wp_t0", "train_eval/score/t0"),
            ("train_eval_wp_t1", "train_eval/score/t1"),
            ("train_eval_wmse", "train_eval/loss"),
            ("train_eval_wmse", "train_eval/wmse"),
            ("train_eval_selected_rows", "train_eval/selected_rows"),
        ):
            value = _number(valid, source)
            if value is not None:
                metrics[target] = value
        for source, target in (
            ("valid_wmse_t0", "valid/wmse/t0"),
            ("valid_wmse_t1", "valid/wmse/t1"),
            ("train_eval_wmse_t0", "train_eval/wmse/t0"),
            ("train_eval_wmse_t1", "train_eval/wmse/t1"),
        ):
            value = _number(valid, source)
            if value is not None:
                metrics[target] = value
        metrics.update(
            {
                "valid/scored_rows": 3_528_273,
                "valid/sequence_count": 1873,
                "train_eval/sequence_count": 32,
                "selection/fixed_endpoint_epoch": fixed_epoch + 1,
                "selection/fixed_endpoint_selected_here": int(epoch_index == fixed_epoch),
                "selection/best_observed_epoch": (best_epoch + 1) if best_epoch is not None else 0,
                "selection/best_observed_here_as_of_log": int(epoch_index == best_epoch),
            }
        )
        summary = valid.get("fit_summary")
        if isinstance(summary, dict):
            objective = _number(summary, "objective_mean")
            if objective is not None:
                metrics["train/objective_window_mean"] = objective
            rates = summary.get("learning_rate_by_group", [])
            if isinstance(rates, list):
                for item in rates:
                    if isinstance(item, dict) and type(item.get("group")) is int:
                        rate = _number(item, "mean")
                        if rate is not None:
                            metrics[f"optimizer/lr/group_{item['group']}/mean"] = rate
        event = bridge._event(
            "evaluation",
            step,
            completed_epoch,
            metrics,
            {
                "run": cell["run_name"],
                "cell_id": cell["cell_id"],
                "panel_sha256": panel_sha,
                "source_epoch_index": epoch_index,
                "valid": {
                    "row_mask": "is_scored",
                    "source": "official full validation, scored rows",
                    "sequence_count": 1873,
                    "scored_rows": 3_528_273,
                },
                "train_eval": {
                    "row_mask": "all_after_warmup",
                    "source": "train only",
                    "sequence_count": 32,
                    "comparable_to_valid": False,
                },
                "selection": {
                    "checkpoint_policy": "fixed",
                    "fixed_endpoint_epoch_index": fixed_epoch,
                    "fixed_endpoint_completed_epoch": fixed_epoch + 1,
                    "best_observed_epoch_index_as_of_log": best_epoch,
                    "best_observed_final_until_more_VALID_records": True,
                },
            },
        )
        evaluation_key = ("evaluation", completed_epoch, step)
        evaluation_was_new = evaluation_key not in known
        bridge._append_evaluation(view, event, known)
        added += int(evaluation_was_new)
        gradients = parsed["gradients"].get(epoch_index)
        if isinstance(gradients, dict):
            grad_metrics = bridge._flatten_gradients(gradients)
            if grad_metrics:
                grad_event = bridge._event(
                    "diagnostics",
                    step,
                    completed_epoch,
                    grad_metrics,
                    {
                        "run": cell["run_name"],
                        "cell_id": cell["cell_id"],
                        "panel_sha256": panel_sha,
                        "gradient_timing": "after unscale and before clipping",
                        "source_epoch_index": epoch_index,
                    },
                )
                added += int(bridge._append_event(events_path, grad_event, known))
    return added


def _append_resource_events(
    view: Path,
    records: list[dict[str, object]],
    cell: dict[str, object],
    panel_sha: str,
    epoch: float,
) -> int:
    events_path = view / "monitor_events.jsonl"
    known = bridge._read_events(events_path)
    added = 0
    for sample_index, record in enumerate(records, start=1):
        gpu = record.get("gpu")
        if not isinstance(gpu, dict) or gpu.get("uuid") != cell["gpu_uuid"]:
            raise bridge.BridgeError("resource snapshot GPU identity differs from its frozen cell")
        metrics: dict[str, int | float] = {}
        free = _number(gpu, "free_mib")
        total = _number(gpu, "total_mib")
        utilization = _number(gpu, "utilization_percent")
        if free is not None:
            metrics["gpu/memory_free_mib"] = free
        if free is not None and total is not None:
            metrics["gpu/memory_used_mib"] = total - free
        if utilization is not None:
            metrics["gpu/utilization_percent"] = utilization
        rss = _number(record, "cohort_owned_rss_bytes")
        disk_free = _number(record, "disk_free_bytes")
        output = _number(record, "output_bytes")
        if rss is not None:
            metrics["cpu/cohort_rss_bytes"] = rss
        if disk_free is not None:
            metrics["disk/free_bytes"] = disk_free
        if output is not None:
            metrics["disk/output_bytes"] = output
        prior = next(
            (
                event
                for (kind, _old_epoch, old_step), event in known.items()
                if kind == "resources" and old_step == sample_index
            ),
            None,
        )
        sample_epoch = prior["epoch"] if prior is not None else epoch
        event = bridge._event(
            "resources",
            sample_index,
            sample_epoch,
            metrics,
            {
                "run": cell["run_name"],
                "cell_id": cell["cell_id"],
                "panel_sha256": panel_sha,
                "gpu_uuid": cell["gpu_uuid"],
                "scope": "assigned-device snapshot; process-group RSS is cohort aggregate",
                "source_timestamp_available": False,
                "epoch_alignment": "latest logged training progress at bridge observation",
            },
        )
        added += int(bridge._append_event(events_path, event, known))
    return added


def update_dashboard(panel: CohortPanel, dashboard_root: Path, selected_runs: set[str]) -> int:
    """Import new events from cohort receipts and trainer logs without running scoring."""
    dashboard_root = dashboard_root.expanduser().absolute()
    if not dashboard_root.exists():
        dashboard_root.mkdir(parents=True, mode=0o700)
    paths = cast(dict[str, object], panel.value["paths"])
    runtime_root = Path(str(paths["runtime_root"]))
    appended = 0
    for raw_cell in cast(list[dict[str, object]], panel.value["cells"]):
        cell = raw_cell
        run_name, cell_id = str(cell["run_name"]), str(cell["cell_id"])
        if selected_runs and run_name not in selected_runs and cell_id not in selected_runs:
            continue
        locations = cell_paths(panel, cell)
        status, receipt = _run_status(
            locations["fit_receipt"], locations["fit_log"], panel.sha256, cell_id
        )
        view = _ensure_view(dashboard_root, run_name, panel.sha256, cell_id)
        metadata = _metadata(panel, cell, status, receipt)
        metadata_path = view / "run_metadata.json"
        encoded = json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if not metadata_path.exists() or metadata_path.read_text(encoding="utf-8") != encoded:
            bridge._atomic_json(metadata_path, metadata)
        log_records = bridge.parse_legacy_log(bridge._read_log(locations["fit_log"]))
        appended += _append_training_events(view, log_records, cell, panel.sha256)
        resource_path = runtime_root / "receipts" / "resources.jsonl"
        resources = _read_resources(resource_path, cell_id)
        appended += _append_resource_events(
            view, resources, cell, panel.sha256, _latest_epoch(log_records)
        )
        if status in {"completed", "failed", "timed_out", "progress_callback_failed"}:
            events_path = view / "monitor_events.jsonl"
            known = bridge._read_events(events_path)
            train_args = cast(list[str], cell["train_args"])
            fixed_epoch = int(train_args[train_args.index("--epochs") + 1])
            parsed_valid = log_records["valid"]
            best_epoch = (
                max(
                    parsed_valid,
                    key=lambda epoch: parsed_valid[epoch].get("weighted_pearson", float("-inf")),
                )
                if parsed_valid
                else None
            )
            terminal = bridge._event(
                "status",
                1,
                0,
                {
                    "run/completed": int(status == "completed"),
                    "run/failed": int(status != "completed"),
                    "selection/fixed_endpoint_epoch": fixed_epoch,
                    "selection/best_observed_epoch": (best_epoch + 1)
                    if best_epoch is not None
                    else 0,
                },
                {
                    "run": run_name,
                    "cell_id": cell_id,
                    "panel_sha256": panel.sha256,
                    "lifecycle_status": status,
                    "selected_fixed_epoch_index_from_log": log_records["selected_fixed"],
                    "best_observed_epoch_index": best_epoch,
                    "valid_records_seen": len(parsed_valid),
                },
            )
            appended += int(bridge._append_event(events_path, terminal, known))
    return appended


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dashboard-root", type=Path)
    parser.add_argument(
        "--run", action="append", default=[], help="run name or cell ID; default is all eight"
    )
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--once", action="store_true", help="bridge one snapshot and exit")
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least one")
    panel = load_panel(args.panel)
    runtime_root = Path(str(panel.value["paths"]["runtime_root"]))
    dashboard_root = args.dashboard_root or runtime_root / "live"
    selected = set(args.run)
    known_names = {
        str(value)
        for cell in cast(list[dict[str, object]], panel.value["cells"])
        for value in (cell["run_name"], cell["cell_id"])
    }
    if selected - known_names:
        parser.error(f"unknown --run selector(s): {', '.join(sorted(selected - known_names))}")
    try:
        while True:
            count = update_dashboard(panel, dashboard_root, selected)
            print(f"bridged {count} new events to {dashboard_root}", flush=True)
            if args.once:
                return 0
            time.sleep(args.poll_seconds)
    except (bridge.BridgeError, OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"W18 dashboard bridge stopped: {type(error).__name__}: {error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
