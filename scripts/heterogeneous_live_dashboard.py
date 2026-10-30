"""Publish actual heterogeneous fits into the existing local native dashboard."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scripts.heterogeneous_dashboard_io import (
        W18,
        evaluation_plan,
        owned_process,
        read_nodes,
        requests,
        split_members,
        view_metadata,
    )
except ModuleNotFoundError:
    from heterogeneous_dashboard_io import (
        W18,
        evaluation_plan,
        owned_process,
        read_nodes,
        requests,
        split_members,
        view_metadata,
    )
from scripts.tl3_dashboard_bridge import (  # noqa: E402
    BridgeError,
    _append_evaluation,
    _append_event,
    _atomic_json,
    _event,
    _flatten_gradients,
    _read_events,
    _sha256_file,
)
from scripts.tl3_finite17_dashboard_core import _cli_args  # noqa: E402
from scripts.tl3_finite17_dashboard_evidence import _parse_log  # noqa: E402
from scripts.tl3_finite17_dashboard_metrics import _cohort_event  # noqa: E402
from scripts.tl3_finite17_dashboard_process import _telemetry  # noqa: E402

# Producer records contain heterogeneous JSON; no model or optimizer is imported.
JsonObject = dict[str, Any]


def context(node: JsonObject, binding: JsonObject | None) -> JsonObject:
    """Attach stable request/source declarations to report-only live observations."""
    return {
        "run": node["run"],
        "node_id": node["id"],
        "request_sha256": None if binding is None else binding["request_sha256"],
        "source_identity": None if binding is None else binding["request"].get("source_identity"),
        "observation_kind": "report_only_owned_command",
        "synthetic": False,
    }


def publish_resources(
    view: Path,
    node: JsonObject,
    binding: JsonObject,
    epoch: float,
    telemetry: tuple[list[JsonObject], list[JsonObject]],
    known: dict,
) -> int:
    """Record actual GPU/PID telemetry only after verifying the owned process instance."""
    process = owned_process(binding)
    if process is None:
        return 0
    pid, uuid = process
    apps, devices = telemetry
    app = next((item for item in apps if item["pid"] == pid and item["uuid"] == uuid), None)
    device = next((item for item in devices if item["uuid"] == uuid), None)
    if app is None or device is None:
        return 0
    metrics = {
        "resource/process_gpu_memory_used_mib": app["memory"],
        "resource/gpu_memory_used_mib": device["memory"],
        "resource/gpu_utilization_percent": device["util"],
    }
    try:
        pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
        metrics["resource/process_rss_bytes"] = pages * os.sysconf("SC_PAGE_SIZE")
        metrics["resource/host_load_1m"] = os.getloadavg()[0]
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    step = max((step for kind, _, step in known if kind == "resources"), default=-1) + 1
    event = _event(
        "resources",
        step,
        epoch,
        metrics,
        {
            **context(node, binding),
            "pid": pid,
            "pid_start_ticks": str(binding["status"]["start_ticks"]),
            "gpu_uuid": uuid,
            "process_identity_verified": True,
            "gpu_process_pid_uuid_verified": True,
        },
    )
    return int(_append_event(view / "monitor_events.jsonl", event, known))


def publish_run(
    root: Path,
    legacy: Path,
    runs: Path,
    node: JsonObject,
    binding: JsonObject | None,
    train: list[int],
    valid: list[int],
    telemetry: tuple[list[JsonObject], list[JsonObject]],
    verified_sources: set[tuple[str, str]],
) -> int:
    """Publish one real fit's status/progress/paired epoch metrics without recomputation."""
    state = "queued" if binding is None else binding["status"].get("status", "requested")
    state = {"complete": "completed"}.get(state, state)
    if state == "running" and owned_process(binding) is None:
        state = "awaiting_terminal_receipt"
    view = view_metadata(runs, root, node, binding, state)
    known = _read_events(view / "monitor_events.jsonl")
    prior_statuses = [value for (kind, _, _), value in known.items() if kind == "status"]
    added = 0
    event_context = {**context(node, binding), "cell_state": state}
    if not prior_statuses or prior_statuses[-1]["context"] != event_context:
        step = max((step for kind, _, step in known if kind == "status"), default=-1) + 1
        added += int(
            _append_event(
                view / "monitor_events.jsonl",
                _event("status", step, 0, {f"run/{state}": 1}, event_context),
                known,
            )
        )
    if binding is None or not binding["status"]:
        return added
    run_root = legacy if node["run"] == W18 else root
    run_dir = run_root / "artifacts/runs" / node["run"]
    log_path = Path(binding["request"]["log"])
    if log_path.is_symlink() or log_path.resolve().parent != (root / "logs").resolve():
        raise BridgeError("fit log must be a regular file under the owned campaign logs")
    log = _parse_log(log_path)
    cli = _cli_args(node["args"])
    batches, windows = int(cli["train_seqs"]) // int(cli["batch"]), 20_000 // int(cli["chunk"])
    updates_epoch, progress_epoch = batches * windows, 0.0
    if updates_epoch * node["epochs"] != node["expected_updates"]:
        raise BridgeError("actual observer recipe update geometry differs")
    for point in log["progress"]:
        if (
            point["total"] != batches
            or not 0 <= point["batch"] < batches
            or not 0 <= point["epoch"] < node["epochs"]
        ):
            raise BridgeError("fit log progress differs from its frozen update geometry")
        step = point["epoch"] * updates_epoch + (point["batch"] + 1) * windows
        progress_epoch = point["epoch"] + (point["batch"] + 1) / batches
        event = _event(
            "train_step",
            step,
            progress_epoch,
            {
                "train/loss": point["loss"],
                "optimizer/lr": point["lr"],
                "throughput/sequences_per_second": point["rate"],
            },
            {
                **context(node, binding),
                "loss_definition": "composite per-window fit objective; not validation WMSE",
            },
        )
        added += int(_append_event(view / "monitor_events.jsonl", event, known))
    validated = evaluation_plan(binding, node, run_dir, train, valid, verified_sources)
    if validated is not None:
        plan, provenance = validated
        for row in log["valid"]:
            epoch = row["epoch"]
            checkpoint = run_dir / f"model_ep{epoch}.pt"
            sha = provenance.get("checkpoint_sha256", {}).get(checkpoint.name)
            if not isinstance(sha, str) or not checkpoint.is_file():
                continue
            if _sha256_file(checkpoint) != sha:
                raise BridgeError("epoch checkpoint differs from actual producer provenance")
            step, metrics, evaluation_context = _cohort_event(plan, node, row, train, sha)
            event = _event(
                "evaluation",
                step,
                epoch + 1,
                metrics,
                {**evaluation_context, "observer_schema": "wnn.heterogeneous-native-dashboard.v1"},
            )
            absent = ("evaluation", epoch + 1, step) not in known
            _append_evaluation(view, event, known)
            added += int(absent)
            gradient = log["gradients"].get(epoch)
            if isinstance(gradient, dict) and (values := _flatten_gradients(gradient)):
                added += int(
                    _append_event(
                        view / "monitor_events.jsonl",
                        _event(
                            "diagnostics",
                            step,
                            epoch + 1,
                            values,
                            {
                                **context(node, binding),
                                "gradient_timing": "after unscale and before clipping",
                            },
                        ),
                        known,
                    )
                )
            progress_epoch = max(progress_epoch, float(epoch + 1))
    added += publish_resources(view, node, binding, progress_epoch, telemetry, known)
    return added


def publish_once(
    root: Path, legacy: Path, runs: Path, verified_sources: set[tuple[str, str]]
) -> JsonObject:
    """Refresh the seven native views from actual request/status/log artifacts."""
    nodes = read_nodes(root)
    train, valid = split_members(nodes)
    bindings = requests(root, nodes)
    telemetry = _telemetry()
    counts, errors = {}, {}
    for node in nodes:
        try:
            counts[node["run"]] = publish_run(
                root,
                legacy,
                runs,
                node,
                bindings.get(node["run"]),
                train,
                valid,
                telemetry,
                verified_sources,
            )
        except (BridgeError, OSError, ValueError, KeyError, TypeError) as error:
            errors[node["run"]] = f"{type(error).__name__}: {error}"
    result = {
        "schema": "wnn.heterogeneous-dashboard-observation.v1",
        "events_added": counts,
        "errors": errors,
        "views": 7,
        "report_only": True,
        "new_fits_started": 0,
    }
    _atomic_json(root / "dashboard-observer-latest.json", result)
    return result


def main() -> None:
    """Observe bounded real training into the already-running dashboard; start no service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--dashboard-runs-root", type=Path, required=True)
    parser.add_argument("--poll", type=int, default=15)
    parser.add_argument("--max-runtime", type=int, default=21_600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.poll <= 60 or args.max_runtime < 1:
        parser.error("poll must be1..60 seconds and lifetime positive")
    root, legacy, runs = (
        args.root.resolve(strict=True),
        args.legacy_root.resolve(strict=True),
        args.dashboard_runs_root.resolve(strict=True),
    )
    if not runs.is_dir() or args.dashboard_runs_root.is_symlink():
        parser.error("native dashboard runs root must be an existing real directory")
    started, verified_sources = time.monotonic(), set()
    while time.monotonic() - started < args.max_runtime:
        try:
            result = publish_once(root, legacy, runs, verified_sources)
        except (BridgeError, OSError, ValueError, KeyError, TypeError) as error:
            result = {
                "schema": "wnn.heterogeneous-dashboard-observation.v1",
                "errors": {"observer": f"{type(error).__name__}: {error}"},
                "report_only": True,
                "new_fits_started": 0,
            }
            _atomic_json(root / "dashboard-observer-latest.json", result)
        print(json.dumps(result), flush=True)
        if args.once:
            raise SystemExit(1 if result.get("errors") else 0)
        time.sleep(min(args.poll, max(0, args.max_runtime - (time.monotonic() - started))))


if __name__ == "__main__":
    main()
