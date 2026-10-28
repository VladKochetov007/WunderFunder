from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from scripts.tl3_finite17_dashboard_core import (
        ADOPTED_NODE,
        SHA256,
        BridgeError,
        _append_evaluation,
        _append_event,
        _canonical_sha256,
        _cli_args,
        _context,
        _event,
        _flatten_gradients,
        _plan_and_split,
        _read_events,
        _read_json,
        _sha256_file,
        _view,
    )
except ModuleNotFoundError:
    from tl3_finite17_dashboard_core import (
        ADOPTED_NODE,
        SHA256,
        BridgeError,
        _append_evaluation,
        _append_event,
        _canonical_sha256,
        _cli_args,
        _context,
        _event,
        _flatten_gradients,
        _plan_and_split,
        _read_events,
        _read_json,
        _sha256_file,
        _view,
    )
try:
    from scripts.tl3_finite17_dashboard_evidence import (
        _fit_log_path,
        _parse_log,
        _verify_provenance,
    )
except ModuleNotFoundError:
    from tl3_finite17_dashboard_evidence import _fit_log_path, _parse_log, _verify_provenance
try:
    from scripts.tl3_finite17_dashboard_metrics import _cohort_event
except ModuleNotFoundError:
    from tl3_finite17_dashboard_metrics import _cohort_event
try:
    from scripts.tl3_finite17_dashboard_final import publish_final_view
except ModuleNotFoundError:
    from tl3_finite17_dashboard_final import publish_final_view
try:
    from scripts.tl3_finite17_dashboard_process import _owned_process, _resource, _telemetry
except ModuleNotFoundError:
    from tl3_finite17_dashboard_process import _owned_process, _resource, _telemetry


def publish_once(
    root: Path,
    state: Path,
    runs: Path,
    *,
    proc_root: Path = Path("/proc"),
    telemetry: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Publish one finite status/log snapshot to the 16 non-adopted dashboard views."""
    root = root.resolve(strict=True)
    state = state.expanduser().absolute()
    if state.is_symlink() or (state.exists() and not state.is_dir()):
        raise BridgeError("finite state path must be a regular directory when present")
    state = state.resolve(strict=False)
    runs = runs.expanduser().absolute()
    if runs.is_symlink() or not runs.is_dir() or runs.resolve(strict=True) != runs:
        raise BridgeError("dashboard runs root must be a resolved directory")
    loaded = _plan_and_split(root, state)
    if loaded is None:
        return {"status": "waiting_for_plan"}
    plan, status, train_groups, valid_groups = loaded
    nodes, states, tasks = plan["nodes"], status.get("nodes"), status.get("tasks")
    if not isinstance(states, dict) or not isinstance(tasks, dict):
        raise BridgeError("finite status omits nodes or task ownership records")
    telemetry_rows = _telemetry() if telemetry is None else telemetry
    added = 0
    for node in nodes:
        node_id, run = str(node["id"]), str(node["run"])
        if node_id == ADOPTED_NODE:
            continue
        args = node.get("args")
        if not isinstance(args, list) or node.get("args_sha256") != _canonical_sha256(args):
            raise BridgeError(f"{node_id}: frozen args hash changed")
        if (
            Path(str(node.get("run_directory"))).resolve(strict=False)
            != root / "artifacts" / "runs" / run
        ):
            raise BridgeError(f"{node_id}: plan run directory is not canonical")
        node_state = states.get(node_id)
        if not isinstance(node_state, dict):
            raise BridgeError(f"{node_id}: finite status omits node state")
        view = _view(runs, plan, node, node_state)
        events = _read_events(view / "monitor_events.jsonl")
        normalized = {"not_started": "queued", "complete": "completed"}.get(
            str(node_state.get("state")), str(node_state.get("state"))
        )
        code = {"queued": 0, "running": 1, "completed": 2, "failed": 3, "blocked": 4}.get(
            normalized
        )
        if code is None:
            raise BridgeError(f"{node_id}: unknown finite status {normalized}")
        added += int(
            _append_event(
                view / "monitor_events.jsonl",
                _event(
                    "status",
                    code,
                    0,
                    {f"run/{normalized}": 1},
                    {**_context(plan, node), "cell_state": normalized},
                ),
                events,
            )
        )
        task = tasks.get(f"fit:{node_id}")
        if not isinstance(task, dict):
            continue
        log_path = _fit_log_path(task, node, state, plan)
        expected_updates, epochs = int(node["expected_updates"]), int(node["epochs"])
        cli = _cli_args(args)
        sequence_count, batch_size, chunk_size = (
            int(cli["train_seqs"]),
            int(cli["batch"]),
            int(cli["chunk"]),
        )
        if (
            epochs < 1
            or expected_updates % epochs
            or batch_size < 1
            or chunk_size < 1
            or sequence_count < batch_size
            or 20_000 % chunk_size
        ):
            raise BridgeError(f"{node_id}: fit update geometry is not integral")
        batches = sequence_count // batch_size
        updates_epoch = expected_updates // epochs
        updates_batch = 20_000 // chunk_size
        if updates_epoch != batches * updates_batch:
            raise BridgeError(f"{node_id}: optimizer updates differ from loader batches")
        log = _parse_log(log_path)
        progress_epoch = 0.0
        for point in log["progress"]:
            if point["total"] != batches or not 0 <= point["batch"] < batches:
                continue
            step = point["epoch"] * updates_epoch + (point["batch"] + 1) * updates_batch
            progress_epoch = point["epoch"] + (point["batch"] + 1) / batches
            added += int(
                _append_event(
                    view / "monitor_events.jsonl",
                    _event(
                        "train_step",
                        step,
                        progress_epoch,
                        {
                            "train/loss": point["loss"],
                            "optimizer/lr": point["lr"],
                            "throughput/sequences_per_second": point["rate"],
                        },
                        {
                            **_context(plan, node),
                            "loss_definition": (
                                "legacy TrainingRun.batch.loss: composite per-window fit objective; "
                                "not validation WMSE"
                            ),
                        },
                    ),
                    events,
                )
            )
        process = _owned_process(task, node, root, proc_root)
        run_dir = root / "artifacts" / "runs" / run
        for row in log["valid"]:
            epoch = row["epoch"]
            if not 0 <= epoch < epochs:
                raise BridgeError(f"{node_id}: VALID epoch is outside frozen schedule")
            checkpoint_path, provenance_path = (
                run_dir / f"model_ep{epoch}.pt",
                run_dir / "provenance.json",
            )
            if (
                checkpoint_path.is_symlink()
                or not checkpoint_path.is_file()
                or provenance_path.is_symlink()
                or not provenance_path.is_file()
            ):
                continue
            provenance = _read_json(provenance_path)
            _verify_provenance(plan, node, provenance, train_groups, valid_groups)
            hashes = provenance.get("checkpoint_sha256")
            checkpoint_sha = hashes.get(checkpoint_path.name) if isinstance(hashes, dict) else None
            if not isinstance(checkpoint_sha, str) or not SHA256.fullmatch(checkpoint_sha):
                continue
            if _sha256_file(checkpoint_path) != checkpoint_sha:
                raise BridgeError(f"{node_id}: checkpoint file hash differs from provenance")
            step, metrics, context = _cohort_event(plan, node, row, train_groups, checkpoint_sha)
            evaluation = _event("evaluation", step, epoch + 1, metrics, context)
            evaluation_key = ("evaluation", epoch + 1, step)
            was_present = evaluation_key in events
            _append_evaluation(view, evaluation, events)
            added += int(not was_present)
            gradient = log["gradients"].get(epoch)
            gradient_metrics = _flatten_gradients(gradient) if isinstance(gradient, dict) else {}
            if gradient_metrics:
                diagnostic = _event(
                    "diagnostics",
                    step,
                    epoch + 1,
                    gradient_metrics,
                    {
                        **_context(plan, node, checkpoint_sha),
                        "source_epoch_index": epoch,
                        "gradient_timing": "after unscale and before clipping",
                    },
                )
                added += int(_append_event(view / "monitor_events.jsonl", diagnostic, events))
            progress_epoch = max(progress_epoch, float(epoch + 1))
        added += _resource(
            view, plan, node, task, progress_epoch, process, telemetry_rows, proc_root, events
        )
    final_view = publish_final_view(root, state, runs, plan, status, train_groups, valid_groups)
    if final_view is not None:
        added += int(final_view["events_added"])
    driver_status = status.get("status")
    terminal_pending = driver_status == "completed" and final_view is None
    return {
        "status": "running" if terminal_pending else driver_status,
        "driver_status": driver_status,
        "terminal_receipt_pending": terminal_pending,
        "plan_sha256": plan["sha256"],
        "nodes": 16,
        "final_blend_published": final_view is not None,
        "events_added": added,
    }
