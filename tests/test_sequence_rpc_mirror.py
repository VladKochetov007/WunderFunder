from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

from scripts.collect_sequence_monitor import CHUNK_BYTES, collect_runs
from scripts.mirror_sequence_rpc import tick
from scripts.sequence_monitor_mirror_io import apply_snapshot
from scripts.sequence_rank0_view import update_rank0_view
from wnn.dashboard.reader import DashboardReader


def _event(step: int = 1) -> dict[str, object]:
    return {
        "schema": "wnn.monitor.v1",
        "kind": "evaluation",
        "step": step,
        "epoch": 1,
        "wall_time": "2020-01-01T00:00:00Z",
        "metrics": {"valid/wp": 0.1},
        "context": {},
    }


def _source(root: Path) -> Path:
    run = root / "cell" / "fit"
    run.mkdir(parents=True)
    metadata = {
        "source": "synthetic schema fixture",
        "config": {"batch_size": 1, "chunk_rows": 2000, "accumulate_chunks": 1},
        "provenance": {
            "world_size": 1,
            "data": {"input_schema": "raw112-v1", "fit_members": [["train", 1], ["train", 2]]},
        },
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata))
    (run / "monitor_events.jsonl").write_text(json.dumps(_event()) + "\n")
    (run / "monitor_evaluations.jsonl").write_bytes((run / "monitor_events.jsonl").read_bytes())
    return run


def test_rank0_adapter_preserves_raw_objective_terms_weights_and_semantics(
    tmp_path: Path,
) -> None:
    native = _source(tmp_path / "native")
    view = tmp_path / "view"
    view.mkdir()
    row = {
        "step": 5,
        "rank": 0,
        "total_loss": 1.4,
        "label_loss": 0.8,
        "kd_loss": 0.2,
        "hint_loss": 0.1,
        "objective_mode": "mixture_wmse_corr",
        "objective_aggregation": "update_mean_wmse_plus_microbatch_correlation",
        "label_loss_is_curvature_scaled": True,
        "objective_wmse": 0.7,
        "objective_correlation_loss": 0.5,
        "wmse_objective_weight": 1.0,
        "correlation_objective_weight": 0.1,
        "microbatch_correlation_surrogate_t0": 0.45,
        "microbatch_correlation_surrogate_t1": 0.55,
        "correlation_scope": "per_microbatch",
        "correlation_prediction_std_floor": 0.01,
    }
    (native / "rank0.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    update_rank0_view(native, view, {})

    event = json.loads((view / "monitor_rank0.jsonl").read_text(encoding="utf-8"))
    metrics = event["metrics"]
    context = event["context"]["native_objective"]
    assert metrics["train/objective_wmse_raw"] == 0.7
    assert metrics["train/objective_correlation_loss_raw"] == 0.5
    assert metrics["train/objective_weight/wmse"] == 1.0
    assert metrics["train/objective_weight/correlation"] == 0.1
    assert metrics["train/objective_correlation_surrogate/t0"] == 0.45
    assert metrics["train/objective_correlation_surrogate/t1"] == 0.55
    assert metrics["train/objective_weight/kd"] == 0.0
    assert metrics["train/native_label_objective"] == 0.8
    assert context["objective_mode"] == "mixture_wmse_corr"
    assert context["label_loss_is_curvature_scaled"] is True
    assert "raw WMSE component" in context["component_semantics"]["train/objective_wmse_raw"]
    assert "curvature-scaled" in context["component_semantics"]["train/native_label_objective"]


def test_delta_chunks_keep_complete_native_bytes_and_preserve_rotations(tmp_path: Path) -> None:
    source = _source(tmp_path / "remote")
    line = (json.dumps(_event()) + "\n").encode()
    journal = source / "monitor_events.jsonl"
    journal.write_bytes(line * 2000 + b'{"partial":')
    view, native = tmp_path / "view", tmp_path / "native"
    cursors: dict[str, object] = {}
    for _ in range(5):
        snapshot = collect_runs(str(tmp_path / "remote"), "l4", cursors)
        assert all(
            chunk["bytes"] <= CHUNK_BYTES for run in snapshot["runs"] for chunk in run["files"]
        )
        cursors = apply_snapshot(snapshot, view, native, cursors, "l4")
    mirrored = native / "l4-cell-fit" / "monitor_events.jsonl"
    assert mirrored.read_bytes() == line * 2000
    assert (view / "l4-cell-fit" / "run_metadata.json").read_bytes() == (
        source / "run_metadata.json"
    ).read_bytes()
    # The same inode is truncated on a safe resume; retain its old local prefix.
    replacement = (json.dumps(_event(999)) + "\n").encode()
    journal.write_bytes(replacement)
    snapshot = collect_runs(str(tmp_path / "remote"), "l4", cursors)
    updated = apply_snapshot(snapshot, view, native, cursors, "l4")
    assert mirrored.read_bytes() == replacement
    assert list(mirrored.parent.glob("monitor_events.jsonl.previous-*"))
    # An already-applied RPC response can be replayed without duplicate rows.
    apply_snapshot(snapshot, view, native, updated, "l4")
    assert mirrored.read_bytes() == replacement


def test_apply_rejects_corrupt_payload_and_cross_host_snapshot(tmp_path: Path) -> None:
    _source(tmp_path / "remote")
    snapshot = collect_runs(str(tmp_path / "remote"), "l4")
    with pytest.raises(ValueError, match="schema or host"):
        apply_snapshot(snapshot, tmp_path / "view", tmp_path / "native", {}, "b300")
    snapshot["runs"][0]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        apply_snapshot(snapshot, tmp_path / "view", tmp_path / "native", {}, "l4")


@pytest.mark.parametrize("labels", [("l4", "b300"), ("l4", "b300", "b300-opt")])
def test_panel_file_queue_keeps_source_times_and_rank_curves_out_of_liveness(
    tmp_path: Path,
    labels: tuple[str, ...],
) -> None:
    nodes = []
    for host in labels:
        remote = tmp_path / host / "remote"
        run = _source(remote)
        if host in ("b300", "b300-opt"):
            (run / "rank0.jsonl").write_text(
                json.dumps(
                    {
                        "step": 1,
                        "epoch": 0,
                        "rank": 0,
                        "total_loss": 1.2,
                        "label_loss": 1.2,
                        "learning_rates": {"adamw": 0.001},
                        "gradient_norm_preclip": 0.5,
                    }
                )
                + "\n"
            )
        # Both B300 recipe panels share the existing physical-host transport queue.
        rpc = tmp_path / ("b300" if host == "b300-opt" else host) / "rpc"
        (rpc / "requests").mkdir(parents=True, exist_ok=True)
        (rpc / "results").mkdir(exist_ok=True)
        nodes.append(
            {
                "host": host,
                "rpc_root": str(rpc),
                "run_root": str(remote),
                "collector_path": str(Path("scripts/collect_sequence_monitor.py").resolve()),
            }
        )
    config = {
        "dashboard_root": str(tmp_path / "view"),
        "native_root": str(tmp_path / "native"),
        "state_path": str(tmp_path / "state.json"),
        "hosts": nodes,
        "interval": 15,
    }
    state = {"schema": "wnn.sequence-monitor-mirror-state.v1", "hosts": {}}
    assert not tick(config, state, 1000)
    for node in nodes:
        pending = state["hosts"][node["host"]]["pending_id"]
        request_path = Path(node["rpc_root"]) / "requests" / f"{pending}.json"
        request = json.loads(request_path.read_bytes())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(request["code"], {})
        result = {
            "id": pending,
            "time": 1001,
            "output": f"WNN_RPC_BEGIN {pending}\n{output.getvalue()}WNN_RPC_END {pending} ok\n",
        }
        destination = Path(node["rpc_root"]) / "results" / f"{pending}.json"
        destination.write_text(json.dumps(result)[:20])
        tick(config, state, 1001)
        assert state["hosts"][node["host"]]["pending_id"] == pending
        destination.write_text(json.dumps(result))
    assert tick(config, state, 1002)
    assert all("pending_id" not in host for host in state["hosts"].values())
    assert list((tmp_path / "l4" / "rpc" / "requests").glob("*.json"))
    acknowledged = next((tmp_path / "l4" / "rpc" / "results").glob("*.json"))
    assert json.loads(acknowledged.read_bytes())["bridge_consumed"] is True
    reader = DashboardReader(tmp_path / "view")
    l4, b300 = reader.get_runs(["l4-cell-fit", "b300-cell-fit"])["runs"]
    assert l4["status"] == b300["status"] == "stale"
    assert "train/loss" in b300["metric_keys"]
    derived = next(event for event in b300["events"] if "train/loss" in event["metrics"])
    assert derived["step"] == 1 and derived["epoch"] == 0.05
    assert derived["context"]["wall_time_is_update_time"] is False
    assert l4["last_event_at"].startswith("2020-01-01")
    if "b300-opt" in labels:
        optimized = reader.get_runs(["b300-opt-cell-fit"])["runs"][0]
        assert "train/loss" in optimized["metric_keys"]
        assert (tmp_path / "native" / "b300-cell-fit").is_dir()
        assert (tmp_path / "native" / "b300-opt-cell-fit").is_dir()


def test_actual_optimizer_samples_survive_the_dense_train_stride(tmp_path: Path) -> None:
    """Keep sampled updates 1 and 50, with actual applied rates distinct from next LR."""
    from wnn.multistage_plot_data import read_training_events

    native = _source(tmp_path / "native")
    view = tmp_path / "view"
    view.mkdir()
    payload = {
        "sampled_update_step": 1,
        "sampling_interval": 50,
        "applied_learning_rates": {"adamw.head": 0.02},
        "groups": {
            "adamw.head": {
                "role": "adamw_matrix",
                "parameter_partition": "head",
                "delta_weight_over_pre_weight_norm": 0.01,
                "applied_lr": 0.02,
            }
        },
    }
    rows = []
    for step in (1, 2, 50):
        row = {"step": step, "total_loss": 0.5, "learning_rates": {"adamw.head": 0.03}}
        if step in (1, 50):
            row["optimizer_update_diagnostics"] = payload | {"sampled_update_step": step}
        rows.append(row)
    (native / "rank0.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    update_rank0_view(native, view, {})
    events, policy = read_training_events(view / "monitor_rank0.jsonl")
    assert [event["step"] for event in events] == [1, 50]
    assert policy["diagnostic_events_seen"] == 2
    assert all(event["kind"] == "diagnostics" for event in events)
    assert events[0]["metrics"]["optimizer_update/group/adamw.head/applied_lr"] == 0.02
    assert events[0]["metrics"]["optimizer/lr/adamw.head"] == 0.03
    assert (
        events[0]["context"]["optimizer_update_groups"]["adamw.head"]["parameter_partition"]
        == "head"
    )
