from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from scripts.tl3_finite17_dashboard_core import ADOPTED_NODE, BridgeError
    from scripts.tl3_finite17_results_validation import _verify_run
except ModuleNotFoundError:
    from tl3_finite17_dashboard_core import ADOPTED_NODE, BridgeError
    from tl3_finite17_results_validation import _verify_run

TERMINAL_STATUSES = {
    "completed",
    "failed",
    "interrupted",
    "budget_exhausted",
    "finished_with_failures",
    "disk_capacity_blocked",
}


def build_snapshot(
    root: Path,
    plan: dict[str, Any],
    status: dict[str, Any],
    train: list[int],
    valid: list[int],
) -> dict[str, Any]:
    """Render only statuses and metrics backed by driver-verified run artifacts."""
    node_states = status.get("nodes")
    nodes = plan.get("nodes")
    fits_used = status.get("fits_used")
    if (
        not isinstance(node_states, dict)
        or not isinstance(nodes, list)
        or len(nodes) != 17
        or type(fits_used) is not int
        or not 1 <= fits_used <= 17
    ):
        raise BridgeError("finite driver status omits its node states")
    cells = []
    node_ids = [node.get("id") for node in nodes if isinstance(node, dict)]
    if len(node_ids) != 17 or len(set(node_ids)) != 17 or ADOPTED_NODE not in node_ids:
        raise BridgeError("finite results plan is not the 17-node graph")
    for node in nodes:
        node_id = str(node["id"])
        state = node_states.get(node_id)
        if not isinstance(state, dict):
            raise BridgeError(f"{node_id}: finite driver status omits the node")
        verified = _verify_run(root, plan, node, state, train, valid)
        cells.append(
            {
                "id": node_id,
                "run": node["run"],
                "state": state.get("state"),
                "phase": state.get("phase"),
                "fit_verified": verified is not None,
                "metrics": verified,
            }
        )
    final_blend = None
    if status.get("status") == "completed":
        try:
            from scripts.tl3_finite17_dashboard_final_validation import validate_final_score
        except ModuleNotFoundError:
            from tl3_finite17_dashboard_final_validation import validate_final_score
        state_dir = Path(str(plan["state_dir"])).resolve(strict=True)
        final_report = validate_final_score(root, state_dir, plan, status, train, valid)
        if final_report is None:
            raise BridgeError("completed finite DAG has no verified fixed blend score")
        blend = final_report["blend"]
        final_blend = {
            "wp": float(blend["wp"]),
            "t0": float(blend["t0"]),
            "t1": float(blend["t1"]),
            "selected_rows": int(blend["selected_rows"]),
            "weights": final_report["run_weights"],
            "clean_gru_reference_wp": final_report["clean_gru_reference_wp"],
            "delta_to_clean_gru": final_report["delta_to_clean_gru"],
            "score_sha256": final_report["score_sha256"],
        }
    counts = {
        name: sum(cell["state"] == name for cell in cells)
        for name in ("not_started", "running", "complete", "failed", "blocked")
    }
    return {
        "schema": "wnn.tl3-finite17-results.v1",
        "plan_sha256": plan["sha256"],
        "source_sha256": plan["source_identity"]["training_source_sha256"],
        "split_manifest_sha256": plan["split"]["split_manifest_sha256"],
        "driver_status": status.get("status"),
        "fits_used": fits_used,
        "planned_fits": len(nodes),
        "fit_states": counts,
        "nodes": cells,
        "final_blend": final_blend,
    }


def format_update(snapshot: dict[str, Any], title: str, only_node: str | None = None) -> str:
    """Format verified metrics without implying a public score or a terminal result."""
    lines = [
        f"## TL3 chronological finite17 — {title}",
        f"Driver `{snapshot['driver_status']}` | fits used {snapshot['fits_used']}/{snapshot['planned_fits']}",
        f"Plan `{snapshot['plan_sha256']}` | planned trainer source `{snapshot['source_sha256']}` | split `{snapshot['split_manifest_sha256']}`",
        "Primary validation is the full official valid membership (`is_scored`). Train-eval is a separate training-only `all_after_warmup` diagnostic.",
        "",
        "| Fit | State / phase | Verified epochs | Fit source SHA | Latest official-valid WP (t0 / t1) | Best observed WP | Train-eval WP |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for node in snapshot["nodes"]:
        if only_node is not None and node["id"] != only_node:
            continue
        metrics = node["metrics"] if isinstance(node["metrics"], dict) else {}
        valid_wp = "-"
        if "latest_valid_wp" in metrics:
            valid_wp = (
                f"{metrics['latest_valid_wp']:.6f} "
                f"({metrics['latest_valid_wp_t0']:.6f} / {metrics['latest_valid_wp_t1']:.6f})"
            )
        train_wp = "-"
        if "train_eval_wp" in metrics:
            train_wp = (
                f"{metrics['train_eval_wp']:.6f} "
                f"({metrics['train_eval_wp_t0']:.6f} / {metrics['train_eval_wp_t1']:.6f})"
            )
        best_wp = (
            f"{metrics['best_observed_valid_wp']:.6f} (ep {metrics['best_observed_valid_epoch']})"
            if "best_observed_valid_wp" in metrics
            else "-"
        )
        epoch = metrics.get("latest_epoch", "-")
        fit_source = str(metrics.get("fit_source_sha256", "-"))[:12]
        phase = node.get("phase")
        state = node.get("state")
        label = f"{state} / {phase}" if phase else str(state)
        lines.append(
            f"| `{node['id']}` | {label} | {epoch} | `{fit_source}` | {valid_wp} | {best_wp} | {train_wp} |"
        )
    final_blend = snapshot.get("final_blend")
    if only_node is None and isinstance(final_blend, dict):
        weights = ", ".join(
            f"`{run}` {weight:.2f}" for run, weight in final_blend["weights"].items()
        )
        lines.extend(
            [
                "",
                "### Terminal fixed blend on official valid",
                (
                    f"WP {final_blend['wp']:.6f} (t0 {final_blend['t0']:.6f}, "
                    f"t1 {final_blend['t1']:.6f}) over {final_blend['selected_rows']:,} "
                    "`is_scored` rows."
                ),
                f"Frozen weights: {weights}.",
                (
                    f"Delta to clean GRU reference ({final_blend['clean_gru_reference_wp']:.6f}): "
                    f"{final_blend['delta_to_clean_gru']:+.6f}."
                ),
                f"Score SHA `{final_blend['score_sha256']}`.",
            ]
        )
    lines.append("")
    lines.append(
        f"State counts: {snapshot['fit_states']}. These are run diagnostics, not a leaderboard score."
    )
    return "\n".join(lines)
