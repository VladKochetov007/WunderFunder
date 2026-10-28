from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from scripts.tl3_finite17_dashboard_core import (
        VALID_GROUPS,
        VALID_ROWS,
        BridgeError,
        _append_evaluation,
        _append_event,
        _atomic_json,
        _canonical_sha256,
        _event,
        _read_events,
        _read_json,
    )
    from scripts.tl3_finite17_dashboard_final_validation import validate_final_score
except ModuleNotFoundError:
    from tl3_finite17_dashboard_core import (
        VALID_GROUPS,
        VALID_ROWS,
        BridgeError,
        _append_evaluation,
        _append_event,
        _atomic_json,
        _canonical_sha256,
        _event,
        _read_events,
        _read_json,
    )
    from tl3_finite17_dashboard_final_validation import validate_final_score

FINAL_VIEW = "tl3c_final_fixed_blend"


def publish_final_view(
    root: Path,
    state_dir: Path,
    runs_dir: Path,
    plan: dict[str, Any],
    status: dict[str, Any],
    train: list[int],
    valid: list[int],
) -> dict[str, Any] | None:
    """Publish one terminal, full official-valid fixed-blend score with no training series."""
    verified = validate_final_score(root, state_dir, plan, status, train, valid)
    if verified is None:
        return None
    score = verified["score"]
    score_sha = verified["score_sha256"]
    run_weights = verified["run_weights"]
    verified_members = verified["members"]
    blend = verified["blend"]
    view = runs_dir / FINAL_VIEW
    if view.is_symlink() or (
        view.exists() and (not view.is_dir() or view.resolve(strict=True).parent != runs_dir)
    ):
        raise BridgeError("fixed blend dashboard view is not an owned direct child")
    view.mkdir(mode=0o700, exist_ok=True)
    run_identity = _canonical_sha256(
        {
            "plan_sha256": plan["sha256"],
            "score_sha256": score_sha,
            "source_sha256": plan["source_identity"]["training_source_sha256"],
            "split_sha256": plan["split"]["split_manifest_sha256"],
            "members": verified_members,
        }
    )
    marker = {
        "schema": "wnn.tl3-finite17-final-dashboard.v1",
        "plan_sha256": plan["sha256"],
        "score_sha256": score_sha,
        "run_identity_sha256": run_identity,
    }
    metadata_path = view / "run_metadata.json"
    if metadata_path.is_symlink():
        raise BridgeError("fixed blend dashboard metadata cannot be a symlink")
    if metadata_path.exists():
        existing = _read_json(metadata_path, 1 << 20)
        provenance = existing.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("native_final_score") != marker:
            raise BridgeError("fixed blend dashboard view belongs to another run")
    _atomic_json(
        metadata_path,
        {
            "source": {"campaign": "TL3 chronological", "node_id": "fixed_final_blend"},
            "config": {
                "run": FINAL_VIEW,
                "node_id": "fixed_final_blend",
                "cell_state": "completed",
                "phase": "terminal_official_valid_score",
                "epochs": 0,
                "evaluation_points": 1,
                "members": verified_members,
                "synthetic": False,
            },
            "provenance": {
                "native_final_score": marker,
                "trainer_source_sha256": plan["source_identity"]["training_source_sha256"],
                "scorer_sha256": plan["scorer_sha256"],
                "recipe_sha256": plan["recipe_sha256"],
                "split_manifest_sha256": plan["split"]["split_manifest_sha256"],
                "final_weights": run_weights,
                "official_valid": {
                    "row_mask": "is_scored",
                    "sequences": VALID_GROUPS,
                    "selected_rows": VALID_ROWS,
                    "metric_id": "global_wp_clip2_v1",
                },
                "exposure": score.get("exposure"),
            },
            "synthetic": False,
        },
    )
    event_files = (view / "monitor_events.jsonl", view / "monitor_evaluations.jsonl")
    for path in event_files:
        if path.is_symlink():
            raise BridgeError("fixed blend dashboard event path cannot be a symlink")
        if not path.exists():
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
    context = {
        "run": FINAL_VIEW,
        "node_id": "fixed_final_blend",
        "run_identity_sha256": run_identity,
        "plan_sha256": plan["sha256"],
        "source_sha256": plan["source_identity"]["training_source_sha256"],
        "split_manifest_sha256": plan["split"]["split_manifest_sha256"],
        "source_timestamp_available": False,
        "evaluation": {
            "type": "single terminal fixed-blend report",
            "row_mask": "is_scored",
            "cohort_count": VALID_GROUPS,
            "selected_rows": VALID_ROWS,
            "metric_id": "global_wp_clip2_v1",
        },
        "members": verified_members,
        "score_sha256": score_sha,
        "exposure": score.get("exposure"),
    }
    events_path = view / "monitor_events.jsonl"
    known = _read_events(events_path)
    added = int(
        _append_event(events_path, _event("status", 2, 0, {"run/completed": 1}, context), known)
    )
    metrics = {
        "valid/score": float(blend["wp"]),
        "valid/score/t0": float(blend["t0"]),
        "valid/score/t1": float(blend["t1"]),
        "valid/selected_rows": VALID_ROWS,
    }
    evaluation = _event("evaluation", 0, 1, metrics, context)
    was_present = ("evaluation", 1, 0) in known
    _append_evaluation(view, evaluation, known)
    added += int(not was_present)
    return {"run": FINAL_VIEW, "score_sha256": score_sha, "events_added": added}
