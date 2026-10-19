"""Observe, validate, export, and grade frozen R01 fits without a fit path."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from delivery_watch_core import (
    EXPECTED_DEADLINE_UNIX,
    HELPER_MANIFEST_SHA256,
    PANEL_SHA256,
    POLL_SECONDS,
    append_event,
    read_object,
    sha256_file,
    verify_helper_bundle,
    verify_runtime_envelope,
    verify_watcher_files,
    write_atomic,
)
from delivery_watch_grades import _grade_pair, _publish
from delivery_watch_phases import _export_one, _fit_one
from delivery_watch_state import _cell_load


def _load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Check the root-supplied config, helper bundle, and frozen panel."""
    if sha256_file(path) != expected_sha256:
        raise ValueError("watcher config SHA-256 differs from the service pin")
    config = read_object(path)
    required = {
        "root",
        "helper_dir",
        "helper_manifest_name",
        "helper_manifest_sha256",
        "absolute_deadline_unix",
        "gpu_uuid",
        "publisher_config_path",
        "publisher_config_sha256",
        "watcher_files",
    }
    if (
        config.get("schema") != "tabdata-r01-delivery-watch-config.v1"
        or not required.issubset(config)
        or config["helper_manifest_name"] != "MANIFEST.json"
        or config["absolute_deadline_unix"] != EXPECTED_DEADLINE_UNIX
        or config["helper_manifest_sha256"] != HELPER_MANIFEST_SHA256
    ):
        raise ValueError("watcher config schema, helper pin, or fixed deadline is invalid")
    script_dir = Path(__file__).resolve().parent
    verify_watcher_files(script_dir, config["watcher_files"])
    root = Path(config["root"]).resolve(strict=True)
    helper = Path(config["helper_dir"]).resolve(strict=True)
    expected_helper = (root / "runtime/r01-delivery-recovery-v2_4").resolve(strict=True)
    if helper != expected_helper:
        raise ValueError("helper directory differs from runtime/r01-delivery-recovery-v2_4")
    verify_helper_bundle(helper, config["helper_manifest_sha256"])
    packet = root / "runtime/b300-r01-v1"
    panel_path = packet / "PANEL.json"
    if sha256_file(panel_path) != PANEL_SHA256:
        raise ValueError("frozen R01 panel hash differs from the reviewed panel")
    jobs = read_object(panel_path).get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 60 or len({j.get("id") for j in jobs}) != 60:
        raise ValueError("frozen R01 panel must contain exactly 60 unique cells")
    return {
        **config,
        "root_path": root,
        "helper_path": helper,
        "packet_path": packet,
        "jobs": jobs,
    }


def _summary(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize delivery outcomes without treating fit failures as scores."""
    fits = [c.get("fit", {}).get("status", "pending") for c in cells.values()]
    exports = [c.get("export", {}).get("status", "pending") for c in cells.values()]
    grades = [c.get("grade", {}).get("status", "pending") for c in cells.values()]
    return {
        "fit_counts": {
            s: fits.count(s)
            for s in ("validated", "rejected", "preserved_partial", "pending", "validation_started")
        },
        "export_counts": {
            s: exports.count(s)
            for s in (
                "exported",
                "rejected",
                "preserved_partial",
                "not_applicable",
                "pending",
                "export_started",
            )
        },
        "fit_terminal_count": sum(
            s in {"validated", "rejected", "preserved_partial"} for s in fits
        ),
        "export_terminal_count": sum(
            s in {"exported", "rejected", "preserved_partial", "not_applicable"} for s in exports
        ),
        "grade_count": grades.count("graded"),
        "grade_terminal_count": sum(
            s in {"graded", "preserved_partial", "not_applicable"} for s in grades
        ),
        "scores_claimed": grades.count("graded"),
    }


def main() -> None:
    """Run the finite delivery monitor under its separately bounded service unit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    config = _load_config(args.config, args.config_sha256)
    verify_runtime_envelope(str(config["gpu_uuid"]))
    root, helper = config["root_path"], config["helper_path"]
    recovery = root / "state/r01-delivery-recovery-v2"
    if recovery.is_symlink() or not recovery.is_dir():
        raise ValueError("recovery tree must be an existing nonsymlink directory")
    state = root / "state/r01-delivery-watch-v1"
    if state.is_symlink():
        raise ValueError("watcher state cannot be a symlink")
    state.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (state / "watcher.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        lock.close()
        raise RuntimeError("another delivery watcher already owns this state") from error
    sys.path.insert(0, str(helper))
    from r01_recovery_delivery import grade_pair
    from r01_recovery_placement import verify_placement

    delivery = sys.modules["r01_recovery_delivery"]
    jobs = config["jobs"]
    cells = {str(job["id"]): _cell_load(state, job) for job in jobs}
    stable: dict[str, tuple[str, float]] = {}
    last_publish = 0.0
    append_event(
        state / "events.jsonl",
        {
            "time_unix": time.time(),
            "phase": "watcher_started",
            "panel_sha256": PANEL_SHA256,
            "deadline_unix": EXPECTED_DEADLINE_UNIX,
        },
    )
    while time.time() < EXPECTED_DEADLINE_UNIX:
        for job in jobs:
            cell = cells[str(job["id"])]
            if cell.get("fit", {}).get("status") in {"pending", "validation_started"}:
                _fit_one(config, state, recovery, job, cell, stable, delivery, verify_placement)
            if cell.get("export", {}).get("status") in {"pending", "export_started"}:
                _export_one(config, state, recovery, job, cell, delivery, verify_placement)
            last_publish = _publish(config, state, force=False, last_time=last_publish)
        pairs: dict[tuple[int, int], dict[str, Any]] = {}
        for job in jobs:
            pairs.setdefault((int(job["fold"]), int(job["seed"])), {})[str(job["arm"])] = job
        for (fold, seed), pair in sorted(pairs.items()):
            if set(pair) != {"wide", "deep"}:
                raise ValueError(f"frozen panel pair is malformed at fold {fold}, seed {seed}")
            last_publish = _grade_pair(
                config,
                state,
                recovery,
                pair["wide"],
                pair["deep"],
                cells,
                grade_pair,
                last_publish,
            )
        last_publish = _publish(config, state, force=False, last_time=last_publish)
        summary = _summary(cells)
        write_atomic(
            state / "STATUS.json",
            {
                "schema": "tabdata-r01-delivery-watch-status.v1",
                "heartbeat_unix": time.time(),
                "panel_sha256": PANEL_SHA256,
                "deadline_unix": EXPECTED_DEADLINE_UNIX,
                **summary,
            },
        )
        if (
            summary["fit_terminal_count"] == 60
            and summary["export_terminal_count"] == 60
            and summary["grade_terminal_count"] == 30
        ):
            break
        seconds_left = max(0, int(EXPECTED_DEADLINE_UNIX - time.time()))
        if seconds_left <= 0:
            break
        time.sleep(min(POLL_SECONDS, seconds_left))
    summary = _summary(cells)
    complete = (
        summary["fit_counts"]["validated"] == 60
        and summary["export_counts"]["exported"] == 60
        and summary["grade_count"] == 30
    )
    write_atomic(
        state / "RESULT.json",
        {
            "schema": "tabdata-r01-delivery-watch-result.v1",
            "status": "complete" if complete else "partial",
            "finished_unix": time.time(),
            "panel_sha256": PANEL_SHA256,
            **summary,
        },
    )
    append_event(
        state / "events.jsonl", {"time_unix": time.time(), "phase": "watcher_finished", **summary}
    )
    lock.close()


if __name__ == "__main__":
    main()
