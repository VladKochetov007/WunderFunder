"""One-shot official grading and optional separate publisher invocation."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from delivery_watch_core import (
    EXPECTED_DEADLINE_UNIX,
    GRADE_TIMEOUT_SECONDS,
    PUBLISH_INTERVAL_SECONDS,
    append_event,
    fresh_attempt_allowed,
    remaining,
    sha256_file,
)
from delivery_watch_receipts import verify_grade_receipt
from delivery_watch_state import _cell_write, _run_child

FIT_TERMINAL = {"validated", "rejected", "preserved_partial"}
EXPORT_TERMINAL = {"exported", "rejected", "preserved_partial", "not_applicable"}
GRADE_TERMINAL = {"graded", "preserved_partial", "not_applicable"}


def _publish(config: dict[str, Any], state: Path, *, force: bool, last_time: float) -> float:
    """Call the separately hash-pinned CPU publisher CLI, never import it."""
    now = time.time()
    if not force and now - last_time < PUBLISH_INTERVAL_SECONDS:
        return last_time
    script = Path(__file__).resolve().parent / "publish_recovery.py"
    digest = config["watcher_files"].get("publish_recovery.py")
    configured_path = config.get("publisher_config_path")
    configured_sha = config.get("publisher_config_sha256")
    if not isinstance(configured_path, str) or not isinstance(configured_sha, str):
        append_event(
            state / "events.jsonl",
            {"time_unix": now, "phase": "publisher_wait", "reason": "publisher config pending"},
        )
        return now
    publisher_config = Path(configured_path)
    publisher_sha = configured_sha
    if (
        digest is None
        or not script.is_file()
        or not publisher_config.is_file()
        or sha256_file(script) != digest
        or sha256_file(publisher_config) != publisher_sha
    ):
        append_event(
            state / "events.jsonl",
            {
                "time_unix": now,
                "phase": "publisher_wait",
                "reason": "pinned publisher/config not present",
            },
        )
        return now
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", WNN_PROJECT_ROOT=str(config["root_path"]))
    argv = [
        sys.executable,
        "-B",
        str(script),
        "--config",
        str(publisher_config),
        "--config-sha256",
        publisher_sha,
    ]
    log = state / "logs" / f"publisher-{int(now)}.log"
    try:
        result = _run_child(argv, env, log, 120)
        append_event(
            state / "events.jsonl",
            {"time_unix": time.time(), "phase": "publisher_finished", **result},
        )
    except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
        append_event(
            state / "events.jsonl",
            {"time_unix": time.time(), "phase": "publisher_failed", "error": str(error)},
        )
    return now


def _grade_pair(
    config: dict[str, Any],
    state: Path,
    recovery: Path,
    wide: dict[str, Any],
    deep: dict[str, Any],
    cells: dict[str, dict[str, Any]],
    grader: Any,
    last_publish: float,
) -> float:
    """Grade one exported same-fold pair once, on the watcher CPU affinity."""
    fold, seed = int(wide["fold"]), int(wide["seed"])
    wide_cell, deep_cell = cells[str(wide["id"])], cells[str(deep["id"])]
    grade_status = deep_cell.get("grade", {}).get("status", "pending")
    if grade_status in GRADE_TERMINAL:
        return last_publish
    if not all(c.get("export", {}).get("status") == "exported" for c in (wide_cell, deep_cell)):
        fit_terminal = all(
            c.get("fit", {}).get("status") in FIT_TERMINAL for c in (wide_cell, deep_cell)
        )
        export_terminal = all(
            c.get("export", {}).get("status") in EXPORT_TERMINAL for c in (wide_cell, deep_cell)
        )
        if grade_status in {"pending", "grade_started"} and fit_terminal and export_terminal:
            terminal_status = "not_applicable" if grade_status == "pending" else "preserved_partial"
            _cell_write(
                state,
                deep_cell,
                "grade",
                status=terminal_status,
                reason=(
                    "one or both paired endpoints failed fit validation or export"
                    if grade_status == "pending"
                    else "prior grade attempt cannot be resumed after endpoint failure"
                ),
                quality_status="not_scored",
            )
        return last_publish
    grade_dir = recovery / "grades" / f"seed{seed}" / f"fold{fold}"
    grade_receipt_path = grade_dir / "grade_receipt.json"
    if grade_receipt_path.exists():
        try:
            receipt = verify_grade_receipt(
                recovery,
                config["root_path"],
                str(wide["id"]),
                str(deep["id"]),
                fold,
                seed,
            )
            _cell_write(
                state,
                deep_cell,
                "grade",
                status="graded",
                receipt_sha256=sha256_file(grade_receipt_path),
                score=receipt.get("score"),
            )
            return _publish(config, state, force=True, last_time=last_publish)
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            _cell_write(
                state,
                deep_cell,
                "grade",
                status="preserved_partial",
                error=str(error),
                quality_status="not_scored",
            )
            return last_publish
    if not fresh_attempt_allowed(
        grade_receipt_path.exists(),
        grade_dir.exists(),
        deep_cell.get("grade", {}).get("status", "pending"),
    ):
        _cell_write(
            state,
            deep_cell,
            "grade",
            status="preserved_partial",
            reason="prior grade attempt exists; no retry",
            quality_status="not_scored",
        )
        return last_publish
    _cell_write(state, deep_cell, "grade", status="grade_started", fold=fold, seed=seed)
    try:
        receipt = grader(
            config["root_path"],
            config["packet_path"],
            config["root_path"] / "runtime/source-a266-lf-subset-v1",
            recovery,
            str(wide["id"]),
            str(deep["id"]),
            timeout_seconds=min(GRADE_TIMEOUT_SECONDS, remaining(EXPECTED_DEADLINE_UNIX)),
        )
        receipt = verify_grade_receipt(
            recovery,
            config["root_path"],
            str(wide["id"]),
            str(deep["id"]),
            fold,
            seed,
        )
        _cell_write(
            state,
            deep_cell,
            "grade",
            status="graded",
            receipt_sha256=sha256_file(grade_receipt_path),
            score=receipt.get("score"),
        )
        return _publish(config, state, force=True, last_time=last_publish)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        RuntimeError,
        TimeoutError,
        subprocess.TimeoutExpired,
    ) as error:
        _cell_write(
            state,
            deep_cell,
            "grade",
            status="preserved_partial",
            error_type=type(error).__name__,
            error=str(error),
            quality_status="not_scored",
        )
        return last_publish
