"""Fail-closed admission helpers for the Oct 5 E427 recovery packet.

This module intentionally contains no process-launch path. Root must review,
seal, and invoke the fresh draft identities after admission becomes possible.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

EXPECTED_FITS = (
    "e464-h093-learned-head-s3-final-v1", "e464-h093-hash-head-s3-final-v1",
    "e464-h093-universal-head-s3-final-v1", "e464-h093-gru-budget-head-s3-final-v1",
    "e464-h094-conditional-head-s3-final-v1", "e464-h094-unconditional-head-s3-final-v1",
    "e464-h094-diagonal-head-s3-final-v1", "e464-h094-gru-budget-head-s3-final-v1",
)
EXPECTED_EXPORTS = tuple(
    f"e464-predict-{arm.replace('_', '-')}-fp32-v1"
    for arm in ("h093_learned", "h093_hash", "h093_universal", "h093_gru_budget",
                "h094_conditional", "h094_unconditional", "h094_diagonal", "h094_gru_budget")
)
EXPECTED_SCORE = "e464-score-all-parent-head-fp32-v1"
EXPECTED_ATTEMPTS = (*EXPECTED_FITS, *EXPECTED_EXPORTS, EXPECTED_SCORE)
EXPECTED_REPORT = "research/campaign_2026-10-04/staging/e464/results/e464_report_all_fp32_v1.json"
EXPECTED_COMPLETED = list(EXPECTED_ATTEMPTS)
STOP_NEW_STAGE_UTC = datetime(2026, 10, 5, 7, tzinfo=timezone.utc)
SUPPORT_RESERVE_BYTES = 4 * 1024**3
HARD_RAM_BYTES = 30_000_000_000
WORKER_RAM_BYTES = 25_769_803_776  # original 24 GiB reservation; immutable
GPU_ALLOCATOR_BYTES = 4_200_000_000
GPU_STOP_BYTES = 4_800_000_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def verify_artifacts(root: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    spec = record.get("spec")
    if not isinstance(spec, dict) or not _sha(spec.get("code_sha256")) or not _sha(record.get("spec_sha256")):
        raise ValueError("attempt is missing authoritative spec/code hash")
    expected = spec.get("artifacts")
    entries = record.get("artifacts")
    if not isinstance(expected, list) or not expected or not isinstance(entries, list):
        raise ValueError("attempt does not declare an exact artifact proof set")
    if len(entries) != len(expected) or {e.get("path") for e in entries if isinstance(e, dict)} != set(expected):
        raise ValueError("attempt artifact proof set is partial or unexpected")
    checked = []
    for item in entries:
        if not isinstance(item, dict) or not _sha(item.get("sha256")) or type(item.get("size")) is not int:
            raise ValueError("artifact proof lacks exact size/hash")
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"artifact does not match attempt proof: {item['path']}")
        checked.append({"path": item["path"], "size": item["size"], "sha256": item["sha256"]})
    return checked


def verify_e464(root: Path) -> dict[str, Any]:
    """Verify exact 17 successful attempts, their artifacts, sequence, and report."""
    campaign = root / "research/campaign_2026-10-04"
    sequence_path = campaign / "checks/e464_sequence_v1.json"
    report_path = root / EXPECTED_REPORT
    sequence = json.loads(sequence_path.read_text())
    report = json.loads(report_path.read_text())
    if sequence.get("status") != "complete" or sequence.get("completed") != EXPECTED_COMPLETED:
        raise ValueError("E464 is not the exact completed 17-attempt sequence")
    if sequence.get("paired_score_path") != EXPECTED_REPORT or sequence.get("paired_comparisons") != 16 or sequence.get("bootstrap_replicates") != 2000:
        raise ValueError("E464 sequence metadata differs from the fixed paired screen")
    report_sha = sha256_file(report_path)
    if sequence.get("paired_score_sha256") != report_sha:
        raise ValueError("E464 sequence does not bind the current report bytes")
    screen = report.get("screen", {})
    if report.get("metric_id") != "global_wp_clip2_v1" or report.get("n_sequences") != 368 or screen.get("folds") != ["holdout"] or len(screen.get("comparisons", [])) != 16:
        raise ValueError("E464 score report does not contain the exact frozen panel/metric")
    receipts = []
    for attempt in EXPECTED_ATTEMPTS:
        p = campaign / "runner/attempts" / f"{attempt}.json"
        record = json.loads(p.read_text())
        if (record.get("id") != attempt or record.get("status") != "finished" or record.get("exit_code") != 0
                or record.get("teardown_verified") is not True or record.get("failure") is not None):
            raise ValueError(f"E464 attempt is not a successful terminal receipt: {attempt}")
        mem = record.get("final_memory_events")
        if not isinstance(mem, dict) or any(type(mem.get(k)) is not int or mem[k] != 0 for k in ("oom", "oom_kill", "oom_group_kill")):
            raise ValueError(f"E464 attempt lacks explicit zero OOM evidence: {attempt}")
        artifacts = verify_artifacts(root, record)
        if record.get("kind") not in ({"cpu"} if attempt == EXPECTED_SCORE else {"gpu"}):
            raise ValueError(f"E464 attempt kind differs from fixed design: {attempt}")
        receipts.append({"id": attempt, "kind": record.get("kind"), "attempt_json_sha256": sha256_file(p),
                         "spec_name": record["spec"].get("name"), "spec_sha256": record["spec_sha256"],
                         "code_sha256": record["spec"].get("code_sha256"),
                         "artifacts": artifacts})
    return {"sequence_path": str(sequence_path.relative_to(root)), "sequence_sha256": sha256_file(sequence_path),
            "report_path": EXPECTED_REPORT, "report_sha256": report_sha, "attempts": receipts}


def verify_e427_prerequisites(root: Path) -> dict[str, Any]:
    """Validate the fit-only stats, public parent, provenance and training data bytes."""
    stats_path = root / "research/campaign_2026-10-04/checks/e427_context_stats_s3_v1.json"
    stats = json.loads(stats_path.read_text())
    parent_path = root / stats["parent_checkpoint_path"]
    provenance_path = parent_path.parent / "provenance.json"
    required = [
        {"path": str(stats_path.relative_to(root)), "sha256": "8e873e9a36f85b109647455c608140ecd594e3c9258288d90e65cbe27e84769d", "size": stats_path.stat().st_size},
        {"path": str(parent_path.relative_to(root)), "sha256": stats["parent_checkpoint_sha256"], "size": parent_path.stat().st_size},
        {"path": str(provenance_path.relative_to(root)), "sha256": stats["parent_provenance_sha256"], "size": provenance_path.stat().st_size},
    ]
    for path, expected in stats["input_sha256"].items():
        full = root / path
        required.append({"path": path, "sha256": expected, "size": full.stat().st_size})
    results = []
    for item in required:
        full = root / item["path"]
        if not full.is_file() or full.stat().st_size != item["size"] or sha256_file(full) != item["sha256"]:
            raise ValueError(f"E427 prerequisite identity changed: {item['path']}")
        results.append(item)
    return {"inputs": results, "stats_parent": stats["parent_checkpoint"], "verified": True}


def check_resource_admission(*, current_bytes: int, hard_limit_bytes: int = HARD_RAM_BYTES,
                             worker_bytes: int = WORKER_RAM_BYTES,
                             support_bytes: int = SUPPORT_RESERVE_BYTES) -> None:
    """Fail closed against observed cgroup usage and unchanged worker/support reservations."""
    vals = (current_bytes, hard_limit_bytes, worker_bytes, support_bytes)
    if any(type(v) is not int or v < 0 for v in vals):
        raise ValueError("RAM admission inputs must be nonnegative integer byte counts")
    reserved = worker_bytes + support_bytes
    available = max(0, hard_limit_bytes - current_bytes)
    if reserved > available:
        raise RuntimeError(f"RAM admission denied: reserve {reserved} > available {available} bytes")


def may_admit_new_stage(now: datetime) -> bool:
    if now.tzinfo is None:
        raise ValueError("timebox requires timezone-aware UTC time")
    return now.astimezone(timezone.utc) < STOP_NEW_STAGE_UTC


def stage_inventory() -> list[dict[str, Any]]:
    """Enumerate every fixed future GPU stage; never shrink budgets after timebox."""
    return ([{"stage": "smoke", "arms": ["aligned", "stale", "noop"], "seconds_each": 900},
             {"stage": "fit", "arms": ["aligned", "stale", "noop"], "seconds_each": 3600},
             {"stage": "export", "arms": ["aligned", "stale", "noop"], "seconds_each": 1800},
             {"stage": "score", "arms": ["aligned", "stale", "noop"], "seconds_each": 1800,
              "device": "cpu"}])


def advise_exact_declared_inputs(root: Path, inputs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hash then issue POSIX_FADV_DONTNEED only on exact declared input files."""
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advice is None or not hasattr(os, "posix_fadvise"):
        raise RuntimeError("POSIX_FADV_DONTNEED unavailable; fresh resource admission must not proceed")
    records = []
    for item in inputs:
        rel = item.get("path")
        if not isinstance(rel, str) or not _sha(item.get("sha256")) or type(item.get("size")) is not int:
            raise ValueError("declared input needs exact path, size, and SHA-256")
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("declared input escapes the repository root")
        if not path.is_file() or path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"declared input identity changed: {rel}")
        records.append({"path": rel, "size": item["size"], "sha256": item["sha256"]})
    return _advise_verified_paths(root, records)


def _advise_verified_paths(root: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advice is None or not hasattr(os, "posix_fadvise"):
        raise RuntimeError("POSIX_FADV_DONTNEED unavailable; fresh resource admission must not proceed")
    result = []
    for item in records:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("declared input escapes the repository root")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.posix_fadvise(fd, 0, 0, advice)
        finally:
            os.close(fd)
        result.append({"path": item["path"], "size": item["size"], "sha256": item["sha256"],
                       "advice": "POSIX_FADV_DONTNEED"})
    return result


def prepare_stage_inputs(root: Path, spec_path: Path) -> list[dict[str, Any]]:
    """Hash this stage's exact current declared inputs, then advise only those paths."""
    spec = json.loads(spec_path.read_text())
    declared = spec.get("inputs")
    if not isinstance(declared, list) or not declared or any(not isinstance(p, str) for p in declared):
        raise ValueError("stage spec must have a nonempty path-only input list")
    if len(declared) != len(set(declared)):
        raise ValueError("duplicate declared stage input")
    records = []
    for rel in declared:
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"stage input missing or escapes root: {rel}")
        records.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return _advise_verified_paths(root, records)


def require_fresh_outputs(root: Path, spec_path: Path) -> None:
    """Refuse to bind an Oct 5 attempt to any previously materialized output."""
    spec = json.loads(spec_path.read_text())
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("stage spec has no declared output set")
    for rel in artifacts:
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("declared output escapes repository root")
        if path.exists():
            raise FileExistsError(f"fresh attempt output already exists: {rel}")
