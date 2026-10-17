"""Bounded subprocess and profile-log handling for the B300 cell runner."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from common import PROFILE_CONFIG, ROOT, sha256_file, utc_now, write_json


def validate_profile_receipt(path: Path, commit: str) -> None:
    """Require a matching successful, overlapping eight-card profile."""
    receipt = json.loads(path.read_text())
    if (
        receipt.get("schema") != "causal_geometry_b300_profile_v1"
        or receipt.get("source_commit") != commit
        or receipt.get("profiled_workers") != 8
        or not receipt.get("overlap_confirmed")
        or receipt.get("profile_config") != PROFILE_CONFIG
        or not receipt.get("strict_fp32")
    ):
        raise ValueError("eight-card profile receipt is incomplete or mismatches this source")


def run_phase(
    phase: str,
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
    record: dict[str, Any],
    record_path: Path,
) -> None:
    """Run one phase, preserving its exact command, log hash and exit status."""
    start = utc_now()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    phase_record = record["phases"].setdefault(phase, {"attempts": []})
    phase_record["attempts"].append(
        {
            "command": command,
            "started_utc": start,
            "finished_utc": utc_now(),
            "exit_code": result.returncode,
            "log": str(log_path.relative_to(ROOT)),
            "log_sha256": sha256_file(log_path),
        }
    )
    write_json(record_path, record)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def parse_profile_events(path: Path) -> dict[str, Any]:
    """Extract measured timing, memory and update records from a profile log."""
    result: dict[str, Any] = {}
    prefixes = {
        "PROFILE ": "timing",
        "PROFILE_RESOURCE_EVIDENCE ": "resources",
        "PROFILE_UPDATE_EVIDENCE ": "updates",
    }
    for line in path.read_text(errors="replace").splitlines():
        for prefix, key in prefixes.items():
            if line.startswith(prefix):
                result[key] = json.loads(line[len(prefix) :])
    missing = set(prefixes.values()) - result.keys()
    if missing:
        raise ValueError(f"profile log is missing records: {sorted(missing)}")
    return result


def clean_profile_id(value: str) -> str:
    """Return a safe, human-readable label for a GPU-specific profile run."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
