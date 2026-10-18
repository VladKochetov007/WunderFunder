"""Bounded, read-only inspection of exact TL3 producer metadata paths."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
from datetime import UTC, datetime
from pathlib import Path


signal.alarm(35)
ROOT = Path("/home/grinchenkoeg/WunderFunder-creative")
RESTORED_PROVENANCE = Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/"
    "restored-propensity-v1/artifacts/runs/c20260929_prop_f0/provenance.json"
)
RUN_NAMES = (
    "prop_f0",
    "c20260929_prop_f0",
    "w18final_teacher",
    "w18final_fbc1k_s5",
    "w18final_family_full",
    "w18final_family_raw",
    "w18final_distilled_s5",
    "w18final_distilled_s6",
)
RUN_TEXT_FILES = (
    "args.json",
    "config.json",
    "resolved_config.json",
    "run.json",
    "manifest.json",
    "provenance.json",
    "history.json",
)
CACHE_NAMES = (
    "w18final_teacher",
    "w18final_fbc1k_s5",
    "w18final_family_full",
    "w18final_family_raw",
)
SOURCE_FILES = (
    "scripts/cache_propensity.py",
    "scripts/predict_any_env.py",
    "scripts/predict_any.py",
    "scripts/predict_rs.py",
    "scripts/predict_fold0.py",
)
MAX_TOTAL_READ = 250_000
MAX_FILE_READ = 25_000
SECRET_KEY = re.compile(r"token|secret|password|credential|authorization|private.?key", re.I)
SECRET_VALUE = re.compile(
    r"(?i)(bearer\s+\S+|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})"
)
read_bytes = 0


def safe(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SECRET_KEY.search(str(key)) else safe(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [safe(item) for item in value]
    if isinstance(value, str):
        return SECRET_VALUE.sub("[REDACTED]", value)
    return value


def inspect_text(path: Path):
    global read_bytes
    result = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    stat = path.stat()
    result["bytes"] = stat.st_size
    if stat.st_size > MAX_FILE_READ or read_bytes + stat.st_size > MAX_TOTAL_READ:
        result["read"] = False
        result["reason"] = "text_read_cap"
        return result
    raw = path.read_bytes()
    read_bytes += len(raw)
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    result["read"] = True
    try:
        result["content"] = safe(json.loads(raw))
    except (json.JSONDecodeError, UnicodeDecodeError):
        text = raw.decode("utf-8", errors="replace")
        result["content"] = SECRET_VALUE.sub("[REDACTED]", text)
    return result


def inspect_run(name: str):
    directory = ROOT / "artifacts/runs" / name
    result = {"name": name, "path": str(directory), "exists": directory.is_dir()}
    if not directory.exists():
        return result
    result["is_symlink"] = directory.is_symlink()
    result["resolved"] = str(directory.resolve())
    result["text_metadata"] = {
        filename: inspect_text(directory / filename) for filename in RUN_TEXT_FILES
    }
    # Endpoint metadata only: never open or hash checkpoint payloads.
    model = directory / "model.pt"
    result["model_endpoint"] = {
        "path": str(model),
        "exists": model.is_file(),
        "bytes": model.stat().st_size if model.is_file() else None,
    }
    return result


def inspect_source(relative: str):
    global read_bytes
    path = ROOT / relative
    result = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    stat = path.stat()
    result["bytes"] = stat.st_size
    if stat.st_size > MAX_FILE_READ or read_bytes + stat.st_size > MAX_TOTAL_READ:
        result["read"] = False
        result["reason"] = "text_read_cap"
        return result
    raw = path.read_bytes()
    read_bytes += len(raw)
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    result["read"] = True
    relevant = []
    for line_number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
        if re.search(r"cache|teacher|family|pred_|propensity|row.group|split|epoch|checkpoint", line, re.I):
            relevant.append({"line": line_number, "text": SECRET_VALUE.sub("[REDACTED]", line)})
            if len(relevant) >= 80:
                break
    result["relevant_lines"] = relevant
    return result


def main():
    root_info = {
        "path": str(ROOT),
        "exists": ROOT.is_dir(),
        "is_symlink": ROOT.is_symlink(),
        "resolved": str(ROOT.resolve()) if ROOT.exists() else None,
    }
    output = {
        "schema": "quality_gap.tl3_producer_trace_probe.v1",
        "observed_utc": datetime.now(UTC).isoformat(),
        "target_root": root_info,
        "restored_propensity_provenance": inspect_text(RESTORED_PROVENANCE),
        "exact_run_metadata": [inspect_run(name) for name in RUN_NAMES],
        "w18_cache_sidecars": [
            {
                "name": name,
                "split": split,
                "metadata": inspect_text(ROOT / "cache" / f"pred_{name}_{split}.json"),
            }
            for name in CACHE_NAMES
            for split in ("train", "valid")
        ],
        "producer_source_candidates": [inspect_source(path) for path in SOURCE_FILES],
        "text_read_bytes": read_bytes,
        "text_read_cap_bytes": MAX_TOTAL_READ,
        "checkpoint_payloads_read": False,
        "prediction_or_propensity_arrays_read": False,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
