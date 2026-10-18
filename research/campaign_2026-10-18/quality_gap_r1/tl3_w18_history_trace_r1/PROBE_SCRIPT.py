"""Read exact archived producer scripts, logs, and base-run text metadata only."""
from __future__ import annotations

import hashlib
import json
import re
import signal
from datetime import UTC, datetime
from pathlib import Path

signal.alarm(55)
ROOT = Path("/home/grinchenkoeg/WunderFunder-creative")
TOTAL_TEXT_CAP = 450_000
PER_FILE_CAP = 60_000
SOURCE_FILES = (
    "research/iterations/2026-10-05-wave18/full_teacher_lane.sh",
    "research/iterations/2026-10-05-wave18/prepare_full_teachers.py",
    "research/iterations/2026-10-05-wave18/full-teacher-manifest.json",
    "research/iterations/2026-10-05-wave18/teacher-strong.log",
    "research/iterations/2026-10-05-wave18/teacher-full.log",
    "research/iterations/2026-10-05-wave18/teacher-raw.log",
    "research/iterations/2026-10-05-wave18/prepare-full-teachers.log",
    "research/iterations/2026-10-05-wave6/scripts/predict_run.py",
    "research/iterations/2026-10-05-wave6/scripts/train_rnn.py",
    "research/iterations/2026-10-02-slow-fast/final_rawW.sh",
    "research/iterations/2026-10-02-slow-fast/ftpairs_wait.sh",
)
RUNS = {
    "fbc1k_s5": ("model_ep3.pt",),
    "ffull_s3": ("model_ep1.pt",),
    "ffull_s4": ("model_ep1.pt",),
    "frawW_s3": ("model_ep3.pt",),
    "frawW_s4": ("model_ep3.pt",),
    "frawW_s3ft": ("model_ep1.pt",),
    "frawW_s4ft": ("model_ep1.pt",),
}
META_FILES = (
    "args.json",
    "config.json",
    "resolved_config.json",
    "run.json",
    "manifest.json",
    "provenance.json",
    "history.json",
)
LOG_FILES = tuple(f"artifacts/logs/{name}.log" for name in RUNS)
SECRET_KEY = re.compile(r"token|secret|password|credential|authorization|private.?key", re.I)
SECRET_VALUE = re.compile(r"(?i)(bearer\s+\S+|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})")
read_bytes = 0

def redact(value):
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if SECRET_KEY.search(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return SECRET_VALUE.sub("[REDACTED]", value)
    return value

def list_summary(value):
    if all(isinstance(x, (int, float, str)) for x in value):
        raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
        return {"count": len(value), "ordered_json_sha256": hashlib.sha256(raw).hexdigest(), "first": value[:5], "last": value[-5:]}
    return {"count": len(value), "item_type": type(value[0]).__name__ if value else None}

def compact(value, key="", depth=0):
    if isinstance(value, dict):
        return {str(k): compact(v, str(k), depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 100 and all(isinstance(x, (int, float, str)) for x in value):
            return list_summary(value)
        if len(value) > 200:
            return {"count": len(value), "item_type": type(value[0]).__name__ if value else None}
        return [compact(v, key, depth + 1) for v in value]
    return redact(value)

def read_text(rel, *, log=False):
    global read_bytes
    path = ROOT / rel
    result = {"path": str(path), "is_file": path.is_file()}
    if not result["is_file"]:
        return result
    stat = path.stat()
    result["bytes"] = stat.st_size
    if stat.st_size > PER_FILE_CAP or read_bytes + stat.st_size > TOTAL_TEXT_CAP:
        result["read"] = False
        result["reason"] = "text_cap"
        return result
    raw = path.read_bytes()
    read_bytes += len(raw)
    result["read"] = True
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")
    if log:
        lines = text.splitlines()
        result["line_count"] = len(lines)
        result["first_lines"] = [SECRET_VALUE.sub("[REDACTED]", x) for x in lines[:12]]
        result["last_lines"] = [SECRET_VALUE.sub("[REDACTED]", x) for x in lines[-12:]]
    elif rel.endswith((".json",)):
        try:
            result["json_summary"] = compact(json.loads(raw))
        except json.JSONDecodeError:
            result["text_preview"] = SECRET_VALUE.sub("[REDACTED]", text[:2000])
    else:
        result["text_preview"] = SECRET_VALUE.sub("[REDACTED]", text[:2500])
    return result

def main():
    sources = [read_text(rel) for rel in SOURCE_FILES]
    runs = []
    for name, endpoints in RUNS.items():
        directory = ROOT / "artifacts/runs" / name
        entry = {"name": name, "path": str(directory), "is_dir": directory.is_dir()}
        if entry["is_dir"]:
            entry["resolved"] = str(directory.resolve())
            entry["metadata"] = [read_text(f"artifacts/runs/{name}/{meta}") for meta in META_FILES]
            entry["endpoints_stat_only"] = []
            for filename in endpoints:
                model = directory / filename
                item = {"path": str(model), "is_file": model.is_file()}
                if item["is_file"]:
                    item["bytes"] = model.stat().st_size
                entry["endpoints_stat_only"].append(item)
        runs.append(entry)
    logs = [read_text(rel, log=True) for rel in LOG_FILES]
    result = {
        "schema": "quality_gap.tl3_w18_history_probe.v1",
        "observed_utc": datetime.now(UTC).isoformat(),
        "target_root": str(ROOT),
        "source_files": sources,
        "base_runs": runs,
        "exact_training_logs": logs,
        "text_bytes_read": read_bytes,
        "text_read_cap_bytes": TOTAL_TEXT_CAP,
        "per_file_cap_bytes": PER_FILE_CAP,
        "checkpoint_payloads_read_or_hashed": False,
        "prediction_or_propensity_arrays_read_or_hashed": False,
        "jobs_or_remote_mutations": False,
    }
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))

main()
