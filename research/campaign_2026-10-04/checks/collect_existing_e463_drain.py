"""Collect one already-submitted TPU job while new research remains paused."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
HANDLE = "vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1-runner"
RECORD = CAMPAIGN / "checks/e463_pause_drain_collection.json"
REPORT = CAMPAIGN / "E463_DRAIN_RESULT.md"
OUTPUT = CAMPAIGN / "kaggle/jax_tpu_realdata_preflight_v1/pause_drain_provider_output"
KAGGLE = str(ROOT / ".venv/bin/kaggle")


def run(command: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """Run a fixed read-only provider command with a bounded observation wait."""
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                          check=False, timeout=timeout)


def main() -> None:
    """Poll only the existing handle; download terminal output without a retry."""
    previous = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    if previous.get("outputs_collected") is True:
        return
    result = run([KAGGLE, "kernels", "status", HANDLE])
    match = re.search(r'KernelWorkerStatus\.(\w+)', result.stdout)
    status = match.group(1) if result.returncode == 0 and match else "OBSERVATION_ERROR"
    record: dict[str, object] = {
        "utc": datetime.now(timezone.utc).isoformat(), "kernel": HANDLE,
        "provider_status": status, "observation_exit_code": result.returncode,
        "outputs_collected": False, "new_launches": 0, "retries": 0,
        "scope": "read-only drain of the existing submitted preflight, not research resumption",
    }
    terminal = status in {"COMPLETE", "ERROR", "CANCELED", "CANCELLED"}
    if terminal:
        downloaded = run([KAGGLE, "kernels", "output", HANDLE, "-p", str(OUTPUT)], 300)
        record["download_exit_code"] = downloaded.returncode
        record["outputs_collected"] = downloaded.returncode == 0
        record["small_raw_json_results"] = []
        if downloaded.returncode == 0:
            summaries = []
            for path in sorted(OUTPUT.rglob("*.json")):
                if path.stat().st_size > 256_000:
                    continue
                try:
                    raw = json.loads(path.read_text())
                except (ValueError, UnicodeError):
                    continue
                if isinstance(raw, dict) and any(k in raw for k in ("status", "passed", "error")):
                    summaries.append({"path": path.relative_to(ROOT).as_posix(),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "reported_status": raw.get("status"),
                                      "reported_passed": raw.get("passed"),
                                      "reported_error": raw.get("error")})
            record["small_raw_json_results"] = summaries
    RECORD.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(
        "# E463 existing TPU job drain\n\n"
        f"Observed {record['utc']}: `{HANDLE}` is **{status}**. "
        f"Terminal outputs collected: **{record['outputs_collected']}**.\n\n"
        "This collector polls and downloads only the already-submitted preflight. "
        "It starts no kernels, performs no scientific retry, and does not resume the paused goal. "
        "Provider completion and raw self-reported checks require independent interpretation; "
        "neither alone establishes full-model TPU or public-score success.\n\n"
        "Raw status, output paths and small result hashes are in "
        "`checks/e463_pause_drain_collection.json`.\n"
    )
    if terminal and record["outputs_collected"]:
        run(["systemctl", "--user", "stop", "wnn-pause-e463-collector.timer"])


if __name__ == "__main__":
    main()
