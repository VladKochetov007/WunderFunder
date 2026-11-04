"""Mirror the hash-verified GitHub final-fit snapshot into the existing local dashboard."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.publish_checkpoint_releases import GitHub  # noqa: E402

REPO = "/repos/VladKochetov007/WunderFunder"
BASE = "research/experiment_drafts/final-submit-all-labeled-20261031/live"
BRANCH = "results/tabdata-final-submit-20261031"
RUN = re.compile(r"final-[A-Za-z0-9][A-Za-z0-9_.-]{0,70}\Z")


def write_atomic(path: Path, payload: bytes) -> None:
    """Replace one owned report file without exposing a partial JSON line."""
    temporary = path.with_suffix(path.suffix + ".new")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("unsafe dashboard report path")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> None:
    """Collect read-only report snapshots until the task's fixed hard deadline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-root", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    target = args.dashboard_root.absolute()
    if target.is_symlink() or not target.is_dir():
        raise ValueError("dashboard root must already be a regular directory")
    api = GitHub(ROOT)
    deadline = 1793523959.0380824
    previous = ""
    while time.time() < deadline:
        ref = urllib.parse.quote(BRANCH, safe="")
        try:
            manifest_record = api.request(
                "GET", f"{REPO}/contents/{BASE}/snapshot.manifest.json?ref={ref}"
            )
            manifest = json.loads(base64.b64decode(manifest_record["content"]))
            record = api.request("GET", f"{REPO}/contents/{BASE}/snapshot.json.gz?ref={ref}")
            compressed = base64.b64decode(record["content"])
            if hashlib.sha256(compressed).hexdigest() != manifest["compressed_sha256"]:
                raise ValueError("changing or invalid compressed snapshot")
            raw = gzip.decompress(compressed)
            if (
                len(raw) > 32_000_000
                or hashlib.sha256(raw).hexdigest() != manifest["payload_sha256"]
            ):
                raise ValueError("invalid decompressed snapshot")
            payload = json.loads(raw)
            if (
                payload.get("schema") != "wnn.final-submit-live-snapshot.v1"
                or payload.get("independent_score_claim") != "forbidden"
            ):
                raise ValueError("unexpected final-fit report role")
            if previous != manifest["payload_sha256"]:
                for run in payload["runs"]:
                    run_id = run["run_id"]
                    if not isinstance(run_id, str) or RUN.fullmatch(run_id) is None:
                        raise ValueError("unsafe dashboard run identifier")
                    folder = target / run_id
                    if folder.is_symlink():
                        raise ValueError("dashboard run folder is a symlink")
                    folder.mkdir(exist_ok=True)
                    metadata = run["metadata"]
                    metadata["provenance"]["controller_status"] = run["status"]
                    events = list(run["events"])
                    status = {
                        "complete": "completed",
                        "failed": "failed",
                        "pending": "scheduled",
                    }.get(run["status"], run["status"])
                    last = events[-1] if events else {"step": 0, "epoch": 0}
                    events.append(
                        {
                            "schema": "wnn.monitor.v1",
                            "kind": "status",
                            "step": last["step"],
                            "epoch": last["epoch"],
                            "wall_time": datetime.now(UTC).isoformat(),
                            "metrics": {},
                            "context": {
                                "status": status,
                                "basis": "hash-verified final-fit director snapshot",
                                "independent_score_claim": "forbidden",
                            },
                        }
                    )
                    write_atomic(
                        folder / "run_metadata.json", json.dumps(metadata, allow_nan=False).encode()
                    )
                    write_atomic(
                        folder / "monitor_events.jsonl",
                        b"\n".join(json.dumps(event, allow_nan=False).encode() for event in events)
                        + b"\n",
                    )
                    write_atomic(folder / "monitor_evaluations.jsonl", b"")
                previous = manifest["payload_sha256"]
                print("FINAL_DASHBOARD_UPDATED", len(payload["runs"]), previous, flush=True)
            if args.once:
                return
        except (RuntimeError, OSError, ValueError, KeyError) as error:
            print("FINAL_DASHBOARD_WAIT", type(error).__name__, str(error)[:160], flush=True)
            if args.once:
                raise
        time.sleep(min(120, max(0, deadline - time.time())))


if __name__ == "__main__":
    main()
