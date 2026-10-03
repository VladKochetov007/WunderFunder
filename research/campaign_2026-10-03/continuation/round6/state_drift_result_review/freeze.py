"""Freeze reviewer artifacts and small provenance; never rehash tensor payloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result = json.loads((HERE / "results.json").read_text())
    # Verify reviewed small provenance files have not changed; no second blob pass.
    assert all(sha(ROOT / name) == expected for name, expected in result["provenance"].items())
    (HERE / "COST.json").write_text(json.dumps({
        "audit_cpu_bodies": 1, "metadata_freeze_cpu_bodies": 1,
        "audit_seconds": result["seconds"], "audit_peak_rss_kib": result["peak_rss_kib"],
        "output_bytes_streamed_once": result["bytes_streamed_once"],
        "input_capture_bytes_read": 0, "GPU_calls": 0,
        "resources": "run_cpu.sh reviewer1core2GiB900s",
    }, indent=2) + "\n")
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name != "HASHES.json")
    hashes = {p.name: sha(p) for p in files}
    (HERE / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("Frozen independent result review: PASS recorded detectable-subthreshold; no second payload pass.")


if __name__ == "__main__":
    main()
