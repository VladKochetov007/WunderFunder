"""Bind the blind pre-mortem artifacts without reading candidate implementations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    current_hash = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    assert current_hash == results["prereg_sha256_before_tests"]
    cost = {
        "experiment_cpu_bodies": 1,
        "artifact_freeze_cpu_bodies": 1,
        "experiment_body_seconds": results["compute_seconds"],
        "experiment_peak_rss_kib": results["peak_rss_kib"],
        "resources": "run_cpu.sh reviewer; one core, 2 GiB, 900 s",
        "gpu_calls": 0,
        "token_cost": "not available from runtime",
    }
    (HERE / "COST.json").write_text(json.dumps(cost, indent=2) + "\n")
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name != "HASHES.json")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (HERE / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("Frozen pre-mortem; prereg hash unchanged; artifacts bound.")


if __name__ == "__main__":
    main()
