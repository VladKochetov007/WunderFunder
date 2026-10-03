"""Freeze the independent source review and confirm consumed actor files unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent / "state_drift"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result = json.loads((HERE / "results.json").read_text())
    assert all(sha(ACTOR / name) == value for name, value in
               result["consumed_actor_artifact_hashes_verified"].items())
    assert sha(HERE / "TEST_PLAN.json") == result["test_plan_sha256"]
    (HERE / "COST.json").write_text(json.dumps({
        "test_cpu_bodies": 1, "metadata_freeze_cpu_bodies": 1,
        "test_seconds_including_hashes": result["seconds_including_hashes"],
        "test_peak_rss_kib": result["peak_rss_kib"],
        "gpu_calls": 0, "model_forwards": 0,
        "resources": "run_cpu.sh reviewer; one core, 2 GiB, 900 seconds",
    }, indent=2) + "\n")
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name != "HASHES.json")
    (HERE / "HASHES.json").write_text(json.dumps({p.name: sha(p) for p in files}, indent=2) + "\n")
    print("Frozen BLOCK review; consumed actor artifacts and pretest plan unchanged.")


if __name__ == "__main__":
    main()
