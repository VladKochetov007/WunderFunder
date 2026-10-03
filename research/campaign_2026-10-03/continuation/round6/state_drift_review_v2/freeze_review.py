"""Freeze terminal critic inventory; no receipts, launches or scientific payload reads."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from identity_controls import extract, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ACTOR = HERE.parent / "state_drift_v2"


def main() -> None:
    result = json.loads((HERE / "results.json").read_text())
    state = json.loads((HERE / "STATE.json").read_text())
    assert state["phase"] == "independent_review_frozen" and result["passed"]
    assert all(sha(ACTOR / name) == value for name, value in result["actor_hashes_verified"].items())
    assert all(sha(ROOT / name) == value for name, value in result["runtime_hashes"].items())
    assert all(sha(ROOT / name) == value for name, value in result["reviewed_native_sources"].items())
    namespace = {"HERE": ACTOR, "cap": SimpleNamespace(sha=sha)}
    own = extract(ACTOR / "run_real.py", {"own_hashes"}, namespace)["own_hashes"]()
    assert own == {name: result["actor_hashes_verified"][name] for name in own}
    inventory = {
        "verdict": state["verdict"],
        "scope": "terminal independent source/instrument review; not GPU authorization or empirical drift evidence",
        "candidate_sha256": own,
        "runtime_sha256": result["runtime_hashes"],
        "source": result["reviewed_native_sources"],
        "runtime_limits": result["runtime_limits"],
        "independent_state_drift_review": "PASS" if state["verdict"] == "PASS" else "BLOCK",
        "author_terminal_state_sha256": sha(ACTOR / "STATE.json"),
        "author_terminal_manifest_sha256": sha(ACTOR / "HASHES.json"),
        "independent_measurement_review_state_sha256": sha(HERE.parent / "capture_measurement_review/STATE.json"),
        "trajectory_sha256": sha(HERE.parent / "real_trajectory.json"),
        "metadata_helper_sha256": sha(HERE.parent / "prepare_capture_metadata.py"),
        "GPU_authorization": False,
        "live_GPU_drain_verified": False,
        "trajectory_limit": "source-inferred initial None and four warm starts; no saved intermediate theta trace",
    }
    (HERE / "candidate_runtime_hashes.json").write_text(json.dumps(inventory, indent=2) + "\n")
    identity = json.loads((HERE / "identity_results.json").read_text())
    (HERE / "COST.json").write_text(json.dumps({
        "CPU_test_bodies": 2, "metadata_freeze_bodies": 1,
        "synthetic_grammar_seconds": identity["seconds"],
        "terminal_targeted_audit_seconds": result["seconds"],
        "peak_test_rss_kib": max(result["peak_rss_kib"], identity["peak_rss_kib"]),
        "actual_tensor_reads": 0, "GRU_forwards": 0, "GPU_calls": 0,
        "actor_control_reruns": 0,
        "resources": "run_cpu.sh reviewer onecore2GiB900s",
    }, indent=2) + "\n")
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name != "HASHES.json")
    hashes = {p.name: sha(p) for p in files}
    assert "candidate_runtime_hashes.json" in hashes
    (HERE / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    assert all(sha(HERE / name) == value for name, value in hashes.items())
    print("Terminal critic review and exact candidate/runtime/source inventory frozen and verified.")


if __name__ == "__main__":
    main()
