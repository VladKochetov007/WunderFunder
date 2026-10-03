"""Freeze bounded specification artifacts and exact source provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    results = json.loads((OWN / "results.json").read_text())
    source_checks = {name: digest(ROOT / name) == expected for name, expected in results["sources"].items()}
    source_checks["executed_fixture_source"] = digest(OWN / "fixtures.py") == results["fixture_source_sha256"]
    source_checks["preregistered_tests"] = digest(OWN / "TESTS.json") == results["prereg_sha256"]
    state = json.loads((OWN / "STATE.json").read_text())
    state.update(phase="frozen", pending=[], last_failure=None,
                 actual_combo_replay_admitted=False, controls=len(results["checks"]),
                 rejected_adversaries=len(results["adversaries"]), source_checks=source_checks,
                 command="bash research/campaign_2026-10-03/continuation/round4/capture_critic/run.sh",
                 executed_inputs="synthetic only; no sampleIDs", actual_backend="CPU FP32/GradScaler CPU/Muon",
                 awaiting="controller-provided frozen builder for independent review")
    (OWN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    prior = OWN.parents[1] / "round3/objective_state_gate"
    hashes = {"own": {p.name: digest(p) for p in sorted(OWN.iterdir()) if p.is_file() and p.name not in {"HASHES.json", "freeze.log"}},
              "permitted_prior": {str(p): digest(p) for p in [prior / "export_schema.json", prior / "REPORT.md"]},
              "source_identity": results["sources"]}
    (OWN / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    if not all(source_checks.values()) or not results["passed"]:
        raise RuntimeError("Frozen source/state checks failed")
    print(json.dumps({"passed": results["passed"], "sources_preserved": source_checks}))


if __name__ == "__main__":
    main()
