"""Freeze independent review outputs and verify the preceding gate stayed frozen."""

import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[4]
gate = here.parent / "objective_state_gate"
gate_hashes = json.loads((gate / "HASHES.json").read_text())
for name, expected in gate_hashes.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
results = json.loads((here / "results.json").read_text())
state = {
    "state": "FROZEN_INDEPENDENT_REVIEW_COMPLETE",
    "verdict": "scoped actor claims supported; no material correction needed; stronger interpretations rejected",
    "compute_command": "bash research/campaign_2026-10-03/continuation/round3/usage_review/run.sh",
    "resource_wrapper": "run_cpu.sh round2-critic onecore/2GiB/900s",
    "cost": results["cost"],
    "domain_diagnostic_cost": json.loads((here / "domain_diagnostic.json").read_text())["cost"],
    "actor_frozen_hashes_verified": results["actor_hashes_verified_unchanged"],
    "preceding_objective_gate_hashes_unchanged": len(gate_hashes),
    "actual_model_forwards": 0, "future_washout_results_read": False,
    "gpu_approval": False, "next": "await controller request; frozen artifacts preserved",
}
(here / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(here.iterdir()) if p.is_file() and p.name != "HASHES.json"}
(here / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
print(json.dumps(state))
