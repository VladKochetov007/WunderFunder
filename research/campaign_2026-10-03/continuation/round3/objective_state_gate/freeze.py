"""Record final artifacts and read-only source identities inside CPU wrapper."""

import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[4]
results = json.loads((here / "results.json").read_text())
assert results["all_pass"]
state = {
    "state": "FROZEN_SYNTHETIC_INSTRUMENT_PASS_ACTUAL_REPLAY_UNAVAILABLE",
    "command": "bash research/campaign_2026-10-03/continuation/round3/objective_state_gate/run.sh",
    "compute_command": "research/campaign_2026-10-03/continuation/run_cpu.sh round2-critic env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research/campaign_2026-10-03/continuation/round3/objective_state_gate/run_checks.py",
    "resources": "CPU1core/2GiB/900s; no GPU", "cost": results["cost"],
    "passed_checks": len(results["tests"]), "actual_scientific_replay": False,
    "gpu_approval": False, "next": "independent frozen matched_usage review",
}
(here / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
files = sorted(p for p in here.iterdir() if p.is_file() and p.name != "HASHES.json")
files += [root / p for p in ("scripts/train_rnn.py", "src/wnn/nn.py", "src/wnn/features.py")]
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(here / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
print(json.dumps(state))
