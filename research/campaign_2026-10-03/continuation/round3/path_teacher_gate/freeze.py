from __future__ import annotations

import hashlib
import json
from pathlib import Path

own = Path(__file__).resolve().parent
initial = json.loads((own / "initial_results.json").read_text())
final = json.loads((own / "results.json").read_text())
for key in ("decision", "modes", "known_generator", "controls", "witnesses", "IDs"):
    assert initial[key] == final[key], key
assert initial["provenance"]["synthetic_arrays_sha256"] == final["provenance"]["synthetic_arrays_sha256"]
cost = {
    "successful_gate_runs": 2,
    "gate_wall_seconds_sum": initial["cost"]["wall_seconds"] + final["cost"]["wall_seconds"],
    "gate_cpu_seconds_sum": initial["cost"]["cpu_seconds"] + final["cost"]["cpu_seconds"],
    "peak_rss_kib": max(initial["cost"]["max_rss_kib"], final["cost"]["max_rss_kib"]),
    "format_and_lint_commands": 2,
    "raw_rows_read": 0,
    "monetary_usage": "unavailable",
}
(own / "COST.json").write_text(json.dumps(cost, indent=2) + "\n")
state = json.loads((own / "STATE.json").read_text())
state.update(phase="report_frozen", formatting_replay_equal=True, lint="passed", cost=cost)
(own / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
manifest = {
    "input_hashes": final["provenance"]["input_hashes"],
    "own_artifact_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(own.iterdir())
        if p.is_file() and p.name != "PROVENANCE.json"
    },
    "scope": "toy only; no raw data; imported frozen modules verified unchanged",
}
(own / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Frozen report; formatting replay equal; final input/artifact hashes recorded.")
