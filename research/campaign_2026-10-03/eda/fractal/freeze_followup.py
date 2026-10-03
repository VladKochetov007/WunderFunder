from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
hypotheses = json.loads((OUT / "HYPOTHESES.json").read_text())
hypotheses["H_stationarity_followup"] = {
    "status": "frozen_after_initial_negative_before_followup_scoring",
    "sample": "same already-exposed first 12 continuation validation groups",
    "seed": 20261004,
    "premise": "Near-zero signed increment DFA slopes can arise by telescoping a stationary encoded feature rather than exploitable market antipersistence",
    "matched_control": "shuffle observed event-time i0_p0 LEVELS then difference; 16 permutations per sequence; compare original differences and initial shuffled-difference null",
    "survival": "absolute median original minus level-shuffled DFA slope over 16..256 <0.05 supports sufficiency of stationary-level explanation, not an equivalence proof",
    "synthetic_checks": "iid normal increments approximately 0.5 and differenced iid normal levels approximately 0 at the same scales",
    "limits": "No inference about underlying raw asset price, no new holdout evidence, no claim about all 112 inputs",
}
(OUT / "HYPOTHESES.json").write_text(json.dumps(hypotheses, indent=2) + "\n")
state = json.loads((OUT / "STATE.json").read_text())
state["phase"] = "stationarity_followup_frozen"
state["pending_work"] = ["run stationary level control", "inspect figure", "notebook", "report"]
(OUT / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
