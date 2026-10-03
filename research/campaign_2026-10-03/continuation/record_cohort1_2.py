from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
roles = ("hft", "linear", "nonlinear", "olympiad", "signal", "statphys", "fractal", "econometrics")
ledger = ROOT / "research/experiments.jsonl"
rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
if any(row.get("kind") == "Codex independent EDA continuation cohorts1+2" for row in rows):
    raise RuntimeError("cohort evidence already recorded")
number = (
    max(
        int(row["id"].split("-")[-1])
        for row in rows
        if row.get("id", "").startswith("E-") and row["id"].split("-")[-1].isdigit()
    )
    + 1
)
reports = {}
for role in roles:
    path = ROOT / f"research/campaign_2026-10-03/eda/{role}/REPORT.txt"
    reports[role] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
record = {
    "id": f"E-{number}",
    "date": datetime.now().astimezone().isoformat(),
    "kind": "Codex independent EDA continuation cohorts1+2",
    "model": "gpt-6.1-sol",
    "reasoning_effort": "medium",
    "reports": reports,
    "result": {
        "universal_event_clock": "failed matched gates across signal/fractal/econometrics;statphys row clock ties",
        "hard_a6_carry": "rejected current input WP.46431 versus.34752",
        "phase8_replacement": "rejected .56277 versus EMA.57485 and placebo.56329",
        "physical_spread_decoder": "rejected all20testedsequences",
        "stationary_level_null": "explains most nearzero incrementDFA",
        "t1_causal_offsets": "weak ridge train-calibrated+ .022698 versusaffine; strong-control testpending",
    },
    "verdict": "retain negative controls; narrow age/activity calibration and target-specific states need strong-baseline component tests; no public improvement established",
    "metric_id": "global_wp_clip2_v1",
    "limitations": [
        "prior sampleexposure",
        "clean-room sequenceintervals notcampaigncomponentCIs",
        "ridge gainsnotstrongbaselinegains",
    ],
}
with ledger.open("a") as output:
    output.write(json.dumps(record) + "\n")
bandit = ROOT / "research/bandit/arms.json"
spec = json.loads(bandit.read_text())
arm = spec["arms"]["arch_two_timescale_state"]
arm.update(
    status="closed tested configuration: both folds negative, pooled -0.001076",
    evidence=arm["evidence"] + "; E-357 confirms fold1 -0.000195, pooledCI [-.002658,.000152]",
)
if "E-357" not in arm["pulls"]:
    arm["pulls"].append("E-357")
spec["arms"]["muon_rnn"]["status"] = (
    "full2fold +.001595; combo seed3 +.002157 vs control, incrementoverMuon unconfirmed; matchedseed4scoringpending"
)
spec["arms"]["muon_rnn"]["evidence"] += (
    "; E-356 combo seed3 minusMuon +.000562 CIcrosseszero; not additive"
)
spec["arms"]["target_stopped_bootstrap"]["status"] = (
    "pooled positive/fold1 negative; jointcombo matchedseed4pending"
)
spec["arms"]["student_chunk1000"]["status"] = (
    "pooled positive/fold1 negative; jointcombo matchedseed4pending"
)
spec["continuation_evidence"] = {
    "ledger": record["id"],
    "reviews": [
        "research/campaign_2026-10-03/continuation/COHORT1_REVIEW.md",
        "research/campaign_2026-10-03/continuation/COHORT2_REVIEW.md",
    ],
    "policy": "Do not sum correlated combo/Muon/chunk/bootstrap effects or promote weak-ridge improvements. Numeric public priors unchanged until strong-control evidence.",
}
bandit.write_text(json.dumps(spec, indent=1) + "\n")
print(record["id"])
