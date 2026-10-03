from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ledger = ROOT / "research/experiments.jsonl"
rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
number = (
    max(
        int(row["id"].split("-")[-1])
        for row in rows
        if row.get("id", "").startswith("E-") and row["id"].split("-")[-1].isdigit()
    )
    + 1
)
records = []
combo = json.loads(
    (ROOT / "research/campaign_2026-10-03/scores/combo_matched_2seed_report.json").read_text()
)
records.append(
    {
        "id": f"E-{number}",
        "date": datetime.now().astimezone().isoformat(),
        "kind": "fullschedule combo two seeds matched controls folds0+1",
        "report": "research/campaign_2026-10-03/scores/combo_matched_2seed_report.json",
        "result": {
            "pooled": combo["pooled"]["paired_component_bootstrap"],
            "folds": {
                key: value["paired_component_bootstrap"] for key, value in combo["folds"].items()
            },
            "public_like": combo["reweighted"]["pub143"]["paired_component_bootstrap"],
        },
        "verdict": "replicates pooled gain on seed4 (+.001788); equal-seed bag +.001854 CI[.000420,.003409]; fold1small/class3negative; freeze additionalfold2+3 confirmation; no public target claim",
        "metric_id": combo["metric_id"],
    }
)
negative = json.loads(
    (ROOT / "research/campaign_2026-10-03/continuation/reproduction/results.json").read_text()
)
records.append(
    {
        "id": f"E-{number + 1}",
        "date": datetime.now().astimezone().isoformat(),
        "kind": "independent CPU strongstudent t1 causaloffset falsifier",
        "report": "research/campaign_2026-10-03/continuation/reproduction/REPORT.txt",
        "result": {
            "pooled": negative["pooled"],
            "gate_passed": negative["gate_passed"],
            "calibration_sequences": negative["calibration_sequences"],
            "evaluation_rows": negative["evaluation_selected_rows"],
            "wall_seconds": negative["wall_seconds"],
            "max_rss_mib": negative["max_rss_mib"],
        },
        "verdict": "REJECT testedfive-state correction; loses t1onbothfolds, meanWP-.000237 versusaffine. Localweakridgegain doesnottransfer. Exposed meta cross-fit notnested parent-training-independent; conditionalcomponentCI exploratory",
        "metric_id": negative["metric_id"],
    }
)
for record in records:
    if any(row.get("kind") == record["kind"] for row in rows):
        raise RuntimeError("already recorded")
with ledger.open("a") as output:
    for record in records:
        output.write(json.dumps(record) + "\n")
bandit = ROOT / "research/bandit/arms.json"
spec = json.loads(bandit.read_text())
spec["arms"]["muon_rnn"]["status"] = (
    "combo two matchedseeds positive; fold0dominates; frozenfold23 confirmationqueued"
)
spec["arms"]["muon_rnn"]["evidence"] += (
    f"; E-{number}: comboS4+.001788, bag+.001854 versus matchedcontrols; class3fold1negative"
)
spec["arms"]["muon_rnn"]["pulls"].append(f"E-{number}")
spec["continuation_evidence"]["strong_offset_verdict"] = {
    "ledger": f"E-{number + 1}",
    "status": "closed testedcorrection",
    "mean_wp_delta": negative["pooled"]["comparisons"]["offset_minus_affine"]["observed_delta"][2],
}
bandit.write_text(json.dumps(spec, indent=1) + "\n")
print([record["id"] for record in records])
