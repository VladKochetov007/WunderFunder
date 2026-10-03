"""Index frozen third-round evidence without treating proxies as public gains."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CONT = ROOT / "research/campaign_2026-10-03/continuation"


def main() -> None:
    """Preserve report hashes and narrow verdicts in the existing ledger."""
    now = datetime.now().astimezone().isoformat()
    specs = [
        (
            "round3 raw-consistent strong GRU history replay",
            "matched_usage",
            "REPORT.md",
            "complete_report_frozen",
            {
                "components": 24,
                "queries": 288,
                "eligible_positions": 46572,
                "fully_equal_last33_model_input_rows": 0,
                "current_full_input_original_queries": 172,
                "raw_response_rms_current_full_input_original": 0.05793,
                "cpu_cache_drift_bound_failures": 7,
            },
            "Functional history use supported on exposed panel; seam support and engineered suffix prevent long-retention/value inference. Sampled-domain ratioWP is not full officialWP. Independent moment review pending; no training proposal.",
        ),
        (
            "round3 input-observation coordinate identifiability witness",
            "coordinate_identifiability",
            "REPORT.txt",
            "complete",
            {
                "toy_rows": 32,
                "identical_input_fields_per_row": 112,
                "physical_quote_rows_differ": 31,
                "physical_move_transitions_differ": 16,
            },
            "Documentation-compatible two-world witness defeats physical identification from frozen supervised coordinate alone. Does not show actual encoder lacks anchors or ambiguity under independently fixed common target law.",
        ),
        (
            "round3 prefix physical event teacher rejection gate",
            "path_teacher_gate",
            "REPORT.txt",
            "report_frozen",
            {"candidate_count": 1, "raw_data_reads": 0},
            "Reject physical event-state teacher justification under changing-reference ambiguity; no general rejection of empirically useful causal representation. No GPU candidate approved.",
        ),
        (
            "round3 exact gradient state capture synthetic instrument",
            "objective_state_gate",
            "REPORT.md",
            "FROZEN_SYNTHETIC_INSTRUMENT_PASS_ACTUAL_REPLAY_UNAVAILABLE",
            {"passed_checks": 16, "actual_scientific_replay": False, "gpu_approval": False},
            "Synthetic CPU FP32/unfusedAdamW instrument passes, detects missing state. Actual fast/Muon/fp16 online capture absent. EMA-selected export does not supply contemporaneous raw/EMA pair, both hidden histories or optimizer/RNG/scaler phase.",
        ),
    ]
    specs.extend(
        [
            (
                "round3 independent sampled history arithmetic review",
                "usage_review",
                "REPORT.md",
                "FROZEN_INDEPENDENT_REVIEW_COMPLETE",
                {
                    "actor_artifacts_verified": 128,
                    "maximum_moment_difference": 7.28e-12,
                    "exact_panel_after511_CPU_wp": 0.6619506180037982,
                    "sampled_identity_wp": 0.6404556770959638,
                },
                "Scoped instrument/arithmetic independently supported. Sampled identity undershoots known domain by.021495; intervention full-domain outputs absent, pairedCI crosseszero. No beneficial corruption or fresh/public confirmation.",
            ),
            (
                "round3 exact engineered suffix washout retention",
                "washout_usage",
                "REPORT.md",
                "complete_report_frozen",
                {
                    "queries": 288,
                    "accepted": 207,
                    "rejected": 81,
                    "components": 24,
                    "accepted_suffix_rows_exactly_equal": 33,
                    "feature_atol": 0,
                    "feature_rtol": 0,
                    "response_rms": 0.027021,
                    "inverse_probability_response_rms": 0.025442,
                    "component_ci95": [0.021176, 0.031012],
                    "identity_rms": 6.86744e-8,
                },
                "Carried recurrent sensitivity survives equal33row fullinput suffix on accepted exposed origins. No predictive-value/support/gain claim. Independent raw/forward review pending.",
            ),
            (
                "round3 anonymous observable stopping simulator",
                "observable_simulator",
                "REPORT.txt",
                "report_frozen",
                {
                    "replication_seeds": 4,
                    "representation_count": 1,
                    "fits": 36,
                    "nominal_history_MSE_reduction_vs_row_percent": 5.20,
                    "nominal_history_MSE_reduction_vs_reset_percent": 5.29,
                    "sign_flip_MSE_reduction_vs_row_percent": -10.99,
                    "stale_MSE_reduction_vs_row_percent": -3.71,
                    "raw_competition_rows_read": 0,
                },
                "Conditional constructive toy; nominal gain and absent/unidentifiable controls pass, both nuisance transports fail. Not real generator, new dual-clock/public evidence or GPU approval. Independent review pending.",
            ),
        ]
    )
    ledger = ROOT / "research/experiments.jsonl"
    existing = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    by_kind = {row.get("kind"): row["id"] for row in existing}
    number = max(int(row["id"][2:]) for row in existing if row.get("id", "").startswith("E-")) + 1
    records = []
    references = {}
    for kind, scope, report_name, phase, result, verdict in specs:
        if kind in by_kind:
            references[scope] = by_kind[kind]
            continue
        directory = CONT / "round3" / scope
        state = json.loads((directory / "STATE.json").read_text())
        if state.get("phase", state.get("state")) != phase:
            raise ValueError(f"report not frozen: {scope}")
        path = directory / report_name
        record = {
            "id": f"E-{number}",
            "date": now,
            "kind": kind,
            "model": "gpt-6.1-sol",
            "reasoning_effort": "medium",
            "reports": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ],
            "result": result,
            "verdict": verdict,
            "evaluation_scope": "explicit synthetic fixture or reused exposed diagnostic panel; no fresh public confirmation",
        }
        records.append(record)
        references[scope] = record["id"]
        number += 1
    with ledger.open("a") as output:
        for record in records:
            output.write(json.dumps(record, allow_nan=False) + "\n")
    bandit_path = ROOT / "research/bandit/arms.json"
    bandit = json.loads(bandit_path.read_text())
    bandit.setdefault("continuation_evidence", {})["round3_frozen"] = {
        "updated_at": now,
        "records": references,
        "decision": "capture actual coupled updates, require full suffix equality, treat physical coordinate as unidentified; no public-gain posterior update",
    }
    bandit_path.write_text(json.dumps(bandit, indent=1) + "\n")
    state_path = CONT / "STATE.json"
    state = json.loads(state_path.read_text())
    state.update(
        updated_at=now,
        phase="round3_frozen_followups_round4_capture_builder",
        round3_evidence=references,
        current_agents="round4/agents.json",
        next="independent usage/capture reviews; finish combo folds; preserve evidence and continueuntil23:00",
    )
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"appended": [record["id"] for record in records], "records": references}))


if __name__ == "__main__":
    main()
