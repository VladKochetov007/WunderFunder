"""Record frozen, independently reviewed diagnostics without performance promotion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CONT = ROOT / "research/campaign_2026-10-03/continuation"


def main() -> None:
    """Append immutable report references and refresh continuation registries."""
    now = datetime.now().astimezone().isoformat()
    ledger = ROOT / "research/experiments.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    number = max(int(row["id"][2:]) for row in rows if row.get("id", "").startswith("E-")) + 1
    specs = [
        (
            "round2 objective estimator and independent coefficient review",
            ["round2/objective/REPORT.md", "round2/spec_critic/objective_review_REPORT.md"],
            {
                "main_mass_cv_chunk1000": 0.1516350673,
                "teacher_proxy_gradient_relative_shift": 0.05106713,
                "sample_sequences": 256,
                "unique_components": 242,
                "components_spanning_LOBO_batches": 12,
            },
            "Narrow estimator identities supported; cached-distillation bypasses running norm. Full-mixture, real lagged EMA and full GRU updates unmeasured; no training promotion.",
        ),
        (
            "round2 future path laws and round3 independent raw decoder audit",
            ["round2/first_passage/REPORT.md", "round3/decoder_audit/REPORT.txt"],
            {
                "selected_origins": 1393,
                "outside_origins": 445,
                "mapped_outside_bound_violations": 417,
                "outside_unknown": 28,
                "synthetic_exact_label_differences": 252,
                "synthetic_correct_physical_moves": 0,
            },
            "Entry-inclusive both-extrema law rejected conditional on supervised coordinate. Physical spread/moves unidentifiable from label differences alone; no physical-generator claim.",
        ),
        (
            "round2 strong GRU history usage diagnostic",
            ["round2/usage/REPORT.md"],
            {
                "components": 6,
                "queries": 342,
                "scored_queries": 14,
                "i0_past256_prediction_rms": 0.336241,
                "i1_past256_prediction_rms": 0.170195,
            },
            "Both streams and older history functionally used. OOD packed interventions and tiny selected score panel do not identify missing predictive information; raw-consistent follow-up running.",
        ),
        (
            "round2 blind specification counterexamples and CPU streaming equivalence",
            ["round2/spec_critic/PREMORTEM.md", "round2/spec_critic/stream_equivalence_REPORT.md"],
            {
                "raw_rows": 512,
                "components": 2,
                "maximum_partition_output_delta": 6.6757e-6,
                "deliberate_hidden_no_reset_delta": 14.72979,
            },
            "Narrow CPU callable invariances pass and instrument detects reset bug. Not native/deployed/GPU/full-sequence parity; no WP gain.",
        ),
        (
            "round3 hybrid exit-law falsification",
            ["round3/hybrid_pair/REPORT.md"],
            {
                "all_origins": 1393,
                "complete_origins": 1007,
                "outside_complete": 260,
                "independent_hybrid_exact_pairs": 0,
                "shared_exit_outside_exact_pairs": 74,
                "shared_exit_interior_exact_pairs": 55,
                "extrema_interior_exact_pairs": 584,
            },
            "Both exact-law representatives rejected. All 74 shared outside matches have historical K=2; none of 85 eligible W>=3 cases match. Conditional association cannot revive general offset law.",
        ),
    ]
    width_path = ROOT / "research/campaign_2026-10-03/scores/width320_prereg_report.json"
    width = json.loads(width_path.read_text())
    specs.append(
        (
            "width320 fixed ep3 twofold confirmation",
            ["../scores/width320_prereg_report.json"],
            width["pooled"]["paired_component_bootstrap"]["WIDTH320_minus_CTRL"],
            "Both fits and exports succeeded; pooled WP delta +.00007249 CI[-.001211,.001313]. No demonstrated benefit from extra width on this fixed twofold panel; no public claim.",
        )
    )
    existing = {row.get("kind"): row["id"] for row in rows}
    references = {}
    records = []
    for kind, paths, result, verdict in specs:
        if kind in existing:
            references[kind] = existing[kind]
            continue
        reports = []
        for relative in paths:
            path = CONT / relative
            reports.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        record = {
            "id": f"E-{number}",
            "date": now,
            "kind": kind,
            "model": "gpt-6.1-sol",
            "reasoning_effort": "medium",
            "reports": reports,
            "result": result,
            "verdict": verdict,
            "evaluation_scope": "exposed diagnostic panels; not fresh public confirmation",
        }
        records.append(record)
        references[kind] = record["id"]
        number += 1
    with ledger.open("a") as output:
        for record in records:
            output.write(json.dumps(record, allow_nan=False) + "\n")
    bandit = ROOT / "research/bandit/arms.json"
    spec = json.loads(bandit.read_text())
    spec.setdefault("continuation_evidence", {})["round2_and_decoder_hybrid"] = {
        "updated_at": now,
        "records": references,
        "decision": "retain distinct mechanisms; no numerical public-gain posterior update from diagnostic proxies",
    }
    bandit.write_text(json.dumps(spec, indent=1, allow_nan=False) + "\n")
    registry_path = CONT / "round2/agents.json"
    registry = json.loads(registry_path.read_text())
    registry.update(phase="frozen_reviewed", updated_at=now, current_registry="round3/agents.json")
    for agent in registry["agents"]:
        agent["status"] = (
            "completed_closed"
            if agent["role"] == "objective"
            else "round2_completed_thread_continues_round3"
        )
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    active = [
        ("coordinate_identifiability", "01a1019d-b660-75c1-bcfe-56fc8a15f897", "Averroes"),
        ("path_teacher_gate", "01a101e6-0368-7002-9b08-952d37d2129d", "Helmholtz"),
        ("matched_usage", "01a1019d-b707-7783-932a-f4267819737d", "Heisenberg"),
        ("objective_state_gate", "01a101f9-30d6-78d3-9a2b-ab61367428ad", "Carver"),
    ]
    (CONT / "round3/agents.json").write_text(
        json.dumps(
            {
                "round": 3,
                "updated_at": now,
                "phase": "bounded_implementation_and_falsification",
                "model": "gpt-6.1-sol",
                "reasoning_effort": "medium",
                "deadline_local": "2026-10-03T23:00:00+03:00",
                "agents": [
                    {
                        "role": role,
                        "agent_id": agent_id,
                        "nickname": nickname,
                        "status": "running",
                        "write_scope": str((CONT / "round3" / role).relative_to(ROOT)),
                        "model": "gpt-6.1-sol",
                        "reasoning_effort": "medium",
                    }
                    for role, agent_id, nickname in active
                ],
            },
            indent=2,
        )
        + "\n"
    )
    state_path = CONT / "STATE.json"
    state = json.loads(state_path.read_text())
    watch = json.loads((CONT / "watch_status.json").read_text())
    state.update(
        phase="round2_reviewed_round3_running",
        updated_at=now,
        current_agents="round3/agents.json",
        evidence_records=references,
        active_training=[job["name"] for job in watch["active_gpu_jobs"]],
        training_status_observed_at=watch["time"],
        next="review raw-consistent history and identifiability gates; score queued width/fold23; monitor and preserve until23:00",
    )
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"appended": [record["id"] for record in records], "records": references}))


if __name__ == "__main__":
    main()
