"""Record bounded measurements and reviews without proxy performance promotion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CONT = ROOT / "research/campaign_2026-10-03/continuation"


def main() -> None:
    """Index frozen report hashes, verification scope and current worker stages."""
    specs = [
        (
            "round4 blind actual update-state specification fixtures",
            "capture_critic",
            "PREMORTEM.md",
            "frozen",
            {"controls": 13, "rejected_adversaries": 11, "actual_GPU_replay": False},
            "CPU FP32/CPU GradScaler/Muon fixtures detect state/order/skip traps; no actual fastCUDA replay or GPU candidate approval.",
        ),
        (
            "round4 samefold teacher and propensity recorded membership audit",
            "teacher_lineage",
            "REPORT.txt",
            "frozen",
            {
                "teachers": 8,
                "propensity_ancestors": 4,
                "recorded_eval_component_intersections": 0,
                "checkpoint_files_hashed": 73,
                "checkpoint_bytes_hashed": 98684359,
            },
            "Recorded fitting pools and graph/order/cache/checkpoint identities support same-fold exclusion. Historical full-source/executed batch attestation absent; no contamination inferred from missing evidence; no prediction-array rehash or gain claim.",
        ),
        (
            "round4 independent exact suffix washout reproduction",
            "washout_review",
            "REPORT.txt",
            "review_frozen",
            {
                "queries_reconstructed": 288,
                "accepted": 207,
                "components": 24,
                "independent_actual_checkpoint_window_pairs": 4,
                "maximum_replay_difference": 0,
            },
            "Narrow carried-recurrent sensitivity independently supported after207 exact33row fullinput suffixes. Four full-prefix checkpoint pairs replay exactly; predictive value/economic support remain unknown.",
        ),
        (
            "round4 finite local state forgetting instrument",
            "state_forgetting",
            "REPORT.md",
            "FROZEN_FINITE_LOCAL_INSTRUMENT_COMPLETE",
            {
                "components": 8,
                "signed_directions": 16,
                "updates": 289,
                "above_numerical_floor_cases_at_end": 16,
                "initial_hidden_RMS_perturbation_fraction": 0.01,
                "endpoint_hidden_difference_fraction_range": [0.172, 0.409],
            },
            "Finite initial-state influence persists under identical inputs, frozen weights. No monotone/global contraction, stability theorem, actual training state-drift or predictive-gain claim; sampled states need not be reachable counterfactual prefixes.",
        ),
        (
            "round4 independent observable simulator reproduction",
            "simulator_review",
            "REPORT.md",
            "complete_report_frozen",
            {
                "author_artifacts_verified": 26,
                "fits_repeated": 0,
                "capacity_control": "matched width/learner, not design rank or penalty geometry",
            },
            "Conditional constructive toy scores/causality/stopping/splits reproduce. No peer-pooling necessity or real candidate established. One input-only peer-innovation bridge test commissioned, no GPU promotion.",
        ),
        (
            "round4 isolated sourcebound online capture prototype",
            "live_capture",
            "REPORT.md",
            "frozen_for_independent_controller_review",
            {
                "CPU_tests": 25,
                "normal_repeat_skip_uninstrumented_CPU_delta": 0,
                "real_fit_panel_identified": True,
                "GPU_launched": False,
                "GPU_parity_established": False,
                "today_fullfit_state_recovered": False,
            },
            "Actual-code tinyCPU capture/replay supported; fresh controlled panel trajectory and panel normalization are explicit departures. CUDA parity/source review pending. Frozen actor shell bypasses campaign admission; controller integration prepared separately. No estimator intervention or live trainer patch applied.",
        ),
    ]
    ledger = ROOT / "research/experiments.jsonl"
    existing = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    kinds = {row.get("kind"): row["id"] for row in existing}
    number = max(int(row["id"][2:]) for row in existing if row.get("id", "").startswith("E-")) + 1
    now = datetime.now().astimezone().isoformat()
    records = []
    references = {}
    for kind, directory, report_name, phase, result, verdict in specs:
        if kind in kinds:
            references[directory] = kinds[kind]
            continue
        folder = CONT / "round4" / directory
        state = json.loads((folder / "STATE.json").read_text())
        if state.get("phase", state.get("state")) != phase:
            raise ValueError(f"not frozen: {directory}")
        report = folder / report_name
        record = {
            "id": f"E-{number}",
            "date": now,
            "kind": kind,
            "model": "gpt-6.1-sol",
            "reasoning_effort": "medium",
            "reports": [
                {
                    "path": str(report.relative_to(ROOT)),
                    "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                }
            ],
            "result": result,
            "verdict": verdict,
            "evaluation_scope": "synthetic/metadata/exposed diagnostic panel; no public confirmation",
        }
        records.append(record)
        references[directory] = record["id"]
        number += 1
    with ledger.open("a") as output:
        for record in records:
            output.write(json.dumps(record, allow_nan=False) + "\n")
    bandit_path = ROOT / "research/bandit/arms.json"
    bandit = json.loads(bandit_path.read_text())
    bandit.setdefault("continuation_evidence", {})["round4_frozen"] = {
        "updated_at": now,
        "records": references,
        "decision": "capture source review before GPU parity; retention supported but deficit untested; teacher recorded exclusion supports shared lineage",
    }
    bandit_path.write_text(json.dumps(bandit, indent=1) + "\n")
    state_path = CONT / "STATE.json"
    state = json.loads(state_path.read_text())
    watch = json.loads((CONT / "watch_status.json").read_text())
    state.update(
        updated_at=now,
        round4_evidence=references,
        phase="round4_reviews_and_peer_bridge_active",
        active_training=[job["name"] for job in watch["active_gpu_jobs"]],
        training_status_observed_at=watch["time"],
        next="finish fold3, review exact capture and pendingpeer bridge; GPU smoke onlyafterindependentreview/exclusiveadmission; researchuntil23:00",
    )
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"appended": [record["id"] for record in records], "records": references}))


if __name__ == "__main__":
    main()
