"""Index independent confirmation and bounded fifth-round instruments."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CONT = ROOT / "research/campaign_2026-10-03/continuation"


def main() -> None:
    """Append once per frozen evidence kind without promoting proxy score gains."""
    specs = [
        (
            "round5 independent loss geometry instrument rejection",
            "round5/mechanism_review",
            "REPORT.txt",
            "review_frozen",
            {
                "actual_captures_read": 0,
                "adapter_version_binding_incompatible": True,
                "independently_accepted_decomposition_bound": None,
                "default_floor_transfer_forbidden": True,
                "author_sources_preserved": True,
            },
            "Synthetic algebra useful, actual captured-update interpretation blocked pending candidate-version binding, finite/skip/participation guards and independent AMP decomposition bound. Default replay floors cannot certify separate-component gradient sum; no science/promotion/threshold relaxation.",
        ),
        (
            "round4 independent v3 private RNG normalization rejection",
            "round4/capture_review_v3",
            "REPORT.md",
            "frozen_v3_independent_review_complete",
            {
                "targeted_assertions": 134,
                "pristine_CPU_delta": 0,
                "NumPy_and_moment_corrections_pass": True,
                "remaining_blocker": "private Torch RNG setter accepts and normalizes noncanonical padding",
                "GPU_approved": False,
                "legacy_E385_recovered_exact_hash": True,
            },
            "Terminal independent review blocks v3 for one actual prestate normalization witness, not observed gradient/GPU corruption. Separate minimal v4 exact private set/get identity required; no tolerances relaxed.",
        ),
        (
            "round5 actual tiny GRU common macrostep credit control",
            "round5/credit_control",
            "REPORT.md",
            "FROZEN_CPU_CONTROL_PROOF_PASS",
            {
                "checks": 58,
                "forward_targets_common_loss_bitexact": True,
                "planted_credit_blocking_pass": True,
                "equal_gradient_null_pass": True,
                "matching_update_clocks": True,
                "GPU_AMP_parity": False,
                "empirical_credit_benefit": False,
            },
            "Terminal synthetic CPU proof separates midpoint hidden detach from common macrostep losses/update cadence. Bound source functions and coefficient oracle checked; no real fit, causal recipe attribution, credit improvement or public score claim.",
        ),
        (
            "round4 final v2 typed moment amendment and exact E385 recovery",
            "round4/capture_review_v2",
            "REPORT.md",
            "frozen_v2_review",
            {
                "prior_record": "E-385",
                "original_report_spec_recovered_exact_hash": True,
                "historical_version": "research/campaign_2026-10-03/continuation/evidence_versions/E385",
                "additional_blocker": "warm momentum_buffer/exp_avg/exp_avg_sq float64 aliases silently cast",
                "GPU_approved": False,
            },
            "Final v2 rejection adds warm-moment type integrity to NumPy gap; original indexed report/spec recovered by exact SHA256 and retained separately. Ledger untouched; indexing now requires terminal agent completion plus verified final bindings.",
        ),
        (
            "round4 isolated v3 exact RNG and moment correction",
            "round4/live_capture_v3",
            "REPORT.md",
            "frozen_v3_request_final_independent_review",
            {
                "CPU_tests": 66,
                "normal_repeat_skip_original_CPU_delta": 0,
                "empirical_CPU_floor": 0,
                "independent_repeats_per_control": 6,
                "GPU_launched": False,
                "CUDA_RNG_engine_validation_executed": False,
            },
            "Builder terminal complete and20protected file hashes match; v1/v2 preserved. Exact typed/byte RNG and source-native moment grammar corrected; final independent review and actual CUDA parity remain pending, not self-approved.",
        ),
        (
            "independent combo fold23 gate and metadata review",
            "round4/combo_confirmation_review",
            "REPORT.md",
            "FROZEN_INDEPENDENT_CONFIRMATION_REVIEW_PASS",
            {
                "gate_passed": True,
                "bulk_model_prediction_loads": 0,
                "real_score_payload_recomputed": False,
            },
            "Combined recipe confirmation supported; scorer toy fixtures and member lineage checked, real prediction payload not independently rescored. Common macrostep credit-vs-cadence experiment preregistered, not run.",
        ),
        (
            "round5 preregistered loss geometry instrument",
            "round5/mechanism_prereg",
            "REPORT.md",
            "preregistered_instrument_tested_awaiting_controller",
            {
                "tiny_controls_passed": True,
                "actual_capture_analysis_executed": False,
                "GPU_launched": False,
                "fits": 0,
            },
            "Runnable CPU geometry instrument frozen before actual CUDA results; component conflict can be local, structural denominators unavailable and TBPTT drift unidentified without same-weight same-prefix reference.",
        ),
        (
            "round5 two input-only observable support tests",
            "round5/observable_support",
            "REPORT.txt",
            "report_frozen",
            {
                "tests": 2,
                "sequences": 24,
                "queries": 288,
                "rows_after_511": 467712,
                "scalar_any_age_mismatches": 40448,
                "production_top_any_age_mismatches": 0,
                "held_fresh_query_membership_changed": False,
                "fits": 0,
                "GRU_forwards": 0,
            },
            "Scalar/any clocks differ off-query but same age-zero strata do not rescue the negative peer gate. Production top-pair and any22 clocks agree on this panel, not universally. Gauge controls limit physical interpretation, not observable forecast validity.",
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
        folder = CONT / directory
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
            "public_confirmation": False,
        }
        records.append(record)
        references[directory] = record["id"]
        number += 1
    with ledger.open("a") as output:
        for record in records:
            output.write(json.dumps(record, allow_nan=False) + "\n")
    path = CONT / "STATE.json"
    state = json.loads(path.read_text())
    watch = json.loads((CONT / "watch_status.json").read_text())
    state.update(
        updated_at=now,
        round5_evidence=references,
        phase="v4_capture_correction_and_instrument_review",
        active_training=[job["name"] for job in watch["active_gpu_jobs"]],
        training_status_observed_at=watch["time"],
        next="Freeze minimal v4 and independent private-engine roundtrip review; default CUDA capture only if approved; instrument decomposition review; continue until23:00 Kyiv.",
    )
    path.write_text(json.dumps(state, indent=2) + "\n")
    path = ROOT / "research/bandit/arms.json"
    bandit = json.loads(path.read_text())
    bandit.setdefault("continuation_evidence", {})["round5_frozen"] = {
        "updated_at": now,
        "records": references,
        "decision": "confirmation accepted, peer proxy stays rejected; no new scientific CUDA arm",
    }
    path.write_text(json.dumps(bandit, indent=1) + "\n")
    print(json.dumps({"appended": [record["id"] for record in records], "records": references}))


if __name__ == "__main__":
    main()
