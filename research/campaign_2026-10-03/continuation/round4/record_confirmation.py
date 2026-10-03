"""Record frozen confirmations and rejected probes without public-score claims."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CONT = ROOT / "research/campaign_2026-10-03/continuation"


def binding(path: Path) -> dict[str, str]:
    """Bind a small frozen report by path and content hash."""
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    """Evaluate the unchanged four-gate confirmation and append new evidence once."""
    report_path = ROOT / "research/campaign_2026-10-03/scores/combo_fold23_prereg_report.json"
    report = json.loads(report_path.read_text())
    primary = report["pooled"]["paired_component_bootstrap"]["COMBO_minus_CTRL"]
    public_like = report["reweighted"]["pub143"]["paired_component_bootstrap"]["COMBO_minus_CTRL"]
    classes = report["pooled"]["by_class"]["3"]
    fold_class3 = {
        fold: values["by_class"]["3"]["COMBO"]["wp"] - values["by_class"]["3"]["CTRL"]["wp"]
        for fold, values in report["folds"].items()
    }
    gates = {
        "positive_primary_mean_and_interval": primary["mean_diff"] > 0 and primary["ci95"][0] > 0,
        "positive_public_like_mean": public_like["mean_diff"] > 0,
        "positive_pooled_class3": classes["COMBO"]["wp"] > classes["CTRL"]["wp"],
        "no_fold_class3_below_minus_002": all(value >= -0.002 for value in fold_class3.values()),
    }
    now = datetime.now().astimezone().isoformat()
    decision = {
        "observed_at": now,
        "gates": gates,
        "recipe_adoption_supported": all(gates.values()),
        "primary": primary,
        "public_like": public_like,
        "pooled_class3_delta": classes["COMBO"]["wp"] - classes["CTRL"]["wp"],
        "fold_class3_delta": fold_class3,
        "bindings": [binding(report_path), binding(CONT / "FOLD23_CONFIRMATION.txt")],
        "limits": "Previously explored folds, combined recipe, fixed epoch3. No untouched-data, isolated-mechanism, or public-score claim.",
    }
    (CONT / "FOLD23_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")
    specs = [
        (
            "combo fixed fold23 confirmation gate",
            [report_path, CONT / "FOLD23_CONFIRMATION.txt"],
            decision,
            "Adopt unchanged recipe for subsequent controlled work if all frozen gates pass; no public score inferred.",
        ),
        (
            "round4 independent v1 capture rejection",
            [CONT / "round4/capture_review/REPORT.md"],
            {
                "GPU_approved": False,
                "blocking_gaps": [
                    "optimizer identity",
                    "integer exactness",
                    "independent CUDA floor",
                ],
            },
            "V1 remains rejected and preserved. V2 remediation must be independently reviewed before GPU parity.",
        ),
        (
            "round4 peer innovation frozen proxy falsifier",
            [CONT / "round4/peer_innovation/REPORT.md"],
            {
                "gate_passed": False,
                "eligible_queries": 287,
                "fit_components": 16,
                "held_components": 8,
                "baseline_MSE": 0.980329,
                "peer_MSE": 0.974071,
                "shift_MSE": 0.969317,
            },
            "Actor frozen negative result; independent review pending. No horizon sweep, real architecture promotion, or information-absence inference.",
        ),
        (
            "round4 independent peer innovation rejection reproduction",
            [CONT / "round4/peer_innovation_review/REPORT.txt"],
            {
                "actor_artifacts_verified": 103,
                "failed_gate_clauses_reproduced": 7,
                "new_fits": 0,
                "GRU_forwards": 0,
                "actor_helpers_imported": False,
            },
            "Fixed proxy rejection independently supported; minor toy shift-independence specification deviation does not rescue real gate. No peer-information absence or saturation claim.",
        ),
        (
            "round4 independent v2 capture typed RNG rejection",
            [
                CONT / "round4/capture_review_v2/REPORT.md",
                CONT / "round4/capture_review_v2/V3_MINIMAL_SPEC.md",
            ],
            {
                "v1_three_gaps_retested_pass": True,
                "normal_repeat_skip_original_CPU_delta": 0,
                "empirical_CPU_floor": 0,
                "GPU_approved": False,
                "remaining_blocker": "NumPy RNG dtype alias coerced during restore",
            },
            "V2 frozen blocked despite pristine parity. Separate minimal v3 must enforce full typed/byte RNG identity; no controller workaround or relaxed comparator.",
        ),
    ]
    ledger = ROOT / "research/experiments.jsonl"
    existing = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    kinds = {row.get("kind") for row in existing}
    number = max(int(row["id"][2:]) for row in existing if row.get("id", "").startswith("E-")) + 1
    records = []
    for kind, paths, result, verdict in specs:
        if kind in kinds:
            continue
        records.append(
            {
                "id": f"E-{number}",
                "date": now,
                "kind": kind,
                "reports": [binding(path) for path in paths],
                "result": result,
                "verdict": verdict,
            }
        )
        number += 1
    with ledger.open("a") as output:
        for record in records:
            output.write(json.dumps(record, allow_nan=False) + "\n")
    state_path = CONT / "STATE.json"
    state = json.loads(state_path.read_text())
    watch = json.loads((CONT / "watch_status.json").read_text())
    state.update(
        updated_at=now,
        phase="confirmation_passed_capture_correction_review",
        active_training=[job["name"] for job in watch["active_gpu_jobs"]],
        training_status_observed_at=watch["time"],
        next="V3 exact NumPy RNG correction and independent capture review; bounded default capture only after approval; research until23:00 Kyiv.",
    )
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    bandit_path = ROOT / "research/bandit/arms.json"
    bandit = json.loads(bandit_path.read_text())
    bandit.setdefault("continuation_evidence", {})["fold23_confirmation"] = decision
    bandit_path.write_text(json.dumps(bandit, indent=1) + "\n")
    print(json.dumps({"appended": [row["id"] for row in records], "gates": gates}))


if __name__ == "__main__":
    main()
