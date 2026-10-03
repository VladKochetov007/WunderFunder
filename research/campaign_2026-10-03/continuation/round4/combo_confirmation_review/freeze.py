"""Freeze review outputs and all small-source identities; no bulk artifact loads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result = json.loads((HERE / "results.json").read_text())
    inputs = json.loads((HERE / "INPUTS.json").read_text())
    assert result["gate_pass"] and result["score_fixtures"]["pass"]
    for name, meta in inputs.items():
        assert digest(HERE / name) == digest(ROOT / meta["source"]) == meta["sha256"]
    for run, meta in result["metadata_members"].items():
        assert digest(ROOT / f"artifacts/runs/{run}/provenance.json") == meta["provenance_sha256"]
        assert digest(ROOT / f"cache/pred_{run}_valid.json") == meta["sidecar_sha256"]
    sources = (
        "scripts/screen_score.py",
        "research/campaign_2026-09-30/audit/campaign_score.py",
        "src/wnn/evaluation.py",
        "src/wnn/gmetric.py",
        "utils.py",
        "scripts/train_rnn.py",
        "scripts/predict_run.py",
        "research/campaign_2026-10-01/jobs/dcurve_fold.sh",
    )
    (HERE / "source_hashes.json").write_text(
        json.dumps({name: digest(ROOT / name) for name in sources}, indent=2) + "\n"
    )
    state = {
        "state": "FROZEN_INDEPENDENT_CONFIRMATION_REVIEW_PASS",
        "gate": result["gates"],
        "primary": result["primary"],
        "command": "bash research/campaign_2026-10-03/continuation/round4/combo_confirmation_review/run.sh",
        "resources": "run_cpu.sh round2-critic CPU1core2GiB900s",
        "cost": result["cost"],
        "prediction_or_model_bulk_loads": 0,
        "real_score_payload_recomputed": False,
        "public_score_claim": False,
        "mechanistic_experiment": "preregistered recommendation only; not executed or approved",
        "queue_or_ledger_modified": False,
        "next": "await controller request",
    }
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    (HERE / "COST.json").write_text(
        json.dumps(
            {
                "audit": result["cost"],
                "scope": "body includes imports; wrapper admission excluded; no GPU",
                "command": state["command"],
            },
            indent=2,
        )
        + "\n"
    )
    hashes = {
        p.name: digest(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "HASHES.json"
    }
    (HERE / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(state))


if __name__ == "__main__":
    main()
