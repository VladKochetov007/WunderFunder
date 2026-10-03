"""Freeze final bounded instrument artifacts; do not touch prior reports."""

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
    selection = json.loads((HERE / "selection.json").read_text())
    assert result["controls"]["all_pass"] and all(case["all_finite"] for case in result["cases"])
    measurement_hashes = json.loads((HERE / "measurement_hashes.json").read_text())
    assert all(digest(HERE / path) == expected for path, expected in measurement_hashes.items())
    panel = ROOT / "research/campaign_2026-10-03/continuation/round3/matched_usage"
    registry = json.loads((panel / "artifact_provenance.json").read_text())["artifacts"]
    inputs = {
        f"seq_{c['panel_index']:02d}.npz": registry[f"seq_{c['panel_index']:02d}.npz"]["sha256"]
        for c in selection["cases"]
    }
    # Recheck selected sources/arrays: integrity hashing does not decode targets.
    assert all(digest(panel / name) == value for name, value in inputs.items())
    inputs["samples.json"] = digest(panel / "samples.json")
    provenance = ROOT / "artifacts/runs/c20261001_d256_s3_f0/provenance.json"
    (HERE / "input_hashes.json").write_text(
        json.dumps(
            {
                "parent_arrays": inputs,
                "source": selection["source_sha256"],
                "checkpoint": selection["checkpoint_sha256"],
                "run_provenance_sha256": digest(provenance),
            },
            indent=2,
        )
        + "\n"
    )
    state = {
        "state": "FROZEN_FINITE_LOCAL_INSTRUMENT_COMPLETE",
        "verdict": "16/16 signed cases retain resolvable initial-state output response through289updates; no uniform stability/burn-in/gain claim",
        "cases": 8,
        "components": 8,
        "lags": [0, 8, 32, 96, 256, 288],
        "compute_command": "bash research/campaign_2026-10-03/continuation/round4/state_forgetting/run.sh",
        "resources": "run_cpu.sh round2-critic CPU1core2GiB900s",
        "cost": result["cost"],
        "full_sequence_forwards": 0,
        "target_mask_arrays_read": False,
        "scoring": False,
        "future_washout_results_read": False,
        "gpu_approval": False,
        "next": "await controller request; no self-approval or follow-on training",
    }
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    (HERE / "COST.json").write_text(
        json.dumps(
            {
                "measurement": result["cost"],
                "scope": "measurement body cost; includes CPU imports, excludes wrapper admission; no GPU",
                "command": state["compute_command"],
            },
            indent=2,
        )
        + "\n"
    )
    files = [p for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "HASHES.json"]
    (HERE / "HASHES.json").write_text(
        json.dumps({p.name: digest(p) for p in files}, indent=2) + "\n"
    )
    print(json.dumps(state))


if __name__ == "__main__":
    main()
