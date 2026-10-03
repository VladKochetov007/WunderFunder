from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OWN = Path(__file__).resolve().parent
REPO = OWN.parents[4]
ACTOR = OWN.parent.parent / "round3" / "washout_usage"


def main() -> None:
    """Check normalization binding and formatting replay, then freeze review."""
    final = json.loads((OWN / "results.json").read_text())
    initial = json.loads((OWN / "initial_results.json").read_text())
    for key in ("statistics", "component_results", "all288_records", "independent_forward_pairs"):
        assert final[key] == initial[key]
    panel = json.loads((ACTOR / "samples.json").read_text())
    checkpoint = torch.load(
        REPO / panel["control"]["checkpoint"], map_location="cpu", weights_only=False
    )
    expected_std = np.maximum(checkpoint["model"]["std"].numpy(), 1e-6)
    for i in range(24):
        with np.load(ACTOR.parent / "matched_usage" / f"seq_{i:02d}.npz") as z:
            assert np.array_equal(z["std"], expected_std)
    state = json.loads((OWN / "STATE.json").read_text())
    state.update(
        phase="review_frozen",
        await_controller=True,
        source_formatting_replay="all measured results equal",
        normalization_binding="all24 cached std arrays equal checkpoint std clipped1e-6",
        distinct_forward_pairs=4,
    )
    (OWN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    cost = {
        "initial": initial["cost"],
        "final": final["cost"],
        "total_review_wall_seconds": initial["cost"]["wall_seconds"]
        + final["cost"]["wall_seconds"],
        "total_review_cpu_seconds": initial["cost"]["cpu_seconds"] + final["cost"]["cpu_seconds"],
        "monetary_usage": "unavailable",
    }
    (OWN / "COST.json").write_text(json.dumps(cost, indent=2) + "\n")
    manifest = {
        "input_sha256": final["input_sha256"],
        "own_artifact_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(OWN.iterdir())
            if p.is_file() and p.name not in ("PROVENANCE.json", "freeze.log")
        },
        "normalization_check": "24/24 cached std arrays equal frozen checkpoint std",
        "status": "frozen; await controller",
    }
    (OWN / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Review frozen; normalization checked; all formatting replay results equal.")


if __name__ == "__main__":
    main()
