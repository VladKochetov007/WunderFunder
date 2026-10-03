from __future__ import annotations

import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent


def main() -> None:
    """Freeze source-bound synthetic instrument review; do not inspect captures."""
    result = json.loads((OWN / "results.json").read_text())
    for path, expected in result["input_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    state = json.loads((OWN / "STATE.json").read_text())
    state.update(
        phase="review_frozen",
        actual_capture_science="BLOCKED",
        independently_accepted_decomposition_bound=None,
        default_floor_transfer_forbidden=True,
        zero_floor_relaxation_forbidden=True,
        author_sources_preserved=True,
    )
    (OWN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    (OWN / "COST.json").write_text(json.dumps(result["cost"], indent=2) + "\n")
    manifest = {
        "input_sha256": result["input_sha256"],
        "own_artifact_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(OWN.iterdir())
            if p.is_file() and p.name not in ("PROVENANCE.json", "freeze.log")
        },
        "actual_capture_reads": 0,
        "status": "review_frozen_blocked_for_actual_capture_science",
    }
    (OWN / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Frozen reviewer blocker; no capture or decomposition allowance accepted.")


if __name__ == "__main__":
    main()
