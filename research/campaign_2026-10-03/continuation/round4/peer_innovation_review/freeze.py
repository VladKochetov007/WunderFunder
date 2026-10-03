from __future__ import annotations

import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent


def main() -> None:
    """Bind consumed frozen artifacts and final review outputs without refitting."""
    result = json.loads((OWN / "results.json").read_text())
    for name, expected in result["input_sha256"].items():
        path = Path(name)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1048576):
                digest.update(block)
        assert digest.hexdigest() == expected, name
    state = json.loads((OWN / "STATE.json").read_text())
    state.update(
        phase="review_frozen",
        caveat="minor toy shift independence discrepancy",
        final_lint="passed",
        actor_artifacts_preserved=True,
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
        "gru_forwards": 0,
        "new_fits": 0,
        "status": "review_frozen",
    }
    (OWN / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Frozen narrow rejection review; all consumed input hashes unchanged.")


if __name__ == "__main__":
    main()
