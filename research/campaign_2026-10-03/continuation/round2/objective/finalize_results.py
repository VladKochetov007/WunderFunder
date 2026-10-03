from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Bundle primary and distinctly labeled exploratory evidence for review."""
    results = json.loads((HERE / "results.json").read_text())
    results["exploratory_mass_alignment_sham"] = {
        "prereg": "PREREG_SHAM.json (after primary scoring, before sham)",
        "results": json.loads((HERE / "sham_results.json").read_text()),
    }
    data = np.load(HERE / "chunk_summaries.npz")
    target_mass = data["bins"][..., 0].sum((0, 1, 2))
    results["target_training_mass_share"] = (target_mass / target_mass.sum()).tolist()
    results["preregistrations_sha256"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("PREREG.json", "PREREG_SHAM.json")
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "target_mass_share": results["target_training_mass_share"],
                "bundled": True,
                "primary_prereg_unchanged": results["prereg_sha256"]
                == results["preregistrations_sha256"]["PREREG.json"],
            }
        )
    )


if __name__ == "__main__":
    main()
