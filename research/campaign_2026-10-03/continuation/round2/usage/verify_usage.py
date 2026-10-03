from __future__ import annotations

import json
import logging

import numpy as np
from inspect_control import digest
from usage_core import NAMES, REPO, ROOT


def main() -> None:
    """Verify final frozen artifact integrity and numerical response reductions."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    provenance = json.loads((ROOT / "artifact_provenance.json").read_text())
    for name, identity in provenance["artifacts"].items():
        if digest(ROOT / name) != identity["sha256"]:
            raise AssertionError(f"Artifact changed: {name}")
    for name, identity in provenance["reference_source_hashes"].items():
        if digest(REPO / name) != identity:
            raise AssertionError(f"Reference source changed: {name}")
    result = json.loads((ROOT / "results.json").read_text())
    with np.load(ROOT / "aggregate_panel.npz") as panel:
        responses = panel["responses"]
        for arm, name in enumerate(NAMES):
            actual = float(np.sqrt(np.mean(responses[:, :, arm, 0] ** 2)))
            expected = result["interventions"][name]["query_rms"]
            if not np.isclose(actual, expected, atol=1e-12, rtol=1e-12):
                raise AssertionError(f"Response reduction mismatch: {name}")
    state = json.loads((ROOT / "STATE.json").read_text())
    if state["pending"] or state["phase"] != "complete_report_frozen":
        raise AssertionError("State is not complete")
    logging.info(
        "Verified %s artifact hashes, live reference hashes, numerical RMS and completed STATE.",
        len(provenance["artifacts"]),
    )


if __name__ == "__main__":
    main()
