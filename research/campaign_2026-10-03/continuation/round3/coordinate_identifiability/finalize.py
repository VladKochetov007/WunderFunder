"""Check bounded saved toy artifacts and freeze their provenance."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
TABLE = OWN.parents[1] / "round2/first_passage/decoder_tables.json"
DOC = ROOT / "research/campaign_2026-10-03/eda/HOST_DATA_DESCRIPTION.txt"
LOGGER = logging.getLogger(__name__)


def digest(path: Path) -> str:
    """Hash one small permitted artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Verify saved outputs without revisiting raw data or choosing new laws."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = json.loads((OWN / "results.json").read_text())
    checks = {
        "prereg_preserved": digest(OWN / "PREREG.json") == results["prereg_sha256"],
        "frozen_table_preserved": digest(TABLE) == results["decoder_table_sha256"],
        "field_documentation_preserved": digest(DOC) == results["field_documentation_sha256"],
    }
    experiment = results["experiment"]
    with np.load(OWN / "two_world_arrays.npz", allow_pickle=False) as arrays:
        checks["saved_inputs_identical"] = bool(np.array_equal(arrays["inputs0"], arrays["inputs1"]))
        checks["saved_labels_identical"] = bool(np.array_equal(arrays["labels0"], arrays["labels1"]))
        checks["saved_physical_paths_different"] = not bool(np.array_equal(arrays["physical0"], arrays["physical1"]))
    checks["negative_controls_pass"] = not any(experiment["negative_controls"].values())
    controls = experiment["positive_controls_extra_assumptions"]
    checks["positive_controls_pass"] = all(
        value is True if name.endswith("breaks_equivalence") else value == 0
        for name, value in controls.items()
    )
    (OWN / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    state = json.loads((OWN / "STATE.json").read_text())
    state.update(
        phase="complete" if all(checks.values()) else "verification_failed",
        commands=[
            "run_cpu.sh round3-hft /usr/bin/time -v .venv/bin/python -B -u identifiability_toy.py",
            "run_cpu.sh round3-hft .venv/bin/ruff check --isolated --no-cache identifiability_toy.py",
            "run_cpu.sh round3-hft .venv/bin/python -B finalize.py",
        ],
        pending=[],
        inspected_sequence_ids=[],
        toy_rows=list(range(32)),
        prior_exposure="Two frozen supervised price codes; no new actual rows",
        figure_inspected=True,
        verification=checks,
        last_failure=None if all(checks.values()) else checks,
    )
    (OWN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    manifest = {
        "inputs": {str(TABLE): digest(TABLE), str(DOC): digest(DOC)},
        "own": {path.name: digest(path) for path in sorted(OWN.iterdir())
                if path.is_file() and path.name not in {"SHA256.json", "finalize.log"}},
    }
    (OWN / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("VERIFICATION %s", json.dumps(checks))


if __name__ == "__main__":
    main()
