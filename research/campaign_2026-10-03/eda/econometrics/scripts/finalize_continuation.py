from __future__ import annotations

import hashlib
import json
import logging
import textwrap

import numpy as np
from continuation_clock_probe import OUT, correlation, moments, save_json


def main() -> None:
    """Check the scorer and seal documentation and current artifact identities."""
    rng = np.random.default_rng(12)
    target = np.clip(rng.normal(size=101), -2, 2)
    prediction = rng.normal(size=101) * 3
    clipped_prediction = np.clip(prediction, -2, 2)
    weight = np.abs(target)
    centered_y = target - np.average(target, weights=weight)
    centered_p = clipped_prediction - np.average(clipped_prediction, weights=weight)
    direct = np.sum(weight * centered_y * centered_p) / np.sqrt(
        np.sum(weight * centered_y**2) * np.sum(weight * centered_p**2)
    )
    pooled = correlation(
        moments(target[:41], prediction[:41]) + moments(target[41:], prediction[41:])
    )
    error = float(abs(direct - pooled))
    assert error < 1e-12
    manifest = json.loads((OUT / "cache/continuation_manifest.json").read_text())
    clock = json.loads((OUT / "cache/continuation_results.json").read_text())
    hazard = json.loads((OUT / "cache/continuation_hazard_results.json").read_text())
    assert not clock["frozen_gate_passed"] and not hazard["frozen_gate_passed"]
    assert len(manifest["train_ids"]) == 48 and len(manifest["valid_ids"]) == 24
    assert not set(manifest["train_groups"]) & set(manifest["prior_train_groups_excluded"])
    prior_card = OUT / "HYPOTHESES_clock_frozen.json"
    assert hashlib.sha256(prior_card.read_bytes()).hexdigest() == manifest["hypothesis_sha256"]
    report_path = OUT / "REPORT.txt"
    report = report_path.read_text().split("\nACTUAL SAMPLED IDS\n")[0]
    report += "\nACTUAL SAMPLED IDS\n"
    for split in ("train", "valid"):
        report += (
            textwrap.fill(f"{split}: " + ", ".join(map(str, manifest[f"{split}_ids"])), width=88)
            + "\n"
        )
    report += f"\nScorer centered formula versus pooled moments: abs error {error:.3g}.\n"
    report_path.write_text(report)
    hypothesis = json.loads(prior_card.read_text())
    hypothesis["followup_card"] = "HYPOTHESES_followup_frozen.json"
    hypothesis["primary_measured_result"] = "cache/continuation_results.json"
    hypothesis["followup_measured_result"] = "cache/continuation_hazard_results.json"
    hypothesis["verdict"] = "both frozen gates failed; exploratory target-specific candidate only"
    hypothesis["original_frozen_card"] = "HYPOTHESES_clock_frozen.json"
    save_json("HYPOTHESES.json", hypothesis)
    state = json.loads((OUT / "STATE.json").read_text())
    state["phase"] = "report_frozen_notebook_verified"
    state["pending_work"] = []
    state["figures_inspected"] = [
        "fig/target_acf.png",
        "fig/continuation_clock_scores.png",
        "fig/continuation_clock_window.png",
        "fig/continuation_hazard_scores.png",
    ]
    state["metric_verification_max_abs"] = error
    state["last_failure"] = None
    state["operational_notes"] = [
        "Notebook apply_patch rejected duplicate delete/add path; corrected by direct owned-file replacement.",
        "No computation or resource failure; no admission retry required.",
    ]
    state["commands"].extend(
        [
            "run_cpu.sh eda-econometrics .venv/bin/ruff format <owned new scripts and notebook>",
            "run_cpu.sh eda-econometrics .venv/bin/ruff check <owned new scripts and notebook>",
            "run_cpu.sh eda-econometrics .venv/bin/marimo export html <owned notebook> -o <owned continuation_notebook.html>",
            "run_cpu.sh eda-econometrics .venv/bin/python <owned scripts/finalize_continuation.py>",
        ]
    )
    state["artifacts"].extend(
        [
            "REPORT.txt",
            "eda_econometrics.py",
            "continuation_notebook.html",
            "cache/continuation_artifact_sha256.json",
        ]
    )
    save_json("STATE.json", state)
    paths = [
        path
        for path in OUT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".marimo" not in path.parts
        and path.name != "continuation_artifact_sha256.json"
        and path.name != "continuation_finalize.log"
    ]
    save_json(
        "cache/continuation_artifact_sha256.json",
        {
            "hash_stage": "post-format source and completed report; original hypotheses verified against run manifest",
            "excluded": [
                "active continuation_finalize.log",
                "generated interpreter caches",
                "hash manifest itself",
            ],
            "files": {
                str(path.relative_to(OUT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
        },
    )
    logging.info("Scorer parity verified: %.3g; report IDs and artifact SHA256 saved", error)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
