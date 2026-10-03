from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Save compact measured aggregates and hashes of continuation artifacts."""
    result = json.loads((HERE / "continuation_results.json").read_text())
    events = json.loads((HERE / "continuation_events_results.json").read_text())
    summary = {"metrics": result["metrics"], "cohorts": {}, "events": {}}
    for split in ["train", "valid"]:
        rows = [r for r in result["records"] if r["split"] == split]
        summary["cohorts"][split] = {
            "seq_ids": [r["seq_ix"] for r in rows],
            "sum_modal_disagreement_range": [
                min(r["sum_state"]["modal_disagreement"] for r in rows),
                max(r["sum_state"]["modal_disagreement"] for r in rows),
            ],
            "sum_modal_disagreement_mean": float(
                np.mean([r["sum_state"]["modal_disagreement"] for r in rows])
            ),
            "sum_within_variance_fraction_mean": float(
                np.mean([r["sum_state"]["within_variance_fraction"] for r in rows])
            ),
            "quantum_classes": sorted({round(r["diagnostic_quantum"], 5) for r in rows}),
        }
    for mask_name in ["unchanged_i0_prices", "unchanged_i0_all", "unchanged_all_inputs"]:
        summary["events"][mask_name] = {}
        for condition in ["aligned", "shifted_control"]:
            totals = {
                key: sum(r[condition][mask_name][key] for r in events["records"])
                for key in [
                    "rows",
                    "sum_changed",
                    "difference_changed",
                    "either_target_changed",
                    "cancelling_target_changes",
                ]
            }
            totals["sum_change_rate"] = (
                totals["sum_changed"] / totals["rows"] if totals["rows"] else None
            )
            summary["events"][mask_name][condition] = totals
    summary["cross_equal_window_groups"] = result["equal_window_group_count"]
    summary["prefix_checks_pass"] = result["prefix_checks_pass"]
    summary["train_fit_rows"] = result["train_fit_rows"]
    summary["valid_scored_rows"] = result["valid_scored_rows"]
    summary["first_witnesses"] = [
        r["unchanged_input_sum_conflict_witnesses"][:1] for r in events["records"]
    ]
    summary["prior_scan_quantum_counts"] = {}
    for split in ["train", "valid"]:
        with np.load(HERE / f"scan1_{split}.npz") as old:
            classes, counts = np.unique(
                np.round(np.minimum(old["q0"], old["q1"]), 5), return_counts=True
            )
            summary["prior_scan_quantum_counts"][split] = list(
                zip(classes.tolist(), counts.tolist(), strict=True)
            )
    (HERE / "continuation_summary.json").write_text(json.dumps(summary, indent=2))
    state = json.loads((HERE / "STATE.json").read_text())
    state["phase"] = "complete_independent_continuation"
    state["pending_work"] = []
    state["last_failure"] = None
    state["resolved_failures"] = [
        "Initial lint failed on imports and zip strict; fixed and verification_final.log passed"
    ]
    state["inspected_figures"] = [
        "fig02_seq0_zoom.png",
        "fig03_price_cols_vs_base.png",
        "continuation_falsifier.png",
        "continuation_events_initial.png",
        "continuation_events.png",
    ]
    state["commands"] += [
        "run_cpu.sh eda-puzzle-a bash research/campaign_2026-10-03/eda/puzzle_a/verify_continuation.sh",
        "run_cpu.sh eda-puzzle-a .venv/bin/python research/campaign_2026-10-03/eda/puzzle_a/continuation_summary.py",
    ]
    state["artifacts"] = sorted(
        set(
            state["artifacts"]
            + [
                "REPORT.txt",
                "eda_puzzle_a.py",
                "eda_puzzle_a.html",
                "continuation_summary.py",
                "continuation_summary.json",
                "PROVENANCE.json",
                "verify_continuation.sh",
                "verification.log",
                "verification_final.log",
            ]
        )
    )
    state["blockers"] = [
        "Unknown source-overlap components",
        "No pristine independent cohort",
        "Exact target law unresolved",
        "Strong baseline intentionally uninspected",
    ]
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))
    files = sorted(
        [
            p
            for p in HERE.iterdir()
            if p.name.startswith("continuation_")
            or p.name.startswith("HYPOTHESES")
            or p.name.startswith("eda_puzzle_a")
            or p.name
            in [
                "RECOVERY_SUMMARY.txt",
                "REPORT.txt",
                "STATE.json",
                "verify_continuation.sh",
                "verification.log",
                "verification_final.log",
            ]
        ]
    )
    manifest = {
        p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in files
        if p.is_file()
    }
    (HERE / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
