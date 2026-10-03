from __future__ import annotations

import json
import resource
import time
from collections import Counter

import numpy as np
from laws import LAWS
from probe import HERE, PARENT, digest, verify_parent


def main() -> None:
    """Deduplicate matches against old offset-two geometry without fitting another law."""
    started = time.monotonic()
    records = json.loads((HERE / "sampled_outputs.json").read_text())
    result = json.loads((HERE / "results.json").read_text())
    initial = json.loads((HERE / "sampled_outputs_initial.json").read_text())
    # New path witnesses may be added, but the frozen law outputs must remain identical.
    output_unchanged = all(
        r["outputs"] == old["outputs"] for r, old in zip(records, initial, strict=True)
    )
    assert output_unchanged
    by_class, by_component, geometry = {}, {}, {}
    matched_witnesses, failed_witnesses = [], []
    for law in LAWS:
        by_class[law] = {}
        by_component[law] = {}
        geometry[law] = {"exact": Counter(), "mismatch": Counter()}
        for r in records:
            output = r["outputs"][law]["aligned"]
            p = np.array(output["prediction_raw"], float)
            complete = bool(np.isfinite(p).all())
            y = np.array(r["target_raw"])
            exact = complete and bool(np.all(np.abs(p - y) <= 0.002))
            for grouping, key in [
                (by_class[law], str(r["fit_code_class"])),
                (by_component[law], str(r["component"])),
            ]:
                local = grouping.setdefault(key, Counter())
                local[f"{r['regime']}_all"] += 1
                local[f"{r['regime']}_complete"] += int(complete)
                local[f"{r['regime']}_exact"] += int(exact)
            if r["regime"] != "outside" or not complete:
                continue
            cls = r["fit_code_class"]
            q, spread = r["quantum"], r["entry_spread_ticks"]
            winner = int(np.rint(y.max() / q))
            excess = -int(np.rint(y.min() / q)) - spread
            kind = "exact" if exact else "mismatch"
            geometry[law][kind][f"class={cls},W={winner},K={excess}"] += 1
            if law == "shared_profitable_exit":
                bucket = matched_witnesses if exact else failed_witnesses
                if len(bucket) < 5:
                    bucket.append(
                        {
                            k: r[k]
                            for k in [
                                "group",
                                "component",
                                "seq_ix",
                                "step",
                                "event_index",
                                "fit_code_class",
                                "quantum",
                                "entry_spread_ticks",
                                "target_raw",
                                "diagnostic_path",
                            ]
                        }
                        | {"law_output": output, "actual_W": winner, "actual_K": excess}
                    )
    converted = {
        law: {key: dict(value) for key, value in table.items()} for law, table in by_class.items()
    }
    components = {
        law: {key: dict(value) for key, value in table.items()}
        for law, table in by_component.items()
    }
    geometry_dict = {
        law: {kind: dict(counts) for kind, counts in table.items()}
        for law, table in geometry.items()
    }
    summary = {
        "law_outputs_unchanged_after_witness_replay": output_unchanged,
        "by_input_class": converted,
        "by_component": components,
        "outside_actual_W_K_geometry": geometry_dict,
        "shared_exact_witnesses": matched_witnesses,
        "shared_mismatch_witnesses": failed_witnesses,
        "dedup_prereg_sha256": digest(HERE / "DEDUP_PREREG.json"),
        "aligned": {law: result["scores"][law]["aligned"] for law in LAWS},
        "paired_controls": result["paired_controls"],
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    verify_parent()
    state = json.loads((HERE / "STATE.json").read_text())
    state["phase"] = (
        "report_frozen" if (HERE / "REPORT.md").exists() else "scoring_summarized_report_pending"
    )
    state["pending"] = [] if (HERE / "REPORT.md").exists() else ["final report"]
    state["last_failure"] = None
    state["resolved_failures"] = [
        "Synthetic non-equivalence assertion initially checked agreeing fixture A; fixed to predeclared fixture B without law changes, before data scoring"
    ]
    state["resolved_failures"].append(
        "Initial lint import/unused/nested-if warnings fixed without numerical changes"
    )
    state["commands"] = [
        line
        for line in (HERE / "COMMANDS.txt").read_text().splitlines()
        if line.startswith("research/campaign_2026-10-03/continuation/run_cpu.sh")
    ]
    state["skeptic_branch_read"] = False
    state["claims"] = {
        "independent_hybrid_exact_law": "rejected",
        "shared_exit_exact_law": "rejected",
        "shared_exit_partial_conditional_association": "measured, needs dedup and independent decoder audit",
        "physical_generator_identity": "unknown",
        "training_representation_proposal": "deferred until both reports frozen and audit comparison",
    }
    state["artifacts"] = sorted(p.name for p in HERE.iterdir() if p.is_file())
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))
    cost = {
        "probe": result["cost"],
        "synthetic": json.loads((HERE / "synthetic_results.json").read_text())["cost"],
        "summary_wall_seconds": time.monotonic() - started,
        "summary_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_seconds": 0,
        "resource_failures": 0,
        "replays": "Initial scoring plus same-cohort replay adding witness fields; output identity asserted, both timed logs retained",
        "scope_log": "controller_command_logs.txt",
    }
    cost["initial_scoring"] = json.loads((HERE / "results_initial.json").read_text())["cost"]
    probe_costs = []
    for filename in ["probe.log", "probe_witness_replay.log", "probe_verified.log"]:
        lines = (HERE / filename).read_text().splitlines()
        if lines and lines[-1].startswith("{"):
            probe_costs.append({"log": filename, **json.loads(lines[-1])["cost"]})
    cost["all_successful_scoring_processes"] = probe_costs
    cost["sum_reported_scoring_wall_seconds"] = sum(c["wall_seconds"] for c in probe_costs)
    cost["sum_reported_scoring_cpu_seconds"] = sum(c["cpu_seconds"] for c in probe_costs)
    (HERE / "cost.json").write_text(json.dumps(cost, indent=2))
    source_hashes = {
        "parent:" + name: digest(PARENT / name)
        for name in [
            "decoder.py",
            "decoder_tables.json",
            "selection.json",
            "sampled_origins.json",
            "REPORT.md",
        ]
    }
    source_hashes.update({p.name: digest(p) for p in HERE.iterdir() if p.suffix in [".py", ".sh"]})
    (HERE / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2))
    manifest = {
        p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in HERE.iterdir()
        if p.is_file() and p.name != "PROVENANCE.json"
    }
    (HERE / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "by_input_class": converted,
                "outside_actual_W_K_geometry": geometry_dict,
                "paired_controls": result["paired_controls"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
