from __future__ import annotations

import json
import math
import os
import re
import resource
from collections import Counter
from datetime import datetime

import numpy as np
from decoder import HERE, ROOT, digest
from path_laws import FAMILIES


def strict_json_value(value: object) -> object:
    """Replace raw diagnostic NaN markers with explicit JSON null without overwriting raw files."""
    if isinstance(value, dict):
        return {k: strict_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strict_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    """Build paired control summaries, compact evidence, plots and source provenance."""
    results = json.loads((HERE / "results.json").read_text())
    records = json.loads((HERE / "sampled_origins.json").read_text())
    (HERE / "sampled_origins_strict.json").write_text(
        json.dumps(strict_json_value(records), allow_nan=False)
    )
    synthetic = json.loads((HERE / "synthetic_results.json").read_text())
    (HERE / "synthetic_results_strict.json").write_text(
        json.dumps(strict_json_value(synthetic), indent=2, allow_nan=False)
    )
    paired = {}
    for family in FAMILIES:
        paired[family] = {}
        for control in ["shift997", "shuffle"]:
            common, aligned_exact, control_exact = 0, 0, 0
            for r in records:
                a = np.array(r["laws"][family]["aligned"]["prediction_raw"], dtype=float)
                b = np.array(r["laws"][family][control]["prediction_raw"], dtype=float)
                if np.isfinite(a).all() and np.isfinite(b).all():
                    y = np.array(r["target_raw"])
                    common += 1
                    aligned_exact += int(np.all(np.abs(a - y) <= 0.002))
                    control_exact += int(np.all(np.abs(b - y) <= 0.002))
            paired[family][control] = {
                "common_complete_origins": common,
                "all_origins": len(records),
                "aligned_exact": aligned_exact,
                "control_exact": control_exact,
                "scope": "Matched conditional comparison, never replaces all-origin denominator",
            }
    geometry = Counter()
    geometry_weights = Counter()
    lattice_errors = []
    for r in records:
        if "label_lattice_residual" in r:
            lattice_errors.append(r["label_lattice_residual"])
        if "target_geometry" not in r:
            geometry["missing_class_origins"] += 1
            continue
        for side, item in enumerate(r["target_geometry"]["exit_side_touch"]):
            if r["target_raw"][side] <= 0:
                continue
            geometry["winner_sides"] += 1
            geometry_weights["winner_sides"] += r["diagnostic_weight"]
            if item["first_exit_side_touch_lag"] >= 1:
                geometry["winner_exit_touched_known_prefix"] += 1
                geometry_weights["winner_exit_touched_known_prefix"] += r["diagnostic_weight"]
                if item["first_touch_running_extreme"]:
                    geometry["winner_touch_running_extreme"] += 1
                    geometry_weights["winner_touch_running_extreme"] += r["diagnostic_weight"]
            else:
                geometry["winner_no_known_touch_within32"] += 1
    transport = Counter()
    for sequence in results["test_samples"]:
        transport.update(sequence["known_transition_transport"])
    replay_identical = digest(HERE / "sampled_origins.json") == digest(
        HERE / "sampled_origins_initial.json"
    )
    summary = {
        "paired_controls": paired,
        "diagnostic_geometry_counts": dict(geometry),
        "diagnostic_geometry_weight_tallies": dict(geometry_weights),
        "eval_mapped_lattice_residual_max": max(lattice_errors) if lattice_errors else None,
        "known_interior_transition_transport": dict(transport),
        "export_fix_replay_sample_hash_identical": replay_identical,
        "families": results["families"],
        "test_IDs": [
            {k: r[k] for k in ["group", "component", "seq_ix", "origins"]}
            for r in results["test_samples"]
        ],
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    strata = {}
    counterexamples = {}
    for family in FAMILIES:
        strata[family] = {}
        counterexamples[family] = []
        for regime in ["interior", "outside"]:
            local = Counter()
            for r in records:
                y = np.array(r["target_raw"])
                is_interior = bool((y <= 0).all())
                if is_interior != (regime == "interior"):
                    continue
                local["all_origins"] += 1
                p = np.array(r["laws"][family]["aligned"]["prediction_raw"], dtype=float)
                if np.isfinite(p).all():
                    local["complete_pairs"] += 1
                    local["exact_pair"] += int(np.all(np.abs(p - y) <= 0.002))
                    if not is_interior:
                        for side in range(2):
                            if y[side] > 0:
                                local["winner_exact"] += int(abs(p[side] - y[side]) <= 0.002)
                            else:
                                local["loser_exact"] += int(abs(p[side] - y[side]) <= 0.002)
                    if len(counterexamples[family]) < 3 and not np.all(np.abs(p - y) <= 0.002):
                        counterexamples[family].append(
                            {
                                "group": r["group"],
                                "component": r["component"],
                                "seq_ix": r["seq_ix"],
                                "step": r["step"],
                                "event_index": r["event_index"],
                                "fit_code_class": r["fit_code_class"],
                                "target_raw": r["target_raw"],
                                "prediction_raw": p.tolist(),
                                "lags": r["laws"][family]["aligned"]["lags"],
                            }
                        )
            strata[family][regime] = dict(local)
    (HERE / "regime_geometry.json").write_text(
        json.dumps(
            {"diagnostic_strata": strata, "explicit_eligible_counterexamples": counterexamples},
            indent=2,
        )
    )
    os.environ["MPLCONFIGDIR"] = str(HERE / ".mplconfig")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = list(FAMILIES)
    counts = [results["families"][f]["aligned"]["exact_fraction_of_complete"] for f in labels]
    axes[0].bar(np.arange(3) - 0.15, counts, width=0.3, label="All complete origins")
    outside_counts = [
        strata[f]["outside"].get("exact_pair", 0) / strata[f]["outside"]["complete_pairs"]
        for f in labels
    ]
    axes[0].bar(np.arange(3) + 0.15, outside_counts, width=0.3, label="Outside complete origins")
    axes[0].set_title("All exact matches occurred in interior pairs")
    axes[0].legend(fontsize=8)
    axes[0].set_xticks(np.arange(3), labels, rotation=15)
    axes[0].set_ylabel("Exact supplied pair / complete decoded pairs")
    bottoms = np.zeros(3)
    for reason in [
        "resolved_pair",
        "missing_origin_class",
        "unknown_origin",
        "unknown_transition",
        "sequence_end",
        "horizon_censored",
    ]:
        vals = np.array(
            [
                results["families"][f]["aligned"]["reason_weight_shares"].get(reason, 0)
                for f in labels
            ]
        )
        axes[1].bar(np.arange(3), vals, bottom=bottoms, label=reason)
        bottoms += vals
    axes[1].set_xticks(np.arange(3), labels, rotation=15)
    axes[1].set_ylabel("Share of ALL sampled-origin label weight")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "family_geometry_coverage.png", dpi=140)
    plt.close(fig)
    source_files = [
        "research/campaign_2026-10-03/continuation/round2/CONTRACT.md",
        "research/campaign_2026-10-03/eda/puzzle_a/REPORT.txt",
        "research/campaign_2026-09-30/data_reconstruction/REPORT.md",
        "research/campaign_2026-09-30/data_reconstruction/probe.py",
        "scripts/cr_mature_common.py",
        "scripts/cr_mature_hit.py",
        "scripts/de_target_law.py",
    ]
    (HERE / "source_context_hashes.json").write_text(
        json.dumps({f: digest(ROOT / f) for f in source_files}, indent=2)
    )
    fit = json.loads((HERE / "fit_results.json").read_text())
    cost = {
        "fit": fit["cost"],
        "test_replay": results["cost"],
        "synthetic": json.loads((HERE / "synthetic_results.json").read_text())["cost"],
        "initial_calibration_and_failed_export_cost": "Not instrumented in surviving result: completed, but result export failed. Do not equate successful replay cost to total campaign cost.",
        "bounded_failed_replays": 2,
        "gpu_seconds": 0,
        "no_installs": True,
        "cpu_quota": "1core2GiB per run_cpu.sh",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "finalize_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    command_log = HERE / "controller_command_logs.txt"
    scope_costs = []
    for line in command_log.read_text().splitlines():
        match = re.search(r"round2-firstpassage-(\d+)-\d+", line)
        if match:
            finished = datetime.fromisoformat(line.split()[0]).timestamp()
            scope_costs.append(max(0, finished - int(match.group(1))))
    cost["captured_scope_wall_seconds_rounded"] = sum(scope_costs)
    cost["captured_scope_count"] = len(scope_costs)
    cost["scope_wall_limit"] = (
        "Controller timestamps are second-resolution and snapshot excludes the currently running finalizer; includes failed commands. Not measured CPU time."
    )
    (HERE / "cost.json").write_text(json.dumps(cost, indent=2))
    state = json.loads((HERE / "STATE.json").read_text())
    state["phase"] = (
        "report_complete" if (HERE / "REPORT.md").exists() else "results_complete_report_pending"
    )
    state["commands_reference"] = "COMMANDS.txt"
    state["commands"] = [
        line
        for line in (HERE / "COMMANDS.txt").read_text().splitlines()
        if line.startswith("research/campaign_2026-10-03/continuation/run_cpu.sh")
    ]
    state["controller_logs"] = "controller_command_logs.txt"
    state["last_failure"] = None
    state["resolved_failures"] = [
        "NumPy integer JSON export failures: two traces retained; normalized counters; final raw samples byte-identical",
        "Initial lint import order and unused import corrected",
    ]
    state["pending"] = [] if (HERE / "REPORT.md").exists() else ["final report"]
    state["claims"] = {
        "exact_generator_identified": False,
        "outside_pair_all_three_laws": "contradicted on eligible decoded sample",
        "trailing_extrema_return_including_entry": "mapped outside losers violate spread loss bound independently of horizon",
        "physical_quote_decoder": "unknown",
        "baseline_or_forecast_gain": "not tested",
    }
    state["inspected_figures"] = ["family_geometry_coverage.png"]
    state["artifacts"] = sorted(p.name for p in HERE.iterdir() if p.is_file())
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))
    manifest = {
        p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in HERE.iterdir()
        if p.is_file() and p.name != "PROVENANCE.json"
    }
    (HERE / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
