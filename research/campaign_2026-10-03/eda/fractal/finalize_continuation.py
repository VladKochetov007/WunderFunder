from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent


def main() -> None:
    """Freeze the independent report and hash all continuation deliverables."""
    clock = json.loads((OUT / "continuation_results.json").read_text())
    stationary = json.loads((OUT / "stationarity_results.json").read_text())
    sample = json.loads((OUT / "sample_manifest.json").read_text())
    audit = json.loads((OUT / "provenance_audit.json").read_text())
    scores = np.array(clock["pooled_wp"])
    panels = np.array(clock["panel_wp"])
    report = [
        "FRACTAL / LONG-MEMORY: BOUNDED INDEPENDENT CONTINUATION",
        "Verdict: reject advancing eventEMA16 and positive signed-increment persistence.",
        "A stationary encoded level null explains most near-zero DFA slopes.",
        "Evidence tier: own matched CPU screen, not independent reproduction or baseline gain.",
        "",
        "RECOVERED PRIOR EVIDENCE",
        "See RECOVERY_SUMMARY.txt; original logs/NPZ/PNGs were preserved.",
        "Old raw WP .4927; raw+EMA16 .5380; EMA64 .5351; EMA1024 .4874.",
        "Five EMA scales .5069; activity/interaction/VR expansions did not improve it.",
        "Detected class routing .4576 versus pooled .5105; oracle class .5127.",
        "Oracle sequence mean shift .5241 cannot be deployed. No target is an input here.",
        "Inspected prior fig1: scale crossovers, not a single invariant exponent.",
        "Inspected prior fig4: negative-q slopes near 9-10 expose zero-floor sensitivity.",
        "Centred bands, full-sequence target quantum/mean and validation PCA are diagnostics.",
        "Class-stat feature selection was exploratory and used validation labels.",
        "",
        "FROZEN PROBE AND MATCHED CONTROLS",
        "HYPOTHESES_initial_frozen.json predates all new scoring; seed 20261003.",
        "32 train, 24 valid groups selected uniformly excluding own conservative exposure.",
        "One sequence loaded at a time; four ridge models, same fit/evaluation rows.",
        "Raw uses 112 inputs. All temporal arms use raw112 + history residual112.",
        "Event = any current change of 22 i0 prices; not a claim about best bid/ask.",
        "Row alpha=2/17; event alpha=2/17 updated only on quote events.",
        "Rate-adjusted row alpha=1-(1-2/17)^causal_EMA64(event); updated every row.",
        "All EMA states initialize at first feature row and reset by sequence.",
        "Clipped response and abs(clipped response) weights; training need_prediction.",
        "Same frozen trace-normalized ridge rule; intercept unpenalized.",
        "Trace-scaled penalties differ by arm; equal scalar penalty is a future sensitivity.",
        "Score pools need_prediction AND is_scored; clips finite y,p to +/-2.",
        "Only training data fit coefficients; no target or mask enters build().",
        "",
        "MEASURED SCORES: arm / t0 / t1 / mean pooled WP",
    ]
    for name, score in zip(clock["arms"], scores, strict=True):
        report.append(f"{name:20s} {score[0]:.6f} {score[1]:.6f} {score.mean():.6f}")
    report.extend(
        [
            f"Event minus row mean: {clock['event_minus_row']:+.6f}.",
            f"Event minus rate-control mean: {clock['event_minus_rate']:+.6f}.",
            f"Frozen survival gate passed: {clock['clock_survives']}.",
            f"Panel1 event-row: {panels[0, 2].mean() - panels[0, 1].mean():+.6f}.",
            f"Panel2 event-row: {panels[1, 2].mean() - panels[1, 1].mean():+.6f}.",
            f"Row16 minus raw: {scores[1].mean() - scores[0].mean():+.6f}.",
            "Absolute scores differ greatly by panel; pooled composition matters.",
            "Fixed 16 events generally spans more rows than 16 rows; results reject the",
            "specified event16 representation, not all event models or all clock mechanisms.",
            f"Feature-prefix parity at 99/500/4096 and future perturbation: max error {clock['causality_max_abs_error']:.1e}.",
            f"Pooled sufficient-statistics vs direct clipped WP: error {audit['metric_sufficient_statistics_vs_direct_max_error']:.3e}.",
            f"Initial fit/scoring/figures runtime: {clock['runtime_seconds']:.2f} seconds.",
            "Provenance audit: 5.91 seconds, maximum RSS 1,370,392 KiB (time -v log).",
            "No GPU, installs, commits, server or resource failure; all compute used run_cpu.",
            "",
            "SCALING FALSIFIER",
            "First 12 frozen valid groups; i0_p0 signed increments on i0 quote-event rows.",
            "DFA1 scales 16..1024; slopes frozen to 16..256; 16 shuffled controls each.",
            f"Median original-minus-shuffled-increment slope: {clock['scaling_median_delta']:+.6f}.",
            f"Positive differences: {clock['scaling_positive_sequences']}/12; frozen persistence gate failed.",
            "Permutation preserves increment marginals but removes all ordering, not just long memory.",
            "Follow-up H_stationarity frozen AFTER seeing initial negatives and BEFORE its scoring.",
            "It reuses the already-exposed same 12 groups; no new holdout confirmation.",
            "Shuffle event-time feature LEVELS, then difference; 16 controls per group.",
            f"Median original-minus-level-null slope: {stationary['median_original_minus_level_null']:+.6f}.",
            f"Stationary-level sufficiency gate: {stationary['stationary_level_explanation_survives']}.",
            f"Synthetic iid-increment slope: {stationary['synthetic']['iid_increment_slope']:.6f}.",
            f"Synthetic differenced-iid-level slope: {stationary['synthetic']['differenced_iid_level_slope']:.6f}.",
            "Interpretation: telescoping stationary encoded levels can flatten DFA of their",
            "increments. Low slopes do not establish exploitable raw-market antipersistence.",
            "Some sequences have original slopes above the level null; exact equality is not claimed.",
            "No inference about all 112 features or asymptotic fractional processes is warranted.",
            "Methods context: Kantelhardt et al. (2002), https://arxiv.org/abs/physics/0202070",
            "uses original/shuffled comparisons to separate order effects from broad marginals.",
            "Our diagnostic is ordinary DFA1; no multifractal claim survives this continuation.",
            "",
            "CAUSALITY AND SAMPLE PROVENANCE",
            "sample_manifest.json maps conservative prior and new row groups to actual seq_ix.",
            "Prior exposure union is conservative: includes all groups named in own scripts,",
            "even where execution cannot be proved. It is not an exhaustive campaign exposure map.",
            "New samples are fresh to that own union, not certified new to the campaign.",
            "provenance_audit.json hashes raw feature, target and mask arrays for each sampled group.",
            "Dataset identities use path, size, mtime_ns and row-group count; not whole-file hashes.",
            "Exact all112-feature 16-row windows screened at every offset in first/last256 rows.",
            f"Cross-sequence boundary links found: {len(audit['boundary_window_cross_sequence_links'])}.",
            "Interior/unsampled links remain unsearched; component disjointness is NOT certified.",
            "No row-independent significance or sequence-bootstrap confidence interval reported.",
            "Controller must regroup paired sufficient statistics by connected components before",
            "campaign uncertainty or integration. No public score or strong-baseline gain forecast.",
        ]
    )
    for source in ("train", "valid"):
        report.append(
            f"NEW {source} row groups: "
            + ",".join(str(row["row_group"]) for row in sample["new_sample"][source])
        )
        report.append(
            f"NEW {source} seq_ix IDs: "
            + ",".join(str(row["seq_ix"]) for row in sample["new_sample"][source])
        )
    report.extend(
        [
            "Prior actual IDs are all listed under own_prior_exposure in sample_manifest.json.",
            "",
            "RANKED NEXT DIRECTIONS (controller phase; not executed here)",
            "1. Reproduce rowEMA16 on component-disjoint data; compare matched raw and temporal",
            "   heads, shared scalar penalty, then strong-baseline residual gain. Kill if the",
            "   component-paired delta is nonpositive or the existing recurrent state subsumes it.",
            "2. Match effective row horizon of event history with causal rate-normalized event",
            "   span versus row16 and rate-adjusted16. Kill if event representation adds nothing",
            "   after horizon matching. Current event16 failure deprioritizes unconditional replacement.",
            "3. Separate short mean reversion from long memory using stationary-feature",
            "   AR/finite-window surrogates vs block-shuffled increments; matched EMA16 predictor.",
            "   Kill fractional-memory idea if short-order null reproduces scaling and prediction.",
            "4. Revisit positive-q volatility clustering with shuffled and finite-memory controls,",
            "   feature-only causal prefixes, no target regime labels, no negative-q zero floor.",
            "   Kill if apparent multifractality is reproduced by marginals/finite-memory controls.",
            "",
            "ARTIFACTS / REPRODUCTION",
            "REPORT.txt, RECOVERY_SUMMARY.txt, STATE.json, HYPOTHESES*.json.",
            "continuation_probe.py, stationarity_control.py, audit_provenance.py: runnable scripts.",
            "clock_coefficients.npz; continuation_raw.npz: per-sequence WP moments and DFA curves.",
            "continuation_results.json; stationarity_results.json; sample_manifest.json; provenance_audit.json.",
            "fig8_causal_clock.png, fig9_shuffle_falsifier.png, fig10_stationary_level_control.png",
            "were visually inspected: both targets lose under event16; all DFA permutation deltas",
            "are negative; stationary-level control reproduces low slopes for most sequences.",
            "eda_fractal.py is an updated runnable marimo app; eda_fractal.html exported successfully.",
            "eda_fractal_prior.py preserves original notebook; initial raw results are preserved.",
            "continuation_probe_executed.py preserves exact initially executed source/hash.",
            "Final script differs only in formatting and strict zip length checks from that source.",
            "Commands (prefix EVERY command with continuation/run_cpu.sh eda-fractal):",
            " .venv/bin/python research/campaign_2026-10-03/eda/fractal/continuation_probe.py",
            " .venv/bin/python research/campaign_2026-10-03/eda/fractal/stationarity_control.py",
            " .venv/bin/python research/campaign_2026-10-03/eda/fractal/audit_provenance.py",
            " .venv/bin/marimo export html research/campaign_2026-10-03/eda/fractal/eda_fractal.py",
            "   -o research/campaign_2026-10-03/eda/fractal/eda_fractal.html",
            "Use repo-root research/campaign_2026-10-03/continuation/run_cpu.sh as that prefix.",
            "All compute commands completed exit0; final ruff checks and notebook export logged.",
            "Blocking integration issues: unknown campaign exposure/components; baseline comparison",
            "reserved for controller. No blocker prevented completion of this independent pass.",
        ]
    )
    (OUT / "REPORT.txt").write_text("\n".join(report) + "\n")
    state = json.loads((OUT / "STATE.json").read_text())
    state.update(
        phase="independent_pass_complete",
        pending_work=[],
        last_failure=None,
        inspected_figures=[
            "fig1_target_scaling.png",
            "fig4_mfdfa.png",
            "fig8_causal_clock.png",
            "fig9_shuffle_falsifier.png",
            "fig10_stationary_level_control.png",
        ],
    )
    state["commands"].extend(
        [
            "run_cpu.sh eda-fractal /usr/bin/time -v .venv/bin/python .../audit_provenance.py (exit0)",
            "run_cpu.sh eda-fractal .venv/bin/marimo export html .../eda_fractal.py -o .../eda_fractal.html (exit0)",
            "run_cpu.sh eda-fractal .venv/bin/ruff check <new scripts> (final exit0)",
        ]
    )
    state["artifacts"].extend(
        [
            "REPORT.txt",
            "provenance_audit.json",
            "eda_fractal.py",
            "eda_fractal.html",
            "artifact_provenance.json",
        ]
    )
    (OUT / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    names = state["artifacts"] + [
        "HYPOTHESES_initial_frozen.json",
        "continuation_probe_executed.py",
        "continuation_probe.py",
        "stationarity_control.py",
        "audit_provenance.py",
        "continuation_probe.log",
        "stationarity_control.log",
        "provenance_audit.log",
        "notebook_export.log",
    ]
    hashes = {
        name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        for name in sorted(set(names))
        if name != "artifact_provenance.json"
    }
    assert hashes["continuation_probe_executed.py"] == clock["script_sha256"]
    assert hashes["HYPOTHESES_initial_frozen.json"] == clock["hypotheses_sha256"]
    save = {
        "sha256": hashes,
        "original_execute_hash_verified": True,
        "compute_exit_codes": {
            "initial_probe": 0,
            "stationarity_control": 0,
            "provenance_audit": 0,
            "notebook_export": 0,
        },
        "limitations": "Independent phase only; no strong-baseline inspection or component certificate",
    }
    (OUT / "artifact_provenance.json").write_text(json.dumps(save, indent=2) + "\n")


if __name__ == "__main__":
    main()
