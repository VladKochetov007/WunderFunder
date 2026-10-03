from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1]
PREFIX = "research/campaign_2026-10-03/eda/puzzle_b"


def main() -> None:
    """Correct recovered sample provenance and freeze the measured report."""
    manifest = json.loads((OUT / "continuation_manifest.json").read_text())
    audit = json.loads((OUT / "ablation_audit.json").read_text())
    prior = {
        split: set(pd.read_csv(OUT / f"seqscan_{split}.csv", usecols=["rg"]).rg)
        for split in ("train", "valid")
    }
    prior["train"].update([0, 1, 2, 3, 500, 5000, 10000])
    for sequence in manifest["sequences"]:
        sequence["prior_own_exposure"] = sequence["rg"] in prior[sequence["split"]]
    manifest["exposure"] = (
        "Corrected from actual prior scan CSV IDs: 2000 random train row groups, "
        "all1873 valid, plus explicit train notebook rows. Initial recovery "
        "range assumption was incorrect; frozen protocols preserved. "
        "Campaign-wide novelty unknown."
    )
    manifest["exposure_counts"] = {
        split: sum(s["prior_own_exposure"] for s in manifest["sequences"] if s["split"] == split)
        for split in prior
    }
    (OUT / "continuation_manifest.json").write_text(json.dumps(manifest, indent=2))
    recovery = (OUT / "RECOVERY_SUMMARY_initial.txt").read_text()
    recovery = recovery.replace(
        "Old seqscan_train.csv covers first 2000 train row groups; seqscan_valid.csv covers first 1900 valid row groups. Those sequences are exposed, including target statistics.",
        "CORRECTION after reading own scan.py and actual CSV IDs: old seqscan_train.csv covers 2000 RANDOM train row groups (seed123), not first2000. seqscan_valid.csv covers ALL1873 valid row groups. Those sequences are exposed, including target statistics. Initial incorrect recovery is preserved as RECOVERY_SUMMARY_initial.txt; frozen protocol exposure assumption is corrected in manifest, not silently rewritten.",
    )
    (OUT / "RECOVERY_SUMMARY.txt").write_text(recovery)
    scores = pd.read_csv(OUT / "continuation_scores.csv")
    ablation = pd.read_csv(OUT / "ablation_scores.csv")
    recurrence = pd.read_csv(OUT / "continuation_recurrence.csv")
    sequence_scores = pd.read_csv(OUT / "continuation_sequence_scores.csv")
    count = sequence_scores.query("stride==7 and arm=='raw'")
    corners = recurrence[recurrence.rg.isin([54, 1385])].copy()
    corners["event_covariance_normalized"] = (corners.repeat_rate - corners.event_rate**2) / (
        corners.event_rate * (1 - corners.event_rate)
    )
    corners.to_csv(OUT / "corner_recurrence.csv", index=False)
    ids = {
        split: [(s["rg"], s["seq_ix"]) for s in manifest["sequences"] if s["split"] == split]
        for split in prior
    }
    report = f"""PUZZLE B — BOUNDED CONTINUATION REPORT
Verdict: clock phase is not supported as a general feature. Lag4 recurrence is an exploratory corner-effect candidate; period6 was negative. No baseline improvement claim.

SCOPE AND RECOVERED EVIDENCE
Abstract-only; own scripts/artifacts and dataset row groups only. No HOST_DATA_DESCRIPTION, other findings, baseline, installs, GPU, commits, or outside-directory edits.
See RECOVERY_SUMMARY.txt for old scores, unsupported interpretations, causal routing flaws and preserved negative metronome probe.
Verification tier C for baseline relevance: deterministic local metric, partially exposed validation, no controller baseline or independent component reproduction.
Initial recovery mistakenly treated the train scan as contiguous. Own scan.py instead samples 2000 random row groups. All1873 valid sequences were previously scanned. Correction and initial summary both preserved.
Frozen HYPOTHESES.json includes the mistaken range assumption; sampling was metadata-only and unchanged. Actual exposure flags are corrected in continuation_manifest.json. No fresh-valid inference.

WHAT ACTUALLY RAN
Frozen global ridge: 32 train sequences; fixed penalty .01*n; train-only scaling; clipped targets; current i0_p0..21/i0_dp0..3 plus causal event-age/rate/clock features.
Primary stride7 uses90976 fit rows; sensitivity stride6 uses106144 fit rows. All {audit["n_scored_rows"]} eligible scored validation rows across24 sequences; no valid stride and no target-scale routing.
Main probe: {manifest["wall_seconds"]:.3f}s, peakRSS {manifest["maxrss_kib"] / 1024:.1f}MiB. Conditional ablation: {audit["wall_seconds"]:.3f}s, peakRSS {audit["maxrss_kib"] / 1024:.1f}MiB. First ablation and extended moment audit have separate logs.
All compute used run_cpu.sh eda-puzzle-b; CUDA hidden. Resource failures: none. No server retained.
Metric: pool weighted sufficient moments, clip finite predictions/targets to[-2,2], weight each target by its own abs(clipped target), compute correlations separately and average. Zero targets have zero weight.
Scores are pooled metric values, not averages of per-sequence correlations. Paired sequence wins are descriptive and do not imply independent trials.
Prefix recomputation, future-suffix mutation, target mutation, and identical-prediction metric self-test passed.
Masks only select evaluation rows. seq_ix/targets/mask never enter main features. Sham fixed RNG phase offsets are experimental controls only.

MEASURED MATCHED TEST (primary stride7)
{scores[scores.stride == 7][["arm", "t0", "t1", "mean", "delta_age", "paired_sequence_wins_vs_age"]].to_string(index=False)}
Period4 delta vs age +.009784 but only10/24 wins: frozen14-win criterion fails. Compared with raw, mean gain is only+.005385.
Period4 improves t0+.021689 vs age and hurts t1-.002121. Period6 delta-.003792; stride6 delta-.010275. Preserve these negatives.
Raw beats age on pooled mean; do not presume event staleness helps.
Stride6 period4 gain+.008421 resembles stride7; wholesale sampling alias does not explain it in this test.
Sham6 goes from+.002793 at stride7 to-.018233 at stride6, showing a fixed row-phase control can be stride-sensitive. Different fit row counts are a caveat.

CAUSAL-FEATURE ABLATION (frozen AFTER first scores; same exposed sample)
{ablation.to_string(index=False)}
Capped-age control absorbs only part of original gain. Phase-only adds+.000064, effectively negligible. Recurrence-only adds+.006931. Full4 adds+.007909 versus cap4 and wins12/24: still below consistency threshold.
Matched sham4 has same capped age and same dimension as full4, adds only+.000840. This separates recurrence contrast from a generic extra-feature/capped-age explanation on this sample.
Full4 reproduction agrees to {audit["full4_reproduction_max_abs_difference"]:.2e}; this is same-author numerical consistency, NOT independent reproduction.
Lag4 recurrence is a causal statistic of available inputs, not proof of a physical causal mechanism. It can proxy a sequence regime or scale without explaining the targets.

CORNERS AND NEGATIVES
Full4-cap4 per-sequence mean deltas: rg54 +.090602; rg1385 +.062196. Other largest positive rg962 +.010892. Negative examples rg824 -.002769, rg1285 -.002746, rg114 -.002031.
rg54 and1385 selected after seeing deltas: exclusion sensitivity is post hoc, not an independent holdout test.
{json.dumps(audit["posthoc_concentration_sensitivity"], indent=1)}
Excluding both reduces delta+.007909 to+.000613. No broad improvement claim survives this concentration check.
Input-event lag3..7 curves differ sharply by sequence (fig11); peaks alone do not establish predictiveness. corner_recurrence.csv records rg54/1385 explicitly.
Scored rows per sequence range {int(count.n.min())}..{int(count.n.max())}; uneven row counts/target weights help corner sequences dominate pooled scores. No resampling of independent rows.
Target ACF and old period plots were inspected (fig08/09); they are descriptive and target-scale-conditioned, not deployable routing evidence.

SAMPLE PROVENANCE
Train (rowgroup,seq_ix): {ids["train"]}
Valid (rowgroup,seq_ix): {ids["valid"]}
Prior own exposure: {manifest["exposure_counts"]["train"]}/32 train;24/24 valid. Exact old sampled IDs are in seqscan_train.csv/seqscan_valid.csv. Other campaign exposure unknown.
Direct selected train-valid seq_ix overlap: {manifest["direct_seq_ix_overlap"]}. Distinct IDs do NOT prove nonoverlapping source windows or connected-component independence.
No valid confidence interval: controller must construct paired connected components before uncertainty estimation. No campaign baseline or score forecast.
Manifest includes source size/mtime/footer SHA256, actual sampled seq_ix, input hashes and scored-step hashes. Artifact hashes in ARTIFACT_PROVENANCE.json.

INSPECTED FIGURES AND RUNNABLE ARTIFACTS
fig08_target_acf.png; fig09_event_periodicity.png; fig10_causal_clock_falsifier.png; fig11_event_clock_corners.png; fig12_clock_ablation.png inspected visually.
eda_puzzle_b.py is a runnable marimo notebook reading measured own outputs; headless HTML export succeeded. eda_puzzle_b_prior.py preserves old notebook unchanged.
continuation_models.npz stores fitted main ridge coefficients. ablation_sequence_moments.npz permits exact concentration recomputation. CSVs preserve scores and per-sequence negatives.
No tracebacks occurred during numeric probes. A patch application failed before file change; fixed. Ruff flagged notebook display expressions/imports and statically conditional initialization; fixed, checks pass. Logs retained; final export clean.

RANKED NEXT DIRECTIONS (CONTROLLER ONLY; NOT RUN HERE)
1. Input-only lag4 recurrence gate on genuinely reserved components, target-specific: current-state predictor versus recurrence augmentation; lag7 sham and phase-only controls. Kill if benefits fail outside rg54/1385-like components or t1 harm offsets t0.
2. Reproduce the two corner effects on new matching input-regime sequences chosen by prefix recurrence only. Match event rate and current state without using targets. Kill if conditional gain vanishes or is explainable by target-derived scale leakage.
3. Integration against controller strong baseline at identical selected rows/components: baseline+lag4 statistic versus baseline, matched sham statistic. Kill if residual gain is zero; do not infer from this weak ridge.
4. Event-age nonlinear response with recurrence interactions: cap-only versus recurrence-only versus interaction. Freeze split and penalty; no clock period sweep. Kill if interaction does not beat recurrence alone or stride-coprime replication fails.
5. Sequence-window provenance audit before uncertainty: connect shared/overlapping source windows, cluster train-valid dependence, rerun paired component evaluation. Stop if overlap invalidates holdout; distinct seq_ix insufficient.

REPRODUCE FROM REPO ROOT (each command separately)
research/campaign_2026-10-03/continuation/run_cpu.sh eda-puzzle-b .venv/bin/python {PREFIX}/scripts/continuation_probe.py
research/campaign_2026-10-03/continuation/run_cpu.sh eda-puzzle-b .venv/bin/python {PREFIX}/scripts/clock_ablation.py
research/campaign_2026-10-03/continuation/run_cpu.sh eda-puzzle-b .venv/bin/marimo export html {PREFIX}/eda_puzzle_b.py -o {PREFIX}/continuation_notebook.html
research/campaign_2026-10-03/continuation/run_cpu.sh eda-puzzle-b .venv/bin/python {PREFIX}/scripts/finalize_continuation.py
Blockers to promotion: exposed valid; unknown source-window components; no independent reproducer; no baseline access by design. Bounded pass itself is complete.
"""
    (OUT / "REPORT.txt").write_text(report)
    results = {
        "phase_claim": "not_supported_in_this_linear_test",
        "period6_claim": "negative_on_fixed_sample",
        "period4_recurrence_claim": "exploratory_concentrated_effect; frozen consistency fails",
        "independent_reproduction": False,
        "exposure_correction": manifest["exposure"],
        "new_scores": "continuation_scores.csv",
        "ablations": "ablation_scores.csv",
    }
    (OUT / "RESULTS.json").write_text(json.dumps(results, indent=2))
    state = json.loads((OUT / "STATE.json").read_text())
    state.update(
        phase="complete",
        pending_work=[],
        last_failure=None,
        inspected_figures=[
            "fig08_target_acf.png",
            "fig09_event_periodicity.png",
            "fig10_causal_clock_falsifier.png",
            "fig11_event_clock_corners.png",
            "fig12_clock_ablation.png",
        ],
        inspected_sequence_ids={
            "train": ids["train"],
            "valid": ids["valid"],
            "prior": "actual IDs in seqscan_train.csv and seqscan_valid.csv",
        },
        prior_exposure_counts=manifest["exposure_counts"],
        blockers_to_promotion=results["period4_recurrence_claim"],
    )
    state["artifacts"] = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "STATE.json").write_text(json.dumps(state, indent=2))
    artifacts = {}
    for path in sorted(OUT.rglob("*")):
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.name != "ARTIFACT_PROVENANCE.json"
        ):
            artifacts[str(path.relative_to(OUT))] = {
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (OUT / "ARTIFACT_PROVENANCE.json").write_text(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
