from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Freeze the evidence report and complete artifact provenance after inspection."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = json.loads((ROOT / "continuation_results.json").read_text())
    followup = json.loads((ROOT / "continuation_hazard_control.json").read_text())
    provenance = json.loads((ROOT / "continuation_samples.json").read_text())
    lines = [
        "STATISTICAL PHYSICS CONTINUATION REPORT",
        "Observed exploratory results; no campaign baseline or public-score claim.",
        "",
        "RECOVERY AND STATUS",
        "RECOVERY_SUMMARY.txt preserves the preceding read-only pass in full.",
        "Prior census covers ALL train/valid sequence targets. Four target scales",
        "and strong t0/t1 anti-correlation exist; target-scale identification is",
        "not a causal inference feature. Waiting times are pooled/overdispersed.",
        "s22_kernel.py uses target increments and target-selected step filtering:",
        "its kernel is not an observed-price process or causal prediction result.",
        "No predecessor saved score establishes s23_physfeat.py improvement.",
        "Original scripts, arrays and figures were preserved without modification.",
        "",
        "FROZEN PROCEDURE / NEW SAMPLE",
        "HYPOTHESES.json was written before sampling/scoring. Seed 610103.",
        "48 train / 32 valid groups, excluding 629/187 conservatively reconstructed",
        "possibly exposed feature groups. Unknown old CLI arguments remain unknown.",
        "No independence from the census or other campaign analysts is asserted.",
        "continuation_samples.json records exact row-group/seq_ix pairs and exposure.",
        "Train steps 611::8; score valid mask on steps >=611. Same rows for all heads.",
        "112 current features, 22 instantaneous i0 price-feature differences, log",
        "dwell, log row-clock event rate. Memory: differences smoothed at half-lives",
        "4/16/64 in event or row time; all 22 indices kept; no quote-depth assumptions.",
        "Matched stale memory uses SAME 66 features delayed 512 rows, with no wrap.",
        "No targets, target scale, future rows or score mask enter feature construction.",
        "No detector labels/target-selected steps are used. State resets per sequence.",
        "Weighted ridge: clipped target weights, train-only feature variance penalty",
        "equivalent to standardized ridge, lambda=.001*sum(weights); no tuning.",
        "Exact pooled clipped global WP; 400 paired sequence bootstrap draws.",
        "Campaign connected components are unknown; CIs are provisional sequence CIs.",
        "Prefix-invariance passed for all five feature designs on the first valid group.",
        "",
        "MEASURED H1: FEATURE-ONLY HISTORY",
        f"Train rows {result['train_rows']}; sampled scored rows {result['valid_scored_rows']}.",
    ]
    for name in result["models"]:
        t0, t1 = result["target_correlations"][name]
        lines.append(f"{name}: mean WP={result['scores'][name]:.6f}; t0={t0:.6f}, t1={t1:.6f}")
    for name, entry in result["deltas"].items():
        lower, upper = entry["sequence_bootstrap_95"]
        lines.append(
            f"Event minus {name}: {entry['event_minus_control']:+.6f}; 95% [{lower:+.6f},{upper:+.6f}]"
        )
    lines += [
        "H1 meets frozen local-proxy criterion versus instantaneous/clock and stale",
        "controls. Recent raw feature history contains useful information here.",
        "NEGATIVE: event clock does not establish superiority over row smoothing;",
        "the event-minus-row CI spans zero and t0 is slightly better with row time.",
        "This does not validate the predecessor's target-derived kernel or its sign.",
        "Stale memory loses to current/instant/clock: added dimensions alone do not",
        "help. However stale and recent histories need not have identical marginals.",
        "A flexible current-state model might absorb this small-ridge history gain.",
        "",
        "MEASURED H2/H3: DWELL AGE VERSUS ACTIVITY MIXTURES",
        "Next price-state event is an evaluation label at t+1, never an input.",
        "Within sequence/exact 22-price state/coarse past-rate stratum, compare",
        "age1..3 against8..31, >=10 observations per arm, harmonic support weights.",
        "H2: young next-event hazard .219593, old .068912; old-young=-.150681",
        "95% [-.196179,-.112111]. Stale-age old-young=-.004952; different support.",
        "HYPOTHESES_FOLLOWUP.json froze a stricter H3 after H2 was observed.",
        "H3 uses real and stale age on identical eligible strata and then adds",
        "a2/a3 matching in fixed .5-wide bins, without changing the sample.",
    ]
    for name, entry in followup["results"].items():
        lower, upper = entry["contrast_bootstrap_95"]
        lines += [
            f"{name}: common strata={entry['common_strata']}; real old-young={entry['real_old_minus_young']:+.6f}; stale={entry['stale_old_minus_young']:+.6f}",
            f"Real-minus-stale contrast={entry['real_minus_stale']:+.6f}; 95% [{lower:+.6f},{upper:+.6f}]",
        ]
    lines += [
        "H3 supports a conditional AGE ASSOCIATION. It challenges a price-state-only",
        "memoryless hazard, not a complete Markov model: volumes, latent activity,",
        "clock-bin residual variation, and batching remain unmeasured explanations.",
        "Hazard prediction concerns update occurrence, not target direction/value.",
        "",
        "RANKED NEXT DIRECTIONS (PROPOSED; NOT RUN)",
        "1 Recent-history prediction versus nonlinear state decoding. Cheapest:",
        "  frozen memory augmentation of a modest flexible current-state head.",
        "  Control: same current head plus stale and row histories, same dimensions.",
        "  Kill: no paired scored-WP gain after flexible state decoding. Inference:",
        "  66 causal EMAs and current features; no detector or target normalization.",
        "2 Age-conditioned memory decay instead of homogeneous clock kernels.",
        "  Cheapest: fixed age x EMA interactions added to the same ridge.",
        "  Control: additive age/EMA plus equal-size stale-age interactions.",
        "  Kill: no target WP gain despite event-hazard association. Inference:",
        "  dwell and activity counters; gates computed from causal raw features.",
        "3 Shared-offset/signed-target geometry rather than two ordinary returns.",
        "  Cheapest: direct vs sum/difference supervised heads on identical features.",
        "  Control: same parameter budget, regularization and scoring rows.",
        "  Kill: reconstruction has no held-out gain; never assume exact target law.",
        "  Inference: predict both latent coordinates from inputs, reconstruct t0/t1.",
        "4 External i1/volume flow after i0 history, outside favored particle lens.",
        "  Cheapest: fixed causal i1/volume innovations appended to history head.",
        "  Control: stale i1/volume histories matched for dimensionality/activity.",
        "  Kill: residual gain disappears. Inference: maintain streams per sequence.",
        "",
        "ARTIFACTS / VERIFICATION / FAILURES",
        "Runnable: prepare_continuation.py, causal_features.py, continuation_probe.py,",
        "continuation_hazard_control.py, continuation_figures.py, finalize_continuation.py.",
        "Runnable marimo eda_statphys.py reads saved measurements; eda_statphys.html",
        "was exported headlessly under wrapper without server or retraining.",
        "Inspected f24_causal_memory_controls.png and f25_conditional_age_controls.png.",
        "Raw outputs: continuation_results.json, continuation_moments.npz,",
        "continuation_coefficients.npz, continuation_hazard_control*.json/npz, logs.",
        f"Main probe seconds={result['seconds']:.2f}; peak RSS={result['peak_rss_mib']:.1f} MiB; CPU only, one thread.",
        "Initial admission exit75 preserved in ADMISSION_FAILURE.txt and raw log.",
        "Controller explicitly authorized same-command retry; no new OOM/crash.",
        "Initial notebook syntax/lint packaging failures preserved in",
        "VERIFICATION_FAILURES.txt, fixed without scientific method changes or refits.",
        "Final ruff and headless notebook checks passed; source/output SHA256 hashes",
        "and dataset sizes are in ARTIFACT_PROVENANCE.json (no whole-dataset hashing).",
        "All computation used continuation/run_cpu.sh eda-statphys. No GPU, installs,",
        "other analysts, campaign baseline, commits or writes outside own EDA scope.",
        "",
        "PRIMARY PAPERS READ IN PRECEDING PASS (MOTIVATION, NOT DATA EVIDENCE)",
        "https://arxiv.org/pdf/1104.4596",
        "https://arxiv.org/pdf/cond-mat/0406224",
        "",
        "SAMPLED SEQUENCE IDS (ALL PREVIOUSLY TARGET-CENSUS-EXPOSED)",
    ]
    for split, records in provenance["samples"].items():
        ids = [str(record["seq_ix"]) for record in records]
        lines += [
            f"{split}: " + ", ".join(ids[start : start + 8]) for start in range(0, len(ids), 8)
        ]
    lines += [
        "",
        "RESUMPTION: controller must reconcile shared campaign components/exposure;",
        "reproduce local history gain against flexible state controls; only then",
        "authorize strong-baseline integration. No active scientific blocker.",
    ]
    (ROOT / "REPORT.txt").write_text("\n".join(lines) + "\n")
    manifest = json.loads((ROOT / "ARTIFACT_PROVENANCE.json").read_text())
    paths = [
        path
        for path in ROOT.iterdir()
        if path.is_file() and path.name not in ["STATE.json", "ARTIFACT_PROVENANCE.json"]
    ]
    paths += [
        ROOT / "data" / name
        for name in [
            "census_train.npy",
            "census_valid.npy",
            "kernel.npz",
            "waits.npy",
            "clock_sample.parquet",
            "dacf.npz",
            "acf_targets.npz",
            "acf_sumdiff.npz",
        ]
    ]
    paths = [path for path in paths if path.name != "continuation_finalize.log"]
    manifest["artifacts"] = {
        str(path.relative_to(ROOT)): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    }
    manifest["verification"] = {
        "figures_visually_inspected": manifest["figures"],
        "prefix_invariance": result["prefix_causality_checks"],
        "ruff": "passed",
        "headless_notebook_export": "passed",
        "source_changes_after_fit": "formatter/import sorting/strict zip only; no scientific refit",
    }
    manifest["unhashed_live_files"] = [
        "STATE.json",
        "ARTIFACT_PROVENANCE.json",
        "continuation_finalize.log",
    ]
    (ROOT / "ARTIFACT_PROVENANCE.json").write_text(json.dumps(manifest, indent=2))
    state = json.loads((ROOT / "STATE.json").read_text())
    state["phase"] = "complete_report_frozen"
    state["pending_work"] = []
    state["next_experiment"] = "flexible stateless control then age x history interactions"
    state["last_failure"] = None
    state["failure_history"] = ["ADMISSION_FAILURE.txt", "VERIFICATION_FAILURES.txt"]
    state["artifacts"] = sorted(
        set(state["artifacts"] + list(manifest["artifacts"]) + ["ARTIFACT_PROVENANCE.json"])
    )
    state["commands"] += [
        "ruff format/check passed",
        "marimo export html completed",
        "finalize_continuation.py completed REPORT/provenance",
    ]
    (ROOT / "STATE.json").write_text(json.dumps(state, indent=2))
    logging.info("Frozen REPORT.txt (%s lines), STATE and provenance.", len(lines))


if __name__ == "__main__":
    main()
