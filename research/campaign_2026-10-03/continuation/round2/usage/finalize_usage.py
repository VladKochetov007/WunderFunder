from __future__ import annotations

import json
import logging

import numpy as np
from inspect_control import digest
from usage_core import NAMES, REPO, ROOT, update_state


def main() -> None:
    """Freeze the report without promoting input sensitivity into gain evidence."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = json.loads((ROOT / "results.json").read_text())
    control = json.loads((ROOT / "control_identity.json").read_text())
    for name, expected in control["source_sha256"].items():
        if digest(REPO / name) != expected:
            raise ValueError(f"Reference source changed during pass: {name}")
    for index in range(6):
        summary = json.loads((ROOT / f"sequence_{index:02d}.json").read_text())
        if not all(
            summary[key]["passed"]
            for key in [
                "baseline_cpu_reference_gate",
                "identity_replay_gate",
                "cached_gpu_vs_cpu_gate",
            ]
        ):
            raise AssertionError("Identity gate failed")
    result["claims"] = {
        "supported": [
            "i0/i1 current and past donor substitutions affect predictions",
            "history responses persist after original inputs resume",
            "i0 256-vs32 history-window contrast is nonzero on every component",
            "synthetic planted80-row memory detected; sham history response zero",
        ],
        "unknown": [
            "missing predictive information in frozen GRU",
            "intrinsic age-dependent kinetics after controlling donor amplitude",
            "predictive gain from a new representation",
        ],
        "not_supported": [
            "event-clock superiority or trunk saturation",
            "a representation-changing GPU proposal",
            "ranking importance from unequal past/current intervention sizes",
        ],
    }
    result["clarifications"] = [
        "organizer_age NPZ field means code ev_i0px age, not official event time",
        "quiet_age counts elapsed rows since any22 i0 price identifiers changed",
        "14 scored queries on5 components cannot establish forecasting gain",
    ]
    result["runtime"]["timing_scope"] = (
        "per-script main body; excludes module imports and wrapper admission"
    )
    (ROOT / "results.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Strong-GRU input-history usage map",
        "",
        "The fold-matched GRU256 responds to both instruments' current and past",
        "streams under frozen causal donor interventions. Identity and planted",
        "history controls pass. This is an input-space usage map, not evidence",
        "that a new representation supplies missing predictive information.",
        "No representation-changing training proposal is justified.",
        "",
        "## Recovery, control and sampling",
        "",
        "The own statphys report found local ridge history gains but unresolved",
        "event-versus-row smoothing. The user reports the later strong-model gate",
        "failed. That negative does not prove saturation; this pass instruments",
        "the existing model rather than repeating a ridge fit.",
        "",
        f"Control: {result['model']}; fixed epoch3, hidden256, two GRU layers.",
        f"Checkpoint SHA256: {control['checkpoint_sha256']}",
        f"Split SHA256: {control['split_sha256']}",
        f"Prediction SHA256: {result['sample_provenance']['prediction_sha256']}",
        "All inspected reference source hashes remained unchanged during the pass.",
        "Actual run fit groups (10535 train/1405 valid) passed component-exclusion",
        "checks against the frozen manifest and feature-only graph. Historical",
        "teacher lineage was not independently reconstructed in this pass.",
        "",
        "PREREG.json froze six interventions before sample/response inspection.",
        "Fold0 sequence IDs ranked by SHA256(round2-usage-v1: + seq_ix); first six",
        "distinct components. Targets, masks, classes, features and predictions",
        "never entered selection. All validation is already campaign-exposed;",
        "all sample targets were in the own census. None match reconstructed old",
        "own feature groups or the own32-sequence CPU pass. Unknown CLI exposure",
        "remains unknown, and no fresh campaign independence is claimed.",
        "",
        "| Row group | seq_ix | Component | Scored fixed queries |",
        "|---:|---:|---:|---:|",
    ]
    for index, sample in enumerate(result["sample_provenance"]["samples"]):
        summary = json.loads((ROOT / f"sequence_{index:02d}.json").read_text())
        lines.append(
            f"| {sample['row_group']} | {sample['seq_ix']} | "
            f"{sample['component']} | {summary['scored_query_count']} |"
        )
    lines += [
        "",
        "## Prefix replay and identity",
        "",
        "Propagate original rows0..4127, including all99 warmup rows. Query511..4095",
        "every64 rows:57 per sequence,342 total. State never resets at a window",
        "boundary. Carry saved full two-layer states before t-256, t-32 or t.",
        "Complete window predictions and original prefix-boundary states are saved",
        "in sequence_00..05.npz, with raw/packed inputs and masks/targets.",
        "",
        "The211 channels partition into i0=125, i1=78, auxiliary=8. Current inputs",
        "include causal deltas, flags, recency and held-event features, not just",
        "raw current state. Auxiliary channels stay original in every arm.",
        "",
        "Six arms: identity; i0_past256; i1_past256; i0_current; i1_current;",
        "i0_past32. Past arms replace the instrument's channels on[t-H,t) with",
        "observed donors64 rows earlier; complete query-row inputs remain original.",
        "Current arms substitute only rowt using t-64. Future inputs resume original",
        "rows through t+32, while the changed hidden state propagates.",
        "",
        "All donors precede affected rows: prefix compatible, no future injection.",
        "These are observed channel bundles in model-input space, not physically",
        "reconstructed market paths. Cross-stream, auxiliary, engineered-clock and",
        "hidden-history consistency can break. OOD movement is not signal importance.",
        "",
        "Synthetic planted80-row i0 memory responds to i0_past256 (RMS .40418),",
        "not short32 or i1 history. Current-only sham has zero history/identity",
        "response; current i1 is positive (RMS .27827). Future canaries pass.",
        "The toy validates intervention discrimination, not real identifiability.",
        "",
        "Independent actual reference: separately loaded original model with",
        "train_rnn.predict(chunk4000, CPU). Instrument: segmented carried-state replay.",
        "CPU float32 bound is atol2e-5 + rtol2e-5, fixed before measurements.",
        f"Reference max error={result['identity']['cpu_reference_max_abs']:.3g};",
        f"identity replay max error={result['identity']['replay_max_abs']:.3g}.",
        f"Cached GPU/CPU max={result['identity']['cached_gpu_max_abs']:.6f},",
        f"max RMS={result['identity']['cached_gpu_max_rms']:.6f}, correlations>.999.",
        "Cached AMP predictions satisfy separately frozen cross-backend bounds;",
        "not bitwise CPU/GPU equality. IDs, row order, raw prefix targets, masks",
        "and warmup align exactly. Original CUDA autocast disables on CPU with a",
        "preserved warning. No GPU executed.",
        "",
        "## Measured usage and propagation",
        "",
        "RMS pools both outputs; per-target values are retained in results.json.",
        "Positive lag means original inputs resumed. Identity query RMS is1.79e-7;",
        "all five real arms exceed the frozen .001 usage threshold.",
        "",
        "| Arm | Query RMS | Lag4 RMS | Lag32 RMS | Scored-query weighted-MSE damage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        entry = result["interventions"][name]
        rms = np.sqrt(np.mean(np.square(entry["rms_by_lag_target"]), axis=1))
        lines.append(
            f"| {name} | {rms[0]:.6f} | {rms[2]:.6f} | {rms[4]:.6f} | "
            f"{entry['scored_query_wmse_damage']:+.6f} |"
        )
    lines += [
        "",
        f"Long256-minus-short32 i0 query RMS={result['i0_long_minus_short_query_rms_raw']:.6f}",
        f"raw, {result['i0_long_minus_short_query_rms_clipped']:.6f} clipped.",
        "All six component contrasts are nonzero (clipped RMS .0242–.1189).",
        "Earlier changes still affect outputs after32 identical altered rows.",
        "This resolves a tested older-history contribution, not an exact horizon",
        "or missing usable history. Current-only substitutions recover faster;",
        "both GRU layers retain history-induced changes after32 resumed rows.",
        "One-row versus256-row and125-channel versus78-channel amplitudes cannot",
        "rank instrument importance or identify a skip/head route.",
        "",
        "## Quiet age and predictive-value limits",
        "",
        "Primary age is elapsed rows since ANY of22 i0 price identifiers changed,",
        "not seconds, best-quote ordering, or inferred target class. The saved field",
        "organizer_age means the CODE'S ev_i0px counter (p0 AND p11 change), not an",
        "official event clock. It disagrees with primary age at47 queries.",
        "",
        "| Age | Queries | Components | i0 current RMS | i0 past256 RMS | i1 past256 RMS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, entry in result["quiet_age"].items():
        arms = entry["interventions"]
        lines.append(
            f"| {label} | {entry['count']} | {entry['components']} | "
            f"{arms['i0_current']['query_rms']:.5f} | "
            f"{arms['i0_past256']['query_rms']:.5f} | "
            f"{arms['i1_past256']['query_rms']:.5f} |"
        )
    lines += [
        "",
        "Quiet age64+ does not erase history sensitivity. But donor displacement",
        "varies with age (current i0 standardized RMS about1.65→.63 from age0 to64+).",
        "Plots expose that confound; intrinsic age-dependent kinetics remain unknown.",
        "",
        "Only14 queries on five components are scored. Apparent i0 error improvements",
        "have component-bootstrap intervals spanning zero. i1-history corruption",
        "increases error (.04264; descriptive95% .00828–.11059), suggesting local",
        "useful-history sensitivity. Tiny OOD tests cannot establish conditional",
        "mutual information, a generalizable correction, or competition WP gain.",
        "No residual head or WP improvement was fitted. Big movement is not unused",
        "signal evidence; a posthoc null closes its specific cue/head, not saturation.",
        "",
        "## Ranked follow-ups",
        "",
        "1. Amplitude/state-matched past donors using only observed feature distances.",
        "   Match displacement, channels and age; kill age-kinetics claims if the",
        "   contrast disappears. No targets in matching.",
        "2. Compare raw-stream replay with causal features recomputed against packed",
        "   substitutions, holding query raw state fixed. Kill physical causal",
        "   interpretations if only inconsistent packed histories respond.",
        "3. Freeze a larger metadata-only component panel and later prefixes with",
        "   fixed cadence before testing targets. Kill apparent i0 improvement if",
        "   it fails transport; use component rather than row uncertainty.",
        "4. Cross-fit a fixed history-cue residual test against a flexible frozen-state",
        "   control and equal-budget shams. A null rejects that cue/head, not every",
        "   memory mechanism. Independent critique precedes integration.",
        "",
        "Representation-changing training proposals: NONE justified. Existing usage",
        "is supported; missing predictive information is unknown. No GPU launch.",
        "",
        "## Cost and retained artifacts",
        "",
        f"Six inference script bodies total {result['runtime']['inference_seconds']:.2f}s;",
        f"longest {result['runtime']['max_command_inference_seconds']:.2f}s;",
        f"peak RSS {result['runtime']['max_rss_mib']:.1f}MiB, CPU one thread.",
        "All computation used run_cpu.sh round2-usage under900s per command.",
        "Measured body time excludes module imports and CPU-slot admission.",
        "No fullvalid forward, fitting, installs, live source/model changes or",
        "writes outside usage/. No current-pass model/resource failure occurred.",
        "Runnable scripts, COMMANDS.txt, PREREG, STATE, results, synthetic controls",
        "and replay/state/window arrays are retained with hashes and sample exposure.",
        "usage_map.png and usage_controls.png were visually inspected. No active",
        "blocker; next step is independent critique/reproduction and matched donors.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    excluded = {"STATE.json", "artifact_provenance.json", "finalize_usage.log"}
    artifacts = {
        path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in ROOT.iterdir()
        if path.is_file() and path.name not in excluded
    }
    manifest = {
        "artifacts": artifacts,
        "live_files_not_hashed": sorted(excluded),
        "checkpoint": {"path": control["checkpoint"], "sha256": control["checkpoint_sha256"]},
        "reference_source_hashes": control["source_sha256"],
        "split_sha256": control["split_sha256"],
        "cached_prediction_sha256": result["sample_provenance"]["prediction_sha256"],
        "sample_ids_and_exposure": result["sample_provenance"]["samples"],
        "visually_inspected_figures": result["figures"],
    }
    (ROOT / "artifact_provenance.json").write_text(json.dumps(manifest, indent=2))
    update_state(
        "complete_report_frozen",
        {
            "pending": [],
            "last_failure": None,
            "commands": (ROOT / "COMMANDS.txt").read_text().splitlines(),
            "claims": result["claims"],
            "artifacts": sorted(artifacts),
            "runtime": result["runtime"],
            "next": "matched donors/larger fixed panel; independent critique",
        },
    )
    logging.info("Frozen REPORT/results/STATE and %s hashes.", len(artifacts))


if __name__ == "__main__":
    main()
