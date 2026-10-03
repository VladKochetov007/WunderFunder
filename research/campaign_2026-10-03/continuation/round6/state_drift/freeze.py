"""Freeze completed CPU checks, source preservation and scoped pending-real report."""

from __future__ import annotations

import json

from protocol import HERE, V4, cap


def main() -> None:
    """Require actual successful CPU controls; never approve a GPU candidate."""
    results = json.loads((HERE / "results.json").read_text())
    if not results["passed"]:
        raise ValueError("cannot freeze a successful instrument from failed checks")
    for name, expected in results["adapter_hashes"].items():
        if cap.sha(V4 / name) != expected:
            raise ValueError("protected adapter changed")
    if cap.identities() != results["source_hashes"]:
        raise ValueError("protected trainer/source changed")
    prior = HERE.parents[1] / "round5/credit_control"
    prior_hashes = json.loads((prior / "HASHES.json").read_text())
    if any(cap.sha(prior / name) != expected for name, expected in prior_hashes.items()):
        raise ValueError("prior credit_control artifact changed")
    state = {
        "state": "FROZEN_CPU_INSTRUMENT_PASS_REAL_UNAPPROVED",
        "checks": results["checks"],
        "actual_captures_opened": 0,
        "actual_cuda_private_rng_validations": 0,
        "actual_binary_drift_tests": 0,
        "gpu_approval": False,
        "cost": results["cost"],
        "preserved_prior_files": len(prior_hashes),
        "resources": "run_cpu.sh round2-critic 1core2GiB900s",
        "next": "independent state-drift critic review; later default-v4 gate and hash-bound trajectory/GPU approval only; real captures unopened",
        "original_prereg_preserved": True,
        "warm_clock_clarification": "separate controller-directed amendment; no scientific threshold relaxation",
    }
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    (HERE / "COST.json").write_text(
        json.dumps(
            {
                "cost": results["cost"],
                "measurement": "main-function compute excludes imports; peak RSS is process lifetime",
                "first_attempt": "failed at missing nested output directory before repeat/drift controls; corrected; shared resource barrier then blocked admission",
                "log_preservation": "first overwritten log recovered explicitly from tool output; admission log and first passing log separately retained",
                "whole_wrapper_cost": (HERE / "resource_cost.txt").read_text().strip(),
                "gpu_cost": 0,
            },
            indent=2,
        )
        + "\n"
    )
    report = """# State-drift instrument: isolated binary local comparison

CPU instrument PASS; real scientific test UNRUN and UNAPPROVED. No real capture, prediction score or training data was opened. No GPU execution, private CUDA RNG restoration, receipt, queue action or trainer/source edit occurred. All new artifacts remain in round6/state_drift.

PREREG.json freezes thresholds and rivals before this actor reads any scientific capture. The only real comparison is stale captured incoming STUDENT h vs a detached current-pre-STUDENT-weight prefix-consistent h. The reference starts at None and runs original full inputs through the actual model and captured student normalization buffers for four 1000-row calls, detaching after each. No optimizer, EMA or scheduler update occurs in that reference. It never resets at query4000 or substitutes EMA weights. The query4000:5000 original source local step changes only student incoming h and restores the same genuine optimizer moments, clocks, scaler/RNG, EMA weights/hidden, cached teacher and target/denominator mathematics.

Pre alone is insufficient provenance. SCHEMA.json requires controller-bound evidence of the actual source entrypoint, initial None, four warm chunk ATTEMPTS, actual completed/skipped counts, full input identity and student mean/std/panel identity. Missing evidence is UNKNOWN and fails closed. Source inspection shows v4 make_environment initializes h/h_ema=None; runner.run_probe real performs four source chunk attempts at starts0/1000/2000/3000 and then captures at4000. That source path does NOT prove a future artifact took that path; the independent receipt and bound trajectory evidence remain mandatory. The sidecar template is not a fabricated scientific trajectory.

The original PREREG.json is preserved. After the controller announced a completed default measurement, but BEFORE this actor opened any actual captured values, the controller clarified that warm AMP skips may occur. PREREG_WARM_CLOCK_CLARIFICATION.json explicitly corrects the initial four-completed/non-skipped assumption: completed count=4-actual skips, checked against EVERY genuine warm Adam integer step, with at least one completed update required. Attempt/step/scheduler/EMA counters remain4 by source semantics. The CPU schema-only skip fixture is NOT an active AMP skip simulation. Scientific thresholds, rivals and floor rules are unchanged. No real pre/post/result values have been opened; the controller-supplied directory string is recorded solely for later binding.

Six independent original/default updates establish their frozen whole-state numerical policy and RMS floors BEFORE the intervention. Six current-weight prefix repeats separately establish hidden floors. Target math must remain exact; EMA output parity uses the independent default floor; clocks/RNG remain exact. Whole unscaled preclip gradient and actual parameter-delta vectors are compared, with no certified component attribution. Every scientific observable must be finite. Floors cannot be estimated from, or relaxed for, the desired stale/reference difference.

Material local drift requires hidden relative RMS>=1%, main output RMS>=1e-4, and whole gradient OR actual update relative L2>=1%; each required effect must exceed five times its independent repeat floor. These deliberately declared local thresholds are not a benchmark improvement threshold. Outcomes distinguish material local drift, detectable subthreshold effects, unresolved numerical-floor effects and invalid/unknown cases. Rivals include locally insensitive hidden state, hidden drift invisible to the head/update, clip/Muon/moments attenuating changed gradient, and descriptor noise. Relative ratios use a documented 1e-12 denominator floor; raw RMS is retained so near-zero reference vectors are visible.

Tiny controls execute actual source RNN and original four-loss/update semantics: fixed weights prefix vs sequential carry bitexact; changing historical warm weights plants drift; unchanged-weight/null has equal predictions, whole gradients and updates; independent repeat floors are retained. The planted effect is DETECTABLE_SUBTHRESHOLD, not material under the unchanged scientific rule. Malformed CPU private RNG is rejected before any global setter. Missing approval and direct CUDA API calls fail before actual CUDA private-engine validation/restoration. CPU protocol is input3/hidden4/B2/T60 and cannot stand in for real CUDA. This is local early fresh-panel trajectory research, not historical full-fit state recovery, inference/WP usefulness or recipe attribution.

The prefix preserves student train/grad-enabled forward context, detaches hidden and discards the graph after each call, with no prefix backward. Immutable full batch handles are reused in the counterfactual observer rather than recopied on host: v4.restore's source-bound verbatim clone and original chunk's read-only batch/loss use are the basis of this memory optimization. Mutable pre-state is actually snapshotted and compared. Counterfactual storage contains only the hidden override and post-state, referring back to the externally hash-bound original pre rather than duplicating the full sequence. Actual CUDA descriptor/host/device-memory behavior remains untested.

Executed CPU command: `bash research/campaign_2026-10-03/continuation/round6/state_drift/run_cpu.sh --output-label cpu_controls_clock_final`. To reproduce after freeze, use a NEW label such as `cpu_controls_independent` (never overwrites floor directories; reproducing results.json changes its hash). Future approved invocation is `run_real.py --pre ... --post ... --trajectory ... --receipt ... --output ...`, only inside the independent controller campaign_job/run_gpu chain. No launch command or approval is supplied by this actor. Both v4 gate/runtime receipt and separate state-drift/source/capture receipt with independent critic PASS are enforced BEFORE actual tensor load/private CUDA validation. Future CUDA execution, host/device memory cost and real private RNG validation remain untested and require controller review; there is no 32-fit or EMA/student factorial expansion.
"""
    (HERE / "REPORT.md").write_text(
        report
        + "\nExecutable CPU evidence:\n\n```json\n"
        + json.dumps(
            {
                "checks": results["checks"],
                "cost": results["cost"],
                "synthetic_planted_effects": results["planted"]["effects"],
            },
            indent=2,
        )
        + "\n```\n"
    )
    hashes = {
        str(p.relative_to(HERE)): cap.sha(p)
        for p in sorted(HERE.rglob("*"))
        if p.is_file() and p.name != "HASHES.json"
    }
    (HERE / "HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps({"frozen": True, "prior_preserved": len(prior_hashes), "real_approved": False})
    )


if __name__ == "__main__":
    main()
