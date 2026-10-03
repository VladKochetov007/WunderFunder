# State-drift instrument: isolated binary local comparison

CPU instrument PASS; real scientific test UNRUN and UNAPPROVED. No real capture, prediction score or training data was opened. No GPU execution, private CUDA RNG restoration, receipt, queue action or trainer/source edit occurred. All new artifacts remain in round6/state_drift.

PREREG.json freezes thresholds and rivals before this actor reads any scientific capture. The only real comparison is stale captured incoming STUDENT h vs a detached current-pre-STUDENT-weight prefix-consistent h. The reference starts at None and runs original full inputs through the actual model and captured student normalization buffers for four 1000-row calls, detaching after each. No optimizer, EMA or scheduler update occurs in that reference. It never resets at query4000 or substitutes EMA weights. The query4000:5000 original source local step changes only student incoming h and restores the same genuine optimizer moments, clocks, scaler/RNG, EMA weights/hidden, cached teacher and target/denominator mathematics.

Pre alone is insufficient provenance. SCHEMA.json requires controller-bound evidence of the actual source entrypoint, initial None, four warm chunk ATTEMPTS, actual completed/skipped counts, full input identity and student mean/std/panel identity. Missing evidence is UNKNOWN and fails closed. Source inspection shows v4 make_environment initializes h/h_ema=None; runner.run_probe real performs four source chunk attempts at starts0/1000/2000/3000 and then captures at4000. That source path does NOT prove a future artifact took that path; the independent receipt and bound trajectory evidence remain mandatory. The sidecar template is not a fabricated scientific trajectory.

The original PREREG.json is preserved. After the controller announced a completed default measurement, but BEFORE this actor opened any actual captured values, the controller clarified that warm AMP skips may occur. PREREG_WARM_CLOCK_CLARIFICATION.json explicitly corrects the initial four-completed/non-skipped assumption: completed count=4-actual skips, checked against EVERY genuine warm Adam integer step, with at least one completed update required. Attempt/step/scheduler/EMA counters remain4 by source semantics. The CPU schema-only skip fixture is NOT an active AMP skip simulation. Scientific thresholds, rivals and floor rules are unchanged. No real pre/post/result values have been opened; the controller-supplied directory string is recorded solely for later binding.

Six independent original/default updates establish their frozen whole-state numerical policy and RMS floors BEFORE the intervention. Six current-weight prefix repeats separately establish hidden floors. Target math must remain exact; EMA output parity uses the independent default floor; clocks/RNG remain exact. Whole unscaled preclip gradient and actual parameter-delta vectors are compared, with no certified component attribution. Every scientific observable must be finite. Floors cannot be estimated from, or relaxed for, the desired stale/reference difference.

Material local drift requires hidden relative RMS>=1%, main output RMS>=1e-4, and whole gradient OR actual update relative L2>=1%; each required effect must exceed five times its independent repeat floor. These deliberately declared local thresholds are not a benchmark improvement threshold. Outcomes distinguish material local drift, detectable subthreshold effects, unresolved numerical-floor effects and invalid/unknown cases. Rivals include locally insensitive hidden state, hidden drift invisible to the head/update, clip/Muon/moments attenuating changed gradient, and descriptor noise. Relative ratios use a documented 1e-12 denominator floor; raw RMS is retained so near-zero reference vectors are visible.

Tiny controls execute actual source RNN and original four-loss/update semantics: fixed weights prefix vs sequential carry bitexact; changing historical warm weights plants drift; unchanged-weight/null has equal predictions, whole gradients and updates; independent repeat floors are retained. The planted effect is DETECTABLE_SUBTHRESHOLD, not material under the unchanged scientific rule. Malformed CPU private RNG is rejected before any global setter. Missing approval and direct CUDA API calls fail before actual CUDA private-engine validation/restoration. CPU protocol is input3/hidden4/B2/T60 and cannot stand in for real CUDA. This is local early fresh-panel trajectory research, not historical full-fit state recovery, inference/WP usefulness or recipe attribution.

The prefix preserves student train/grad-enabled forward context, detaches hidden and discards the graph after each call, with no prefix backward. Immutable full batch handles are reused in the counterfactual observer rather than recopied on host: v4.restore's source-bound verbatim clone and original chunk's read-only batch/loss use are the basis of this memory optimization. Mutable pre-state is actually snapshotted and compared. Counterfactual storage contains only the hidden override and post-state, referring back to the externally hash-bound original pre rather than duplicating the full sequence. Actual CUDA descriptor/host/device-memory behavior remains untested.

Executed CPU command: `bash research/campaign_2026-10-03/continuation/round6/state_drift/run_cpu.sh --output-label cpu_controls_clock_final`. To reproduce after freeze, use a NEW label such as `cpu_controls_independent` (never overwrites floor directories; reproducing results.json changes its hash). Future approved invocation is `run_real.py --pre ... --post ... --trajectory ... --receipt ... --output ...`, only inside the independent controller campaign_job/run_gpu chain. No launch command or approval is supplied by this actor. Both v4 gate/runtime receipt and separate state-drift/source/capture receipt with independent critic PASS are enforced BEFORE actual tensor load/private CUDA validation. Future CUDA execution, host/device memory cost and real private RNG validation remain untested and require controller review; there is no 32-fit or EMA/student factorial expansion.

Executable CPU evidence:

```json
{
  "checks": {
    "planted_changing_weights_hidden_drift": true,
    "actual_CPU_warm_counts": true,
    "schema_counts_attempts_vs_completions": true,
    "inconsistent_warm_moment_clock_rejected": true,
    "fixed_prefix_vs_sequential_carry_bitexact": true,
    "unchanged_weight_null_no_difference": true,
    "repeat_floors_independent_and_exact": true,
    "planted_drift_affects_prediction_gradient_update": true,
    "no_receipt_fails_before_capture_load_or_cuda_validation": true,
    "cuda_API_guard_before_private_validation": true,
    "unknown_trajectory_rejected": true,
    "private_rng_rejects_before_global_setter": true,
    "source_adapter_preserved": true
  },
  "cost": {
    "wall_seconds": 1.9341411509958562,
    "cpu_seconds": 1.9250348279999998,
    "peak_rss_kib": 791620
  },
  "synthetic_planted_effects": {
    "hidden": {
      "rms_difference": 0.00010427319904294595,
      "relative_l2": 0.00029560378577240566
    },
    "prediction": {
      "rms_difference": 5.221707399326394e-06,
      "relative_l2": 1.4461511916449864e-05,
      "reference_rms": 0.3610761744342056
    },
    "gradient": {
      "rms_difference": 2.7295483759770382e-06,
      "relative_l2": 3.5632213197046436e-05,
      "reference_rms": 0.07660339145600115
    },
    "update": {
      "rms_difference": 1.6573744773289125e-08,
      "relative_l2": 0.0008778911501998457,
      "reference_rms": 1.887904299925592e-05
    }
  }
}
```
