# Strong-control recovery and F0 residual diagnostic

[Latest 18:40 UTC continuation](CONTINUATION_20261018T1840Z.md): F3 wide stage1 is independently accepted and deep stage1 is healthy. The [eight-file W18 metadata pass](tl3_w18_checkpoint_execution_r1/README.md) completed with independent acceptance. Actual target identities and creative propensity provenance still need closure; [bounded next steps](TL3_CLOSURE_NEXT_STEPS_20261018T1840Z.md) define the remaining scope. No new quality score or clean TL3 fit.

**Measured result, 18 October 16:29 UTC:** [same-row comparison and interpretation](strong_control_evaluation_r4/records/RESULTS.md) are complete. The restored historical Main T3 control scores **0.6811541566** versus the new Q8 package's **0.6738263577** on 808,863 identical F0 rows. Difference **+0.007327799**, conditional component-bootstrap 95% interval **[+0.00238446,+0.01352521]**. The first target supplies 69.81% of the difference; the retrospective `|clipped label| ≥ 1` bin supplies 75.24% of its additive contribution. These describe this exposed panel, not public causality or deployable label-based rules. Canonical float64 grade and diagnostic calculations agree. All export/alignment/diagnostic/grade outputs are captured; do not repeat completed work.

**Public incumbent identity:** the [100-row user transcript](public_submission_trace_r1/README.md) identifies **`tl3.zip` / `1lmehfi3` / 0.66832 / 39m12s / 656MB**. Its archived recipe is 0.34 distilled s5 + 0.33 tied-lead s5 + 0.33 tied-lead s6. Portal payload SHA remains unverified. These production models fit train+valid; the restored Main T3 control is a different recipe. The [reviewed clean counterpart plan](TL3_CLEAN_COUNTERPART_PLAN_20261018.md) specifies the still-needed source/config/ancestor closure and F0-excluding reconstruction. Do not score the all-data production checkpoints as held-out F0 or tune weights against public.

The bounded [TL3 source-closure packet](tl3_source_closure_r1/README.md) now resolves a command-line ambiguity: `fttall_s5@ep3` supplied tied-lead arguments, while both resolved `init` values are empty, so it is not a model-weight ancestor. Exact teacher/cache and propensity ancestry remain unproven; allocator OOM warnings are not per-batch update receipts. The packet is metadata-only and does not certify a clean-F0 counterpart.

The [actual checkpoint metadata extraction](tl3_checkpoint_metadata_execution_r1/README.md) completed on eight exact team checkpoints at17:01UTC, with stored configs and tensor schemas, exact hashes, exit0 and independent acceptance. Its [preparation](tl3_checkpoint_metadata_r1/README.md) remains frozen. All seven predictor configs include heldout valid; propensity has no stored fitting args. The [named ancestor follow-up](tl3_ancestor_followup_r1/README.md) led to the completed [two-parent fin256 metadata pass](tl3_fin256_metadata_execution_r1/README.md); propensity/target producer gaps remain. Do not repeat the eight-file extraction.

**Rank1:** F0 and F1 fits, exports, grades, terminal checks, result collection and independent acceptance are complete. [F1 acceptance](../priority25_execution_r4/receipts/RANK01_F1_S1001_RESULT_ACCEPTANCE_V2.json). Two paired cells per arm are not CP30. [F3 execution](../priority25_execution_r4/f3_beleriand_execution_r1/README.md) was admitted and launched at17:06UTC under its unchanged reviewed plan. Wide stage1 now has verified 29,760 successful updates; at 18:40 UTC deep stage1 reaches epoch0 batch540/1488, fresh heartbeat and zero new memory events. It is running, not a third completed cell.

**Provider state:** the [18:32 UTC observer and current live official status](provider_observation_20261018T1835Z/README.md) confirm zero nonterminal jobs among the latest 100, unchanged balance 41,652,791, successful restrictions access and continuing incident 2136 affecting DataSphere. No recovery or blind paid retry. The Mac performs only light orchestration, metadata work and tiny synthetic tests.

Earlier dated continuation files preserve the SSH outage, restoration, preparation and consumer failures. They do not override the completed result above. [Detailed findings](FOLLOWUP_RESULTS.md).

The [reviewed w18 history trace](tl3_w18_history_trace_r1/README.md) identifies direct teacher endpoints and the source-defined raw composite (coefficient sum 0.8125, with no normalization). It preserves the missing actual array-row and execution bindings, including absent fine-tune logs. The [reviewed propensity history](tl3_propensity_history_trace_r1/README.md) finds a structurally compatible legacy writer but no actual producer identity; it cannot certify fitting groups or normalization. The old [six-endpoint template](tl3_w18_checkpoint_preparation_r1/README.md) remains immutable preparation. A separate eight-file helper and fresh admission completed the [actual W18 metadata pass](tl3_w18_checkpoint_execution_r1/README.md), including the two observed fmu128 parents. Saved optimizer declarations are present in these eight files; exact runtime/source and fitting memberships remain unbound. Completed metadata loads are not repeated.

## Residual diagnostic preparation

This is a source-bound, descriptive decomposition of the current F0 wide, deep,
fixed FP32 75/25 mixture, and selected-row Q8 predictions. It does not compare
against the strong incumbent, establish a cause for the public-score gap, fit a
calibrator, select a new blend, or evaluate held-out data.

Run `diagnostics/diagnose_f0_residuals.py` with the pinned `diagnostics/INPUTS.json`.
The input manifest binds the 808,863-row compact bundle, selected Q8 prediction
array, fold and validation-parquet identities, and extraction/selection/native
grade receipts by SHA-256. It verifies that those receipts agree before reading
prediction data. It recomputes the 75/25 mix using the frozen float32
multiply-then-add order and checks all four endpoint scores against their
source receipts.

The script reads at most the 64 MiB compact bundle and 32 MiB external prediction
file, rejects more than one million selected rows, and processes calculations
in 50,000-row chunks. It does not scan the 5.6 GB parquet or read checkpoints.
Sequence position is unavailable because the compact source contains no
source-bound `step_in_seq`; the diagnostic records this and does not infer a
position.

For each model and target, the report records clipped weighted means and global
centered moments. Regime contributions use the same global centers, so label-bin
and component contributions sum to the official pooled score, and their paired
contribution deltas sum to the model endpoint delta. The shared label bins are
retrospective summaries, not candidate features. Predicted-magnitude and raw
clipping bins are model-specific and descriptive; the script never reports
paired gains over those changing memberships. Within-regime Pearson values are
also descriptive only. The calibration slope/intercept are the weighted
prediction-on-target regression, alongside weighted bias and means.

Example command on Beleriand:

```bash
/home/grinchenkoeg/WunderFunder/.venv/bin/python \
  research/campaign_2026-10-18/quality_gap_r1/diagnostics/diagnose_f0_residuals.py \
  --inputs research/campaign_2026-10-18/quality_gap_r1/diagnostics/INPUTS.json \
  --output /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/diagnostics-v1/RESULT.json
```

This is the inner command only. Use the reviewed resource wrapper, physical
scratch and source bindings in [the diagnostic runbook](diagnostics/README.md).
The diagnostic has already completed. This command documents the procedure;
do not rerun or overwrite its existing outputs.

The command creates a JSON report and matching CSV table. Both are create-only;
choose new output names for each run. Synthetic invariants are in
`diagnostics/test_residual_analysis.py` and can run without model data:

```bash
PYTHONPATH=src:research/campaign_2026-10-18/quality_gap_r1/diagnostics \
  python -m unittest test_residual_analysis -v
```

## Frozen F0 control export preparation

The historical preparation plan is in
[`export_preparation/README.md`](export_preparation/README.md). Its
[`F0_MEMBERSHIP_COMPARISON.json`](export_preparation/F0_MEMBERSHIP_COMPARISON.json)
confirms exact ordered fit/evaluation/exclusion memberships and component-map
equality between the historical Main F0 split and the current canonical F0
split, while documenting their different manifest and `chain_folds.npz`
identities. The preparation names the restored CP5 and propensity checkpoints and the
missing historical cache. The reviewed compatibility overlay and successful export are now recorded in
[`export_execution_r1`](export_preparation/export_execution_r1/README.md). No new training was needed.

The isolated [`strong_control_evaluation_r1`](strong_control_evaluation_r1/README.md)
packet preserves the first unexecuted consumer preparation. The separately
reviewed R4 consumer completed the actual alignment and grade documented above;
the R2/R3 runtime failures remain preserved.

## Evidence and operational index

- [Two fin256 parent configs](tl3_fin256_metadata_execution_r1/README.md): successful CPU-only extraction of both exact epoch3 parents,7.87MB, unchanged source hashes, exit0/noCUDA/no new memory events. Recorded configs are scratch/empty-distill with auxiliary-next/structure losses and production train+valid; `prop_f0` remains a learned-weighting dependency. The [separate preparation/review](tl3_fin256_metadata_r1/README.md) is preserved and must not be rerun.
- [Propensity and w18 producer trace](tl3_producer_trace_r1/README.md): recovered Main propensity recipe is a different checkpoint from creative TL3 `prop_f0`; the latter's fitting provenance and w18 target-cache producer bindings remain unresolved. Exact-path absence claims are bounded, not proof no archived copy exists.

- [Propensity recovery](propensity_recovery_r1/RESULT.md): two additional exact
  archive files restored and verified, 187,249 bytes, exit 0; closes that export
  dependency without loading a model or occupying another GPU.

- [Restoration](recovery/README.md): ten exact checkpoint/provenance files,
  34,842,630 bytes, were restored successfully; the unit exited zero and was
  captured/stopped. This is not prediction or tensor verification.
- [Asset map](CONTROL_ASSET_MAP.json): source and ancestry inventory; use the
  actual restoration receipt for updated availability.
- [Diagnostic preparation and independent review](diagnostics/README.md):
  pinned inputs, three passing synthetic tests, resource and collection bounds.
  [Actual outputs](diagnostics/results/README.md) are now collected and hash-verified.
- [Historical trajectory](EXISTING_KD_TRAJECTORY.json) and
  [arithmetic interpretation](TRAJECTORY_INTERPRETATION.json): matched warm-parent
  continuation evidence, separating fitting and excluded-group trajectories.
- [Plot source](plot_existing_kd.py) and [launch](PLOT_LAUNCH.json): aggregate-only
  CPU plotting exited 0; [capture and visual review](plots/README.md) are complete.
- [SSH incident](SSH_ACCESS_INCIDENT_20261018T1350Z.json) and
  [bounded rechecks](SSH_ACCESS_RECHECK_20261018T1358Z.json): TCP connects but
  banner exchange timed out before authentication. These are historical
  observations, superseded by the successful [access recovery](ACCESS_RECOVERY_20261018T1426Z.json).
- [Public baseline](PUBLIC_BASELINE_20261018.json): dated team/leader snapshot;
  its then-missing incumbent ID is superseded by the [user-supplied submission
  trace](public_submission_trace_r1/README.md). The trace binds tl3's ID, filename
  and reported score, and joins the archived local recipe by filename; uploaded
  portal bytes remain unverified.
- [Provider status](PROVIDER_STATUS_20261018T1355Z.json): the official DataSphere
  incident remains visible; this does not explain Beleriand SSH failures.

[Seventh bounded SSH recheck](SSH_ACCESS_RECHECK_20261018T1405Z.json) after backoff also failed before authentication; no job was modified. Further polling is deferred to the existing heartbeat.

[14:19 UTC SSH recheck](SSH_ACCESS_RECHECK_20261018T1419Z.json): one bounded attempt after backoff again timed out before authentication. F1 health and plot outputs remain unobserved; no job or admission changed.

- [Actual TL3 checkpoint metadata](tl3_checkpoint_metadata_execution_r1/README.md) — eight exact checkpoint hashes/configs recovered on CPU at 17:01 UTC; propensity matches its cache sidecar, while its fitting membership remains absent. Direct teachers disclose seed-specific parents and valid-only fine-tuning. No new fit or clean-F0 certification.

- [Named ancestor follow-up](tl3_ancestor_followup_r1/README.md) — exact fin256 parent paths are now known; missing source/config/membership and w18 producer records remain explicit. One small readonly probe, no repeated model extraction.

- [17:10 UTC continuation](CONTINUATION_20261018T1710Z.md) — F3 actual admission and running sequence, new preflight memory baseline, accepted TL3 checkpoint metadata and remaining exact ancestry/cache-source work.
