# Strong-control recovery and F0 residual diagnostic

**Incumbent identified from the user's 18 October transcript:** [all 100 submission rows](public_submission_trace_r1/README.md) identify **`tl3.zip` / `1lmehfi3` / 0.66832 / 39m12s / 656MB**, displayed 23:52 on 15 October (timezone unspecified). The local inventory records its fixed recipe as 0.34 distilled s5 + 0.33 tied-lead s5 + 0.33 tied-lead s6. This supersedes the earlier missing-ID claim; the portal payload SHA remains unverified. A clean held-out counterpart of the complete train+valid production recipe is still needed. The reconstructed Main T3 control is a separate recipe, not the exact tl3 ensemble.

**15:28 UTC:** [Full source and loader preflight passed](export_preparation/preflight_execution_r2/records/README.md), with exact raw receipt captured and no model or GPU run. F1 is fully collected and accepted. The historical strong-control export is the next operation. Its extraction R1 was found incompatible with the frozen producer sidecar before execution and is being replaced by isolated R2; no score or prediction was modified.

**Current status, 18 October 15:22 UTC:** F1 result collection, 57-file final-path verification and independent acceptance are complete ([acceptance receipt](../priority25_execution_r4/receipts/RANK01_F1_S1001_RESULT_ACCEPTANCE_V2.json)). The root-gated CPU preflight is running as `preflight-r2-20261018t1520z`; its first tar attempt failed before extraction because `--gzip` was omitted, and the retained failure was followed by a successful separate recovery into the exact empty root. No preflight result/pass is recorded. See the [preflight packet](export_preparation/preflight_execution_r2/README.md); earlier handoffs are dated history.


**Update 14:40 UTC:** SSH is restored. [Recovery observation](ACCESS_RECOVERY_20261018T1426Z.json)
confirms healthy F1 deep epoch 5 and no new memory events. The
[residual diagnostic](diagnostics/results/README.md) has now run successfully;
the [historical plot](plots/README.md) is collected and visually checked.
Both CPU units exited 0 and were stopped after capture. The [current continuation](CONTINUATION_20261018T1440Z.md) records next actions; the
[14:00 handoff](CONTINUATION_20261018T1400Z.md) preserves the preceding access failure.
[Findings](FOLLOWUP_RESULTS.md) separate actual results, historical evidence and
the still-pending comparison with strong-control predictions.

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

The no-training, not-admitted export plan is in
[`export_preparation/README.md`](export_preparation/README.md). Its
[`F0_MEMBERSHIP_COMPARISON.json`](export_preparation/F0_MEMBERSHIP_COMPARISON.json)
confirms exact ordered fit/evaluation/exclusion memberships and component-map
equality between the historical Main F0 split and the current canonical F0
split, while documenting their different manifest and `chain_folds.npz`
identities. The preparation names the restored CP5 and propensity checkpoints and the
missing historical cache. An isolated compatibility path is being prepared
and reviewed; no strong-control inference or training has been admitted.

The isolated [`strong_control_evaluation_r1`](strong_control_evaluation_r1/README.md)
packet prepares same-row alignment and a fixed comparison against this
historical CP5 control after its full-valid export arrives. It preserves the
existing diagnostic inputs and remains unexecuted pending the real export and
sidecar hashes.

## Evidence and operational index

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
