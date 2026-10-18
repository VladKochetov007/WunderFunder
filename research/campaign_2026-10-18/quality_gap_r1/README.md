# Strong-control recovery and F0 residual diagnostic

Current status and next actions are in the
[continuation handoff](CONTINUATION_20261018T1400Z.md).
[Findings](FOLLOWUP_RESULTS.md) separate completed restoration and historical
evidence from diagnostic work that has not run on actual arrays.

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
The diagnostic has not been staged or run on actual prediction arrays.

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
identities. The preparation names the restored CP5 checkpoint, the still-needed
propensity checkpoint and old cache, and the exact-source and separately
reviewed compatibility paths. No inference or training was admitted.

## Evidence and operational index

- [Restoration](recovery/README.md): ten exact checkpoint/provenance files,
  34,842,630 bytes, were restored successfully; the unit exited zero and was
  captured/stopped. This is not prediction or tensor verification.
- [Asset map](CONTROL_ASSET_MAP.json): source and ancestry inventory; use the
  actual restoration receipt for updated availability.
- [Diagnostic preparation and independent review](diagnostics/README.md):
  pinned inputs, three passing synthetic tests, resource and collection bounds.
  Actual residual measurements remain pending.
- [Historical trajectory](EXISTING_KD_TRAJECTORY.json) and
  [arithmetic interpretation](TRAJECTORY_INTERPRETATION.json): matched warm-parent
  continuation evidence, separating fitting and excluded-group trajectories.
- [Plot source](plot_existing_kd.py) and [launch](PLOT_LAUNCH.json): aggregate-only
  CPU plotting was launched; terminal capture and visual review are pending.
- [SSH incident](SSH_ACCESS_INCIDENT_20261018T1350Z.json) and
  [bounded rechecks](SSH_ACCESS_RECHECK_20261018T1358Z.json): TCP connects but
  banner exchange times out before authentication. Training health is unobserved.
- [Public baseline](PUBLIC_BASELINE_20261018.json): dated team/leader snapshot,
  without an exact incumbent ZIP binding.
- [Provider status](PROVIDER_STATUS_20261018T1355Z.json): the official DataSphere
  incident remains visible; this does not explain Beleriand SSH failures.

[Seventh bounded SSH recheck](SSH_ACCESS_RECHECK_20261018T1405Z.json) after backoff also failed before authentication; no job was modified. Further polling is deferred to the existing heartbeat.

[14:19 UTC SSH recheck](SSH_ACCESS_RECHECK_20261018T1419Z.json): one bounded attempt after backoff again timed out before authentication. F1 health and plot outputs remain unobserved; no job or admission changed.
