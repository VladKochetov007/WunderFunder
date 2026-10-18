# Historical F0 strong-control same-row evaluation

**Status: prepared, not admitted or executed.** This packet aligns the restored
Main F0 CP5 strong-control prediction to the exact 808,863 rows in the existing
F0 residual diagnostics. It is a historical strong-control pipeline, not a
hash-bound exact public incumbent. The frozen diagnostics and the earlier
[`strong_control_evaluation_r1`](../strong_control_evaluation_r1/README.md)
packet remain unchanged. R1 was stopped before execution because it expected
sidecar fields absent from the frozen H390 predictor; R2 validates the actual
H390 sidecar and binds the export contract and captured guarded attempt.

[`PREPARATION_MANIFEST.json`](PREPARATION_MANIFEST.json) binds the helper,
tests, documentation, frozen inputs, H390 predictor identity, and export
execution source hashes. Actual contract, receipt, sidecar, and prediction
hashes are captured only after the guarded export arrives.

The collector requires a contract with schema
`quality_gap.f0_strong_control_export_contract.r1` and a passing capture
receipt with schema `quality_gap.f0_strong_control_export_receipt.r1`. It
checks the exact `clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1` run with output
name `clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1`, epoch 5,
the H390 predictor SHA-256 `30647bb7…69554dd0`, source manifest SHA-256
`41a5452c…b3b236fa`, old split SHA-256
`09263ad4…79f9316a3d`, and control checkpoint SHA-256
`e30577ff…b776eda`. The command must use
`--campaign-full-valid-feature-only`, FP32 inference, the old split, and the
reviewed overlay-first `PYTHONPATH`; training and grading must be false.
The receipt must bind that exact contract, attempt ID, successful guarded exit,
raw attempt JSON, the exporter's utility source hashes and exact worker command,
prediction and sidecar bytes, and the captured strict-FP32 pre-inference record.
The worker's original record remains at `records/<attempt>/WORKER_PRE_INFERENCE.json`;
capture binds a byte-identical create-only archive at
`records/<attempt>/CAPTURED_WORKER_PRE_INFERENCE.json` as captured raw evidence.

R2 follows the frozen sidecar schema: `campaign_fit_scope` must equal
`full-valid-feature-only`; `evaluation_membership_sha256` and
`campaign_evaluation_membership_sha256` must be null; and `row_groups` must be
the explicit full-valid permutation. R2 derives the ordered row-group digest
from that list and compares it to the capture receipt. It does not require the
nonexistent H390 fields `campaign_export_scope` or `feature_only`, and it does
not treat the full-valid row-order digest as evaluation-membership proof.

The expected prediction is a memory-mapped float32 array shaped
`[1873, 20000, 2]` with 299,680,128 file bytes. R2 hashes it in bounded chunks,
checks finiteness in 32-group slices, and maps each physical parquet row group
to its prediction row through the captured sidecar order. It reads only six
columns—`seq_ix`, `step_in_seq`, `need_prediction`, `is_scored`, `t0`, and
`t1`—from the 468 evaluation row groups. It does not load checkpoint tensors
or replay features.

The 5.6 GB validation parquet is not hashed again. The collector validates the
existing Q8 full-panel receipt and source identities, fresh file size and
mtime, the full footer digest, and all 1,873 group lengths. It verifies the
old and canonical F0 split identities and their pinned ordered-membership and
component-map comparison. After applying `need_prediction AND is_scored`, it
requires labels, row-group IDs, and component IDs to match the compact
diagnostic bundle bit-for-bit. The compact control NPY, alignment receipt,
and derived inputs are create-only.

## Collection after captured export

Root must pass the actual hashes printed by export preparation. Do not replace
these values with the H390 predictor hash or an inferred row-order digest. Use
a new output directory:

```bash
uv run --no-sync python \
  research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r2/collect_control.py \
  --repo-root /path/to/WunderFunder-maintenance \
  --valid-parquet /path/to/B/datasets/valid.parquet \
  --prediction /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root/cache/pred_clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1_valid.npy \
  --sidecar /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root/cache/pred_clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1_valid.json \
  --export-contract /path/to/export_execution_r1/contracts/ACTUAL_ATTEMPT.json \
  --expected-export-contract-sha256 ACTUAL_CAPTURED_CONTRACT_SHA256 \
  --expected-sidecar-sha256 ACTUAL_CAPTURED_SIDECAR_SHA256 \
  --export-receipt /path/to/export_execution_r1/records/ACTUAL_ATTEMPT/EXPORT_RECEIPT.json \
  --expected-export-receipt-sha256 ACTUAL_CAPTURED_EXPORT_RECEIPT_SHA256 \
  --output /path/to/B/quality-gap-r1/strong-control-eval-r2
```

On success, `DERIVED_INPUTS.json` adds the aligned control as an NPY model and
fixed comparisons against wide, the existing FP32 mixture, and Q8. Run the
existing unchanged `diagnose_f0_residuals.py` with those derived inputs, then
run `grade_control.py` with the actual SHA-256 of its result. The grade helper
checks its endpoint scores against canonical `wnn.gmetric.global_wp` and
computes paired 2,000-draw component-bootstrap intervals using
`wnn.evaluation.paired_component_moments` and
`paired_component_bootstrap(centered=True)`, seed 271828. It does not tune or
select a blend.

Planned collector bounds are 0.5 CPU, 1.5 GB memory, 300 seconds, CUDA hidden,
and physical B scratch. This comparison is conditional on fixed predictions
and one previously inspected F0 panel; it does not measure training-seed or
public-transfer variation. Tests use tiny synthetic arrays and a synthetic
Parquet fixture only. No model/checkpoint, full prediction, real parquet
payload, compact-bundle arrays, or remote execution was used during packet
preparation.

Run bounded fixtures with:

```bash
PYTHONPATH=src:research/campaign_2026-10-18/quality_gap_r1/diagnostics:research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r2 \
  uv run --no-sync python -m unittest discover \
  -s research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r2 \
  -p 'test_*.py' -v
```
