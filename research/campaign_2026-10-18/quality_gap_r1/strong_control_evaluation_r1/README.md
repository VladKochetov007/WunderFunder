# Historical strong-control same-row evaluation

**Status: code prepared, not admitted or executed.** This packet prepares a
no-training comparison of the restored Main F0 CP5 strong control on the same
808,863 rows already used by the frozen Q diagnostics. It is a historical
strong-control pipeline, not a hash-bound exact public incumbent. The original
diagnostics, inputs, receipts, and extraction artifacts remain unchanged.
[`PREPARATION_MANIFEST.json`](PREPARATION_MANIFEST.json) seals the helper,
tests, README and frozen metadata pins; actual export identities are left
unbound until they arrive.

The full-valid feature-only prediction must arrive with its actual export
receipt and producer sidecar. Do not guess their hashes. Root must capture both
exact hashes and pass them as `--expected-export-receipt-sha256` and
`--expected-sidecar-sha256`. The helper verifies the sidecar's run/epoch,
checkpoint SHA-256 `e30577ff…b776eda`, old split SHA-256 `09263ad4…f9316a3d`,
full-valid scope, FP32 dtype and prediction digest, along with an explicit
row-group mapping for all 1,873 groups. It memory-maps the approximately
300 MB full prediction, checks finiteness in bounded row chunks, and reads only
`seq_ix`, `step_in_seq`,
`need_prediction`, `is_scored`, `t0` and `t1` from the 468 evaluation parquet
groups. It does not replay features or load a model/checkpoint tensor.

The local `valid.parquet` identity is checked against the existing Q8 full
grade receipt, whose full-file SHA-256 is
`2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782`. The
collector does not hash all 5.6 GB again: it verifies that receipt's pinned
full-file hash, fresh file size and mtime, the historical split's footer SHA,
Parquet group count and every group length, then validates source sequence IDs
and step order in each selected group. This is a conservative inherited
full-file identity, not a newly computed full-file hash.

For the old/canonical split difference, the collector checks both exact split
file hashes and the pinned `F0_MEMBERSHIP_COMPARISON.json`. It keeps the
checkpoint's old split hash in the control receipt while requiring ordered
evaluation membership and validation sequence/component identities to match
the canonical split used by the existing diagnostic bundle. After applying
`need_prediction AND is_scored`, it requires the extracted labels, row-group
IDs and component IDs to match the compact bundle bit-for-bit and by the
bundle's existing array hashes. The compact control NPY, alignment receipt and
derived diagnostic inputs are create-only.

## Arrival-time collection

Run only after root captures the actual prediction, sidecar, export receipt,
parquet path and both receipt hashes. Choose a new output directory. This is
the inner command; root owns the execution wrapper and resource gate.

```bash
uv run --no-sync python \
  research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r1/collect_control.py \
  --repo-root /path/to/WunderFunder-maintenance \
  --valid-parquet /path/to/B/datasets/valid.parquet \
  --prediction /path/to/B/cache/pred_clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1_valid.npy \
  --sidecar /path/to/B/cache/pred_clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1_valid.json \
  --expected-sidecar-sha256 ACTUAL_CAPTURED_SIDECAR_SHA256 \
  --export-receipt /path/to/captured/export-receipt.json \
  --expected-export-receipt-sha256 ACTUAL_CAPTURED_EXPORT_RECEIPT_SHA256 \
  --output /path/to/B/quality-gap-r1/strong-control-eval-v1
```

On success, `DERIVED_INPUTS.json` appends the aligned control as an NPY model
with its paired-bundle and alignment-receipt bindings. It adds fixed
comparisons against wide, the existing FP32 mixture and Q8. It performs no
blend search, calibration, or tuning. Run the existing unchanged diagnostic
with `--inputs strong-control-eval-v1/DERIVED_INPUTS.json` to obtain its
float64 `global_wp_clip2_v1` grades and descriptive regime contributions.
The planned collector bound is 0.5 CPU, 1.5 GB memory and 300 seconds, with
CUDA hidden and physical scratch supplied by the root execution wrapper.

After that diagnostic exits successfully, the independent grade helper checks
its endpoint scores against `wnn.gmetric.global_wp` and computes paired
2,000-draw component-bootstrap intervals with
`wnn.evaluation.paired_component_moments` and
`paired_component_bootstrap(centered=True)`. The same seed (271828) is reused
for each fixed comparison. Its command requires the actual diagnostic-result
SHA captured after that run:

```bash
PYTHONPATH=/path/to/WunderFunder-maintenance/src:/path/to/WunderFunder-maintenance/research/campaign_2026-10-18/quality_gap_r1/diagnostics \
  uv run --no-sync python \
  research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r1/grade_control.py \
  --inputs /path/to/B/quality-gap-r1/strong-control-eval-v1/DERIVED_INPUTS.json \
  --diagnostic-result /path/to/B/quality-gap-r1/strong-control-eval-v1/RESULT.json \
  --diagnostic-result-sha256 ACTUAL_CAPTURED_DIAGNOSTIC_RESULT_SHA256 \
  --output /path/to/B/quality-gap-r1/strong-control-eval-v1/PAIRED_COMPONENT_GRADE.json
```

The grade is conditional on these fixed predictions and one repeatedly
inspected F0 panel; it does not quantify training-seed or public-transfer
variation. The bootstrap resamples complete source components and recomputes
the pooled official metric, rather than averaging component correlations.
Both collection and grade are create-only and require distinct fresh output
paths. Tests use tiny synthetic arrays only; they do not open the real bundle,
control export, parquet payload, checkpoint or remote data.

Run the bounded fixtures with:

```bash
PYTHONPATH=src:research/campaign_2026-10-18/quality_gap_r1/diagnostics:research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r1 \
  uv run --no-sync python -m unittest discover \
  -s research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r1 \
  -p 'test_*.py' -v
```
