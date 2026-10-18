# Historical F0 strong-control same-row evaluation R4

**Status: prepared for independent review; not admitted or executed.** R4 is a
minimal consumer repair for the captured full-valid H390 feature-only export.
It preserves R3's extraction, row alignment, and grading inputs. R2 and R3
remain unchanged, including their failed collection receipts.

R3 passed static metadata review, then the root-gated real alignment reached
source extraction and compact-bundle comparison before failing while it
rehash-bound captured evidence. `validate_receipt()` returned the captured
pre-inference path/hash and captured-sidecar hash, but `validate_export()`
rebuilt a smaller mapping and dropped those three fields. The preserved
`records/R3_ALIGN_FAILURE_V1.json` records the `KeyError` for
`captured_preinference_path`. R4 forwards the entire validated capture-binding
mapping alongside the explicit prediction, contract, receipt, and row-map
fields consumed by `collect()`.

The added positive tiny test calls the real `collect()` and `validate_export()`
with producer-shaped metadata and a 4-row synthetic prediction/bundle. It
checks the six post-extraction hashes, the captured-evidence fields in
`ALIGNMENT_RECEIPT.json`, the model and alignment bindings in `DERIVED_INPUTS.json`,
and the diagnostic loader's actual strong-control hash/shape/finiteness and
row-alignment acceptance path. The test bypasses only the historical baseline
receipt chain, which remains bound by the frozen diagnostic inputs. The other
19 tests continue to check captured producer JSON/source identities, sidecar
ordering, extraction masks, create-only behavior, grading math, and previous
path/layout repairs. None of R4's preparation tests load the full prediction,
checkpoint, or validation-parquet payload, or contact a remote system.

The captured export is bound by contract SHA-256
`c534c0a23eacc2d174864393ab4ac0c38debd34b39b1d897d9023e2d2e487204`, receipt
SHA-256 `13d75acf3e20abea5c1bce8aa3ae21fbae67f32e0e7b0fee4412882800813197`,
sidecar SHA-256 `d8ca5c734cc7fac000d530b8d0cc6797fa22cb12bcad8fff25e00f8d1cd43d28`,
and prediction SHA-256
`3a6e7348334fa8bffe31776abcbd8c3803e80b03f8a3f92f2995ac15be65bc30`. The
receipt records a finite FP32 prediction shaped `[1873, 20000, 2]` with
299,680,128 bytes, `campaign_fit_scope=full-valid-feature-only`, null
evaluation-membership hashes, and an explicit full-valid row-group order. The
row-order digest does not establish evaluation-membership proof.

## Root-gated collection

Run only after root admission, with the bounded wrapper and a new create-only
output directory. The command binds the captured export:

```bash
uv run --no-sync python \
  research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r4/collect_control.py \
  --repo-root /home/grinchenkoeg/computer-vision-homework-20261018 \
  --valid-parquet /home/grinchenkoeg/computer-vision-homework-20261018/data/datasets/valid.parquet \
  --prediction /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root/cache/pred_clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1_valid.npy \
  --sidecar /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root/cache/pred_clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1_valid.json \
  --export-contract /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root/research/campaign_2026-10-18/quality_gap_r1/export_preparation/export_execution_r1/contracts/computer-vision-homework-quality-gap-control-export-v1.json \
  --expected-export-contract-sha256 c534c0a23eacc2d174864393ab4ac0c38debd34b39b1d897d9023e2d2e487204 \
  --expected-sidecar-sha256 d8ca5c734cc7fac000d530b8d0cc6797fa22cb12bcad8fff25e00f8d1cd43d28 \
  --export-receipt /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root/research/campaign_2026-10-18/quality_gap_r1/export_preparation/export_execution_r1/records/computer-vision-homework-quality-gap-control-export-v1/EXPORT_RECEIPT.json \
  --expected-export-receipt-sha256 13d75acf3e20abea5c1bce8aa3ae21fbae67f32e0e7b0fee4412882800813197 \
  --output /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/strong-control-eval-r4
```

Keep the canonical import path
`/home/grinchenkoeg/computer-vision-homework-20261018/source-a266b036/src`
unchanged. The root's metadata layout supplement provides byte-identical
`src/wnn/evaluation.py` and `src/wnn/gmetric.py` at the B project root for the
later grade receipt's source-hash fields.

The collection reads only six source columns from the 468 evaluation row
groups, checks labels and identities against the existing compact bundle, and
publishes `strong_control_scored.npy`, `ALIGNMENT_RECEIPT.json`, and
`DERIVED_INPUTS.json` with create-only writes. This is a historical strong
pipeline control, not the exact public incumbent; it performs no training,
grading, blend search, or calibration. The planned execution bound remains
0.5 CPU, 1.5 GB memory, 300 seconds, CUDA hidden, and physical B scratch.

Run the tiny local tests with:

```bash
PYTHONPATH=src:research/campaign_2026-10-18/quality_gap_r1/diagnostics:research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r4 \
  uv run --no-sync python -m unittest discover \
  -s research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r4 \
  -p 'test_*.py' -v
```
