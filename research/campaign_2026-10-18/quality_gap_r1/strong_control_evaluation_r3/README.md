# Historical F0 strong-control same-row evaluation R3

**Status: prepared for independent review; not admitted or executed.** R3 is an
isolated consumer packet for the captured full-valid H390 feature-only export.
It preserves the R2 collection, row-alignment, and grading logic. R2 remains
unchanged, including its failed pre-payload align receipt.

R2 stopped before opening the prediction because its contract validator
required the measured Python executable path to equal the command's
`.venv/bin/python` path. The captured contract distinguishes those correctly:
the command uses `/home/grinchenkoeg/WunderFunder/.venv/bin/python`, while the
runtime receipt records the resolved uv interpreter at
`/home/grinchenkoeg/.local/share/uv/python/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13`.
R3 validates each path in its own role and pins the measured interpreter's
version, byte count, SHA-256, and package versions. Guard and utility hashes
remain pinned to the captured B-stage evidence.

`fixtures/` contains only the small captured JSON contract, export receipt,
guard attempt, sidecar, and pre-inference evidence. Tests validate the actual
contract and sidecar schemas, recompute the guard source-map digest, and hash
the current exporter utilities. The contract fixture test changes only its
contract-path field in a temporary copy so the local path-binding check can
run; all producer runtime, command, utility, and artifact pins remain intact.
The guard test similarly rewrites the temporary contract path and rebinds that
single source-map entry before checking the captured command and source map.
No prediction bytes, checkpoint tensors, validation parquet payload, network,
or remote execution were used in packet preparation or tests.

The actual captured export is bound by contract SHA-256
`c534c0a23eacc2d174864393ab4ac0c38debd34b39b1d897d9023e2d2e487204`, receipt
SHA-256 `13d75acf3e20abea5c1bce8aa3ae21fbae67f32e0e7b0fee4412882800813197`,
sidecar SHA-256 `d8ca5c734cc7fac000d530b8d0cc6797fa22cb12bcad8fff25e00f8d1cd43d28`,
and prediction SHA-256
`3a6e7348334fa8bffe31776abcbd8c3803e80b03f8a3f92f2995ac15be65bc30`. The
receipt records a finite FP32 prediction shaped `[1873, 20000, 2]` with
299,680,128 bytes, `campaign_fit_scope=full-valid-feature-only`, null
evaluation-membership hashes, and an explicit full-valid row-group order.
R3 does not reinterpret the row-order digest as evaluation-membership proof.

## Root-gated collection

Run only after the root admits this packet and supplies the current guarded
stage. Use a new create-only output directory. The captured values are shown
below so the command binds the actual export hashes:

```bash
uv run --no-sync python \
  research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r3/collect_control.py \
  --repo-root /home/grinchenkoeg/computer-vision-homework-20261018 \
  --valid-parquet /home/grinchenkoeg/computer-vision-homework-20261018/data/datasets/valid.parquet \
  --prediction /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root/cache/pred_clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1_valid.npy \
  --sidecar /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root/cache/pred_clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1_valid.json \
  --export-contract /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root/research/campaign_2026-10-18/quality_gap_r1/export_preparation/export_execution_r1/contracts/computer-vision-homework-quality-gap-control-export-v1.json \
  --expected-export-contract-sha256 c534c0a23eacc2d174864393ab4ac0c38debd34b39b1d897d9023e2d2e487204 \
  --expected-sidecar-sha256 d8ca5c734cc7fac000d530b8d0cc6797fa22cb12bcad8fff25e00f8d1cd43d28 \
  --export-receipt /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root/research/campaign_2026-10-18/quality_gap_r1/export_preparation/export_execution_r1/records/computer-vision-homework-quality-gap-control-export-v1/EXPORT_RECEIPT.json \
  --expected-export-receipt-sha256 13d75acf3e20abea5c1bce8aa3ae21fbae67f32e0e7b0fee4412882800813197 \
  --output /home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/strong-control-eval-r3
```

Keep the canonical import path `/home/grinchenkoeg/computer-vision-homework-20261018/source-a266b036/src` unchanged. The root's metadata layout supplement provides byte-identical `src/wnn/evaluation.py` and `src/wnn/gmetric.py` at the B project root for the later grade receipt's source-hash fields.

The collection still reads only the six source columns from the 468 evaluation
row groups, verifies labels, row-group IDs, and component IDs against the
existing compact bundle, and publishes `strong_control_scored.npy`,
`ALIGNMENT_RECEIPT.json`, and `DERIVED_INPUTS.json` with create-only writes.
The following independent grade continues to use canonical `wnn.gmetric`
scoring and paired component bootstrap; it does not tune a blend. Planned
collection bounds remain 0.5 CPU, 1.5 GB memory, 300 seconds, CUDA hidden, and
physical B scratch.

Run the tiny local fixtures with:

```bash
PYTHONPATH=src:research/campaign_2026-10-18/quality_gap_r1/diagnostics:research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r3 \
  uv run --no-sync python -m unittest discover \
  -s research/campaign_2026-10-18/quality_gap_r1/strong_control_evaluation_r3 \
  -p 'test_*.py' -v
```
