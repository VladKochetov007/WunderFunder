# Submission ZIP to native-vector parity checker R1

This code-only delivery adds [`scripts/check_submission_zip_native_parity.py`](../../../scripts/check_submission_zip_native_parity.py) and focused tests at [`tests/test_submission_zip_native_parity.py`](../../../tests/test_submission_zip_native_parity.py). Its eight bounded unit tests pass. The checker imports the shipped ZIP in a fresh process, requires the selected native library to come from the requested ISA directory, streams only the first eight ordered F0 evaluation row groups, and compares required callbacks bitwise with a memory-mapped float32 `.npy` reference. Warmup callbacks must return `None`. The first complete sequence is replayed around the second (A-B-A), with explicit native reset before every group. ZIP, vector, split and data byte hashes are required command-line pins; extraction is temporary, path/special-file checked, size bounded and rejected on RAM-backed filesystems. The checker reads no labels and computes no metric.

The R1 QAT fixture supplied by the controller is:

- ZIP: `artifacts/qat_renewable_r1/submission_candidates_r1/seed3/kotiki_qat_lsq_xteam_s3_20261013.zip`, SHA-256 `967b7173981ab2df44f66d6e420d6bc0e2c861773ba4fee46d45ea4e297bc90e`.
- Native reference: `artifacts/qat_renewable_r1/final_blend_grade_r2/blend_qat_s3.npy`, shape `(468, 20000, 2)`, float32, SHA-256 `b574cbaa06623b83a84c9aa3c7b9d169cc6103c0abbf3c34e874ee85c66d449c`.
- F0 split: `research/campaign_2026-09-29/artifacts/splits/fold0.json`, SHA-256 `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`.
- Validation data: `datasets/valid.parquet`, SHA-256 `2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782`.

The original comparison pair uses `artifacts/team_models_20261013/xteam_h4.zip` and `artifacts/qat_renewable_r1/final_blend_grade_r2/blend_baseline.npy` (confirmed basename). Freeze and pass the original ZIP and baseline-vector hashes before running it; those values are intentionally not guessed here. A non-bitwise result is preserved in the JSON receipt with differing row/value counts and maximum absolute difference, and the command exits nonzero. This allows the controller to distinguish packaging/native-library drift from an exact match without treating a small numeric delta as success.

Run one ISA level per process because `wnn_native` is cached by Python after import. Create the new output directory before checking; the helper never overwrites a receipt or creates its parent. A QAT-seed3 v3 invocation is:

```sh
PYTHONPATH=src WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1100000000 WNN_RUNTIME_MAX=180s scripts/limited \
.venv/bin/python scripts/check_submission_zip_native_parity.py \
  --label qat-lsq-xteam-seed3-f0-v3 \
  --zip artifacts/qat_renewable_r1/submission_candidates_r1/seed3/kotiki_qat_lsq_xteam_s3_20261013.zip \
  --reference artifacts/qat_renewable_r1/final_blend_grade_r2/blend_qat_s3.npy \
  --split research/campaign_2026-09-29/artifacts/splits/fold0.json \
  --data datasets/valid.parquet --repo-root . \
  --output artifacts/qat_renewable_r1/zip_native_parity_r1/qat_s3_v3.json \
  --level v3 \
  --expected-zip-sha256 967b7173981ab2df44f66d6e420d6bc0e2c861773ba4fee46d45ea4e297bc90e \
  --expected-reference-sha256 b574cbaa06623b83a84c9aa3c7b9d169cc6103c0abbf3c34e874ee85c66d449c \
  --expected-split-sha256 09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d \
  --expected-data-sha256 2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782
```

For Python 3.11, invoke the same checker through the offline interpreter with `--require-python311` and clean `PYTHONPATH`/`LD_LIBRARY_PATH`; it verifies the actual ZIP's `wnn_native.abi3.so` import and selected ISA path in that interpreter. `--level v2`, `avx2`, and `v3` must each run in a separate process. This helper and its unit tests are implementation evidence only. The controller owns execution and interpretation of the generated receipt; no parity execution or quality approval is included in this note.
