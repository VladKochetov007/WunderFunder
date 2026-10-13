# QAT deployment R2 full-fold consumer clearance — 2026-10-13

**Decision: GO for the controller-admitted full-fold baseline comparison using the frozen R2 source and manifest hash below.** This is an independent consumer-validity clearance; the root controller still owns the run's CPU/RAM admission and collection. I did not launch or score the 468-group run.

## Exact admitted inputs

- R2 manifest: `artifacts/qat_renewable_r1/deployment_frozen_r2/manifest.json`, SHA-256 `8894af708dcba1500fc8c1b431d30468a29a9999531dd7133ec982c3b1d19556`.
- Evaluator source: `artifacts/qat_renewable_r1/deployment_source_r2/scripts/evaluate_qat_deployment.py`, SHA-256 `6e2c32088afbcca517d14bb40d08359eb03d93cdc829812f15f8a89ba5316f8a` (same current `scripts/evaluate_qat_deployment.py` hash).
- Native library: `artifacts/native_qat_r1/wnn_native.abi3.so`, SHA-256 `a4390ecf7bcd54bb6826b43f5dcf6d7a40064c64ed1450bdc1780c34c2a6e108`.
- Source-qualified F0 split: SHA-256 `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`; exactly 468 evaluation groups.

The run command uses the evaluator in the reviewed MAIN checkout and must pin the hash, for example:

```sh
.venv/bin/python scripts/evaluate_qat_deployment.py run \
  --output artifacts/qat_renewable_r1/deployment_frozen_r2 \
  --manifest-sha256 8894af708dcba1500fc8c1b431d30468a29a9999531dd7133ec982c3b1d19556 \
  --case native_q8
```

Repeat with each fixed case (`fp32`, `native_q8`, `packed_q8`, `recurrent_w4`, `input_w4_recurrent_w8`) under the controller's admitted wrapper. The evaluator checks the exact source snapshot, packager, engine, archive hashes, native library, split identity, ordered full-fold group list, and fitting-only profile membership before callbacks. Output directories are created immutably; keep the frozen manifest and inputs unchanged.

## Independent preflight

- R2 `checked_manifest` passed against the actual frozen manifest, all source files, four archives, and actual native library.
- The root's `DEPLOYMENT_GUARDS_R2.json` receipt binds this source/manifest and records rejection of a wrong manifest SHA, partial evaluation panel, and reordered panel. R2 corrects the R1 consumer gap: R1 checked split identity but did not require the requested evaluation list/order to equal the split's full list.
- I separately re-imported packed Q8 and both W4 transports. Exported native code arrays and scale bits matched the frozen archive arrays exactly; packed Q8 also matched the native-Q8 reference export.
- On actual parquet groups 1604 and 1295, sequence IDs and `(20,000,112)` input shapes matched the frozen manifest. The exact official selection mask reconstructed as `is_scored & need_prediction` matched the evaluator mask.
- Native-Q8 and packed-Q8 `EnsemblePredictor` callbacks reproduced A after A→B→A exactly, matched the stored two-trajectory predictions bitwise, and matched each other. The separate native state getter returned read-only `(2,256)` arrays; 1,000-row state snapshots matched the stored native/packed Q8 state receipt.
- Recomputed `global_wp_clip2_v1` scores from replayed Q8 predictions, labels and masks matched stored `t0`, `t1`, `wp`, and selected-row count.

The prepared profile remains only two complete fitting-component trajectories. The cleared full-fold run is exposed F0 development evidence, not a pristine holdout or a public-score forecast. Local callback measurements are not an EPYC guarantee. This clearance covers baseline serving endpoints built from the unchanged CP5 parent; trained-QAT endpoints need their own exact packed export and matched FP32-continuation PTQ comparison.

Related IDs: `H-QAT-W4-20261013`, `H-QAT-A8-20261013`, `H-QAT-LSQ-20261013`, `E-QAT-DEPLOYMENT-20261013-R1`.
