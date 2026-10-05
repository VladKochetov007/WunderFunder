# E479 independent grader v1 metadata incident

The original grader source `score_recurrent_adaptive.py` and its first failed
runner attempt are preserved unchanged. Attempt
`e479-learned-candidate-cpu-test-grade-v1` (quarantine id
`0616a765a7d36760d566876cec93f3ff1222cdc07786badc5be645ab1768b7a7`)
exited 1 with `ScientificFailure: command exited 1` before producing a grader
artifact. The captured cause was `verify_arms` expecting `batch` and
`chunk_rows` in each arm object. The actual completed E479 fit manifest records
those recipe fields once at the common manifest level (`batch: 4`,
`chunk_rows: 256`); arm records contain the fit-specific order, update count,
checkpoint, and telemetry, but omit those duplicate fields. This was a grader
schema assumption defect, not a fit failure or scientific negative. No test
prediction/label read occurred in the failed attempt.

`score_recurrent_adaptive_v2.py` is the separate correction. It validates the
common cadence at report scope and accepts absent per-arm cadence fields while
rejecting any contradictory values if present. It checks actual arm fields,
fit-order permutations, all telemetry rows, checkpoint hashes and model state
layouts. Its `--verify-fit-only` path recomputes the normalizer from fit-role
feature columns, compares it exactly with the fit manifest, strictly loads all
16 checkpoints, and stops before reading any test-role Parquet row. Full score
mode calls that verification first, then starts test-row reads. The scorer
fails closed on either pre-existing output destination before receipt, data,
or checkpoint work.

The runner fit attempt that v2 targets is
`e479-learned-candidate-sixteen-fit-v2`; its report attests the single
`fit_manifest.json` artifact. The verification command is:

```bash
.venv/bin/python research/campaign_2026-10-05/recurrent_activation_adaptive/score_recurrent_adaptive_v2.py \
  --run-dir research/campaign_2026-10-05/recurrent_activation_adaptive/runs/sixteen_v1 \
  --seal research/campaign_2026-10-05/recurrent_activation_adaptive/e479_sixteen_fit_v2.sealed.json \
  --attempt research/campaign_2026-10-04/runner/attempts/e479-learned-candidate-sixteen-fit-v2.json \
  --verify-fit-only --device cpu
```

Source hashes and validation:

- v2 scorer: `c9edc310bc8aefe9ddc2df43538a9fac45335ad8d5a9923d93bcf08398ff3965`
- v2 tests: `08012f4fe0a55eec4485f3c3fde27a416417612075c33ed7d8df81a8fe8ef6a6`
- Ruff and `py_compile` passed; the three synthetic tests passed.

The actual `--verify-fit-only` command passed under a transient user systemd
scope limited to CPUQuota=100% (one logical CPU), MemoryMax=2,147,483,648 bytes,
and MemorySwapMax=0. It printed
`all_fit_receipts_and_checkpoints_verified_test_rows_unread`, `arms: 16`, and
`test_rows_read: false`. It verified fit manifest SHA256
`9f6b4f9ed3994a857511a67254eff1f5de01e062e51d95b05da8fd9b1b964317` and loaded
all 16 hash-matched state dictionaries with strict model layout, finite tensors,
and identity normalization buffers. Normalization was recomputed using only
the 24 fit-role feature groups. The command stopped before test-role row access
and produced no score, result, or prediction export.
