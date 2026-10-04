# E450 seed-4 package parity worker

`verify_seed4_package.py` is a source-only CLI for a later bounded CPU worker.
It does not package or modify a submission. The expected archive is pinned to
SHA-256
`c4dc258b529a52267a5e108ef0936f7919221cbdd17913270c8eb3482557f94a` and must
contain the single prescribed merge
`c20261004_combo_s4_final@ep3|c20261004_combo_head_s4_final@ep0:1.0` at i16
precision.

The worker runs `scripts/verify_submission.py` in three fresh subprocesses over
the exact 20 full streams in `cache/valid_subset20.parquet`: i16/v3, i16/v2,
and native f32/v3. It requires 400,000 rows and the official metric for each
run, exact target/mask equality, bitwise i16 v2/v3 predictions, maximum i16 to
native-f32 error no greater than `3e-4`, and absolute official WP difference no
greater than `5e-6`. The callback script checks every first-99-row warmup
response and finite predictions on required rows.

It also checks that the packaged callback, native libraries, package members,
merged weights, source checkpoints, and ZIP match the pinned provenance. A
separate short child reconstructs the parent/head merge from the actual seed-4
checkpoints, verifies every non-head tensor is bitwise equal, and verifies the
packaged merged weights are bitwise equal to that merge. The existing
`scripts/head_transfer_packages.py --reset` path replays full stream A, stream
B, then A again and checks that a changed future suffix cannot alter earlier
outputs.

The report includes a diagnostic comparison of native f32 callback outputs to
the arithmetic average of the actual full-VALID parent and head FP32 caches.
That diagnostic has no pass/fail threshold. Every model checkpoint, FP32
cache/sidecar, target cache, package, callback dataset, and checker dependency
is hashed before use and rechecked after verification.

A suitable later invocation is:

```text
python research/campaign_2026-10-04/staging/e450/verify_seed4_package.py \
  --output research/campaign_2026-10-04/checks/e450_seed4_native_parity_v1
```

The intended worker cap is one CPU, 2 GiB RAM, no GPU, and 1,800 seconds. The
checker filters incidental callback timing fields from its readiness JSON and
does not run a quiet-window timing comparison. A successful result reports
`parity_passed_runtime_pending`; it does not establish runtime eligibility,
public score, or manual-upload readiness. The quiet-window runtime review is a
separate gate.
