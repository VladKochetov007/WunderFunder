# E420 native factorized parity preparation

This directory contains the parity harness, source inspection, bounded build and
synthetic-test records, and sealed-compatible draft specifications for E420. It
does not contain a zero-update export. The root owner must admit the draft before
the export or real-data parity command is run. There is no timing in this phase.

## Bound source and stream

The source is the existing fixed epoch-3 checkpoint
`/home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/runs/c20261003_w384_s3_f0/model.pt`,
SHA-256 `b5181f3aa948765f22a8c3f23f6e293ba715b52ffb91eb6eac02d6b7e30a8b8d`.
Read-only inspection found `epoch=3`, `checkpoint_epoch=3`, seed 3, fold 0,
hidden 384, two GRU layers, 211 inputs, and 23 outputs (two scored plus 21
auxiliary outputs). Training precision is FP16; checkpoint inference tensors
are FP32; `eval_ema=true`. The model is an ordinary, unfactorized GRU.
`SOURCE_INSPECTION.json` contains the exact shape and identity report.

E409's hash-bound streams are row groups 0, 1 and 2 of
`/home/vlad/development/wunder/wnn_connectome_starterpack/cache/valid_subset20.parquet`,
SHA-256 `cd6a349b3655bb6f98314e5db391e69b1e1ea7f12ab18f1d15c61cefd28daa37`.
Each contains 20,000 rows. E409's manifest is
`research/campaign_2026-10-04/E409_width_runtime/manifest.json` in this
worktree. E409 used `model_ep3.pt` from the same run (SHA-256
`90b3ef746971e220efaf1c64e7a51302f29fc3f3836edd3ebc00de3c78847be3`); E420
uses the requested `model.pt` and records both identities separately.

## Export draft

The zero-update control is exactly the rank-128 export helper already in this
tree. It saves the four FP32 left/right pairs in factor metadata and stores the
effective `left @ right` products in the ordinary inference state. The full
23-output head remains intact. Run this only after the root admits preparation:

```bash
systemd-run --user --scope --slice=wnn-campaign.slice \
  -p CPUQuota=100% -p MemoryMax=2G -p MemorySwapMax=0B \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  /tmp/wnn-c20261004-native-factors/scripts/export_factorized_control.py \
  /home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/runs/c20261003_w384_s3_f0/model.pt \
  --output /tmp/wnn-c20261004-native-factors/research/campaign_2026-10-04/E420_factorized/zero_update_w384_r128.pt \
  --rank 128
```

Expected factor shapes are `[1152,128]` for left and `[128,384]` for right on
`gru.weight_ih_l0`, `gru.weight_hh_l0`, `gru.weight_ih_l1`, and
`gru.weight_hh_l1`. The unmodified W384 parent remains a separate approximation
comparison. It is not the native factor parity reference.

## Native parity contract

`probe_factorized_native.py` rebuilds each E409 stream as raw 112 features plus
the 99 causal event features. It verifies batched and streaming event features
bitwise, then runs the complete 20,000-row sequence without resetting the
recurrent state. It compares each native build with Torch running the effective
dense matrices from the zero-update control. It checks reset/replay, portable
versus SIMD bit equality, v2 versus v3 bit equality, and byte-for-byte round-trip
of each saved factor array.

The predeclared representation gate is maximum absolute error at most `3e-4`
over both scored output columns on every row, and absolute pooled WP difference
at most `5e-6` on the E409 masks. Original-parent error and WP are descriptive
approximation context and do not set the gate. Storage reports
`total_streamed_weights` including the `factor_weights` subset; recurrent state,
scratch and import-only dense recurrent bytes remain separate fields.

Only after parity passes and a root quiet-window admission may a separate timing
task be specified. No public runtime conversion is planned from this probe.

## Bounded implementation checks

The v2 and v3 native libraries were built from the same source with the explicit
scorer flags captured by `native_build_command.txt`. The build used one CPU and
a 2 GiB memory cap, completed in 37.83 seconds, and peaked at 373,436 KiB RSS.
The build output records the native library hashes. Each library passed the
synthetic factor harness, native factor tests, and quantized-export regression
tests: 11 tests per build, with peak RSS below 737 MiB. Outputs and resource
records are in `v2_test_output.txt` and `v3_test_output.txt`; library hashes and
build settings are summarized in `native_build_manifest.json`. These checks do
not read the W384 checkpoint or E409 data.

`e420_zero_update_export_v1.draft.json` specifies the fixed epoch-3 W384,
rank-128, zero-update SVD control export. The export command is also rendered in
`export_command.draft.sh`. `e420_full20k_native_parity_v1.draft.json` specifies
the subsequent three-stream parity run against that represented dense control.
Both are drafts only; neither command has been launched. Their resource caps
are one CPU and 2 GiB RAM, with GPU use disabled.

## Source closure

Exact hashes are in `source_closure.sha256`. The closure includes the W384
checkpoint and its provenance, E409 manifest and data, the export and SVD
implementation (`scripts/export_factorized_control.py`,
`src/wnn/training_factorized.py`, `src/wnn/nn.py`), the event-feature and scorer
reference (`src/wnn/features.py`, `utils.py`,
`scripts/probe_factorized_native.py`), plus Cargo metadata and every native Rust
source used by the v2/v3 factor builds. The built libraries and output report
will be hashed separately after the root admits the export and parity run.
