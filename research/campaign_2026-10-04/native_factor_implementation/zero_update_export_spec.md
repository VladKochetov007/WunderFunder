# H-070 zero-update export and parity proposal

This is a preparation specification only. It does not create an export or claim a
scientific result. E-420 remains implementation and parity work; no training,
scoring, or timing has started.

## Source and transformation

- Source checkpoint: `/home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/runs/c20261003_w384_s3_f0/model.pt`
- Source SHA-256: `b5181f3aa948765f22a8c3f23f6e293ba715b52ffb91eb6eac02d6b7e30a8b8d`
- Source is the fixed epoch-3, seed-3, fold-0 W384/two-layer GRU parent. Its
  recorded training precision was FP16; export and inference tensors are FP32.
- Apply `scripts/export_factorized_control.py` with rank 128 to the checkpoint
  without optimizer steps. Preserve every non-recurrent tensor and the source
  checkpoint. The four exact saved factor pairs must be
  `gru.weight_{ih,hh}_l{0,1}.left: [1152,128]` and
  `gru.weight_{ih,hh}_l{0,1}.right: [128,384]`, in FP32.
- Keep both each saved `left`/`right` pair and its FP32 product `left @ right`
  in the export metadata/checkpoint. Record each source matrix hash and
  truncation residual. The product is the independent dense reference; original
  W384 weights are a separate approximation context only.

## Required parity before any timing

1. Bind the source checkpoint, export script and native source hashes before
   preparing the output. Refuse overwrite; record the zero-update declaration
   and confirm the source checkpoint hash above.
2. On a full 20,000-row real sequence, compare `factor_f32` native inference
   against a dense model loaded with the exact saved FP32 `left @ right`
   matrices. Preserve recurrent carry across the full sequence. Also compare
   reset/replay output and the portable fixed-order kernel. Record max absolute
   and RMS errors, bitwise portable equality, and exact byte round-trip of all
   eight factor arrays.
3. Run the same frozen inputs and tensors on the v2 and v3 native builds and
   require repeatable predictions. Report the ordinary W384 parent comparison
   separately as approximation context; do not use it as the factor parity
   reference.
4. Report storage fields with their accounting relation explicit:
   `total_streamed_weights` includes `factor_weights`, which is a labeled subset;
   recurrent state, scratch, and import-only dense recurrent bytes are separate.

Only after all full-sequence parity gates pass may a separately frozen timing
phase compare factorized native inference with dense native inference. Timing
must use the same rows, process affinity, build flags, warm-up, reset policy and
trial order. No score or runtime improvement follows from parameter counts or
the zero-update export alone.
