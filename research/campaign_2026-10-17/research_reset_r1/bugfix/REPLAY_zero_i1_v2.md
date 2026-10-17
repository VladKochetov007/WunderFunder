# Zero-i1 CPU replay

This is a bounded inference diagnostic using the deployed predictor restore function and the existing `dftt_zi1_s3/model_ep1.pt` checkpoint. No optimizer or fit step ran.

- Checkpoint SHA-256 before/after: `82b7c93aa9cdfbad03cf9c62b419727f0ff537ddc42ad1a88c765d01bd648c68` / `82b7c93aa9cdfbad03cf9c62b419727f0ff537ddc42ad1a88c765d01bd648c68`; unchanged: **True**.
- Selected 4 of the deterministic strict425 panel before model scores: `[3, 6, 14, 15]`.
- CPU threads: intra-op 1, inter-op 1; wall time 82.1s.
- Manual expected ZERO_COLS equals patched restore bitwise: **True**; expected columns are `52..103`.
- Metric: `global_wp_clip2_v1` on the same selected groups and canonical labels/masks.
- Bad behavior (`ZERO_COLS=None`): WP 0.06289742 (t0 0.08175472, t1 0.04404011).
- Patched restore: WP 0.72994782 (t0 0.74194941, t1 0.71794622).
- Largest bad-vs-fixed prediction absolute difference: 76.069533.

This four-group slice is diagnostic only. It is not a full strict425 ranking and does not establish a model-quality gain.

See `REPLAY_zero_i1_v2.json` for checkpoint/source/data hashes, exact group membership, per-group comparisons, and timing.
