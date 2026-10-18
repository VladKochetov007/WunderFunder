# F1 wide epoch-2 checkpoint collector (preparation)

This small collector covers one generation only: the wide F1/seed1001 continuation checkpoint at epoch index 2 for recovery attempt `computer-vision-homework-r01-wide-f1s1001-ep25-v2`. It imports the recovery binder from `../checkpoints/f1_checkpoint_common.py` only after checking its exact SHA-256. The binder enforces the recovery-plan hash and the single allowed attempt-ID substitution; this collector also binds the unchanged base F1 plan contract and the exact verified V2 wide stage1 receipt/state at `B/runtime/receipts/RANK01_WIDE_F1_S1001_STAGE1_VERIFICATION.json`.

It copies only `model_ep2.pt`, `provenance.json`, and `training_state.pt` to `checkpoint_snapshots/rr18_r01_wide_f1_s1001_e6/epoch2`. The create-only collection receipt is written to `B/runtime/receipts/RANK01_WIDE_F1_S1001_EP2_COLLECTION.json`. Before and after the copy it checks source file inode, size, modification time and SHA-256; it also checks the copied model hash against producer provenance. The producer counters must be 44,790 optimizer attempts/completions and 4,479 batches after three completed epochs, with no skipped updates or nonfinite gradient steps. The provenance resume pointer must be the stage1 state SHA `8cb3dde79550355303cbede0d0953864046b7594c16af648423d4a3c9af92255`, `next_epoch=2`, and the exact wide stage1 training-state path. It does not load Torch or inspect checkpoint tensors.

No remote collection has been run with this helper. The local tests use the actual V2 stage1 verification receipt plus synthetic producer-shaped counters and file generations. A successful collection would establish a stable byte copy and relevant metadata only; tensor validity, the actual `training_state.pt` payload, measured resume parity, quality, and grading remain unchecked.

The intended invocation is only after independent review, fresh headroom checks, and an outer systemd admission with 25% CPU, 512 MB memory, no swap, and a two-minute runtime cap:

```text
python research/campaign_2026-10-18/priority25_execution_r4/f1_recovery_r1/intermediate/collect_f1_wide_epoch2.py
```

See `PREPARATION_MANIFEST.json` for file/input hashes and local validation.
