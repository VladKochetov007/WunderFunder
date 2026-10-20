# EMA successor local packet

This folder prepares the immediate second fixed arm, `fg19_child_ema_f0_s1001_fit_v1`. The local `REMOTE_SUBMISSION.py` body has now been rendered against the actual, accepted, and exactly stopped CONTROL receipts. Its identity is recorded in [EMA_FINAL_BINDING_RECEIPT_V1.json](EMA_FINAL_BINDING_RECEIPT_V1.json). The body is 163,040 bytes, SHA-256 `f83ac3120a2e445f1752fcd026123a0e8a21ba53594d27b398462a5236ffebd6`; its trainer argv SHA-256 is `a20b4ede3b501daa6cd39e847c212ca3d6215dde81e130a8ab09a48004beb449`.

This is local preparation only. No EMA source was staged, and no EMA unit was admitted or launched. The final binding receipt explicitly keeps admission readiness false pending independent review and the root-owned fresh gates.

## Frozen experiment

The EMA arm keeps the accepted parent epoch-3 initializer, inherited normalization, fold-0 split and fit membership, current homogeneous T3 cache, shared dictionary, seed 1001, seeded batch order, B16/C2000, six fixed epochs, and checkpoint epoch 5. It keeps CONTROL's WMSE target, mean-variance head 1, optimizer, schedule, batch semantics, and remaining trainer arguments. Only two argv values change from CONTROL: the run name and `--representation-mode base` to `--representation-mode ema`. No resume, profiling, max-batch, evaluation, or unrelated weighting flags are present.

The target is `0.1 * clip(label, -2, 2) + 0.9 * clip(cache, -2, 2)`. The homogeneous S3/S4/S5 mean has its `0.8125` factor applied once when the accepted cache is built; the fit consumes that cache at scale 1.0. It uses the original clipped-label-magnitude WMSE weights with canonical `main_valid` behavior. The accepted parent checkpoint and preprocessing are retained; the child does not initialize from CONTROL weights.

In canonical source commit `68ff682977b822f38351aac0f621146f5f13b440`, `representation_mode=ema` appends 256 columns to the inherited 211-column input: columns 0:128 are exact zero dictionary/sparse slots; columns 128:240 contain causal signed EMA of 112 raw input innovations; columns 240:256 are exact-zero padding. Half-life is 32 rows, row zero is zero, and history resets at each input sequence before recurrent chunking. This is input history, not parameter/model EMA. The resulting input width is 467, and the unchanged WMSE recipe leaves the fixed trailing log-variance outputs unused.

The local recipe preserves the accepted all-three shape-profile receipts and fixed one-owned-GPU/aggregate-two-CPU allocation. Historical 150-fit/resource snapshots in the design document do not define this current allocation.

## Actual CONTROL predecessor receipts

The body embeds the exact bytes from:

- `records/child_control_fit_v1/TERMINAL_ACCEPTANCE.json` — 11,883 bytes, SHA-256 `92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2`.
- `records/child_control_fit_v1/exact_stop/STDOUT.raw.json` — 3,089 bytes, SHA-256 `600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012`.

The acceptance is schema `fg19-child-control-fit-terminal-acceptance.v1`, status `ACCEPTED`, invocation `e66e0d0703314e4fb5d47dcdbfcaf202`, with 44,760 successful updates, 4,476 completed batches, zero skipped or nonfinite updates, `next_epoch=6`, and selected epoch 5. It records full source verification, a finite model/optimizer, and selected model/training-state tensor identity. The exact-stop receipt binds that acceptance and invocation and records the exited unit stopped and absent afterward. The local binder's receipt validator and embedded-byte checks passed.

## Frozen snapshots and current status

`EMA_RECIPE_BINDING_V1.json` and `EMA_LOCAL_CHECKS_V1.json` remain unchanged historical snapshots from before the actual CONTROL receipts existed. Their unbound/body-absent status describes that earlier check and is not the current body status. Use `EMA_FINAL_BINDING_RECEIPT_V1.json` for the current local binding and body identity. It records the recipe, binder, receipt, body, and argv hashes, and explicitly records `source_staged=false`, `unit_admitted=false`, `successor_launched=false`, and `admission_ready=false`.

The rendered body is prepared for independent source review. Root owns the later actual admission and launch path; this packet performs no remote action. Do not overwrite or rerender the frozen body.
