# Conditional TL3 student member adaptation

**Status: `NOT_ADMITTED`.** This packet describes a production route to adapt the existing plain TL3 student after the fold-0 and any admitted fold-1 WMSE/NLL comparisons are graded. It does not convert weights, train, export, or create a submission archive. The metadata-only plan is in [TL3_MEMBER_ADAPTATION_PLAN_R1.json](TL3_MEMBER_ADAPTATION_PLAN_R1.json).

## Proposed endpoint

Replace only TL3's `weights_s5.npz` member (`distilled_s5@ep1`, coefficient 0.34) with a final all-data NLL-adapted version of that same student. Keep `weights_tl5.npz` and `weights_tl6.npz` at 0.33 each. Keep the original `solution.py` and all three `wnn_cpp` binaries. This preserves three model calls per row and the existing support for the tied members' 263-wide input projection; the Rust native frontend does not support that layout.

The selected plain member has a metadata-compatible canonical `RNNModel`: 211 inputs, H256, two GRU layers, one hidden readout layer, two outputs, and the 16 state entries including `mean` and `std`. Convert it only after strict shape/key/state round-trip checks. The source archive's teacher, initializer, and fitting membership remain unknown; its description calls the parents all-data, which is not verification. Preserve that ancestry as unknown and use the derived member for production only.

## Conditional training path

If the F0/F1 evidence supports NLL and the controller admits a production fit, initialize from the converted two-output student, append two zero-initialized logvariance rows (2→4), and train one full-data production student fit. The family and auxiliary objectives remain off. Carry forward the judged F0/F1 NLL schedule: strict FP32, chunk-normalized wMSE target path with the Gaussian NLL mean/variance objective, alpha 0.9, cache scale 1.0, fixed CP1, and no fit-time evaluation. Use the existing `.8125` full-data history target cache only after a production-only cache validator confirms complete membership and ancestry. The current NLL trainer rejects its `history` cache variant and requires clean-fold cache validation, so no runnable production argv is frozen here.

After fitting, export the first two output rows as the serving member. Keep the other two TL3 weights, their coefficients, the wrapper, and the native libraries byte-identical. A new archive must have its own hash and exact three-member manifest; the original archive remains unchanged. Since the TL3 source ancestry is unknown and the adaptation uses all supplied labels, its local scores cannot be called clean-fold validation.

## Source basis and limits

- The original ZIP hash is `d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`; its member map and wrapper were inspected without loading NPZ arrays or importing Torch.
- The existing `team_member_export_lib` provides canonical shape inference and strict loading, but its package allowlist does not include TL3. The create-only [TL3 conversion CLI](../../../../scripts/convert_tl3_student_checkpoint.py) pins the archive and every entry hash, and writes only under an explicit new directory on physical scratch. It has not been run.
- No matching TL3 Torch checkpoint is present in the local run artifacts; the only available initializer is the original NPZ member.
- The original wrapper dynamically quantizes its three members through `wnn_cpp`; FP32 PyTorch and native outputs are not expected to match exactly. Separate exact tensor/state round-trip checks from native-versus-FP32 diagnostic deltas.
- The full-data history cache has complete target coverage and coefficient sum 0.8125, but its `history` variant is not accepted by the pinned clean-fold NLL trainer. Do not relabel it as a clean-fold `source` cache.

The converter's runtime checks cover strict loading, finiteness, positive normalization scales, and exact checkpoint tensor round-trip. It does not run forward or native comparisons; native-versus-FP32 tolerance, package ABI/runtime and prediction quality remain unmeasured. Its output keeps unknown ancestry and `training_admission: false`; it does not add a production validator. No conversion, fit, prediction export, package, grader, or original ZIP was changed.
