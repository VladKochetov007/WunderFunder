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
- The [conversion CLI](../../../../scripts/convert_tl3_student_checkpoint.py) reuses the existing canonical shape/strict-load helpers; the [archive helper](../../../../scripts/tl3_source_archive.py) binds the exact original ZIP and source receipt. The converter is integrated and statically reviewed; actual conversion remains pending.
- No matching TL3 Torch checkpoint is present in the local run artifacts; the only available initializer is the original NPZ member.
- The original wrapper dynamically quantizes its three members through `wnn_cpp`; FP32 PyTorch and native outputs are not expected to match exactly. Separate exact tensor/state round-trip checks from native-versus-FP32 diagnostic deltas.
- The full-data history cache has complete target coverage and coefficient sum 0.8125, but its `history` variant is not accepted by the pinned clean-fold NLL trainer. Do not relabel it as a clean-fold `source` cache.

No converter, cache validator, fit, prediction export, package, grader, or original ZIP was changed.

## Exact source preservation

The [source archive content receipt](SOURCE_ARCHIVE_CONTENTS_R1.json) records
streaming SHA256 hashes, byte sizes and CRCs for every original ZIP entry.
Root verified the archive hash and filled exact member hashes in the plan.
This binds eventual conversion and preservation of the other two models,
wrapper and libraries without interpreting tensor values or running a model.
A candidate remains NOT_ADMITTED; these are source bytes, not quality evidence.

## Converter readiness and execution boundary

Integrated actor revisions through1a2c6259 provide a create-only CPU converter.
Ruff format/check and Torch-free CLI help pass. Independent Luna/max static
review passes after fixing a missing filename constant and binding the actual
imported `wnn.nn` module path/hash. Partial failed-attempt outputs are retained.
Unknown training membership/ancestry and `training_admission:false` survive
in the converted artifact; no clean-fold initializer or final-fit admission
is claimed by conversion.

After the current GPU fit/export/grade chain drains, root can execute conversion
under the existing delivery authority with an explicit1.2GBCPU-only/no-swap
bound and source `PYTHONPATH` fixed to this checkout's `src`. Use a new physical
scratch directory, for example:

```sh
env CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD/src" \
  WNN_MEM_MAX=1200000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 \
  scripts/limited .venv/bin/python scripts/convert_tl3_student_checkpoint.py \
  --output-dir /home/vlad/development/wunder/research_tmp/tl3-student-conversion-r1
```

The converter intrinsically verifies source/member hashes, strictFP32 state,
positive scales, exact tensor round-trip and code identity. Add one bounded
fixed-input source/checkpoint CPU output comparison before accepting conversion.
Native dynamic-int8 deltas, exact serving quality/runtime and the separate
production cache/membership adapter remain outstanding. No conversion or model
execution has occurred at this recording.

## Actual CPU conversion and transport acceptance

[Conversion receipt](CONVERSION_RECEIPT_R1.json) binds all source archive entries, the exact imported canonical model source and the3,912,179-byte checkpoint SHA4e91c705b43d9dfcf279592089162d4c7147188491d58ca13cf90dd4cecbfb2b. Payloads stay at `/home/vlad/development/wunder/research_tmp/tl3-student-conversion-r1/`. [Forward acceptance](FORWARD_TRANSPORT_RECEIPT_R1.json) passes source-NPZ versus checkpoint-loaded output/state bitwise equality for seeded CPUFP32 synthetic input2×257×211, full and chunks99/101/57, and reset/replay equality. Chunk/full diagnostic max difference0.0. Both processes used one CPU,1.2GBRAMhard,zero task swap,CUDAhidden. This establishes transport, not original quantized-native parity, quality, clean ancestry or training admission. No production fit or candidate ZIP exists.
