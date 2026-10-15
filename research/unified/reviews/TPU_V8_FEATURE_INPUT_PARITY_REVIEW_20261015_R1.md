# TPU V8 feature-input parity review R1

This read-only source audit follows the one-sequence native/PyTorch/TPU comparison after the native and PyTorch paths agreed but both differed from the TPU export. I read the bounded source/JSON receipts and inspected the frozen bridge and local data/feature builders. I did not import Torch, load NPZ payloads, read a Parquet row group, or run a model.

## Confirmed source path

- The producer exports a contiguous FP32 tensor built by `RawSequenceStore.read("valid", 254)` → `load_sequence_supervision(...)` → `event_features(raw)` → `np.concatenate((raw, event), axis=-1)`. Its SHA-256 covers `(1, 20,000, 211)` before chunking. The source is the V8 pinned bridge `e465` (`bridge.py` SHA `788e83eb...1990bb`) and its pinned shared `data.py`/`features.py` (`a93da840...bc7e00`, `284c119d...12bfeee`).
- The V8 producer and local `SequenceFile.read(254)` both select ordered `FEATURE_COLUMNS`, read one Parquet row group without threads, stack columns, cast to FP32, transpose and make the 112-wide raw array contiguous. The local `data.py` and `features.py` are byte-identical to the pinned source files. The mounted producer `valid.parquet` identity is SHA `2861ee30...418782`, 5,615,509,128 bytes, with footer SHA `4e0a6ad9...a7d93b`.
- The local R3 diagnostic does not replay the producer feature call. It concatenates `event_features_torch(raw)` instead of NumPy `event_features(raw)`. The NumPy builder computes `log1p` from integer recency counters and then casts to FP32; the Torch builder first casts the counters to FP32 and calls `torch.log1p`. This can produce different FP32 bits in the three recency channels even though the source definitions and intended values match. That is a code-level explanation for differing hashes, not yet an observed feature-difference measurement.

## Evidence and limit

The frozen [PyTorch reference receipt](../../campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/PYTORCH_REFERENCE_R3.json) reports local Torch-built input hash `45ba5620...efb4655` versus producer hash `68af663f...1d3f08`; its input-identity check is false. On the same EMA weights, native and PyTorch main-two outputs agree within max absolute `1.43e-6`, while PyTorch and TPU differ by max absolute `0.0077724` on the first two outputs. The [native compatibility receipt](../../campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/NATIVE_COMPATIBILITY_R2.json) independently reports the same scale of native/TPU difference.

The comparison therefore localizes the discrepancy to a confounded input/backend comparison, not yet to JAX arithmetic or the trained TPU weights. R3 does not assert the local row’s `seq_ix` against a producer field or hash the 112 raw block separately. Although the group index, dimensions, source-code hashes, and Parquet file identity are pinned, the R3 result itself does not demonstrate byte-identical raw inputs before feature construction.

## Decisive bounded follow-up

From the exact local group-254 raw array, assert `seq_ix == 267134787`, hash the contiguous FP32 `(20,000,112)` bytes, and build both 211-wide arrays from that same raw block: the pinned NumPy producer path and the Torch serving path. Compare per-feature bit mismatches/ULPs and hashes to the two recorded input hashes. If the producer-style NumPy hash matches `68af...`, replay the same frozen EMA on both local arrays to measure how much the feature-builder difference accounts for the output gap. Keep backend tolerance unchanged until inputs are matched; if NumPy and Torch paths fail to reconcile the hashes, investigate source bytes/group selection before analyzing recurrent arithmetic.

This is an input-path diagnosis only. It is not a model score, a completed training-recipe validation, or evidence that the TPU update is invalid. The one-update fitting-sequence result remains exposed plumbing evidence, and full multi-update TPU training should wait for the matched-input native compatibility check.
