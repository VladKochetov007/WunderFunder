# Native comparison implementation state

Source is an isolated successor of our own frozen TPU worker and generic JAX
GRU; earlier sealed closures remain unchanged. GPU reference identity is the
frozen R3 producer, and its successful shared artifact contract is still pending.
Provider metadata remains the supported TpuV5E8; runtime device kind is unknown.

The successor batches input-to-gate projections before each recurrent layer's
scan, retains detached carry and exact weighted global reductions, separately
times real batch32 and checks padded batch30, and records five synchronized warm
update samples per loss. The gated native BF16 profile checks finite updates,
FP32 masters/moments and persisted updated-state reload. Native BF16 fit requires
both hash-bound FP32 lifecycle evidence and its own full-shape profile receipt.
FP32 and BF16 production branches share initialization, masks, order and schedule.

CPU attempt R1 reached its120-second support limit before producing a completed
XML. This preceded the batched-gate core change and establishes no parity claim
for that change. The lock was released for GPU R3 dispatch. New tiny H8 tests
compare forward/gradient/AdamW against the original own core and verify BF16
compute with FP32 master/moment persistence; admission tests reject unadmitted
native precision and the obsolete producer. These checks are pending until R3
push and first status complete. Actual-device lifecycle/full-shape gate remains
mandatory before fitting, including the modified core hash.

The existing GPU R3 trainer already supports native FP16 via `--precision fp16`
with CUDA autocast, GradScaler, unscaled global gradients, clip and skip traces.
Its CUDA-event StageTimer synchronizes at report. Current R3 remains FP32 and is
not rewritten. An identified native GPU precision successor must retain exact
shared tensors/order/schedule and explicitly report scaler skips; source support
is not a completed native comparison. Root owns admission and allocation.
