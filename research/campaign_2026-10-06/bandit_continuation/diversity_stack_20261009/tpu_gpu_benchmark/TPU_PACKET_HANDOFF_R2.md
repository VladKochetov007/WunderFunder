# Matched FP32 loss-pair TPU handoff

Owner: Sol diversity delivery. Provider launch and spending authority: root.

The packet consumes the completed GPU producer's shared initialization, normalization,
350-sequence epoch orders, 220 pre-update OneCycle LR/beta1 rows and actual first
B32/C2000 fixture. The current intended GPU producer mount is
`vladkochetov07/wnn-loss-pair-gpu-20261009-v2`; the parsed-cache mount is
`vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1`.

`seal_tpu_packet.py` freezes the source closure and creates the exact embedded
bootstrap, metadata and pending admission record. Pending records prohibit provider
launch and fitting. A root record must bind the final source manifest, completed
GPU training-contract SHA, cache identity and both kernel IDs. Reseal into a new
directory when authority or source changes; preserve existing candidates.

Runtime order is fail closed: exact source validation; admission validation; nested
mount discovery; shared artifact byte/size checks; streamed cache validation with a
new mount-specific receipt; normalization/init/order/schedule checks; actual-device
forward, gradient, AdamW, reload/reset and prefix-causality lifecycle checks; both
losses' discarded full-shape cold/warm updates and padded-replica reduction check;
then a separately authorized matched FP32 pair only if the finite runtime permits.
Native BF16 is a separate future arm. A runtime deferral is execution evidence,
not a scientific rejection of either loss.

The profile records memory and synchronized timing. The pair estimate includes
warm updates, measured cold-minus-warm compile reserve and host/export reserve;
one warm step remains an uncertain estimate. Root may allocate a successor after
reviewing actual device evidence. Report implemented Torch/CuDNN versus JAX
throughput at the matched recipe, one T4 versus the eight-chip TPU pod and the
measured two-T4 aggregate separately. No hardware multiplier has been measured yet.

Offline evidence: `TPU_PACKET_OFFLINE_TESTS_R3.xml` (four passes) covers actual CPU producer
metadata/NPY byte accounting, the exact extracted bootstrap and both entrypoint
imports, pending admission refusal, artifact mutation rejection and exact first-batch
fixture cache/order/normalization identity (including three semantic mutations).
`ACTUAL_GPU_WRITER_CONSUMER_CHECK_R1.json` records the immutable GPU R1 writer's
real contract collector passing the TPU consumer with synthetic binary fixtures;
this is interface evidence, not actual-data or device parity. Carry to GPU R2 only
after checking its immutable collector delta and full module hash.

Preserved `sealed_tpu_candidate_r1` is historical and unadmitted. It targets GPU
v1 and predates the corrected actual producer profile-receipt hash key. Root's
independent exact-archive import review of that candidate is retained in main.
The corrected successor must be sealed after GPU R2 source freeze and reviewed
before dispatch. Completed GPU outputs and their contract SHA remain the concrete
dependency for TPU execution. No large cache arrays need laptop download.
