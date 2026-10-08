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

Offline evidence: `TPU_PACKET_OFFLINE_TESTS_R4.xml` (four passes, zero skips,
19.01 seconds) covers actual CPU producer
metadata/NPY byte accounting, the exact extracted bootstrap and both entrypoint
imports, pending admission refusal, artifact mutation rejection and exact first-batch
fixture cache/order/normalization identity (including three semantic mutations).
`ACTUAL_GPU_WRITER_CONSUMER_CHECK_R2.json` records the immutable GPU R2 writer's
real contract collector passing the TPU consumer with synthetic binary fixtures;
this is interface evidence, not actual-data or device parity. The R2 collector
adds the auxiliary fold0 receipt and source provenance fields; all seven consumed
shared-role semantics match R1. The actual R2 collector was executed again and
passed byte/schema validation plus first-batch cache/order/normalization checks.

Frozen producer source: `8924deec1eaf1a50cb31d4c42e042d24093b4be6`;
packet/source-asset commit: `0704ad3e1d44d1cc69b137b5c81aa5fe6faa174e`;
worker SHA: `8b59b258de30dbb8b51e6d7463b9ca6c4fdb47e1ac6e59f2772cbf9c085707c3`.
The sealed successor records these and the producer packet SHA in its source
manifest. These pins identify the prepared producer; its completed runtime
training contract still needs a root SHA pin before execution.

Corrected TPU candidate identities:

- Source manifest: `7eee31c549bcd57b9d9ccbd5c3bed882f37be18b748d57bf808ac8e2f53ca248`.
- Archive: `1f538427c02d69f3c4017e053c163883f69fb9f65fad71d6d1f7ccb9e457d74a`.
- Bootstrap: `178a85b5b099aeda3d8a98455b451fd8a6df9441de9b21467ff7795c638985d5`, 81,975 bytes.
- Seventeen source files; 3300-second worker and 3600-second provider envelope.

Preserved `sealed_tpu_candidate_r1` is historical and unadmitted. It targets GPU
v1 and predates the corrected actual producer profile-receipt hash key. Root's
independent exact-archive import review of that candidate is retained in main.
The corrected `sealed_tpu_candidate_r2` follows GPU R2 source freeze and requires
root review before dispatch. Completed GPU outputs and their contract SHA remain the concrete
dependency for TPU execution. No large cache arrays need laptop download.
