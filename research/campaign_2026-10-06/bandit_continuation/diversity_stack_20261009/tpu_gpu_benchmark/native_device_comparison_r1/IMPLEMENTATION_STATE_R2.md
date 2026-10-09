# Native comparison implementation state R2

R2 is source preparation only. No TPU parity, full-shape profile, fit,
prediction export, device timing, or native precision comparison has run.
Provider admission and fit flags remain false; the completed GPU R4 training
contract hash is pending.

The TPU closure now pins the exact parsed-cache reader and bridge-test source
from GPU R4 commit `1bc0fbacf53cdc08a8c4664c6f66f38f8980092c`, checking module,
test, packet, and GPU kernel hashes while sealing. The packed TPU source
consumes the R4 producer identity. Before compiling, it opens the mounted
cache through the pinned reader, reconstructs the producer's ordered batch,
and checks exact feature, target, and training-mask equality against the
producer's hash-bound profile fixture.

The actual-data FP32 TPU gate now requires the exact producer OneCycle trace,
uses its first learning-rate/beta1 row, reports the discarded update's loss,
global gradient norm, and clip scale, saves its prediction fixture, and
records runtime device kind/count and synchronized host-plus-device timing.
The separate BF16 gate uses the same schedule row and compares loss, gradient
norm, clip scale, and weighted/raw prediction drift to the FP32 fixture while
checking FP32 master and optimizer state persistence.

The R4 reader owner reports five focused producer/cache bridge tests passing
in 3.18 seconds. This agent has not rerun that producer-specific suite. The
full native lifecycle suite and full-size H256 CPU gradients remain
unverified. The older bounded CPU attempt ended at its 120-second limit before
the current batched-gate and producer-binding changes; it is not evidence for
these changes.

The focused native CPU tests then passed 5/5 in 7.75 seconds under one CPU,
1.5 GB, and a 120-second runtime cap; the receipt is
`CPU_NATIVE_TINY_TESTS_R2.json`. They compare the H8 batched-input core with the
original own JAX core, exercise BF16 master/optimizer persistence for both
losses, and check admission guards. This is CPU-only unit evidence, not TPU
numerical parity.

The exact generated V2 Kaggle bootstrap and packaged runtime passed the
offline source validator; `PACKET_SEAL_R2.json` records packet/archive hashes
and the `fit_admitted=false` result. The embedded root admission still denies
push, fit, and BF16 and has no GPU training-contract hash.

The remaining sequence is: successful admitted GPU R4 execution and exact
training-contract pin, root admission, actual TPU startup with cache-to-producer
fixture equality, actual eight-device FP32 parity and memory gate, then a
separately admitted BF16 gate and bounded matched fits. No native result or
speedup claim exists until artifacts from these stages are complete and
reviewed.
