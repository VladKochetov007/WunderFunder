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
in 3.18 seconds. This agent has not rerun them. The native source changes made
after the prior CPU lifecycle receipt remain untested because root's local
recovery checks have priority. The older bounded CPU attempt ended at its
120-second limit before the current batched-gate and producer-binding changes;
it is not evidence for these changes.

The pending validation sequence is: offline source/packet integrity and tiny
tests, actual GPU R4 completion and exact training-contract pin, root admission,
TPU source startup with cache-to-producer fixture equality, actual eight-device
FP32 parity and memory gate, then separately admitted BF16 gate and bounded
matched fits. No native result or speedup claim exists until artifacts from
these stages are complete and reviewed.
