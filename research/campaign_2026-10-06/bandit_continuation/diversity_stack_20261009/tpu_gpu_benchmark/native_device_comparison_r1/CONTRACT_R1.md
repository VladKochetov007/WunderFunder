# Native GPU/TPU comparison

Compare the same independently initialized GRU training logic using native
Torch/cuDNN/CUDA on T4 and JAX/XLA on the eight-device TPU allocation. This
measures the implemented hardware **and software** stacks. No speed multiplier
or time-to-score result exists yet. Root retains provider and fit admission.

The matched FP32 run is the correctness anchor: raw112 + causal event99 inputs,
GRU256×2, linear projection, GELU, 350 complete training sequences, 468 exposed
evaluation sequences, the two epoch −1 normalization batches, shared fresh
initial tensors, AdamW, OneCycle LR/beta1 and 220 attempted updates. WMSE and
weighted Huber δ1 use targets clipped ±2 and their absolute magnitude as weights;
fit mask is row≥99. Evaluation remains global_wp_clip2_v1 with the fixed scored
mask and clipping ±2. GPU R3's successful producer contract will supply exact
initialization, normalization, order, schedule and actual pre-update fixture.

Effective batch is 32 sequences and truncated-BPTT chunk length is 2000. Each
chunk computes one global weighted numerator divided by the global weight sum,
then clips the complete global gradient norm at 1 and applies one AdamW update.
TPU shards batch32 as four sequences per chip, sums local numerator gradients
over the global denominator, and updates replicated parameters identically.
The final batch has 30 real sequences: TPU pads two with zero weights; GPU uses
30. Carry pre-update recurrent state detached across chunks; reset between
sequence batches. Padding must not enter normalization, loss or evaluation.
Do not average replica means or change update count/schedule for the remainder.

Native precision is a separate, admitted arm after actual FP32 forward,
gradient, AdamW-update, reload, reset and prefix-causality checks pass. T4 uses
native FP16 autocast with FP32 master parameters/AdamW accumulators and recorded
GradScaler scale/skip behavior; TPU uses BF16 compute with FP32 masters and
accumulators. These are unequal dtypes. T4 BF16 is not a native comparison arm.
Count all attempts, completed updates and skips; retain LR/beta1/global unscaled
gradient norm/clipping traces. Compare loss and prediction drift as well as time.

Keep effective batch32 in this successor. Device-specific layout, compilation,
buffer reuse and dispatch are permitted. Microbatch accumulation may be added
only as an identified successor with verified global weighted reduction,
recurrent carry, full-gradient clipping and one update per original chunk.
Changing effective batch or schedule is a scientific arm, not a throughput gain.

Separate setup/input verification, cold load/compilation, synchronized warm
updates, full fit, prediction/export and full worker wall time. GPU timing uses
CUDA events or synchronization around the measured region; TPU timing blocks
the whole result tree. Enqueue latency is never device execution time. Discard
warmup updates and restore exact initialization before fitting. Report device
model/count and software versions. Measure single-T4 arm throughput, observed
two-T4 concurrent pair aggregate, and eight-chip TPU pair wall time separately.
The two T4s run disjoint loss fits concurrently; the eight-device TPU allocation
runs one fit over eight cores and the pair sequentially. The selected supported
provider metadata is TpuV5E8; its actual device kind and count remain unverified
until the runtime receipt. Single-fit latency and whole-allocation pair
throughput are separate measures, never a chip-to-chip speed ratio. Record
provider-reported accelerator quota consumption separately from worker wall and
device time, including whether quota accounting charges the session or devices.

Fixed-work throughput uses identical attempted sequence/chunk work and useful
completed updates, with padding reported separately. Time-to-comparable-score
requires actual saved predictions evaluated on the same 468 IDs, checkpoint
timestamps and the official metric. Report the observed endpoint/epoch score
gap and uncertainty; do not invent an interpolated time-to-target from one
endpoint. Preserve poorer individual cells because their blend contribution
may differ. Save weights, normalization, predictions/membership sidecars,
resource peaks, synchronized timings and diagnostics for every completed arm.

## Reuse and necessary gaps

Keep the previous sealed TPU R1/R2 closures immutable. Reuse their own generic
JAX reset-after GRU, matched objective/reduction, eight-device worker, strict
producer validator, lifecycle/device gate, feature adapter and retained exports
in a newly identified source closure. Rebind only to successful GPU-v3 producer:
source cbf77aa485f072cf92bc123b240ce16c78a5eb30, packet
ea31663b696b857c31f3fe1eb2a0ff5dd0e46ad4950423fe4fbd189c0cb91068,
kernel vladkochetov07/wnn-loss-pair-gpu-20261009-v3. Completed contract SHA is
pending; metadata is not a completed-device receipt.

The current GPU reference already uses native cuDNN; the TPU core uses compiled
lax.scan and eight-device pmap. Necessary successor changes are the GPU-v3
binding, explicit native-precision branch gated by FP32 evidence, repeated
synchronized warm full-shape timing, measurement provenance and comparison
outputs. Our generic GRU also computes the input-to-gate matrix multiply inside
each recurrent step. The isolated `rnn_native.py` batches that independent
projection over the chunk before scanning only hidden-to-hidden gates. This
preserves the reset-after equations and causal row dependencies while adding
a chunk-sized gate buffer. Tiny forward/gradient/AdamW comparisons and actual
FP32/full-shape memory checks must pass; performance is unmeasured. Investigate
further dispatch/buffer reuse from actual timing. No new runner framework, data reparse, local GPU fit
or provider action is authorized by this contract.
