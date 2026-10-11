# Mixed precision, Q5–Q7 and quantization-aware fine-tuning

User steering received 2026-10-11. Root owns planning; status is proposed
bounded experiments, not launched fits. These extend the complete-endpoint
policy in ADR0033 and preserve the current clean-fold/local/Kaggle branches.

## What we should optimize

Choose the complete delivered blend under ZIP, causal/reset and one-CPU runtime
constraints. Compare its official held-out score directly with the strongest
feasible Q8 endpoint on identical groups. Extra members, low correlation,
compression ratio, or quantization error alone cannot establish a better blend.
All folds are exposed; full-data outputs give fidelity evidence only. Do not
subtract a precision delta on one pair from a different recipe's fold score.

Current measured screen: the H390 pair loses about0.000071 WP under Q8 versus
FP32 and0.001801 under 15-level rowwise rounding versus Q8 on ten in-sample
complete sequences. These results do not close mixed-bit or larger-blend arms.
The selective-FP16 storage probe reduces a wide Q8 pair17.272→9.591MB without
changing the loader/backend. Actual larger endpoint grading remains required.

## Mixed precision is three different mechanisms

1. **Transport precision:** different on-disk dtypes/codes, restored before
   inference. Our current loader casts NPZ tensors to contiguous float32, then
   constructs Q8VNNI. Selective FP16 storage already works with that interface.
2. **Model/member precision:** retain an important member at Q8 and compress a
   complementary member more. The complete weighted mixture earns inclusion;
   different rounding of the same seed is not an independent training seed.
3. **Matrix/gate runtime precision:** genuinely different native compute paths
   within one model. This needs kernel/ABI/packing changes and real CPU timing;
   the current global precision selector does not establish this capability.

Keep normalization, input projection, biases, final head and recurrent state at
their audited representation initially. Screen recurrent ih/hh matrices and
head.0 separately, then GRU gate blocks. Do not presume which gate is sensitive
from its name. A small precision upgrade can matter more than another member
when it protects a recurrent error that propagates for20,000 causal rows.

## Q5, Q6 and Q7: mathematical and hardware distinctions

For a finite matrix row with maximum absolute value M, a symmetric b-bit grid
uses Q=2^(b−1)−1 and s=M/Q. Rounding to nearest gives q=clip(round(w/s),−Q,Q)
and restored weight w_hat=sq. For this no-clipped-outlier construction,
|w_hat−w|≤s/2. Zero rows use zero codes/scale. Our native dynamic Q8 uses the
255-level symmetric grid; the previous Q4 diagnostic used15levels.

| Bits | Signed levels | Maximum rounding error / M | Nominal packed weight bits |
|---|---:|---:|---:|
|4|15|1/14|4|
|5|31|1/30|5|
|6|63|1/62|6|
|7|127|1/126|7|
|8|255|1/254|8|

These bounds concern individual weights, not WP loss or long-sequence behavior.
Linearizing a recurrent transition gives
delta_h[t] ≈ J_h[t] delta_h[t−1] + J_W[t] delta_W + delta_activation[t].
Products of state Jacobians determine amplification/attenuation over time;
small weight RMSE need not imply small endpoint error. Measure long complete
sequences, target-specific WP, tail errors and reset replay.

Our `native/src/vnni.rs` packs byte weights in blocks shaped for unsigned-byte
activation × signed-byte weight dot products. Intel documents that VNNI byte
dot-product interface. It does not provide a dedicated Q5/Q6/Q7 operation for
this kernel. [Intel primary documentation](https://www.intel.com/content/www/us/en/developer/articles/technical/whats-new-in-llvm-for-4th-gen-intel-xeon-processor.html)

**Project inference:** Q5–Q7 can be efficient *storage*: pack once, unpack/load
once to the existing Q8 layout. The steady-state weights still occupy bytes, so
that path has no automatic runtime-cache or matvec-speed benefit. On-the-fly
packed inference saves fetched bits only if it retains packed storage; shifts,
masks/unpacking and a different layout then add work. Odd bit widths crossing
byte boundaries are engineering costs, not invalid precisions. Benchmark the
complete callback; do not infer speed from ZIP size or alignment alone. Q4's
nibble layout is simpler, but speed still needs a kernel and measurements.

The current FP32 NPZ representation of grid-rounded weights relies on archive
compression, not a real fixed-bit binary format. Record exact ZIP bytes. A new
packed-format loader must validate actual codes/scales, padding and finite
zero rows against real producer artifacts. Native dynamic Q8's preserved-code
import is currently missing; the existing q8r/q8x import cannot be substituted.

## Can fine-tuning recover the loss?

Yes, this is a credible hypothesis rather than an observed gain here. Ordinary
float fine-tuning followed by quantization does not train against quantization
error. Quantization-aware training (QAT) uses float master parameters and fake
quantization during the forward pass, with a surrogate gradient through the
rounding operation. Learned step-size quantization explicitly learns quantizer
steps alongside model weights. Its reported image-model results do not establish
transfer to our GRUs. [LSQ primary paper](https://arxiv.org/abs/1902.08153)

RNN-specific work treats gate/interlink quantization as an important part of
the method, supporting a recurrently faithful test rather than importing a
Transformer recipe unchanged. [RNN quantization paper](https://arxiv.org/abs/1611.10176)

For our candidate, model the **actual** deployment composition: storage rounding
followed by rowwise dynamic Q8 weights/activations, normalization folding, gate
nonlinearities and FP32 recurrent state. Weight-only fake Q6 is a first explicit
partial-fidelity arm, not exact Q8VNNI QAT. Learned scales are meaningful only if
the final consumer preserves them; today's packer recomputes row maxima, so a
learned clipping scale can be lost at export. Check the consumer before fitting.

Use our own audited checkpoint and fitting groups; any teachers/initializer/
propensity ancestry must exclude the evaluated groups. Keep the original target
treatment, true-label/KD alpha, optimizer, schedule and batch fixed. Compare QAT
against a matched ordinary fine-tuning control with the same update budget, plus
the unchanged checkpoint under PTQ. Use identical declared seeds. No large CPU
fits, teammate weights, evaluation-based calibration, or public arithmetic.

## Bounded priority queue

| Card | Hypothesis/test | Cost/owner | Falsifier or next gate |
|---|---|---|---|
|H-QMIX-01|FP16 transport + Q8 runtime permits teacher trio or student/trio without losing conditional quality.|Actor already building; root grades fixed complete endpoints.|Exact ZIP/runtime failure or measured endpoint worse than strongest admissible Q8; report scope, preserve materially different precision arms.|
|H-QMIX-02|Q7/Q6/Q5 rowwise grids, same Q8 kernel, can provide a better quality/ZIP tradeoff than uniform Q4.|No-fit first: Q7 and Q6 on the frozen teacher trio; Q5 only if extra capacity is useful. CPU1/1.2GB/600s per native panel, root admission.|No capacity or complete-score advantage over selective FP16/Q8; no runtime-speed claim.|
|H-QMIX-03|Allocate low bits to less sensitive recurrent/head matrices and protect the rest at Q8.|At most3 prespecified policies after single-block sensitivity observations; calibration on fitting groups only; no weight search on complement.|Complete endpoint loses its conditional value; a covariance/correlation story alone is insufficient.|
|H-QMIX-04|Short, causal QAT + own-teacher/true-label objective recovers useful low-bit endpoint quality.|Prepare GPU implementation only after a low-bit endpoint has useful capacity; matched PTQ/ordinary-FT/QAT control, one own seed initially, explicit bounded updates and actual-device profile.|No matched quantized-endpoint improvement or runtime/export mismatch; follow useful uncertain signals with a bounded next seed/fold, not a universal CI-zero kill.|

H-QMIX-02/03/04 are recorded proposals, not dispatched or GPU-queue entries.
Current full-data packaging and clean-F0 teacher/student comparisons retain
priority. QAT has no spending or fit admission in this note. Root will freeze
and review a concrete successor before any GPU execution.
