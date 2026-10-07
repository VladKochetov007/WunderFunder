# TPU utilization: primary-source reading and Challenge2 application

Read2026-10-07. This is a source-grounded research brief, not measured TPU performance.
E463v3 is queued; its frozen code is unchanged. TPU fits require real-device parity,
full-shape gradients/reload and HBM profiling before production. Localcaps stay6CPU,
20GBRAM,5GBVRAM; cloudusesseparateactualheadroom. Independentlytrainallourweights.

## Findings from our implementation

`research/campaign_2026-10-04/staging/jax_tpu/src/wnn_jax/rnn.py` already uses
`lax.scan`, but `_gru_cell` computes both input and hidden projection inside that scan.
The input projection is independent of previous hidden state; move it outside the scan
per layer. Our existing JAX bridge preserves PyTorch reset-after GRU semantics, which
must survive any Flax/XLA replacement. Do not replace it with an arbitrary GRU API and
assume identical gates, bias ordering or initialization.

`cloud_preflight/tpu_probe_v3.py` uses globalbatch16/localbatch2 across eight replicas,
with duplicated synthetic inputs and averaged gradients. This is a hardware/numerical
diagnostic, not a scientifically trained model or an efficient production allocation.
Replicated identical data does not establish real-data throughput or sharded loss parity.

## Source map and components to apply

| Primary source / code example | Transferable component | Application / limitation |
|---|---|---|
| [Google TPU performance guide](https://docs.cloud.google.com/tpu/docs/performance-guide) | Profile inputs, matrix padding and per-device batch; tune dimensions to hardware | Localbatch2 is a candidate utilization bottleneck. Profile larger perchip batches with full backward HBM, not just allocate most memory. Do not apply a universal1024batch rule blindly to long sequences. |
| [JAX scan documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) | Rolled scan lowers to one loop operation and controls graph size; tune partial unrolling | Existing scan already avoids huge Python-unrolled graphs. Test unroll1/2/4/8 after hoisting input gates; compilation can grow, recurrence remains sequential. |
| [Flax recurrent source](https://github.com/google/flax/blob/main/flax/linen/recurrent.py) | Runnable RNN/scan layout examples, fixed carry and lengths | Use as API/layout reference, preserve our Torch reset-after equation and exact carried state; changing cell implementation is a separate arm. |
| [Flax LM1B training example](https://github.com/google/flax/blob/main/examples/lm1b/train.py) | Actual compiled training-loop, replicated state and input delivery patterns | Adapt fixed-shape batches, bounded prefetch, checkpointing and step profiling. This example is not our loss/causal semantics. |
| [JAX persistent compilation cache](https://docs.jax.dev/en/latest/501/compilation-cache.html) | Reuse compiled programs rather than recompiling repeated shapes | Cohorts with same architecture/shape but new seeds can amortize compilation. Cache compatibility depends on environment/HLO; only use trusted own cache, report cold/warm compile times. |
| [JAX buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html) | Reuse dead input buffers in compiled update | Donate parameter/optimizer/carry state when caller no longer needs it. Never reuse donated arrays; do not donate inputs still shared with another seed or grader. |
| [JAX checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html) | Trade recomputation for activation memory | Test rematerialization of recurrent blocks when gradients limit HBM. Keep TBPTT boundaries and carried states unchanged. Benchmark total stepcost, not memory alone. |
| [Google BF16 explanation](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) | Low-precision matrix multiplication with higher-precision accumulation | BF16 activations/matmuls, FP32 masterweights/Adam moments/loss sums. This changes numerics; test gradients, fullshape update/reload and official metric before promotion. |
| [JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html) | Host timing may measure enqueue rather than completed computation | Synchronize timed warmup/endpoint with block_until_ready; separate compile, transfer, compute and export. Avoid Python scalar conversion every step. |
| [JAX Pallas TPU matmul example](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html) / [pipelining example](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html) | Tiled matrix kernels, fused epilogues and overlapped memory transfer | Candidate for measured input-gate/activation bottlenecks after XLA baseline. A generic Pallas matmul is not a fused GRU backward kernel; include gradients and compile cost. |

[Google v5e specifications](https://docs.cloud.google.com/tpu/docs/v5e) list16GB HBM
per chip. Eight chips do not make128GB automatically available to each replica. Verify
actual Kaggle topology/HBM instead of assuming specification memory is free. Keep20%
HBM headroom including temporary compiler/backward buffers. Budget host pinned buffers separately in host RAM.

## Original illustrative input-hoisting example

This is one-layer GRU pseudocode using JAX APIs, not a replacement trainer or tested
TPU kernel. Localenvironment has noJAX; no execution/parity claim is made. It keeps
PyTorch's gate order(reset,update,new) and reset-after candidate equation. Matmul
reassociation can change floating point rounding even though the algebra matches.

```python
import jax
import jax.numpy as jnp
from jax import lax


def gru_chunk(x, h0, w_in, w_rec, b_in, b_rec):
    # x:[B,T,I], w_in:[3H,I], w_rec:[3H,H].
    input_gates = x @ w_in.T + b_in

    def tick(h, gates):
        ir, iz, inn = jnp.split(gates, 3, axis=-1)
        hr, hz, hn = jnp.split(h @ w_rec.T + b_rec, 3, axis=-1)
        reset = jax.nn.sigmoid(ir + hr)
        update = jax.nn.sigmoid(iz + hz)
        candidate = jnp.tanh(inn + reset * hn)
        h_next = (1.0 - update) * candidate + update * h
        return h_next, h_next

    h_final, rows = lax.scan(
        tick, h0, jnp.swapaxes(input_gates, 0, 1), unroll=2
    )
    return jnp.swapaxes(rows, 0, 1), h_final

forward = jax.jit(gru_chunk)
```

Causality is preserved: each output depends only on current/past rows. The time-independent
input projection is larger and potentially more efficient; storing all three gates adds
activation memory. Compare against current bridge with independent forward/state/gradient
and oneAdamW-step checks, then measure TPU compilecost, steady throughput and HBM. No
claimed speedup until actual-device evidence. Never reset hidden state at chunk boundary;
reset only at sequenceboundary and detach at the frozen TBPTT/update boundaries.

## Allocation options for our competition

1. **Matched GRU execution arm:** hoist input gates, retain batch16/currentloss/seed/
   update budget, tune scan unroll. This is the closest backendoptimization test.
2. **Independent model-per-chip arm:** train distinct own seeds or mechanisms on chips,
   each with the original batch16, rather than shard one16batch into8tinybatches. Keep
   params, Adam state, RNG keys and outputs separate; no cross-model gradient average.
   Multiple chips per model may still win; compare completed independent fits per hour.
3. **Larger-batch scaling arm:** profile full-shape perchip batch8→16→32→64 only while
   memory allows; freeze changed optimizer/update/row-exposure/LR rules as new science.
   Do not call larger globalbatch a faithful same-recipe speedup.
4. **TPU-shaped teacher arm:** independently train wider recurrent, causalTCN or
   chunk-attention/Transformer teachers whose dense computations can use matrixhardware;
   compress complementary predictions/representations into a compliant deterministic
   student. This is our newarchitecture exploration, not a claim Maxim usedTransformer.
5. **Custom kernel arm:** after profiling proves a specific fusion/dispatch bottleneck,
   evaluate Pallas gate/activation kernels with numerical and backward checks. No blanket
   exclusion, but initialscanning/IO improvements are cheaper discriminating tests.

These arms get null public-gain priors. Efficiency can enable more exploration, larger
teachers or independentseed cohorts; it does not itself establish a higher publicscore.

## Critical real-data sharding check

Our training loss uses chunk-normalized weighted sums. If replicas see different label
mass, averaging their already normalized losses is generally not the original global
chunk loss. Duplicated diagnostic data hides this mismatch. In production reduce the
weighted numerators and denominators globally, or sum numerator gradients and divide
by the global fixed-label weightmass. Test with deliberately unequal shard masses.
Keep official evaluator clipping independent of this training objective.

Independent review correction: the primary WMSE oracle below covers only one term.
`wnn_jax.bridge.combo_sbd_loss` also has EMA consistency, three structural CE heads
and next-row MSE+BCE, with distinct weight masses. Preserve each component's global
numerator/denominator (primary: absolute clipped label times main weight; consistency
and each structural head: main weight; next-row losses: next weight). Compare full
distributed gradients and an AdamW update against a concatenated unsharded oracle,
including unequal and zero-mass shards. Global primary loss parity alone is insufficient.

## First bounded device profile after E463 lands

Freeze input/weightancestry and profile identity once; save real-deviceinventory. Compare
currentGRU versus input-hoistedGRU at same shapes, FP32 first then separateBF16. Report
compiletime, cold/warm cache, 20completed updates/synchronized wall, inputwait, HBMpeak,
finitegradient/reload checks and checkpoint/export roundtrip. Then allocate one bounded
model-per-chip versus data-sharded cohort, measuring completed jobs per allocatedhour.
Do not silently turn the queued diagnostic into a fit or spend a new TPU allocation
before realdevicechecks. Preserve its frozenbytes and schedulerwait evidence.

## Additional architecture arm: parallel linear recurrence

[Resurrecting Recurrent Neural Networks for Long Sequences](https://arxiv.org/abs/2303.06349)
provides a useful smallcomponent: linear/diagonal recurrent state enables paralleltraining
while retaining recurrentinference. [JAX associative_scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html)
implements associativeprefix operations. For `h[t]=a[t]*h[t-1]+b[t]`, compose two diagonal
affine maps as `(a2*a1, a2*b1+b2)`. This gives causal prefixstates without ordinary
nonlinearGRU sequential dependence. A compact teacher/featurebranch can combine these
states with nonlinear readouts or residual recurrentfeatures, then distill into ourstudent.

This is a newarchitecture/representation arm, not a drop-in exactGRU acceleration.
Complex-valuedLRU states and initialization choices need explicit numerical/serving
semantics. Test diagonal recurrence versus sequential reference, causal prefixes,
gradients and chunk-state reset; benchmark actualTPU forward+backward+memory and a
matched complete student/ensemble endpoint. Preserve earlier negativeSSM evidence,
but this TPU-shaped teacher interaction is not closed by unrelated small-model results.

For shardedloss, a concrete oracle is numerators[1,9], weightmasses[1,99]: mean local
normalizedlosses≈0.54545 whereas globalweightedloss=0.1. Production must preserve the
latter when replicating the frozen chunk objective; the identicalsyntheticdiagnostic
cannot expose this difference.


## Architecture selection as an exploration multiplier

Hardware fit is a research variable: compare completed useful candidates per allocated
hour, then evaluate individual predictions and matched blend/distillation contribution.
A faster weaker model can be valuable as a complementary teacher. Standalone quality,
complementarity and execution cost are separate observations; none is presumed.

| Architecture arm | Hardware rationale (hypothesis) | First concrete comparison | Causal serving requirement |
|---|---|---|---|
| Dilated residual TCN | Convolutions parallelize over time; avoids per-row nonlinear recurrence | Compact causal TCN versus own GRU at matched data/exposure; full backward profile and residual/complementarity check | Left-only padding; retain receptive-field history across chunks and reset at sequence boundaries |
| Block-recurrent causal attention | Dense projections/attention within blocks, state across blocks; bounded attention memory | Small fixed block sizes, e.g.128/256, against own recurrent teacher; compile/HBM/complete-fit cost before scale | Strict causal mask within each block; previous-block state only; verify prefix invariance |
| Diagonal linear recurrence plus nonlinear readout | Associative prefix training, compact recurrent state at inference | H216 sequential/prefix parity, then useful teacher contribution | Preserve exact recurrence state/reset and deterministic readout |
| Hoisted GRU / independent-chip cohort | Improve existing architecture execution and allocation | H214/H215 matched profile and completed-seeds/hour | Existing GRU equation and per-model ancestry unchanged |

The TCN paper provides an authors' implementation using residual dilated convolutions
and trimming padded future outputs:
[paper](https://arxiv.org/abs/1803.01271),
[actual source](https://github.com/locuslab/TCN/blob/master/TCN/tcn.py).
Its offline convolution example does not implement our streaming state; add and test
history handling. Do not import its weight-normalization/checkpoint API blindly.

[Block-Recurrent Transformers](https://arxiv.org/abs/2203.07852) supplies the transferable
component of parallel computation within blocks with recurrent state between blocks.
This is a source-grounded architecture proposal, not a claim its language-model results
transfer to Challenge2. Our adaptation must enforce strict within-block causality;
full-block bidirectional features would violate the prediction task.

Proposed first architecture allocations: one small TCN recipe and one diagonal-recurrence
recipe, with a small block-attention teacher as a separate deeper arm. Use independently
trained weights and existing data membership. Freeze representative shapes, profile
actual TPU/GPU forward plus backward, then bounded fits. Report compiler/startup/data
cost as well as step throughput; compare complete endpoints, not isolated kernels.
Keep a cheap matched GRU execution lane alongside architecture exploration. No new
numerical reward or automatic production admission is created by this reading.

Reviewed correction2026-10-07: original brief remains in commit c438422; this successor separates host RAM from HBM and expands distributed-loss parity to all existing auxiliary terms. The queued diagnostic remains unchanged.
