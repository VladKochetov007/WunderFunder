# Unlaunched D1 synthetic GPU profiler, parity amendment v4

No GPU execution. Root owns launch after independent numerical and integration
CPU proofs, all four clean controls and equal budget/cost authorization. Existing
1080-second per-arm fit cap is conditional; no fit is launched by this runner.
Earlier unlaunched runner/prereg versions are preserved under revisions/.

## Exact workload

Match the SSM profile recipe: actual trainer FastEMA, forward_chunk and detach_state;
fused AdamW(lr1e-3, weight_decay1e-2), FP16 autocast/GradScaler, teacher consistency1,
EMA .999, auxiliary structure .3 and next .3. Seed3, batch32, full20k, chunk2000,
input211/output23, ten attempts and exactly ten successful optimizer/EMA updates.
Required exclusive --arm CLI values and exclusive JSON/PT paths; no batch or
accumulation changes. No real competition data, labels, masks or scores.

Resident raw112, causal event features yielding input211, synthetic y2, weights,
three structural label arrays and next rows/up arrays. Warmup99 weights zero while
all rows update model state. Raw22 codes change every8 rows across17 cyclic states;
endpoint every16 or113 rows plus11 natural endpoint origins. This includes code
refreshes without endpoints and same-code endpoint writes. Exact raw float32 a6 is
retained. All arms share the same generated input. Synthetic loss labels never
reach clock extraction or recurrent state.

Record forward(student+teacher), backward, parameter steps before/after, AMP scales
and skips, EMA steps, allocator peaks and sampled total device memory. Abort above
5e9 bytes, nonfinite tensors/gradients, invalid state grammar or unequal/skipped
updates. Root's continuous device monitor and actual loader/intermediate memory
check remain necessary; stage samples cannot prove absence of transient peaks.

## Explicit correction of the earlier precision assumption

Root's source inspection found no matmul-precision or TF32 setter in the actual
trainer/resources/export path. --fast does not enable TF32-high. Earlier proposed
forced high/TF32 policies in unlaunched v3 are superseded and preserved, not applied.
This runner changes no inherited precision switches beyond the existing standard
configure_torch call (cudnn benchmark false/deterministic true). Record actual
fresh-process defaults: float32 matmul precision, CUDA matmul TF32, cuDNN TF32,
deterministic/benchmark flags and reduced-precision reduction flags. No optional
TF32-high diagnostic is needed. FP32 export has no autocast.

## Saved final-weight parity evidence

After ten successful updates, freeze final student weights, disable outer autocast
and perform FP32 reset forward on exact batch1 first2000 resident input rows. Save
input, GPU output and all six state tensors with final weights in the exclusive PT.
Record actual backend flags. Independent CPU literal replay requires no GPU rerun
or RNG regeneration. The input retains source a6 precision and event/Walsh boundaries.

Primary CPU literal versus saved GPU FP32 gate, frozen before profile: maximum
absolute output difference3e-5, floating state difference .003; integer counters and
previous raw float32 prices exact. These are the unchanged CPU numerical tolerances.
No observation-dependent relaxation. Any failure blocks real fitting.

Also save a separate FP16 AMP reset output/state diagnostic with the same final
weights, input and inherited flags. Require all tensors finite, S/Z FP32 and exact
raw prices/counters versus FP32 reset; record output and all state dtypes. No AMP
numerical closeness threshold is claimed. Actual training uses AMP; export uses FP32.
No model source or numerical representation change in this profiler amendment.
