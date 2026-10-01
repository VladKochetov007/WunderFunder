# CTXMEM-01 — small contextual event-memory reserve (not allocated)

This is a concrete architecture alternative to repeated attention/dual-clock experiments. No empirical probe, fit or readout is authorized by this draft. It is a reserve behind the already allocated experiments, with a fixed-serving runtime gate.

## Mechanism and closest controls

Existing native128GRU exposes its current contextual recurrent output `h_t` to the head. Historical event attention and event-GRU wrote the instantaneous projected raw input `z_t`, so a past event token did not explicitly contain the preceding interval's accumulated recurrent context. Store low-dimensional filtered **h_t** instead. There is no learned event recurrence, attention, price-level decoder or assertion that the existing proxy is a physical quote event.

A dense historical memory of128states is unnecessary. Fix a16×128 orthonormal projection P using rows0,8,16,...,120 of the order128 Sylvester Hadamard matrix, divided by√128. Save the exact float32 matrix in checkpoint metadata. It is deterministic, independent of outcomes, and requires no feature-fit labels. Compute q_t=P·h_t in float32 after the current row's GRU update. Projection is untrained; all original backbone parameters remain frozen.

The existing event flag is `x[0] != prev[0] AND x[11] != prev[11]`, exactly the original99-feature definition. At an event update two16-dimensional summaries:

`m_fast ← λ4*m_fast + (1−λ4)*q_t`, `λ4=2^(−1/4)`;

`m_slow ← λ32*m_slow + (1−λ32)*q_t`, `λ32=2^(−1/32)`.

At row0 initialize both memories to q_0. On other non-event rows hold them unchanged. State resets only at seq_ix boundaries. Initial-state handling, half-lives and projection are fixed before an outcome; no half-life, rank or projection sweep belongs to this screen.

Both arms have a zero-initialized32→128 additive head-preactivation map W_extra; all original head weights are copied exactly. Candidate inputs are `[m_fast, m_slow]`. **Matched current-state control** inputs are `[q_t, q_t]`. Both compute the same q and memory states during training to match compute and RNG; only the selected head input differs. They therefore have equal parameter count, width and update budget. The control feature is a linear transformation of h_t already in the ordinary head, so it is algebraically foldable into the old head weights. Its role is to control the optimizer/parameterization effect of another input path. The candidate introduces a temporal summary rather than extra instantaneous representational capacity. This does not isolate which timescale or which event-content mechanism matters; that would require another allocated comparison.

Critic caveat: the control's duplicatedq has rank16 while the two summaries can have rank32. This is inherent to the hypothesis that two histories add information beyond one current state, not a claim of equal information content. Replacing the control by last/penultimate event states would test a different question (smoothing versus two explicit lags) and must be decided before any fits, not swapped after a result. The plain two-epoch anchor remains a second required comparator.

## Training implementation that is actually affordable

Use ref/refs4 parents at epoch3, seeds3/4 and folds0/1, same two-epoch head-only recipe as TRANSHEAD01, aux settings unchanged. Freeze inp/gru/P. Original normalization, raw features and per-sequence recurrent reset stay identical. Four candidate and four matchedcontrol fits; model_ep1 fixed. Primary replaces both h128 slots in the equal-quarter V1-like ensemble. Gate: memory−current control≥.0003 pooled and eachfold≥0, and memory−plain anchor≥0 eachfold and pooled. If serving overhead cannot fit inside fixed-four budget, reject regardless of the local point estimate.

Do **not** run a Python CUDA loop over every event to compute the IIR states: thousands of tiny kernel launches can dominate training. Two feasible implementations, with one chosen and frozen before allocation:

1. Since the trunk is frozen, compute q for the current trainingchunk onGPU, copy only16float32values per row plus eventflags to CPU, run compiled `scipy.signal.lfilter` on the compact event list for each sequence/channel, scatter held states, then transfer32features back. The model's chunk state becomes `(GRUstate, m_fast, m_slow, initialized)`; exact state continuity is required. For batch16×chunk2000 this transfers roughly6MB round-trip, not a complete112feature or256hidden cache. This introduces synchronization; benchmark synthetic chunks before accepting. It is the simplest auditable first implementation.
2. A small torch C++/CUDA scan could avoid copies but adds build risk and is not preferred under the overnight deadline.

The lfilter recurrence must use the same double coefficients/native rounding policy. For parity, the native and research feature calculators should accumulate memory in float64 from float32q, then cast the32head channels tofloat32. This makes recurrence across chunks well defined and avoids silently changing accuracy with chunk length. Computing P·h_t across AMP and native paths still inherits ordinary GRU rounding differences; real trained-model prefix parity is mandatory. Synthetic preflight must demonstrate prefix, non-event holds, first-row initialization, per-sequence reset and arbitrarychunk boundaries.

No frozen hidden-state disk cache is necessary. A full `[h,z]` cache for four OOF backbones would be about77GB even infloat16 and would spend disk/IO for a short head experiment. The compact transfer path deliberately avoids it.

## Serving cost and hard spending gate

Per128member perrow: fixed projection16×128=2048 multiply-adds; head extension128×32=4096; at an event two16-coordinate updates add about64 multiply-adds. Total≈6208 multiply-adds plus small loop/state overhead. Ordinary128model dense arithmetic is approximately128×211 inputprojection +12×128² two-layerGRU +128×256 head≈256k multiply-adds. The added arithmetic is≈2.4% per128member. Across the actual four-member blend (one256 and three128), extending two128slots adds roughly0.7% of dense arithmetic. These counts are **not** runtime measurements: extra GEMV calls, copying, activation behavior and Python callback overhead can dominate small matrices.

Persistent extra state is two16-double vectors (256bytes/member), q and feature scratch, plus P (8KiB shared or per-model) and W_extra (16KiB/model infloat32). There are no per-row allocations and no attention ring. Implement q projection and extra head GEMV in the existing C++ Predictor so there is no additional Python callback.

Before training: prototype random weights and fixed same-row inputs, alternating unmodified versus memory kernel measurements on one CPU using the actual recovered C++ engine. Require fixed-four median overhead≤3% with repeated trials; report absolute projected minutes and scheduler conditions. If the known-good fixed-four package lacks sufficient timing margin, no amount of theoretical MAC counting promotes this branch. A ten-percent overhead on the two extended members, as seen in the separate68-channel transition prototype, would be materially worse than the arithmetic estimate and must not be hidden.

## Expected evidence and main ways it can fail

A positive result would support the utility of a small explicit history of contextual states for the current frozen backbone and serving mixture. It would not establish physical event semantics, long-range missing information in principle, or a new hidden-test best. Pure-chain evaluation remains exposed. Most likely negatives: h_t already summarizes all useful event history; P loses the rare useful direction; fixedclock smooths unrelated code events; adding longhistory increases domain sensitivity; or native cost erases the smallscoregain. A failed fixedrank/halflife recipe does not license a rank/half-life sweep this night.
