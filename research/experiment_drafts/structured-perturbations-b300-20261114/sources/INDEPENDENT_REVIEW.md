# Independent source and experiment review

Reviewed on 14 November 2026 by a separate audit agent. This is a read-only
implementation review followed by this note; no model import, test, numerical
probe, inference, profiling or training was executed. The review supports
publishing a prepared draft for receiver admission. It is not B300 acceptance,
measured parity, successful execution or a quality result.

## Scope and source comparison

Inspected `payload/wnn/training/sequence_perturbations.py`, `runtime.patch`,
`probe_invariants.py`, `perturbation_probe.py`, `prepare_runtime.py`, `receiver.py`,
`readout.py`, `run.sh`, the published recipe/configuration shapes and
`GPU_COMMANDS.md`. Compared their call signatures, CLI flags and result fields
against native source `8626000a4b965763a70011a0291a891aa6535d7e`, including model
construction, attention masks/backends, precision, compilation, training loop,
profiles/reload, prediction/scoring, epoch indexes and the native paired grader.

No deterministic wrong-API or producer/consumer schema blocker was found in
those inspected paths. This statement does not assert that the code has run.

## Findings repaired before handoff

1. **Initialization comparison could hide drift.** The first probe version
   loaded baseline weights into each arm before comparing outputs. The revised
   source compares original state keys and tensors under the common seed before
   any replacement; baseline weight replacement was removed.
2. **Mask helper tests did not demonstrate that chosen keys disappeared.**
   Efficient-versus-math comparison reused the same attention implementation,
   while independent mask-count checks alone could admit an ignored mask. The
   revised probe changes only chosen dropped keys and requires invariant output,
   then changes a protected recent key and requires sensitivity. Both replay the
   same CUDA RNG before attention.
3. **Model integration could ignore history corruption.** Eval parity and
   same-arm training replay alone could pass if the model never dispatched the
   perturbation helper. The revised model probe requires equal cache-empty first
   chunks, a nonzero treatment-versus-clean difference with eligible cached
   history, and zero difference for non-history controls. Eval identity remains
   a separate requirement. Input staleness is a training-loop hook, so its
   primitive invariants plus the inspected patch insertion are the relevant
   source evidence; the model-only probe correctly does not apply that hook.
4. **RNG evidence needed a narrower claim.** The native profile verifies
   serialized model/optimizer/scheduler/RNG roundtrip; the small probe verifies
   replay in a warmed compiled model. Neither proves a fresh-process stochastic
   resume. The wrapper exposes no resume stage, and receiver instructions now
   require separate admission before resuming an interrupted scientific fit.
5. **Statistical output needed explicit scope.** The readout returns per-seed
   component intervals and the descriptive mean of two seed deltas; it does not
   compute an interval for that mean. Native `p_le_0` is a bootstrap sign fraction,
   not a calibrated p-value. README now states both limits and makes no Holm or
   formal significance claim.

The repaired checks are present in source and still require execution on B300.

## Causality and implementation observations

- Input corruption chooses a fixed-size subset of the same eligible raw i1
  positions for IID/burst arms. Row0 and the incomplete final block are protected.
  Past-value gather indices are constructed by cumulative last-observed position,
  so replacing a later input value cannot change an earlier imputed value.
  The whole sequence is transformed before native chunking. The private input
  generator is indexed by seed/epoch/rank/batch position and does not consume
  global model RNG. Targets and loss masks remain the native tensors.
- History corruption trims keys with the native helper and constructs the
  additive mask in the resulting relative coordinates. The native rolling
  causal condition is retained alongside the random historical-key mask.
  Current-chunk causal eligibility and the most recent32 cached keys are
  protected. Backing mask width uses native alignment; visible Q/K/V shapes and
  mask dtype agree with the selected efficient or math SDPA path. Stored K/V are
  not modified by applying the attention mask.
- Constructor and config fields connect the new mode to the block. Native
  `load_config` injects the declared precision into the model settings. The
  treatment creates no parameters, and evaluation dispatches to native clean
  attention. Actual same-seed/eval equality, CUDA kernels and gradients remain
  receiver measurements.
- Historical masks consume the native device RNG. The fixed configs disable
  activation checkpointing and retain native block compilation. This review
  makes no unsupported checkpoint-recomputation or cold-resume guarantee.

## Commands and evidence path

Prepared runtime extraction is isolated from existing source checkouts. Exact
source/payload manifests, clean packet identity and the original split are
checked before admission. Native train, predict and score argument names agree
with the pinned source. The receiver's profile fields and parameter count match
the native recipe's schema and model shape. Readout validates fixedE4/3280-update
terminal, checkpoint, producer, epoch-index, export and official-score identities.
It requires a common GPU pair per seed, as the receiver commands specify.

Stage timeouts, owner/allocation metadata, logs and immutable outputs are
prepared; they are not a new allocation. The wrapper neither launches the whole
panel automatically nor uploads a submission. The actual receiver must verify
available resources and approve each real-shape profile before fitting. No
timing, peak-memory, finite-gradient, RNG-replay or full-valid receipt exists
from this review.

## Scientific limits retained

The ten fits are two six-fit screens sharing two fresh controls, not ten
independent mechanisms. Both seed repetitions use one repeatedly exposed
official split. Input arms match replaced-row count and eligibility, but not
the age of held observations. History arms match marginal eligible-key removal,
but not removed attention mass; a wrapped block can be two chronological spans.
This does not prove label invariance or replicate the public/private shift.

The +0.001 mean-WP and both-seeds-positive rule is a follow-up allocation
heuristic. Structured-minus-IID contrasts diagnose the frozen policies with the
above confounding limits. All six contrasts, per-target scores, partial results
and negative outcomes remain reportable. Old controls, early-epoch peaks and
leaderboard feedback cannot replace this fixed endpoint comparison.

No unresolved deterministic blocker was identified in the final inspected
preparation. Receiver numerical admission and real-shape profiles are required
before any scientific execution; successful publication must not be described
as passing those gates.
