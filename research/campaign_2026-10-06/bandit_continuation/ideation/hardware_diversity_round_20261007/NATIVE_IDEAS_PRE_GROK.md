# Native hypotheses committed before the independent Grok call

Frozen at 2026-10-07 13:05 UTC, before sending Grok any ideation prompt. These
are proposals from the assigned native model (`gpt-6-luna`, maximum reasoning),
not results. Public-gain priors are null. The target is a useful, independently
trained, causal teacher/component per measured compute-hour; every candidate
must reset state by sequence, emit on each required row, and keep the official
`global_wp_clip2_v1` evaluator fixed. Exposed validation remains a development
screen only.

The active hardware evidence is narrow: the E522 local profile measured 49.53
sequences/s at batch 16/zero workers, 50.49 at batch 32/zero workers (+1.94%
with substantially more VRAM), and 51.72 at batch 16/one worker (+4.42%). A
two-worker profile hit its scoped RAM cap and was killed. This motivates compact
single-worker candidates; none of those local numbers predicts Kaggle/T4 speed.
E520's cloud profile repeatedly spent most of its bounded pre-CUDA time reading
the same raw train bytes for identity verification. Any new teacher must reuse
the reviewed same-worker verified-input handoff only after its independent
review/admission; this task proposes ideas, not pipeline changes.

I screened out another causal TCN, block attention, fixed diagonal recurrence,
and a widened GRU as the main novelty: those directions are already represented
by H-217/H-218/H-216 and E-521. They remain useful controls, but repeating their
labels would not create fresh hypotheses.

## Card N1 — Random causal reservoir with a streaming readout

- **Lens:** unconventional dynamical systems / streaming algorithms.
- **Directed source:** Herbert Jaeger, *The “Echo State” Approach to Analysing
  and Training Recurrent Neural Networks*, GMD Report 148 (2001; corrected
  2010 erratum version), especially
  the sections on echo-state networks and output/readout training. The source
  describes fixed recurrent dynamics with a trained output map; this is a
  source fact, not evidence that the method helps this task. Primary author-hosted
  PDF: <https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf>.
- **Random-library transfer:** A seeded uniform draw from the 14 existing PDF
  paths recovered by the library footprint selected Vertox, *TWAP & VWAP — More
  important than you think* (2023), p. 4. The exact short excerpt and selection
  receipt are in `RANDOM_LIBRARY_PASSAGE.md`. I transfer only its recursive
  numerator/denominator bookkeeping pattern to bounded weighted-readout
  sufficient statistics; the article is not evidence for a market-label
  mechanism here.
- **Intervention:** From a fixed, recorded seed, initialize a small sparse
  reservoir (initial candidate: 32–64 states, bounded recurrent spectral
  radius, tanh/leaky update) over a low-rank projection of the 112 channels.
  Train only a two-target ridge/readout head from scratch. Stream per-batch or
  per-sequence weighted sufficient statistics; never retain all hidden rows.
  Use clipped training labels and label-magnitude weights only if recipe audit
  confirms those exact semantics. Keep official scoring clipping/weights
  unchanged. The random reservoir and fitted readout are this team's own model;
  no external or teammate weights enter.
- **Mechanism hypothesis:** Fixed nonlinear recurrences may span different
  transient/decay features from a learned GRU while eliminating recurrent
  backpropagation and most optimizer state. That may produce a cheap, useful
  teacher or student target, even if its standalone WP is lower.
- **Observable and direction:** At a matched data pass, complete optimizer
  updates / wall-second should exceed the compact GRU, while its residual
  covariance against the fixed C1 anchor is lower than a matched random-feature
  control. Both are hypotheses; no numerical threshold or public-score delta is
  assigned here.
- **Cheapest falsifier:** First check causal prefix invariance, sequence reset,
  deterministic seed replay, and exact streaming/readout parity on synthetic
  short sequences. Then compare actual-device full-update profiles at one small
  candidate width and the already-admitted GRU shape. If throughput does not
  improve or RAM/VRAM is worse, stop before fitting. Only after a separate
  recipe/evidence review, fit one preregistered seed on the existing complete
  chain-excluded fold and compare alone against the frozen anchor; use the
  exposed fold only as development evidence. Consider it a distillation source
  only if an independent student test is separately allocated.
- **Falsifier:** Verified profile shows no useful throughput advantage, or the
  completed candidate has neither standalone value nor a distinct residual /
  teacher signal under the preregistered comparison.
- **Expected cost/artifact:** One short profile, then one small independently
  initialized fit only if admitted; outputs are a deterministic checkpoint,
  provenance manifest, row-aligned prediction export, measured profile, and
  same-fold anchor comparison. Fit cost is unknown until the profile.

## Card N2 — Shared semantic group encoder before a small recurrent core

- **Lens:** representation symmetry / tensor factorization.
- **Directed source:** Zaheer et al., *Deep Sets*, NeurIPS 2017, Sections 2–3,
  on shared transformations and pooled set representations. Primary paper:
  <https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf>.
- **Intervention:** Treat only the documented repeated instrument/type blocks
  as related: `i0` and `i1` each have price-like, volume-like, trade-price and
  trade-volume groups. Apply a shared small per-group encoder, add explicit
  instrument/type IDs, pool to a compact per-row representation, and pass it to
  a small causal recurrent state. Keep the eight `a*` fields on an explicit
  separate path. Do not impose a permutation over numbered levels: the data
  documentation explicitly says the indices do not imply sorted depth.
- **Mechanism hypothesis:** Weight sharing can encode repeated field grammar
  with fewer input parameters and may emphasize cross-instrument relations
  differently from a flat 112-channel projection; the compact recurrent core
  may improve experiments/hour.
- **Observable and direction:** At a matched parameter budget, peak memory and
  complete-step profile should fall or hold while a chain-held-out predictor
  retains enough information to match the flat control. Teacher complementarity
  is a separate endpoint, not inferred from parameter count.
- **Cheapest falsifier:** Synthetic channel-label and permutation tests first:
  swapping i0/i1 must follow explicit instrument embeddings; arbitrarily
  permuting numbered levels must not be silently treated as valid symmetry.
  Then one measured profile and a small matched-control fit. A no-history
  encoded-row model is a cheap input-screen, not the final causal claim.
- **Falsifier:** Either the channel contract cannot be preserved, the shared
  encoder loses predictive content versus an equally small unshared projection,
  or the measured profile does not improve memory/throughput.
- **Expected cost/artifact:** One implementation-parity test, one actual-device
  profile and, if still attractive, one complete-fold independent checkpoint
  with feature-group provenance and anchor comparison. No teacher cache is
  required for this first test.

## Card N3 — Closed-form input-dependent retention (“liquid” microstate cell)

- **Lens:** continuous-time dynamical systems / control.
- **Directed source:** Hasani et al., *Liquid Time-constant Networks*, AAAI 2021,
  especially the model construction and time-constant discussion. The paper
  describes first-order dynamics with nonlinear gates and state-coupled varying
  time constants; it does not establish performance on these uniformly indexed
  challenge rows. Official proceedings:
  <https://ojs.aaai.org/index.php/AAAI/article/view/16936>.
- **Intervention:** A small stable state update with a learned, bounded
  per-dimension decay rate, evaluated in closed form (no ODE solver). An
  input-only, causal activity scalar derived from current and previous
  standardized feature innovations may modulate elapsed *model time*; initialize
  it to the ordinary row-clock case. Add a stale-activity control and a fixed
  decay control. This makes the mechanism explicit and distinguishable from a
  second event-clock GRU.
- **Mechanism hypothesis:** The useful memory horizon may depend on how much
  the observed book/trade representation changed; a small state with
  input-dependent retention may represent quiet persistence and sudden changes
  more efficiently than a wide gated recurrent teacher.
- **Observable and direction:** At equal state size and training exposure,
  activity-conditioned retention should improve prediction or teacher residual
  structure over fixed decay and stale-activity controls without adding a
  numerical solver cost. A benefit is uncertain because row intervals may not
  represent elapsed wall time.
- **Cheapest falsifier:** Check bounded-state and prefix/reset invariants;
  compare fixed, aligned-activity, and stale-activity cells on short synthetic
  inputs, then profile the exact full update at a small hidden width. Only the
  aligned-vs-stale same-fold contrast can test the proposed signal.
- **Falsifier:** The learned decay saturates to a constant, aligned activity
  fails to beat stale activity, or exact full-update wall/VRAM cost exceeds the
  compact GRU without evidence of a distinct teacher.
- **Expected cost/artifact:** A small custom cell, parity tests and one actual
  device profile before any fit. Preserve all three matched variants if a
  scientific comparison is admitted; one isolated win is not enough to claim
  activity causality.

## Card N4 — Fixed causal scattering features as a low-backprop teacher

- **Lens:** signal processing / robustness.
- **Directed source:** Stéphane Mallat, *Group Invariant Scattering*, 2012,
  Sections 1–2 and 4. The paper constructs cascades of wavelet transforms,
  modulus operations and local integration, and discusses higher-order
  statistics of stationary processes. Primary preprint:
  <https://arxiv.org/abs/1101.2286>.
- **Intervention:** Project the raw input causally to a small number of
  feature-family streams, then apply a tiny bank of causal multi-rate filters
  and modulus/envelope features. Feed those features to a linear or very small
  recurrent readout trained from scratch. Use left-only histories and preserve
  filter state across chunks, resetting only on sequence boundaries. Avoid
  global pooling or future-dependent normalization.
- **Mechanism hypothesis:** Fixed band-limited envelopes may expose transient
  and persistent patterns across scales without learning every convolution;
  they may create a fast teacher with different errors from a GRU.
- **Observable and direction:** Fused filtered feature construction should be
  faster / lower-memory than training a full deep temporal stack, and the
  representation should retain scored-row predictive signal versus an
  equally-sized raw linear control. Complementarity must be measured separately.
- **Cheapest falsifier:** Offline synthetic impulse/step tests for causal delay,
  chunk-equivalent state and reset, then a complete-update profile using
  memory-bounded streams. Compare one preregistered filter bank against the
  same readout on raw projected inputs.
- **Falsifier:** The feature bank violates prefix invariance, loses nearly all
  signal against the raw control, or its filter compute erases the saved
  backpropagation cost.
- **Expected cost/artifact:** Small deterministic kernels/filters and a compact
  readout checkpoint with feature-bank seed/config provenance; profile cost is
  unknown until actual-device measurement.

These four proposals are not ranked by expected public delta. Root/controller
owns priority, allocation, source admission, recipes, and independent grading.
