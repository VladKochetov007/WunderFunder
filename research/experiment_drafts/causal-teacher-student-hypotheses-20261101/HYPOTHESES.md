# Eight causal teacher–student research cards

**Status: proposals only. None is runnable, profiled, fitted, graded or ready for
submission.** Adapters, source-qualified data/cache bindings, receiver numerical
acceptance and B300 profiles are outstanding. These cards specify prospective
comparisons, not commands or a commitment to launch every study. Each requires a
separate frozen execution recipe, claimed worker and independently reviewed source
before admission. No local tests, inference or training accompany this packet.

## Shared evaluation and cost contract

- The working reference is canonical TL3, fixed W18/tiedlead-s5/tiedlead-s6 weights
  0.34/0.33/0.33, reported **0.692377425** on exposed full official valid. The
  [PR52 report](https://github.com/VladKochetov007/WunderFunder/pull/52#issuecomment-6532373933)
  records TL3 + 10% GALAM at 0.692496447 (+0.000119022); a separate
  [PR53 report](https://github.com/VladKochetov007/WunderFunder/pull/53#issuecomment-6545810833)
  reports TL3 + 10% LSTM at 0.692502063 (+0.000124638). The two additions were
  not rescored together; their tiny difference does not establish superiority.
  The six positive KD pairs in PR53 are on an exposed 8260/2065 train-only proxy,
  not a fresh holdout or public-score evidence.
- Each two-arm screen below proposes **two control and two treatment fits**, paired
  by proposed seeds 1001 and 1002 on the same source-qualified component split.
  Card 2 proposes three arms, two seeds each, for **six** fits. The eight primary
  panels therefore total **34 fresh fits**, including the teacher fits in cards 5
  and 6. This is an inventory, not authorization to run all panels. Teacher
  prerequisites, cross-fit teachers, student transfer, cache exports and engineering
  profiles are additional and must be counted explicitly before admission.
- Two seeds on one split do not separate split and training variance or establish
  significance. For each panel freeze the exact base checkpoint/ancestry, cells,
  fit components, transform membership, effective batch, updates, optimizer,
  schedule, endpoints and teacher-cache bytes before starting. Unspecified recipes
  here are **not** executable specifications. Existing models or controls are
  reusable only after matching the exact split, seed, source, normalizers and full
  ancestry; no reuse or reduced cost is assumed merely because a PR exists.
- Use the unchanged official `global_wp_clip2_v1`, scoring mask, clipped weighted
  pooled correlation per target and their prescribed aggregate. Report each
  paired seed delta, both targets, subgroup counts, runtime/memory and all failures.
  Prespecify +0.0005 WP as a *screening* effect of practical interest for each
  changed student on the same data; teacher-only cards also require a separately
  budgeted student-transfer or deployable-ensemble check before promotion. Smaller
  or mixed results are uncertain, not proof of no effect. A fixed-model paired
  component bootstrap measures evaluation sensitivity, not seed variance or an
  independent confirmation. Negative outcomes constrain the tested recipe only.
- Derive any inner selection split from full train components; all teachers,
  normalizers, initializers and cache producers must exclude that split's held-out
  components. Official valid must be absent from fitting and is development-exposed
  reporting data, not a fresh holdout. Preserve exposure history, sample coverage,
  data/source hashes and conditional uncertainty. Component disjointness does not
  prove chronological separation when true timestamps are unavailable.
- At prediction time every value, gate, state and normalization statistic is
  causal: observed prefix through the current row only, complete sequence resets,
  no true past target or future inputs. Fit all learned thresholds/transforms on
  eligible fitting data only; test prefix invariance and reset behavior on the
  receiving host before GPU admission. Any cross-fitting must also exclude the
  scored component from every producer's teacher and initializer ancestry.
- Scientific screens default to FP32 weights, activations, gradients and optimizer
  state, with TF32/autocast off. The explicit BF16/BF16x9 amendment for B300
  Transformer work can be selected only as a separate declared, consistently
  matched recipe, with FP32 master/loss/optimizer state and no IEEE-FP32 parity
  claim. Do not quantize scientific training. Submission-stage compression is a
  separate measured decision. Competition execution is Linux/Python 3.11, one
  vCPU, 16 GB RAM, 60 minutes, uploaded ZIP at most 20 MB and no network; these
  are requirements, not measured acceptance of any proposal here.

## 1. Distil temporal changes in addition to levels

**IDs:** `H-CST-TEMPORAL-DELTA-KD-20261101` / `E-CST-TEMPORAL-DELTA-KD-20261101`.

**Lineage:** connection/student; related to [PR29](https://github.com/VladKochetov007/WunderFunder/pull/29)
and [PR47](https://github.com/VladKochetov007/WunderFunder/pull/47).
Temporal alignment in [DistilTS](https://arxiv.org/abs/2601.12785) motivates
examining dynamics; this loss is our proposal, not an established paper result.

- **Mechanism and comparison:** reuse one source-qualified fixed teacher and the
  same GRU student. Control is ordinary label plus output KD; treatment additionally
  matches the causal teacher/student prediction changes between `t` and `t-k`,
  using a fixed train-chosen `k` and coefficient. Mask sequence starts, warmup and
  missing teacher outputs. Keep initialization, teacher targets, data, sample
  order and total updates matched; keep label and level-KD coefficients unchanged.
  Normalize each loss over the same eligible rows and log gradient scales. Changing
  the total effective gradient scale remains an alternative explanation; a scale-matched
  control needs two additional seed fits under a separate allocation.
- **Cost:** 2+2 fresh student fits; zero new teacher fits *only if* the exact
  teacher and complete causal caches are qualified for the chosen split. Otherwise
  count new teacher fits and cache generation separately. The added difference
  loss is training-only and does not add CPU inference modules.
- **Decision and failure:** improvement in paired official WP and stable transition
  slices supports useful temporal information; worse or mixed paired deltas, or
  excessive sensitivity to noisy teacher changes, narrow this particular `k` and
  coefficient. Check both targets and high-change rows without selecting on valid.

## 2. Reweight KD on teacher disagreement

**IDs:** `H-CST-DISAGREEMENT-KD-20261101` / `E-CST-DISAGREEMENT-KD-20261101`.

**Lineage:** connection/student; a refinement of uncertainty-aware KD already
discussed by the team, and of the fixed heterogeneous pool in PR52.

- **Mechanism and comparison:** freeze the same three causal teacher predictions
  and student recipe across three arms: constant KD weights; a bounded multiplier
  that increases with teacher disagreement; and a bounded multiplier that
  decreases with disagreement. Determine thresholds on fit components only,
  normalize mean KD weight consistently, retain the same label loss, and do not
  use labels, future rows or valid data to compute serving-time features.
- **Cost:** 2 control + 2 upweight + 2 downweight = **6 fresh student fits**.
  Qualified exact teacher checkpoints/caches are assumed reused only after
  verifying membership and hashes; otherwise teacher fits/cache exports are extra.
  Teacher-disagreement computation is training-only; no extra CPU model is needed.
- **Decision and failure:** the paired control contrasts plus predeclared
  high-disagreement slices distinguish whether disagreement highlights useful
  structure or teacher noise. An aggregate improvement with harm to a target or
  rare groups is a tradeoff, not automatic promotion. In-sample teacher targets
  may themselves overfit, so record that limitation.

## 3. One streaming state, multiple gated heads

**IDs:** `H-CST-SHARED-STATE-GATED-HEADS-20261101` / `E-CST-SHARED-STATE-GATED-HEADS-20261101`.

**Lineage:** deployable student; extends multi-head/family targets and PR52's
fixed mixtures rather than asserting a new architecture.

- **Mechanism and comparison:** control uses a shared recurrent backbone with two
  small heads and fixed equal averaging. Treatment uses the same backbone and heads
  with a small causal input/state-based gate. Provide matching gate capacity in
  the control but force equal weights, and explicitly report trainable-parameter
  differences, update budgets, gate parameters and FLOPs. Report a separately
  measured existing one-head baseline if its ancestry exactly matches. Only the
  gate combination differs in the primary comparison.
- **Cost:** 2+2 student fits; qualified teacher targets may be reused, with
  missing teacher fits and exports separately budgeted. Every row updates one
  shared state; a second hidden recurrent backbone invalidates the claimed saving.
  Package and time the complete gated student on the receiver.
- **Decision and failure:** paired student gains and favourable score/latency versus
  fixed heads support useful specialization. Gate collapse, quality loss in either
  target, or CPU overhead erasing the benefit limits this formulation.

## 4. Regime-conditioned teacher mixture from out-of-fold predictions

**IDs:** `H-CST-REGIME-TEACHER-MIX-20261101` / `E-CST-REGIME-TEACHER-MIX-20261101`.

**Lineage:** connection/student; complements PR52 and is inspired by diverse
inductive priors in [TIPS](https://arxiv.org/abs/2603.16985).

- **Mechanism and comparison:** on train components only, produce source-qualified
  out-of-fold (OOF) causal predictions for two teacher families. Control distils
  their fixed equal mixture. Treatment distils a bounded mixture whose weights are
  predicted by a small gate from the current/past prefix. Gate training uses only
  OOF predictions and eligible train labels, never valid or future inputs. Use the
  same OOF producers, student, seeds and update budget for both arms; cross-fit the
  gate or isolate its fit components from its target-generation components to
  audit label memorization. The gate is for constructing training targets, not an
  implicit extra model at CPU inference.
- **Cost:** 2+2 fresh student fits **plus** OOF prerequisites. With two teacher
  families and two component folds, at least four source-clean teacher fits are
  needed if no exact matching OOF artifacts exist; lightweight gate cross-fits,
  causal prediction caches and their compute are extra. Reusing a teacher trained
  on the gate's own components as an OOF producer is invalid.
- **Decision and failure:** require paired improvement over the same OOF fixed
  mixture and check regime-specific scores and gate stability. A stronger teacher
  in one arm, collapse to one teacher, memorized regimes or negligible improvement
  would not support the gating explanation. No external market weights are needed.

## 5. Multi-scale causal Transformer memory

**IDs:** `H-CST-MULTISCALE-TRANSFORMER-20261101` / `E-CST-MULTISCALE-TRANSFORMER-20261101`.

**Lineage:** teacher; complements [PR48](https://github.com/VladKochetov007/WunderFunder/pull/48)'s
width-versus-context screen rather than relabelling it.

- **Mechanism and comparison:** control is the same-size causal Transformer with
  a fixed recent window. Treatment keeps that recent window and adds compressed
  summaries of older prefixes, masking every summary until all of its inputs are
  observed. Match number of train rows, seeds, updates and endpoints; report actual
  FLOPs, GPU-hours and parameter count. An equal-wall-time comparison, if needed,
  is a separately declared panel with extra endpoints/fits, outside the 34-fit
  inventory. No pooling block can see the future.
- **Cost:** 2+2 fresh Transformer teacher fits, included in the primary 34. Any
  student KD fits, teacher cache construction, longer-context profiles and CPU
  serving work are additional. Five epochs or 30 epochs must be selected in the
  future frozen recipe; neither is claimed optimal here.
- **Decision and failure:** paired teacher gains across sequence-age slices and
  adequate subsequent student transfer would support useful longer history.
  No gain, degraded recent-context quality, future leakage, or a compute increase
  disproportionate to benefit constrains this summary design.

## 6. Gated DeltaNet or Mamba hybrid as a causal teacher

**IDs:** `H-CST-PARALLEL-STATE-TEACHER-20261101` / `E-CST-PARALLEL-STATE-TEACHER-20261101`.

**Lineage:** teacher; complements the unrun [PR24](https://github.com/VladKochetov007/WunderFunder/pull/24)
architecture handoff. [Gated DeltaNet](https://arxiv.org/abs/2412.06464) and
[Mamba-2](https://arxiv.org/abs/2405.21060) offer parallel training mechanisms;
their reported benchmark or kernel speedups are not measured on this task.

- **Mechanism and comparison:** control is a causal Transformer teacher;
  treatment is one explicitly chosen DeltaNet/window-attention or Mamba hybrid
  teacher. Use matched input/data/seed/update budget and comparable capacity, and
  separately measure GPU wall time and memory. Fix architecture before fitting;
  kernel availability and causal streaming behavior require receiver profiling.
- **Cost:** 2+2 fresh teacher fits, included in the primary 34. Student-transfer
  fits and teacher prediction exports are separate. Parallel training does not
  eliminate recurrent/streaming state during inference; a CPU-native implementation
  and timing are not assumed.
- **Decision and failure:** stronger teacher quality at useful GPU cost followed
  by a separately measured student gain supports the direction. A faster kernel
  without better end-to-end training, no transfer, unsupported kernels, or poor
  streaming quality does not establish a deployable improvement.

## 7. Past-only scale and discrete-level input encodings

**IDs:** `H-CST-CAUSAL-SCALE-ENCODING-20261101` / `E-CST-CAUSAL-SCALE-ENCODING-20261101`.

**Lineage:** input representation; related to [PR35](https://github.com/VladKochetov007/WunderFunder/pull/35)
and prior discrete-occupancy/clock work. [TradeFM](https://arxiv.org/abs/2602.23784)
motivates testing scale-stable representations, not assuming our anonymized
features have a particular financial meaning.

- **Mechanism and comparison:** both arms keep the same raw inputs and model;
  control uses existing features, treatment adds bounded past-only normalized
  changes and train-fitted discrete-level indicators. Fit bin boundaries and
  fallback scales on eligible train components, use only prefix statistics at
  each step, and match parameter/compute budgets or record their difference.
- **Cost:** 2+2 fresh model fits; any distinct teacher training is extra. Extra
  preprocessing and input width increase streaming cost, so independently verify
  deterministic streaming parity, ZIP size and one-vCPU runtime.
- **Decision and failure:** paired gains across component/target slices support
  transfer beyond feature scale. Instability under causal startup, loss of absolute
  information, no gain or costly preprocessing narrows this encoding. Do not
  assume physical price/volume semantics for anonymous fields.

## 8. Robustness over prefix-defined regimes

**IDs:** `H-CST-PREFIX-REGIME-ROBUSTNESS-20261101` / `E-CST-PREFIX-REGIME-ROBUSTNESS-20261101`.

**Lineage:** student/objective; extends earlier regime diagnostics, not a claim
of a new discovery.

- **Mechanism and comparison:** predefine train-only groups from causal prefix
  change density, bounded scale and saturation; thresholds come only from fit
  components. Compare ordinary label/KD WMSE with the same objective plus a mild,
  explicitly fixed group-robust penalty, keeping model, data and updates matched.
  Group membership must be available from the prefix and never from future labels.
- **Cost:** 2+2 fresh student fits; exact qualified teachers/caches may be reused,
  otherwise separately count them. The training penalty need not add any inference
  feature or runtime path if only the objective changes.
- **Decision and failure:** evaluate official pooled WP, both targets and
  predeclared group slices on equal scored rows. Better weak-group behaviour with
  material pooled-score loss is a tradeoff; overfitting group thresholds, mixed
  seeds or uniformly lower WP narrows this design. Never substitute the mean of
  per-sequence correlations for the official metric.

## Engineering prerequisite and permitted sources

Before interpreting any served quality claim, compare the **exact** candidate
ZIP on identical rows against its source-bound Torch reference, then compare
native FP32 and submission-stage INT8, especially around 4000-row chunk boundaries,
sequence resets and warmup masks. PR51 records a *possible* row-change reset
mismatch; no root cause is established by the difference between the exposed
valid score and a public score on different data. This is a receiver-side
engineering check, not a new fit or a claim that the canonical W18 exporter exists.

Use only data allowed by the [competition rules](https://wundernn.io/connectome/docs/rules).
Open-source pretrained models are allowed only if publicly available and they do
not contain external market data; verify actual training provenance before use.
Names such as Chronos or TradeFM alone do not establish eligibility. Do not use
external private data or infer future features in teachers or students.
