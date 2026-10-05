# Activation research retrospective — 5 October 2026

## Scope and accounting

At the end-of-window accounting, the activation campaign has **236 registered
fit slots**. Training completed and received an official grade for **224**. The 12 E488 fits and its independent grade are complete; the score and arithmetic proof are recorded in [the E488 report](context_mixture_duration/REPORT.md).
The other 12 registered slots did not produce training results: the six E485
continuations remained unadmitted after the native/custom optimizer probe
failed, and the six E489 trajectories were prepared but not started. E489's
first probe entrypoint failed before CUDA setup or cohort access; a versioned
import-path repair is prepared for a later window. Neither E485 nor E489 is a
negative scientific result.

Most graded activation experiments are small, one-epoch diagnostics: 24 fit
groups, 16 historically exposed evaluation components, and 474 updates per
arm. Their paired component intervals condition on the fitted checkpoints.
They do not measure uncertainty over training seeds, do not correct the many
registered/exploratory comparisons, and are not public-score estimates. The
recorded public best remains **0.6660**. This campaign has not established an
expected score of 0.7000, and none of these pilot deltas should be added to the
public score.

## Findings that survived controls

**Activation effects depend on placement, optimizer, and schedule.** E470/E472
found positive sine-minus-GELU point estimates in all six seed-by-optimizer
comparisons across three fixed seeds, but AdamW's gains over GELU were small
and intervals often crossed zero. MuonAdamW changed both the ranking and the
effect size; optimizer comparisons themselves reversed across seeds. E474
localized the change and found that embedding effects changed sign between
seeds, while head-only effects were near zero. Applying both changes also
showed a non-additive placement interaction. These results argue for matched
controls and interaction tests, not for one universally preferred activation.
See [E470/E472](activation_optimizer/THREE_SEED_ACTIVATION_REPORT.md) and
[E474](activation_placement/REPORT.md).

**The early sine result does not identify periodic curvature as the cause.**
E473's matched GELU-plus-linear control had larger point estimates than
GELU-plus-sine by about 0.0010 under AdamW and 0.0018 under MuonAdamW; the
AdamW difference was uncertain and the Muon difference was positive on this
panel. E481 later compared a bounded sine candidate with a matched-center
cubic candidate across three seeds and two optimizers. Sine and cubic usually
moved together relative to baseline, while their difference was smaller and
seed/optimizer dependent. This keeps local slope/amplitude as live
explanations; it does not establish that a linear or cubic replacement is
better. See [E473](activation_mechanism/REPORT.md) and
[E481](recurrent_activation_curvature/REPORT.md).

**Recurrent candidate perturbations merit transfer tests, not deployment
claims.** In E471/E475, positive alpha helped the H32 Muon recipe at all three
seeds, while AdamW effects varied in sign. E478 moved the bounded candidate
to a two-layer H64 topology: alpha 0.1 had positive point estimates at each
of three seeds (+0.00270, +0.01067, +0.00448); two of three conditional
component intervals excluded zero. E479's learned alpha effects were small
and changed with seed and optimizer. E481's matched cubic controls show that
the H64 result is not yet specific to sine curvature. All used the exposed
panel. See [E471/E475](recurrent_activation/THREE_SEED_REPORT.md),
[E478](recurrent_activation_scale/REPORT.md), and
[E479](recurrent_activation_adaptive/REPORT.md).

**Momentum changes the context but has no universal setting.** E483 found
positive sine-minus-baseline intervals in all nine seed-by-Muon-momentum
comparisons. The momentum ranking and shape-by-momentum interaction changed
sign across the three seeds. This is a consistent fixed-panel shape contrast
inside the tested recipes, not proof that momentum caused the gain or that
one beta is best. See [E483](activation_momentum/REPORT.md).

**A large one-seed schedule effect did not replicate.** E476's warmup/cosine
schedule beat a constant-at-equal-arithmetic-mean-LR control by +0.0673 and
+0.0727 for the two AdamW activation arms; the Muon intervals crossed zero.
E480 held the protocol fixed at two new seeds. All eight primary
warmup/cosine-minus-constant-mean point estimates were negative, with six
intervals below zero. Equal summed learning rate does not imply equal
optimizer-update area, but the sign reversal is enough to reject a
seed-robust schedule claim from E476 alone. Preserve both directions; do not
select only the first seed. See [E476](activation_schedule/REPORT.md) and
[E480](activation_schedule_replication/THREE_SEED_REPORT.md).

**Head mixtures and context routing remain unresolved.** E477's fixed mixed
head and dropout effects changed with seed. E484 tested fixed, learned-static,
recurrent, projection, and self-context mixtures. Recurrent-minus-static
intervals crossed zero in both seeds, recurrent-minus-self reversed sign,
and fixed-mixture-minus-GELU intervals crossed zero in both seeds. The small
differences among gate variants do not show that context adds useful
information. E488 extended the four arms to three continuous epochs over
three seeds and retained all 24 saved snapshots. At epoch 3, fixed-mixture
minus GELU was +0.0089295, +0.0043077, and +0.0086897; two conditional
component intervals excluded zero and one crossed it. Recurrent-context minus
static-mixture was +0.0000838, -0.0000127, and +0.0001658, with all three
intervals crossing zero. At epoch 1, all three recurrent-minus-static points
were negative with intervals excluding zero. The epoch-3-minus-epoch-1
duration interaction was positive for all seeds, with only one of three global
intervals excluding zero. Thus the larger fixed-mixture effect is consistent
across these runs, while the extra recurrent gate contributes very little on
this panel; this does not show that context adds information or predict public
transfer. See [E477](activation_head_mixture/REPORT.md),
[E484](activation_context_mixture/REPORT.md), and the [E488 report](context_mixture_duration/REPORT.md).

**Frozen-weight swaps show training/inference dependence, not pure optimizer
causality.** E486 rescored 18 existing E483 checkpoints under both baseline
and sine recurrence equations. All nine training-by-inference interaction
intervals were positive, while only one of nine inference-only swap intervals
excluded zero. The diagonal cells repeat E483, so they are consistency checks
rather than new fits. Changing the recurrence at inference also changes hidden
states, output scale, and clipping; the interaction cannot isolate
optimization or prove a representation mechanism. See
[E486](activation_counterfactual/REPORT.md).

**Hardware replication exposed numerical sensitivity.** E482 reran the six
E481 recipes at the same seed on T4 workers. The cubic-minus-sine Muon
interval barely excluded zero on T4, while the same-seed local interval
crossed zero. This is hardware/optimizer numerical sensitivity, not a new
seed or independent market confirmation. See
[E482](activation_t4_replication_v2/REPORT.md).

The small sequence-component side study also found configuration dependence:
spline heads helped some memory configurations, while attention ranged from
competitive to substantially worse depending on the state model. Those
single-seed exposed-panel results motivate matched capacity and function
controls; they do not establish that splines or attention transfer to the
strong incumbent. See [sequence-component results](sequence_components/REPORT.md).

## Literature context

The primary papers already collected in
[the activation interpretation note](ACTIVATION_LITERATURE_INTERPRETATION.md)
are useful motivation, not external confirmation of these results:

- [Mining Generalizable Activation Functions](https://arxiv.org/abs/2602.05688)
  motivates searching activation shapes against generalization behavior. It
  does not predict this financial sequence setting or identify an optimizer.
- [Activate or Not: Learning Customized Activation (CVPR 2021)](https://arxiv.org/abs/2009.04759)
  motivates conditional activation switches. E484 is a related comparison,
  not a reproduction of ACON.

The combined evidence weakens a simple story that periodic activation shape
alone explains the pilot gains. Placement, optimizer, schedule, local slope,
and training/inference recurrence interact. The studies do not establish a
single cause or a robust winner.

## Framework, resource, and incident lessons

The most reliable results came from pinning source, split, raw input, common
initialization, optimizer recipe, and runner attempt before independent
scoring. The official global metric and target slices must use the exact
selected rows and masks, and raw predictions must be checked for finiteness
before clipping. Several early grader versions failed against real receipts
because they assumed artifact fields or path bases that the worker did not
produce. Those failures were retained and repaired in new versions before
test scoring; they were not model failures. The E477 mask/label alignment
repair is a concrete reminder that equal array shapes and synthetic tests do
not replace an actual sealed-receipt check.

Numerical parity is part of the experiment contract. E485's custom
recurrent/optimizer path failed its strict update comparison and its six
trajectories were not admitted. E487 then replayed identical saved gradients
and found that BF16 orthogonalization failed the registered native/custom
update gates while FP32 passed. That is evidence about a one-step numerical
implementation, not prediction quality. E489 therefore needs its own
all-seed FP32 recurrence and clipped-update probe; the E487 result alone does
not authorize it.

Transport needs the same testing discipline as training. The first E482
remote worker did not receive its `--output` argument through the orchestration
entrypoint and exited before GPU work; the versioned correction completed the
six T4 fits. E489's v1 probe likewise omitted a Python import directory and
failed before CUDA or cohort loading. The v2 import repair has only passed a
limited `--help` regression and has not been probed. A successful CLI help,
synthetic fixture, or source review is not a GPU receipt.

Resource enforcement must be real and measured. Setting `WNN_MEM_MAX` without
invoking `scripts/limited` does not impose a hard memory ceiling; the E489
clarification records this for its first preflight command, and the canonical
preflight was rerun through the systemd cgroup wrapper with a 1 CPU / 2 GiB
cap. Small H64/H128 fits generally used well under 2 GiB and below 1 GiB of
observed GPU memory, but these numbers do not bound the larger production
model, full-data preprocessing, CPU inference, or packaging cost. Honor the
one-GPU-worker and parent RAM/VRAM budgets, record OOM counters and teardown,
and reserve enough runtime for source validation and cleanup. Admission after
the preregistered cutoff is not a useful partial pass: E489's first controller
admission was recorded at 07:00:32.527, so no six-trajectory study began.

## Next window

1. **Re-review and probe E489 v2 first.** Bind its exact entrypoint, actor,
   receipt, and all parent hashes. Repeat the actual B4×256 native parity and
   strict update-delta checks at all three seeds; time both continuous chunks
   for all six cells including input transfer. Use actual provider-charged
   probe time in the full-study budget. Start the six continuous FP32 H128
   trajectories only if the strict gates, memory limits, timing projection,
   and cutoff all pass.
2. **Transfer only after the strongest training procedure is represented.**
   A useful larger study should incorporate the complete H256 teacher/EMA/head
   procedure, not just copy a small GRU activation into a different recipe.
   Keep a strong baseline and matched controls, use component-separated
   evaluation with less-exposed evidence, and preserve independent fit seeds.
3. **Measure deployability before building a ZIP.** Establish native CPU
   inference cost and parity against the Python model, then test the scorer's
   ABI and runtime limits on the exact artifact. A GPU metric gain has no
   submission value if the one-CPU inference path is too slow or numerically
   different.
4. **Spend the next budget on information gain.** Prefer one experiment that
   separates the remaining slope/curvature, optimizer, or schedule hypotheses
   on stronger data over another large pilot grid on the same exposed panel.
   Keep all negative and sign-reversing replications visible, and do not make
   a public forecast until the transfer and packaging evidence supports one.

No additional fits, sweeps, or public-score claims are proposed in this
retrospective.
