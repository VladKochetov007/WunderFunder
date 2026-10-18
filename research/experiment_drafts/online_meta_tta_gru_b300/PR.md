# [Experiment] E-ONLINE-META-TTA-GRU-B300-20261018-R1: meta-trained causal online adaptation

## 1. Status, lineage and owner

**Status:** proposal with a first research primitive and synthetic profiler.
No actual B300 profile, task-data fit, task score, student distillation, or
competition runtime has been measured. The packet is **not launch-ready**.
Hypothesis `H-ONLINE-META-TTA-GRU-B300-20261018-R1`; experiment
`E-ONLINE-META-TTA-GRU-B300-20261018-R1`. Mechanism family: teacher/student
connection through test-time learning. Parent is cold H512×1; it does not
inherit a Kalman or auxiliary-teacher checkpoint.

The branch is based on `research-team`
`da90e68e13e8af4f95856f99c4c41a3fdb7cc315`, whose shared-policy `main` base is
`24e9001570de158db9a5031e632dbb0b4477145f`. Code is in
[`fast_adapter.py`](fast_adapter.py) and the no-data GPU shape profiler is
[`profile_episode.py`](profile_episode.py). The full packet is in this folder's
[`README`](README.md) and [`recipe`](recipe.json).

This is distinct from [PR #14](https://github.com/VladKochetov007/WunderFunder/pull/14)
(fixed causal innovation features), [PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17)
(causal Kalman state features), and [PR #18](https://github.com/VladKochetov007/WunderFunder/pull/18)
(static next-input self-supervision). PR #14 constructs observed-input residual
features without online parameter updates. PR #17 supplies a causal filter
state. PR #18 uses next-input loss as a static training auxiliary. This proposal
uses next-input loss to update per-sequence fast weights and meta-trains those
updates through subsequent task-query loss. It is an interaction with PR #18's
auxiliary, not a new independent prediction target.

## 2. Why it should work

The callback receives 20,000 ordered inputs per sequence but no target labels
or scoring mask. Target-error adaptation is therefore unavailable. A
next-observed-input objective supplies a causal update signal; an outer task
loss can train the update initialization so it adapts only when the change
helps target prediction. This could address within-sequence regime changes
while preserving the normal GRU backbone. The falsifying observation is
lower input reconstruction loss without better next-block official WP, or an
adapted teacher whose value disappears in the student.

The decoder reads the fast-adapter-transformed recurrent state. If it bypassed
the adapter, the input loss would have zero gradient with respect to the fast
weights. The task loss is never used in the inner update. It appears only as
the outer training objective on fitting groups. Measure input/task-gradient
cosine on fitting data as a mechanism diagnostic; it is not a substitute for
the score comparison.

## 3. Compute advantage

If the team assigns eight B300 cards, independent paired seed/fold fits could
run in parallel; current card availability and topology have not been verified.
DDP is not proposed for these recurrent models. The extra calculation is a
rank-16 update every 256 observed rows and a second-order meta-gradient through
the updates within each 2,000-row TBPTT chunk. Whether that fits at batch 8 and
improves optimization is unknown until actual-device profiling. Use strict
FP32. Do not infer speed from advertised B300 specifications or the synthetic
profiler.

Competition value must arrive through the H256×2 student, not teacher-only
quality. Stage 1 compares three teachers (90 fits total); Stage 2 compares
three matched students (another 90 fits). Both stages are included because a
teacher gain does not guarantee transfer. Do not drop the student phase just
because the first small set of teacher runs looks weak. No direct adaptive
inference submission is proposed under the one-CPU runtime limit.

## 4. Frozen comparison

All teacher arms use cold H512×1 models on the existing 211 inputs (raw112 plus
causal event99). All student arms use cold H256×2 models. For each fold/seed,
arms share fitting membership, raw normalization, input construction,
optimizer/update budget, checkpoint rule, and fold-clean ancestry. Six epochs;
fixed `model_ep5`; batch 8; 2,000-row chunks; AdamW LR `1e-4`, betas
`(0.9, 0.999)`, weight decay `0.01`; OneCycle warm-up fraction `0.05` over the
fixed update budget; task gradient clipping `1.0`; strict FP32, no autocast,
GradScaler, BF16/FP16, or TF32. These settings follow the current cold H512×1
Rank-1 recipe. The worker must pin that recipe's exact trainer/config SHA
before fit admission.

For every teacher arm, the supervised task term is the canonical clipped-label
weighted MSE: `y_c=clip(y,-2,2)`, `w=abs(y_c)*main_valid`,
`sum(w*(prediction-y_c)^2)/max(sum(w),1e-8)`, per training chunk. T0 uses only
this term; T1 and T2 add `0.1 * mean SmoothL1` on standardized next raw inputs.
T2 additionally uses that same self-supervised loss to update fast weights.

Fast update: rank-16 residual adapter on recurrent state, one fixed SGD step
per observed support block at LR `1e-3`, per-sequence global adapter gradient
clip norm `1.0`, proximal coefficient `0.01`. Next-input loss is mean
SmoothL1 on raw112 standardized with the immutable first 112 entries of that
cell's canonical model `mean`/`std` buffers; there is no second or online
normalizer. Reject nonfinite inputs at data ingress and record the rejected
count; never replace them silently. Add this auxiliary loss with fixed
coefficient `0.1` to both T1 and T2 outer objectives. T2 also uses the same
input loss for its inner update; T1 does not update fast weights. No parameter
sweep is part of this comparison. The trainer caller must leave autograd enabled
around adapter updates, including during teacher rollouts; only the task
inference path can run without gradients.

| Stage | Arm | Treatment |
|---|---|---|
| Teacher | T0 | Supervised-only reference |
| Teacher | T1 | Same model plus rank-16 adapter and next-input auxiliary; no online updates |
| Teacher | T2 | Same as T1 plus causal fast updates and outer meta-gradient through updates |
| Student | S0 | Supervised-only H256×2 reference |
| Student | S1 | Distilled H256×2 from matched T1 teacher |
| Student | S2 | Distilled H256×2 from matched T2 teacher's causal rollout |

Freeze the student objective exactly. Let `y_c = clip(y, -2, 2)` and
`t_c = clip(teacher_prediction, -2, 2)`. For S1/S2 use
`target = 0.5*y_c + 0.5*t_c`, `w = abs(y_c) * main_valid`, and
`sum(w*(student_prediction-target)^2) / max(sum(w), 1e-8)` over the two main
outputs in each training chunk. S0 uses the same weighted-MSE reduction with
`target=y_c`. Disable bootstrap, reanchor, cache scaling, and family targets.
This is the existing trainer's clipped-label/teacher blend at
`--distill-alpha 0.5`, `--distill-boot-h 0`, `--distill-reanchor 0`,
`--distill-cache-scale 1`, and `--distill-family-weight 0`; its loss branch
uses weighted MSE. Match six epochs, optimizer, scheduler and fitting
membership across all student arms. Freeze teacher rollout and student cache
hashes per cell before fitting. Current code does not yet wire these recipes
into the canonical trainer or cache builder.

### Update schedule

Process rows 0–98 as warm-up. Before prediction at row 99, adapt from the
already-observed transitions `h[t] -> x[t+1]`, `t=0..97`; the row-98 hidden
state has no in-support successor. Predict each following 256-row block using
the current fast weights. After that block's predictions are fixed and all
its inputs have arrived, adapt from internal adjacent pairs only, excluding
the boundary pair that would consume the next block's first input. Carry fast
weights across later blocks in this sequence. Reset recurrent state, adapter,
event history and counters on every sequence change. Do not revise predictions
already returned.

For a 20,000-row sequence, treat the final 189 rows after the 99-row warm-up
and 77 complete 256-row blocks as one partial block. Apply the same predict,
then update ordering and use only its internal adjacent pairs.

During training, retain the ordinary row schedule (support rows 0–98, then
256-row query blocks) even when a block crosses a 2,000-row TBPTT boundary.
At every exact boundary, carry the GRU state, event-feature state, pending
partial-block input/hidden values, and fast-weight values while detaching their
graphs. When a split query block completes, compute its input-loss update from
the whole block: values buffered before the boundary stay detached, while
post-boundary values remain connected to the current chunk. Apply the update
only after all block predictions are fixed. This preserves test-time ordering
while explicitly truncating only the meta-gradient at the TBPTT boundary. For
the first chunk, 99 support rows plus 1,901 query rows make exactly 2,000 rows;
the query region contains seven full 256-row blocks and 109 rows of the eighth.
The real two-chunk trainer profile must also exercise completion of that split
block and its update. The fixed fold scaler is stateless and never reset.

The GRU, task head, adapter initialization, and decoder are outer-trainable;
only the per-sequence adapter weights change in the inner step. At `seq_ix`
changes, reset GRU state, event state, fast weights, pending block buffers, and
block counters; retain the same immutable fit-only scaler. This is a specified
implementation requirement, not behavior wired into this PR yet.

## 5. Thirty paired runs

Each of the six arms uses all 30 cells: F0/F1 seeds 1001–1008 and F2/F3 seeds
1001–1007. Candidate and control are paired on identical source-qualified
fitting/evaluation components and seed identities. This is 90 teacher fits and
90 student fits if both stages complete. A synthetic shape profile is an
engineering gate, not a scientific cell.

Use clean per-cell teacher, initializer, normalizer and preprocessing
ancestry. No outer-fold group or target may enter fitting, selection, or the
distillation ancestry for its evaluation. Do not repurpose a cross-fold
checkpoint under a new fold flag. All four folds have prior exposure; label
results as robustness evidence, not pristine holdout confirmation.

## 6. Validation and significance

Use unchanged `global_wp_clip2_v1`, its fixed prediction/scored mask,
clipping, and weighting. Grade complete outer-fold components; aggregate
official sums/counts correctly. Primary teacher contrast is T2−T1; primary
transfer contrast is S2−S1; practical student endpoint is S2−S0. Also report
T1−T0 and S1−S0. The primary statistic is the equal-weight mean over four
folds of paired seed means, not a row-IID test or seed-averaged prediction
endpoint.

Report all fold means and fitting-seed variability. For seed uncertainty, draw
10,000 bootstrap samples of eight seed identities with replacement from
1001–1008, using shared multiplicities across folds. In F2/F3, omit identity
1008 and renormalize sampled multiplicities over the available seven cells;
redraw a sample if a fold receives zero total available multiplicity. For
evaluation-unit uncertainty,
use a separate 10,000 paired source-component bootstrap within each fold,
sharing multiplicities across arms and seeds. Report the intervals separately;
they condition on exposed folds and fixed ancestry. Apply Holm correction
within each stage across exactly three pairwise contrasts: teacher T2−T1
(primary), T2−T0, and T1−T0; student S2−S1 (primary), S2−S0, and S1−S0.
A confidence interval crossing zero is inconclusive and
may justify a targeted follow-up. Thirty cells do not by themselves guarantee
power or calibrated coverage.

## 7. Leakage and representativeness

The callback has no target or mask access. Inner updates use only causal
observed inputs; outer task labels are fitting-only. The raw112 auxiliary
target uses `model.mean[:112]` and `model.std[:112]` from the same
fold-training model, frozen across the episode; the adapter does not calculate
or update normalization statistics. All means/scales and
teacher/student ancestors are fitted within each cell's training membership.
The 99-row warm-up, block schedule, event-feature history and sequence reset
must match at train, teacher-rollout and inference replay. Do not normalize
from a complete sequence or use held-out query targets in the adaptation
path. Consume unscored inputs as usual because input-only adaptation cannot
depend on the score mask.

Before a real fit, bind source-qualified fold manifests, duplicate/source
membership checks, data hashes, the real producer format and a bounded data
reader profile. The local train-Parquet footer attempt previously exceeded its
memory cap; this PR does not read local Parquet. Existing folds are exposed,
and anonymous chronology limits claims about unseen future regimes.

## 8. Decision-changing result

`+0.001` absolute WP over a matched endpoint is a provisional useful-effect
reference, not a universal rejection threshold. Teacher success requires T2
to improve on T1 with no severe fold collapse and acceptable profiled cost.
Submission value additionally requires S2 to improve on S1 and to be useful
against S0. If input reconstruction improves but WP does not, retain that as
mechanism evidence and test a bounded alternative objective or timescale if
the gradient-alignment diagnostics justify it. If teacher improvement does
not transfer, the online teacher may still be a representation or ensemble
component, but no student claim follows. Uncertainty or a weak early run does
not close the online-learning direction. Any follow-up changes one declared
factor and keeps its own ancestry and comparison.

## 9. How to run

Inspect actual hardware and environment on the assigned worker first:

```bash
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
nvidia-smi topo -m
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
```

With the pinned project environment and experiment branch checked out, the
only currently implemented command is a synthetic strict-FP32 shape and
second-order-autograd probe:

```bash
PYTHONPATH=src python -m research.experiment_drafts.online_meta_tta_gru_b300.profile_episode \
  --hidden 512 --layers 1 --rank 16 --batch-size 8 \
  --initial-support-rows 99 --block-rows 256 --tbptt-rows 2000 --device cuda
```

It performs no data load and emits a JSON record with device, elapsed episode
time, process peak RSS, peak allocated/reserved VRAM, mean per-update gradient
diagnostics, finite-gradient status, and synthetic losses. The synthetic raw
inputs align with the model's first 112 columns, but task labels are random and
independent of inputs. Its outer task loss matches the project's clipped-label
weighted-MSE reduction; its inner update uses the proposed SmoothL1 input
reconstruction loss. This checks shape and second-order autograd behavior, not
the full recipe, data path, or model quality. Process RSS is not aggregate
host/cgroup memory. Save output on physical worker storage with the command and
environment manifest. The synthetic stream bypasses the production data loader
and event-feature builder; this is not an actual-B300 profile, fit command, or
throughput claim.

**Missing before any scientific fits:** integrate the adapter and causal block
schedule, including cross-chunk partial blocks and event-state handling, into
the modular trainer; bind actual fold/data manifests and fit-only normalizers;
implement matched T0/T1/T2 objectives and clean T1/T2
teacher cache construction for S1/S2; profile the real loader and complete
second-order 2,000-row and split-block updates at batch 8; and wire progress, checkpoint/resume,
prediction export, independent grading, and artifact reporting. Freeze the
exact source/config/data hashes and issue/cell ownership only after those
checks. Use one fit per B300 process. No DDP, queue manager, reduced precision,
quantization, paid rental, or billing change is part of this proposal.

## 10. Evidence to return

For every profile and scientific cell, return source/config/environment/data
and split hashes; fold/seed/arm/run ID; actual GPU UUID/topology and software;
all optimizer attempts and successful updates; per-block inner input loss,
outer task loss, per-group LR, preclip task and fast-gradient norms, clipping
fraction, proximal/update norms, nonfinite count; official train/validation WP
and row/component counts; loader, forward/backward, update, validation, export
and grading times; peak host/GPU memory and utilization when available;
checkpoint/resume and prediction hashes; and failures/exclusions.

Record gradient-cosine probes only on fitting membership. Preserve separate
evidence and weights archives with file/archive SHA256 and approved private
storage locations; do not put data or private rows in Git. Every valid result,
including negative and inconclusive results, receives an append-only H/E
ledger event and a unified result note.

## 11. Results and trace

There is no measured B300 profile, task-data result, teacher/student gain,
runtime, or leaderboard impact. The original hypothesis-card text was committed
before independent criticism and remains preserved with a dated addendum. The
first review returned ITERATE: it confirmed
the adapter-to-decoder gradient path and second-order query path by static
inspection, and identified the profile length, normalization contract,
cross-chunk state handling, exact student loss, bootstrap treatment,
multiplicity correction, and initial-card wording as gaps. The revised packet
records each correction and the still-open trainer/data/profile work. The code
remains an adapter/profile starter, not an integrated trainer.

A second static closure review verified the synthetic input alignment,
canonical weighted-MSE outer loss, per-update diagnostic aggregation, PR #14
distinction, and card addendum wording. It also returned ITERATE before
publication because the eight-card parallel plan sounded like confirmed
availability; the PR now makes it conditional on a team assignment and says
availability/topology remain unverified. The final static review of committed
tree `056a7bd2e17b7fa39dd4f07726dd8e73e476a79d` returned PASS for those
corrections. No tests, profiles, fits, or data reads were performed.

External source claims are limited to the following: [Sun et al., arXiv:1909.13231](https://arxiv.org/abs/1909.13231)
motivate self-supervised updates and discuss gradient alignment in a convex
analysis, with image-corruption experiments; [Kim et al., arXiv:2501.04970](https://arxiv.org/abs/2501.04970)
study time-series adaptation but use partially observed future labels that are
unavailable here; [Wu et al., arXiv:2602.00073](https://arxiv.org/abs/2602.00073)
report regime-dependent time-series and financial effects; [Tandon et al., arXiv:2512.23675](https://arxiv.org/abs/2512.23675)
meta-learn long-context self-supervised updates for language modeling. These
papers motivate a test; none establishes a Challenge2 gain.

The source note and idea record are linked from the unified
[sources](../../unified/sources/README.md) and
[ideas](../../unified/ideas/README.md) indexes. PR number, URL, final head SHA,
and any issue/cell claim are added after publication; no result or dispatch is
implied by opening this draft.
