# Meta-trained causal online adaptation for a GRU teacher

**Status:** draft implementation and synthetic shape profiler only. No B300
allocation or profile is recorded; no task-data fit, student distillation,
score, or runtime measurement exists. `launch_ready` remains false until the
trainer and data bindings below are completed and an actual B300 profile passes.

**Lineage:** `H-ONLINE-META-TTA-GRU-B300-20261018-R1` /
`E-ONLINE-META-TTA-GRU-B300-20261018-R1`, teacher-to-student connection.
Source base: `main` `24e9001570de158db9a5031e632dbb0b4477145f`,
`research-team` `da90e68e13e8af4f95856f99c4c41a3fdb7cc315`.
**Draft PR:** [#20](https://github.com/VladKochetov007/WunderFunder/pull/20)
targets `research-team`; its branch was first published at
`d987b0d667c8f9bb789c8168700c02a4756f1d01`. The independently reviewed
starter implementation is commit `056a7bd2e17b7fa39dd4f07726dd8e73e476a79d`.

## Question

Can a GRU learn, through an outer task loss, when a small input-only online
update helps its next-block target predictions? If so, can a conventional GRU
student distill that behavior without online gradient updates at serving?

This is intentionally separate from [causal innovation features
(PR #14)](https://github.com/VladKochetov007/WunderFunder/pull/14), the
[Kalman smoother study
(PR #17)](https://github.com/VladKochetov007/WunderFunder/pull/17), and
[static next-input auxiliary study
(PR #18)](https://github.com/VladKochetov007/WunderFunder/pull/18). PR #14
adds fixed causal residual features; PR #17 adds a causal statistical state
estimate; PR #18 uses next-input prediction as a training auxiliary. This
proposal uses the next-input loss to update per-sequence fast weights online,
then learns the update initialization through task loss on later query blocks.
It is a meta-learning interaction with PR #18, not an independent auxiliary
signal.

## Causal update contract

The model receives the existing 112 raw columns plus 99 causal event features
(211 total). The online decoder predicts standardized raw-112 inputs using the
first 112 entries of the same immutable `model.mean`/`model.std` buffers fitted
by the canonical model on that cell's fitting data. The adapter does not fit or
change normalization statistics. Reject and count nonfinite values at data
ingress; do not silently replace them. The update does not read task labels,
scoring masks, validation targets, or future rows. Keep autograd enabled while
computing the inner update, including during teacher rollouts.

1. Reset GRU state, fast weights, event-feature history, pending block buffer,
   and counters when the sequence ID changes. The fit-only scaler stays fixed.
2. Process warm-up rows 0–98. Before predicting row 99, update the rank-16 fast
   adapter using only transitions `h[t] -> x[t+1]` for `t=0..97`; row 98 has
   no in-support successor. This update therefore uses inputs through row 98.
3. Predict the next 256 rows with the current fast weights. Once the full block
   has arrived and its predictions are fixed, update from its internal pairs
   only (`t` through block-end-minus-one). Do not revise predictions already
   returned, and do not include a transition crossing into the next block.
4. Carry the fast state across later blocks of this sequence only. Use the same
   order during training teacher rollouts and inference replays. Treat the
   final remainder shorter than 256 rows as one partial block and adapt from
   its internal transitions only after its predictions are fixed.

`fast_adapter.py` implements the shared rank-16 residual transform and the
input-only update. The decoder consumes the transformed recurrent state, so
its loss has a gradient with respect to the fast weights. It uses one fixed
SGD update per observed support block (`lr=1e-3`, per-sequence gradient norm
clip 1.0, proximal penalty 0.01); these are frozen proposal settings, not tuned
or validated. The meta-training proposal differentiates through inner updates
within each 2,000-row training chunk. The 99-row support prefix plus 1,901
query rows form the first exact chunk: seven complete 256-row blocks and 109
rows of the eighth. At a TBPTT boundary, carry GRU/event states, fast-weight
values, and pending block values but detach their graphs. If a 256-row block
straddles the boundary, finish it before adapting; buffered pre-boundary
activations are detached and post-boundary activations remain connected to the
current chunk. The required real two-chunk profile must exercise this path.
The GRU, task head, adapter initializer, and input decoder are outer-trainable;
only per-sequence adapter weights change in the inner update. Trainer
integration is not implemented.

## Frozen comparison

All six arms use fresh, fold-clean models, the same source-qualified fitting
membership and seed in each paired cell, six epochs, fixed epoch-5 checkpoint,
strict FP32, batch 8, 2,000-row chunks, AdamW (`lr=1e-4`, `betas=(0.9,0.999)`,
`weight_decay=0.01`), OneCycle warm-up fraction 0.05, and gradient clipping 1.
These settings follow the current Rank-1 cold H512×1 recipe. Teacher outer
task loss and student loss use the canonical project implementation. Fixed
next-input auxiliary weight is 0.1 in both auxiliary teacher arms; no per-arm
loss-weight search is planned in this comparison.

| Stage | Arm | Model / treatment |
|---|---|---|
| Teacher | T0 | H512×1 supervised-only reference |
| Teacher | T1 | H512×1 plus rank-16 adapter and next-input auxiliary; no online updates |
| Teacher | T2 | Same as T1; causal 256-row fast updates and meta-gradient through each update |
| Student | S0 | H256×2 supervised-only reference |
| Student | S1 | H256×2 distilled from the matched static T1 teacher |
| Student | S2 | H256×2 distilled from the matched adapted T2 teacher |

All student arms use the same explicit weighted-MSE distillation objective.
With `y_c=clip(y,-2,2)` and `t_c=clip(teacher,-2,2)`, S1/S2 use
`target=0.5*y_c+0.5*t_c`, weights `abs(y_c)*main_valid`, and the weighted
two-output MSE normalized by the sum of weights in each chunk. S0 uses the same
loss with target `y_c`. The recipe disables bootstrap, reanchor, cache scaling,
and family targets. This is the existing trainer's `--distill-alpha 0.5`
target-blend path. Six-epoch schedule and clean ancestry are matched. Teacher
outputs for distillation are generated by a causal 20,000-row replay with the
exact schedule above. The
serving candidate is S2; direct online adaptation at inference is out of scope
until its CPU, memory, package-size, and deterministic reset behavior are
measured separately.

The teacher phase is 3 arms × 30 paired cells = 90 fits. The student phase is
another 3 × 30 = 90 fits, for 180 total if both phases are run. The student
phase is required to establish competition value; teacher quality alone is not
a submission claim. One fit per assigned B300 is the initial execution shape;
use up to eight independent cards only after actual topology and per-card
profiles are recorded. DDP is not proposed for these fits.

## Evaluation and uncertainty

Use the unchanged `global_wp_clip2_v1` metric and its fixed scored/prediction
mask. The primary teacher contrast is T2−T1; the primary transfer contrast is
S2−S1. S2−S0 is the practical endpoint contrast; T1−T0 and S1−S0 are
secondary. Compute each cell's official WP on its complete source-qualified
outer components. The primary statistic is the equal-fold mean of paired seed
means, with folds weighted 1/4 each and seeds 1001–1008 in F0/F1 and 1001–1007
in F2/F3. Report fold means, seed variation, and per-source/sequence-age
descriptives separately.

Use 10,000 paired seed-identity bootstrap draws: sample eight identities with
replacement from 1001–1008 and share multiplicities across folds; for F2/F3,
omit 1008 and renormalize over the available seven cells. Redraw if a fold gets
zero available multiplicity. Use a separate 10,000 paired source-component
bootstrap within each fold, with multiplicities shared across arms and seeds.
Intervals condition on four previously inspected folds; they do not quantify
new-regime or public-transfer uncertainty. Apply Holm correction within each
stage across exactly three pairwise contrasts: T2−T1 (primary), T2−T0,
T1−T0; and S2−S1 (primary), S2−S0, S1−S0. A provisional `+0.001` absolute WP improvement is a useful
priority reference, not a kill threshold. Zero-crossing intervals, lower
teacher scores, or an early weak run justify targeted follow-up when the
mechanism or student transfer remains unresolved.

## Leakage, coverage, and risks

Task labels are used only by the outer supervised loss on fitting components.
The inner update and all evaluation adaptation use observed inputs only. The
raw112 auxiliary uses `model.mean[:112]` and `model.std[:112]`, frozen across
each episode. Normalization, teacher initialization, distillation targets, and any
preprocessing statistics must be fitting-fold clean. No evaluated component
may appear in its teacher or student fitting ancestry. Do not fit full-sequence
statistics. Process all rows in order, including warm-up and unscored rows.
All current folds are exposed; this comparison is robustness evidence, not a
fresh holdout or proof of future-regime coverage.

The central risk is that self-supervised next-input gradients do not align with
the target task, or help only transiently. Record inner input loss,
task-gradient/inner-gradient cosine on fitting data, update norm, clipping
fraction, and official WP by sequence age and source. A falling reconstruction
loss is not evidence of task improvement. A teacher gain that does not survive
S2−S1 is not submission value. The full risk register and commands are in
[`PR.md`](PR.md) and [`recipe.json`](recipe.json).

## Code and execution readiness

- `INITIAL_CARD.md`: original hypothesis-card text committed before the
  independent mechanism critique, preserved with a dated post-review addendum.
- `fast_adapter.py`: research primitive, not wired into the canonical trainer,
  evaluator, prediction exporter, or student cache builder.
- `profile_episode.py`: synthetic strict-FP32 shape and second-order-autograd
  probe. Inputs and task labels are random and independent, so its outputs are
  engineering diagnostics only, not task-quality or recipe-fidelity evidence.
- No real-data loader binding, per-cell trainer command, fold-clean teacher
  registry, or reporting integration is included. Do not start the 180-fit
  matrix until those are completed and source/data hashes are frozen.

From the repository root on an assigned B300 worker, after installing the
project's pinned environment, the currently implemented profile path is:

```bash
PYTHONPATH=src python -m research.experiment_drafts.online_meta_tta_gru_b300.profile_episode \
  --hidden 512 --layers 1 --rank 16 --batch-size 8 \
  --initial-support-rows 99 --block-rows 256 --tbptt-rows 2000 --device cuda
```

This synthetic profile uses exactly 2,000 rows: 99 support rows and 1,901
query rows (seven 256-row blocks and 109 rows of the eighth). Its outer loss
uses the project's clipped-label weighted-MSE reduction, while the inner
update uses the proposed SmoothL1 input reconstruction loss. The synthetic
labels are independent of the inputs, so neither loss value represents model
quality. The probe checks tensor shape, second-order autograd, and rough memory
behavior only; it does not reproduce the full recipe or certify B300 capacity.
It records device VRAM and process peak RSS; the latter does not replace
aggregate host/cgroup memory. Record actual GPU UUID/topology, driver,
CUDA/PyTorch versions, aggregate host and device memory, elapsed
forward/backward/optimizer times, finite-gradient status, and exact stdout.
The synthetic path does not build real event features. Before fits, the first
real-data profile must exercise the production loader, causal event-state
integration, and exact meta-update loop across a 2,000-row boundary, including
completion of a split 256-row block.

See [`PR.md`](PR.md) for the 11-section handoff, and the unified
[idea](../../unified/ideas/ONLINE_META_TTA_GRU_B300_20261018_R1.md) and
[literature source](../../unified/sources/ONLINE_META_TTA_GRU_B300_20261018_R1.md)
records. The draft PR is a proposal for the B300 team; no fit dispatch is
implied.

Two independent static implementation reviews returned ITERATE, then a final
static review of commit `056a7bd2e17b7fa39dd4f07726dd8e73e476a79d` returned
PASS for the starter packet. Their findings and author corrections are recorded in the
[review note](../../unified/reviews/ONLINE_META_TTA_IMPLEMENTATION_REVIEW_20261018_R1.md).
The packet now fixes the 2,000-row synthetic shape, standardized-input
contract, cross-chunk update semantics, student objective, profiler scope, and
uncertainty procedure. Trainer integration, actual B300 profiling, and all
scientific fits remain pending; the static PASS does not authorize launch.
