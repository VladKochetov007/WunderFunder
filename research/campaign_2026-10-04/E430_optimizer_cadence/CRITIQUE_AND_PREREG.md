# E430: optimizer cadence, loss-mass pooling and recurrent credit

**Draft only. No fit or score has been admitted or run.** The three staged full-recipe specs use
fold 0, seed 3, fixed epoch 3, and the same E423 source data, teachers, architecture, objective,
optimizer, AMP mode and resource caps. The historical E423 run is context, not the E430 control.

## What the trainer does

The combo recipe fits four epochs of a two-layer GRU (`H=256`, batch 32) on the same
train-plus-development population with `wmse`, structural and next-event auxiliary heads,
propensity weighting, the two cached since teachers, `alpha=0.5`, and stopped teacher bootstrap
at horizon 8. E423 kept the 1,000-row chunk/update cadence and only changed the EMA clock. Its
reported row-clock arm used an effective decay of 0.999499875 per 1,000 rows, while its fresh
control used 0.999 per attempted update; the row-clock arm scored 0.000574 lower in global WP,
with an interval crossing zero. E423 therefore does not test accumulation or longer recurrent
credit.

The relevant code path has two distinct regimes. The ordinary path performs one optimizer step
per `--chunk`; its hidden value is carried to the next chunk after detaching the graph. The
controlled macro path activates when `--update-rows` is set. It computes one denominator from the
whole update window, forwards each `--chunk` micro-window, backpropagates each contribution while
the parameters are still unchanged, cuts hidden-state gradients at `--detach-rows`, and performs
one optimizer/scheduler/scaler/EMA attempt after the macro window. With `chunk=1000`,
`update-rows=2000`, `detach-rows=1000`, the state values cross the internal boundary but their
gradient does not. `chunk=2000` instead keeps the graph connected across those 2,000 rows.

The main loss weights remain based on the real clipped targets, `abs(clip(y)) * main_valid`; the
blended teacher target does not change that mass. `distilled_macro_losses` sums the micro-window
numerators and divides every contribution by the complete macro-window mass. This is the right
pooled weighted objective. Averaging two separately normalized chunk losses is a different
estimator whenever their weight masses differ. The difference is systematic at the first boundary:
with 99 warmup rows, the first 1,000-row window has 901 supervised rows and the next has 1,000,
before propensity, since and target-magnitude weights vary further.

For bootstrap, the trainer substitutes teacher prediction at `t+8` only when the label run remains
unchanged strictly beyond that row and the row is in the teacher sequence; otherwise it retains
the observed label. That is a denoising hypothesis, not an unbiased target estimate. Cached
teacher predictions are averaged before use and must remain bound to fold 0. The EMA consistency
teacher is separate: it is called on each micro-window and its recurrent state is detached between
micro-windows.

All three specs set `--ema-reference-rows 1000`. This retains decay 0.999 for a 1,000-row update
and uses `0.999**2` for a 2,000-row update, matching the nominal EMA decay per processed row.
That removes the simple row-clock mismatch from the cadence comparison, but cannot make teacher
trajectories identical: the 2,000-row candidate still observes and updates its student less often.
The trainer also advances the learning-rate scheduler and EMA on every attempted step even when
GradScaler skips the optimizer update; E423 recorded 12 skipped attempts among 29,840 (EMA still
advanced 29,840 times). E430 must report attempts, successful updates, skips and EMA updates for
each epoch and arm.

## Fixed three-arm contrast

The staged recipes are:

- `chunk1000`: ordinary 1,000-row update and detach cadence.
- `accum2000_trunc1000`: two 1,000-row forward/backward contributions, one 2,000-row optimizer
  update, with a 1,000-row gradient detach.
- `chunk2000_connected`: one 2,000-row connected recurrent graph and update.

The primary practical contrast is `accum2000_trunc1000` versus `chunk1000`: it asks what happens
when the same stream rows feed fewer optimizer steps while credit stays truncated at 1,000 rows.
That contrast includes the intended consequences of cadence—optimizer moments, weight-decay
applications, gradient clipping and schedule granularity—and must be described as an operational
cadence change, not a pure estimator-only effect. The `accum2000_trunc1000` versus
`chunk2000_connected` comparison matches macro-step count, nominal EMA decay per row and pooled
loss mass; its main intended distinction is cross-microchunk gradient credit, with microkernel
segmentation as a smaller numerical/performance difference.

All arms use the same order seed and frozen epoch 3. `--no-eval-during-fit` withholds the clean
evaluation fold from fitting and checkpoint choice. After fits, compare common-FP32 exports on the
same fold-0 evaluation groups and paired components, with global WP primary and class/t0/t1 plus
legacy sequence means diagnostic. Do not promote from a single exposed fold/seed. The connected
2,000-row arm has the highest activation-memory risk: E423's 1,000-row recipe peaked at about
4.52 GB observed VRAM against the 4.8 GB hard stop. Preserve the 4.2 GB allocator cap and 4.8 GB
stop; any resource stop is a failed run retained as evidence, not a reason to raise the limit.

## Synthetic pooling discriminator

`check_pooling.py` has no model or dataset dependency. It uses the first-window warmup count plus
unequal synthetic row weights to distinguish one pooled numerator/denominator from the mean of
separately normalized chunks. Its fixed example has masses 901 and 4,000, local losses 0.1 and 0.8,
and therefore pooled loss `(901*0.1 + 4000*0.8)/(901+4000) ≈ 0.6713`, versus the equal chunk mean
0.45. E430's trainer path is expected to implement the former for an accumulated update.

## Remaining interpretation limits

The 1,000-to-2,000 primary contrast changes the optimizer's number of applications, the schedule's
step grid, clipping frequency and weight-decay applications along with accumulation. EMA's row
clock normalizes decay but not its dependence on each newly updated student. AMP skips also leave
the scheduler and EMA clocks advancing. The 2,000-row connected arm may exceed the established
VRAM envelope. These constraints make the three-arm matrix useful for mechanism discrimination,
but a single fold/seed cannot estimate a stable generalization effect or a public-score change.
