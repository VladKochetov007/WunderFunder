# Training telemetry, callbacks and dynamics experiments (2026-10-09)

Root (Claude Code) note. Purpose: stop learning from final scores alone. Until now a run left only a final
checkpoint, epoch wall time, AMP-skip counts and an optional per-epoch gradient quantile summary
(`--grad-stats`). We could not say whether the 4-epoch budget saturates, whether a schedule helps by
cooling down or by annealing noise, whether any parameter family under-trains, or whether clipping bites.

## What each run records now (opt-in, default path unchanged)

Snapshot `continuous_execution_E529/h230_source_v5_trace` (= v4 + telemetry; mainline module
`src/wnn/training_trace.py`, test `tests/test_training_trace.py`, analysis `scripts/analyze_training_trace.py`).

| Flag | Output | Signal | Cost |
|---|---|---|---|
| `--trace-every N` | `trace.jsonl` kind=`step` | total loss, `loss_family` (distillation), lr and beta1 per param group, AMP loss scale, pre-clip total and per-family gradient norms (`gn_*`, captured between unscale and clip) | 1 host sync per 64 rows; negligible |
| `--trace-state-every M` | kind=`state` | weight norm `w_*`, update ratio `ur_*` = ‖Δw‖/‖w‖ since previous state row, optimizer second-moment RMS `v_rms_*`, momentum norm `m_norm_*` per family (`rnn_hh`, `rnn_ih`, `inp`, `head`, `bias`) | a parameter copy every M steps |
| `--intra-epoch-points K` | `intra_eval.jsonl`, `INTRA` log lines | evaluation (wp, t0, t1) K-1 extra times per epoch; **logging only, never used for selection** | one eval per point (about 1 min on fold1) |
| `--intra-snapshots` | `model_ep{E}p{B}.pt` | intermediate checkpoints for offline averaging and re-scoring | ~4 MB each |
| existing `--grad-stats` | history.json | per-epoch gradient quantiles and clip rate | already there |

Use together with `--no-eval-during-fit` removed (the epoch-end score also lands in `history.json`).
Selection stays `--checkpoint-policy fixed`, so these runs remain comparable with non-telemetry runs; the
only difference is that intra-epoch evaluation does not consume training RNG but does interleave GPU work.

Exposure: evaluation on the outer fold is monitoring of already-exposed folds. Any rule that *acts* on a
score (early stop, best-checkpoint, plateau LR) must use an inner dev split carved from the fit data, not the
outer fold.

## Queued now (all fold1, clean fit, combo recipe seed 3 unless stated)

Vast `refill_jobs/w5_*` (telemetry on): `tr_base`, `tr_wsd` (warmup-stable-1-sqrt cooldown), `tr_ep6` (6 epochs),
`tr_lr2` (lr 2e-3), `tr_lrh` (lr 5e-4), `tr_swa` (`--swa-frac 0.25`), `tr_b16` (batch 16).
Local `00d_h230-f1-s3-trace-fit` : H230 student with telemetry (warm-start distillation regime, includes `loss_family`).

## Ideas, hypotheses and the signal that decides them

| ID | Idea | Decided by | Status |
|---|---|---|---|
| H-297 | Fixed 4-epoch teacher budget is under- or over-trained | eval curve of `tr_base` and `tr_ep6`: slope of wp over the last epoch vs seed noise (±0.002) | queued |
| H-298 | Cooldown, not shape, creates the late gain (WSD vs onecycle) | `tr_wsd` vs `tr_base` curves; gain should appear only after decay starts | queued |
| H-299 | Parameter families train at very different relative speeds; per-family LR multipliers help | `ur_*` across families in `tr_base`, response to lr 2x and 0.5x | queued |
| H-300 | Gradient clipping at 1.0 rarely or often binds; adaptive percentile clipping (AutoClip) is better | fraction of steps with `gn_total` > 1 and its time profile | queued (analysis) |
| H-301 | Averaging intra-epoch snapshots (LAWA/SWA) beats the last checkpoint at equal cost | offline re-score of `model_ep*p*.pt` averages vs `tr_swa` online SWA | queued |
| H-302 | Critical batch size: batch 16 vs 32 differ in gradient-noise scale and final wp | `gn_total` variance and eval curve of `tr_b16` | queued |
| H-303 | In distillation the family loss share and its gradient interact with the main loss; weight 0.1 is arbitrary | `loss_family` vs `loss` and `gn_head` in the local trace run; informs alpha-0.5 and family-off arms | queued |
| H-304 | Within-run eval jitter between nearby intra points is a cheap estimate of evaluation noise, separating it from seed noise | variance of consecutive intra points around a plateau vs seed-to-seed spread | queued (analysis) |

Not implemented yet, need the curves first (do not build blind):
- Plateau / early-stop / reload-best callbacks on an inner dev split (needs a carved fit subset).
- Update-ratio controller: rescale each family LR to a target ‖Δw‖/‖w‖ (LARS/LAMB-style) if H-299 shows imbalance.
- Spike rollback: restore the last snapshot and skip data when loss or gradient norm jumps (AMP scale trace shows stability).
- Annealed distillation alpha driven by student-teacher gap (Annealing-KD), if `loss_family` and the curve show the teacher term saturating.
- Tail-weight curriculum driven by per-sample loss, if the late-epoch curve is flat.

## Analysis recipe

`scripts/analyze_training_trace.py <run names> --out <lineage>/plots --steps-per-epoch 7460` writes one PNG
(loss EMA, lr, beta1, gradient norms and per-family norms, update ratio, second-moment RMS, weight norm,
eval curve) plus a JSON summary. Collect runs with `vast_teachers_r1/controller.py pull`; snapshots and
`trace.jsonl` come with the run directory.
