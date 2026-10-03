# Research continuation and next decisions

The ten original Claude EDA framings were continued as active Codex R&D agents
using gpt-6.1-sol at medium effort. Each pass executed a bounded falsifier,
preserved earlier evidence and failure logs, and saved runnable probes, a notebook,
inspected figures, sample IDs and provenance. The optimizer builder and independent
strong-control reproducer also wrote and ran code. See agents.json for thread IDs.

## What survives the stronger controls

The full-schedule Muon + chunk1000 + stopped-bootstrap combination is the leading
measured candidate. On the same two component-excluded folds, seed3 beats its
matched control by .002157 and seed4 by .001788. The equal-seed combination bag
beats the equal-seed control bag by .001854, component95% interval [.000420,.003409].
The public-like mixture diagnostic gains .001869, interval [.000213,.003706].
Fold0 dominates: the bag gains .003159 there and .000483 on fold1. Class3 loses
on fold1. This supports additional fixed-fold confirmation, not a public-score
claim. The incremental benefit over Muon alone remains unconfirmed.

Source: scores/combo_matched_2seed_report.json, E-359. Folds2/3 now have frozen
commands and a scoring specification in FOLD23_CONFIRMATION.txt and
scores/combo_fold23_prereg.json. The unchanged procedure follows the two width320
jobs. CPU scoring runs automatically after complete exports. No checkpoint is
chosen from interim validation.

The strongest new EDA lead, five causal state offsets for t1, failed transport
to the strong GRU256. On437 evaluation sequences/128 components, t1 loses .000475
versus affine calibration; mean WP loses .000237 and both folds are negative.
The weak-ridge +.02 to +.04 t1 gain was a recalibration opportunity in that small
model, not a new improvement for the strong student. Close the tested correction.
Its exposed meta cross-fit is not nested parent-training-independent, so the
component interval is an exploratory conditional interval, not fresh confirmation.
Source: continuation/reproduction/REPORT.txt, E-360.

## Independent mechanisms and contradictions

| Original framing | Controller interpretation |
|---|---|
| HFT | Physical-spread decoding fails; economic names of anonymized columns remain tentative. |
| Linear | Dense EMA beats phase splitting. A large score shift on a different sample exposes composition sensitivity. |
| Nonlinear | Linear state recalibration beats threshold bins on weak ridge; strong-student correction subsequently fails. |
| Olympiad | Label constancy does not justify holding predictions. Current price-code information explains the anchor gap; clocks/flow controls fail. |
| Signal | Own-group filter clocks have a heterogeneous pooled gain and fail the frozen half-split gate. |
| Statistical physics | Recent history helps the small ridge; event versus row clock is unresolved. Dwell-age hazard association does not establish target-value prediction gain. |
| Fractal | Fixed eventEMA16 loses strongly; stationary encoded levels explain most near-zero increment DFA slopes. |
| Econometrics | Universal event clocks fail; t0 and t1 respond differently, and scored versus all-required populations rank models differently. |
| Puzzle A | A timeless current-price target-sum lookup is falsified. Labels still stay constant within unchanged-price runs on the tested subset. |
| Puzzle B | Period6 and phase alone fail. Lag4 recurrence's gain is concentrated in two sequences and fails consistency; no broad promotion. |

Puzzle A's repeated-state witnesses have mixed-sign targets. They do not refute
the historical conditional sum-to-spread law on interior, known-code rows.
Similarly, an a6 label-boundary detector and an optimal prediction-update rule
are different claims. The apparently conflicting reports become consistent when
their domains and controls are retained. Never use target-derived classes or
full-sequence summaries as inference inputs.

Broad event-clock, long-memory, phase, carry, physical-decoder and calibration
claims should not be reopened from a weak-control gain. Prior rejected sequence
recentring, class scaling, output EMA and structural supervision remain closed
(E-287,E-288,E-295,E-312,E-314,E-315). Recent-history/recurrence features are
diagnostic proposals; existing recurrent state may already absorb them.

## Implementation options and selection

opt2/REPORT.txt, FLAGS.txt and PLAN.txt describe optimizer switching with explicit
moment handoff, warmup/SGDR, head LR multipliers, module clipping, annealed gradient
noise, Sobol initialization and antithetic ordering. Eighty-one CPU tests and six
synthetic fits pass; default-off AdamW/Muon parity is bitwise on the tested CPU
path. This is implementation evidence, not forecasting efficacy. QMC initialization
loses the synthetic fit and has unfavorable matrix geometry; rank it last.
Stratification helps only when strata explain gradients in its synthetic test.
QMC batch metadata integration and exact fused-GRU zoneout are explicitly deferred.

Independent review found normalization confounding in antithetic order and a
launch helper bypassing controller admission. The controller fixed both in the
saved prototype; 39 focused CPU tests pass, including actual normalization and
switch-boundary checks. The original review and failure evidence remain intact.
See opt2/CONTROLLER_CORRECTIONS.txt and continuation/opt2_review/REVIEW.txt.
The trainer patch remains unapplied; full GPU/fused/EMA parity is still pending.
Future proposals must use gpu_queue.sh/run_gpu.sh and the campaign monitor;
opt2/PLAN.txt now contains admitted-job payloads only. After width/additional-fold evidence,
the best narrow optimizer proposals are late Muon-to-AdamW and module clipping,
with same-seed full-schedule and grouped no-op controls. Avoid bundles whose
mechanisms cannot be separated. Require GPU/fused/EMA parity before integrating.

Best reported public remains .6641; >=.6660 is unproven. Do not sum correlated
local gains or turn a proxy point estimate into a public claim. No upload occurred.

## Runtime and recovery

The persistent watcher checks every minute, retaining incident IDs/journal cursors,
aggregate OOM deltas, failed/nonzero jobs and vanished units. Scope terminal logs
survive CPU failures. New launches fail closed; diagnosed retries preserve original
records and use revised conditions. Monitoring errors remain visible. See
RECOVERY_POLICY.txt and RUNTIME_MIGRATION.md for the repaired scorer incident.

Aggregate limits are12cores/36GiB/no swap. One GPU family contains monitor and
execution together at8cores/24GiB; four CPU slots each have1core/2GiB. Observed
VRAM cap5.5decimalGB, allocator ≤4.8GB; active recipes retain4.6GB. Actual deployed
width service uses Type=notify, separate admission/execution deadlines and the
verified shared GPU family. All current research scopes have stayed within caps.

HANDOFF.md and RESTORE.md identify snapshots, raw Claude evidence, saved research
artifacts, active jobs, hashes and the paused restoration procedure. Dataset/model
backing files remain external dependencies; all newly generated research state
and measured outputs are saved on disk. ETA.json is an estimate, not a deadline.
