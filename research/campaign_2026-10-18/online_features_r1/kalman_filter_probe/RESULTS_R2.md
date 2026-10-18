# Causal Kalman input probe, R2 results

## Question and scope

This is a prediction-only diagnostic of whether fixed causal filters improve a
frozen GRU when four raw price-like columns are replaced by posterior levels.
The three arms use the same checkpoint, event-feature path, normalization,
sequence boundaries and rows: unchanged raw inputs, scalar random-walk
posterior levels, and local-level/local-trend posterior levels. No model weights
were updated. The local-trend state is causal and resets between sequences.

The replay ran first on the first 16 source-qualified F0 evaluation groups,
without requesting labels or scoring masks. Only after all three caches were
written did the separate grader read scored targets and masks for those groups.
The panel was selected before grading, but F0 is exposed and the panel is small;
this is discovery evidence, not clean validation. See the exact
[execution receipt](replay_execution_r3.json), [grade JSON](grade_r2.json),
[grader source](grade_replay_r2.py), and frozen [R2 recipe](recipe_r2.json).

## Official metric results

| Input arm | Pooled WP | T0 | T1 | Difference vs raw |
|---|---:|---:|---:|---:|
| Raw | 0.66670529 | 0.73098520 | 0.60242538 | — |
| Random-walk posterior | 0.66682468 | 0.73085002 | 0.60279935 | +0.00011940 |
| Local-trend posterior | 0.66704354 | 0.73097460 | 0.60311249 | +0.00033825 |

The local-trend point difference is almost entirely in T1 (+0.00068711); T0
changes by −0.00001060. The paired fixed-endpoint source-component bootstrap
uses 12 components and 10,000 draws. Its 95% interval is
[−0.00001915, +0.00071564] for local trend versus raw. The random-walk interval
is [−0.00050730, +0.00065399]. Both intervals are conditional on these fixed
predictions and selected F0 groups; they do not cover training variation,
selection uncertainty, other folds or public transfer. The local-trend
interval barely crosses zero, so this is an uncertain positive screen, not a
confirmed gain.

An independent CPU scorer, implemented separately from `grade_replay_r2.py`,
reproduced the three scores and both bootstrap intervals to floating-point
precision after checking row-group/sequence alignment, prediction hashes,
checkpoint/normalization identity, the official 99-row warmup and feature-only
replay receipts. Its `p_le_0 = 0.0319` for local trend is only the bootstrap
fraction with a nonpositive delta; because the two-sided percentile interval
still crosses zero, it is not a two-sided significance result. See the
[independent audit receipt](independent_regrade_r3.json) and
[independent scorer](independent_regrade_r3.py). The first bounded invocation
failed before data reads because the frozen pre-run recipe had no `metric` key;
the corrected audit used the official `WARMUP=99` contract and completed.

Only four of the 16 individual sequence WP scores rose under either filtered
input. That count is descriptive: the official metric pools weighted moments
across rows and is not the mean of sequence scores. Source components move in
opposite directions, so the pooled difference is heterogeneous.

## Long-sequence position diagnostic

These four equal-width time bins were added after the whole-panel grade to check
whether a causal state estimate behaves differently later in a sequence. They
are post-hoc, have no separate uncertainty interval, and remain exposed.

| Rows | Scored rows | Raw WP | Random-walk delta | Local-trend delta |
|---|---:|---:|---:|---:|
| 99–4,999 | 6,633 | 0.6947910 | +0.0004528 | +0.0002534 |
| 5,000–9,999 | 5,791 | 0.6523917 | −0.0012381 | −0.0003353 |
| 10,000–14,999 | 7,558 | 0.7544278 | −0.0000171 | +0.0001351 |
| 15,000–19,999 | 11,356 | 0.5946195 | +0.0003818 | +0.0006244 |

The late-bin point is larger, but the middle bin is negative and the mask
density changes across bins. This does not show a reliable long-horizon benefit
or test-time learning effect. The mechanism tested here is online causal
filtering of four input streams; the GRU parameters remain frozen.

## Interpretation and next experiment

The result is not “replace inputs with a Kalman filter and ship it.” The local-
trend transform slightly changes the frozen model's output and has a small
positive pooled point difference on this selected panel. Because the model was
trained on raw inputs, replacement also creates a train/inference representation
mismatch. The next meaningful test should retain raw input and expose posterior
level, slope, standardized innovation and uncertainty as learned features, so
training can select when to use them. The B300 draft implements this as matched
raw/EMA/fixed/local-trend GRU teachers and includes a matched student-cache
substitution; judge the competition payoff at the student endpoint, not teacher
WP alone. See [draft PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17).

Do not turn this screen into a submission or claim that it improves competition
score. First profile the full112 FP32 path on the actual B300 host, then run the
paired clean-fit recipe, build the same-fold student cache, and independently
grade across the frozen fold/seed matrix. The 5 GB local replay completed with
no recorded OOM events; process-RSS and worker-cgroup peak readings differ and
are both preserved in the execution receipt.
