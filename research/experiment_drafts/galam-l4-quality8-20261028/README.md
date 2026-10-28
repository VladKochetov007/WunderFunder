# GALAM quality screen: one baseline and seven candidates on eight L4s

Latest consultation: [common20-epoch cap / hybrid optimizer proposal](V2_DISCUSSION.md).
It is not yet materialized or admitted; the checked recipes below remain v1.

Prepared for consultation, 28 October 2026. Zero scientific fits and zero GPU
profiles. The user requests one independent fit per GPU, prioritizing quality.
This eight-fit wave replaces the earlier prospective six-fit/two-seed panel;
historical recipes and results remain unchanged. Model dimensions are declared
choices for the schematic reconstruction, not recovered original GALAM settings.

| Provisional GPU | Recipe | Change from baseline | Parameters | Final epoch |
|---|---|---|---:|---:|
| 0 | [A0 baseline](configs/a0-baseline.json) | Full LSTM GALAM, W64 | 376426 | 5 |
| 1 | [A1 capacity](configs/a1-width128.json) | W128; mixer/FFN widths256 | 1423978 | 5 |
| 2 | [A2 recurrent depth](configs/a2-depth2.json) | Two LSTM layers | 409706 | 5 |
| 3 | [A3 alternative recurrence](configs/a3-gru72.json) | GRU W72, approximately matched capacity | 358170 | 5 |
| 4 | [A4 longer attention](configs/a4-history256.json) | History256 instead of64 | 376426 | 5 |
| 5 | [A5 longer gradients](configs/a5-tbptt256.json) | TBPTT256 instead of64 | 376426 | 5 |
| 6 | [A6 optimization](configs/a6-lr2e4.json) | AdamW LR2e-4 instead of1e-4 | 376426 | 5 |
| 7 | [A7 longer horizon](configs/a7-horizon10.json) | Ten epochs with a retimed cosine schedule | 376426 | 10 |

Common: seed1001, four heads, one global/local attention block, context64,
chunk64, batch16, AdamW weight decay0.01, 5% warmup then cosine, gradient clip1,
FP32, train-only normalization, all-epoch checkpoints. Batch16 is an unprofiled
starting recipe, not an optimum. A common batch may be amended after engineering
profiles and before scientific admission; every affected config/hash/profile
must be regenerated. No silent per-arm batch/LR changes.

Why these arms: capacity, recurrent depth/type, readable past, gradient horizon,
optimization and training duration can each limit quality. An attention-head
count sweep is lower priority without baseline evidence. Local-only ablation,
SWA/EMA with train-only BatchNorm recalibration and combined improvements are
follow-ups, not additional fits hidden inside this eight-fit wave.

Important distinctions: GRU removes the cell-state branch and changes4W fusion
to3W, so it is a complete architecture alternative. Longer attention keeps
detached past-chunk values; longer TBPTT additionally carries gradients and
reduces updates per epoch by about4x. Timestamp scale1 can saturate the relative
lag-bias Tanh for long lags; log lag/bias diagnostics. Ten epochs also retimes
the whole LR schedule. Same integer seed does not match tensors across shapes.

Use the official earlier train / complete later valid boundary (ADR0044), whole
sequences and train-only ancestry. No seq_ix chronology or random primary folds.
Score all valid is_scored rows after warmup with pooled global_wp_clip2_v1,
including t0/t1. Report every epoch; primary checkpoint is each arm's fixed final
epoch, with no best-valid checkpoint selection. This is an exposed-valid,
single-seed breadth screen, not significance or private-score confirmation.

[Server audit](SERVER_AUDIT.json) records the incomplete train and approved RSNA
deletion. Initial /data2 free space1.27GB rose to74.58GB after deletion, enough for
the remaining20.92GB receipt-recorded train ranges plus an estimated50GiB buffer.
One source dataset serves all eight fits. A complete local train candidate also
exists in the October18 audit staging folder; reverify its pinned SHA before use.
No download has started in this preparation.

Remaining receiving work: rehash/resume missing ranges and verify complete
Parquet; audit source-qualified row groups, targets and linked components; resolve
the legacy unconditional row-group528 skip against complete-coverage admission;
repair explicit train-eval mask handling and labeling: the existing local train
candidate footer confirms10607groups/212140000rows and no is_scored, whereas the
present monitoring cohort setup requires it. Use an explicitly labelled audited
training diagnostic mask; preserve official valid is_scored unchanged. This is
a real launch blocker, not already-fixed code. Prepare an isolated compatible
runtime and measure real profiles.
The host driver570/CUDA12.8 cannot simply take the repository's cu130 lock.
[NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
requires580+ for ordinary CUDA13 compatibility. Do not update a shared driver as
a preparation shortcut. Host Torch2.8 is a candidate requiring explicit acceptance,
not a tested equivalent to the local CPU Torch2.14 checks.

The model currently loops through time in Python. GPU memory capacity alone
cannot establish efficient full-fit speed. Profile batches16/32/64/128 with the
actual monitored recipe, memory headroom, legal BatchNorm tail and fixed membership.
Measure rows/s, data wait, train/eval/checkpoint durations and shared eight-reader
contention. Total workload is45 fit-epochs; wall time is unknown until measured.
The engineering profile must exercise sampled diagnostics (at least50 updates
with the present interval25); a separate bounded validation timing probe is also
required. Profiles are not competition fits and produce no scientific scores.

The existing dashboard reads local run files and supports run/config selection,
train-eval/valid loss and score, per-target metrics, LR/gradients, model diagnostics,
and CPU/RAM/GPU/disk sensors. Remote collection/tunneling and durable detached
service ownership remain receiving setup, not already-operating remote monitoring.

See [PR contract](PR.md), [inspection receipt](configs/INSPECTION.json),
[portfolio review](PORTFOLIO_REVIEW.json) and [data review](DATA_REVIEW.json).
Full training awaits consultation on these recipes, the measured time budget and
receiving admission. GitHub PR publication is not claimed by this local packet.
