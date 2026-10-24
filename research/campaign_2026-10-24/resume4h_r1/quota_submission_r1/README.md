# Fixed quota submission replay panel

The [completed result and ZIP handoff](RESULTS_AND_HANDOFF_R1.md) ranks the
S3/S4 replacement first and S3 second for the explicit exploratory quota
request. Pair deltas versus TL3 are +0.003742/F0 and +0.000205/F2; this is
fitted/unknown-ancestry sampled evidence, not clean CV or a public forecast.
All native callbacks and the independent scalar result review pass.
The [handoff receipt](HANDOFF_RECEIPT_R1.json) binds the delivered ZIP identities,
review and exploratory priority. No competition upload or public forecast.

This folder records the fixed native replay panel and a separately gated grade
for three exact TL3 ZIPs: the original baseline, an H390 S3 replacement, and an
H390 S3/S4 replacement. The [selection](SELECTION_R1.json) deterministically
samples 32 source-qualified components each from F0 and F2, with one complete
sequence per component. The [admitted replay plan](REPLAY_PLAN_R1.json) binds
the rows, source files, ZIP hashes, member rosters, and resource limits.

[replay_package.py](replay_package.py) processes one package per process. It
reads only sequence metadata, `need_prediction`, and the 112 input features;
it checks ABI metadata, package/member/native hashes, finite outputs, and an
A–B–A reset replay, then writes prediction and selected-input hashes. It does
not load targets or score masks. The [independent review](INDEPENDENT_REPLAY_REVIEW_R1.md)
confirmed exact selection and package pins. All three native replays completed;
the [cross-replay gate](CROSS_REPLAY_GATE_R1.json) records matching ordered
64-sequence feature and `need_prediction` hashes across candidates, plus the
prediction payload identities. Replay timings are callback diagnostics, not
competition runtime estimates.

The [grade adapter](grade_packages.py) and
[draft grading recipe](GRADE_PACKAGES_RECIPE_DRAFT_R1.json) freeze the next
comparison: official `global_wp_clip2_v1` scores for baseline, S3, and S3/S4 on
the selected F0/F2 groups; primary deltas are S3−baseline and S3/S4−baseline,
with S3/S4−S3 secondary. It verifies all source, ZIP, native, receipt, ordered
input, prediction-header, and prediction-payload pins before target-array
extraction. It
then streams the pinned target arrays to physical scratch, maps them from disk,
and processes one selected group at a time using the official mask and row
index ≥99. Each fold reports endpoint scores and paired component bootstrap
intervals (2,000 draws, fixed seed); the equal-weight mean of the two fold
deltas is descriptive only. Prediction and target hashes are checked again
after scoring.

The grade completed under the separate [root admission](GRADE_ADMISSION_R1.json)
after [independent source review](GRADE_PACKAGES_INDEPENDENT_REVIEW_R1.md).
Its [execution receipt](GRADE_EXECUTION_R1.json) records3.545seconds,
399,998,976-byte peak and zero OOM/swap; [all results](QUOTA_GRADE_R1.json)
and the [independent scalar review](QUOTA_GRADE_RESULT_REVIEW_R1.md) are saved.
The recipe and admission use separate pins to avoid a circular hash dependency.
This is exposed validation evidence, not clean CV: H390 S3/S4 used train and
validation data, TL3 ancestry is unknown, and the 32 components per fold are a
sample rather than a full-fold endpoint. There are no new fits, coefficient
search, competition upload, or public-score forecast.

Run only after root admission and final recipe pinning:

```sh
WNN_MEM_MAX=400000000 WNN_THREADS=1 WNN_CPU_QUOTA=100% \
WNN_RUNTIME_MAX=120s WNN_SCOPE_UNIT=wnn-quota-grade-r1-20261024 \
scripts/limited .venv-py311-profile/bin/python \
  research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/grade_packages.py \
  --recipe research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/GRADE_PACKAGES_RECIPE_DRAFT_R1.json \
  --recipe-sha256 <root-frozen-recipe-sha256> \
  --admission-sha256 <root-grade-admission-sha256>
```

The candidate ZIPs and build receipts are indexed in
[the quota inventory](../../../../submissions/quota_20261024/README.md). Root
owns grade admission and execution.

The exact replay receipts are [baseline](baseline_REPLAY_RECEIPT_R1.json),
[S3](s3_REPLAY_RECEIPT_R1.json), and [S3/S4](s3_s4_REPLAY_RECEIPT_R1.json).
The latter two have final [S3 resource counters](s3_EXECUTION_R1.json) and
[S3/S4 resource counters](s3_s4_EXECUTION_R1.json), with no OOM or swap.
The baseline's attempted post-exit [snapshot](baseline_RESOURCE_SNAPSHOT_R1.json)
must be read with its [erratum](baseline_RESOURCE_ERRATUM_R1.json): its empty
scope path caused a root-cgroup CPU counter to be read, so that counter is
excluded. The successful baseline replay and an earlier live memory observation
remain valid; final baseline cgroup counters are unavailable.
