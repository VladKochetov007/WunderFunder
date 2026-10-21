# From-scratch causal geometry/discrete feature comparison R4

## Question and current status

R4 compares a from-scratch H512×1 GRU with the 60 causal geometry/discrete
features against a matched-width copy whose 60-channel suffix is zeroed. The
feature family contains running moments, fitted arc residuals, anchor occupancy,
exact-level identity/occupancy, event dominance and dwell. The fitting prior
is learned from eligible fitting sequences only. Both arms use the same seed,
source-qualified membership, model initialization, training order and fixed
training schedule. The detailed preregistration is the
[campaign plan](../../campaign_2026-10-17/geometry_from_scratch_r4/plan.json).

**Status: attempt R4 passed fresh resource admission and started pair 0's
fold-0 seed-42000 control fit at 2026-10-17 21:27:24 UTC. Startup source,
input and split verification passed; no R4 score is recorded yet.** Attempts R1
and R2 stopped before fitting due to entrypoint and input-hash errors; R3 was
rejected while the periodic monitor held a full-core reservation. All three
have zero fits and zero OOM events. Admission R3 binds the corrected identities.
The user requested training both arms and a comparison relevant to
whether this feature family merits a submission path. All 30 paired seed/fold
runs are planned; early scores do not stop the direction. The campaign README links the preserved
[feasibility manifest](../../campaign_2026-10-17/geometry_from_scratch_r4/all_eligible_membership_20261017.json),
which checked component metadata without opening Parquet or staging feature
arrays; it is not the fit membership.

The four fixed CampaignSplit manifests each exclude evaluation-linked
components from fitting. Their held-out slices contain 141–149 components and
443–505 valid sequences. Folds 0 and 1 have been used in prior geometry work;
all folds are labeled exposed or potentially exposed development evidence.

## Recipe and evidence to collect

The fixed comparison uses H512×1, FP32, AdamW, OneCycle, weighted MSE, batch 8,
2,000-row chunks and three complete epochs. The frozen plan records fold-specific
budgets of 44,760, 44,790, 44,850 and 44,640 optimizer updates for folds 0–3.
Epoch 2 is graded for both arms; no best-seed choice is allowed. Each completed
epoch saves an atomic continuation state. Thirty same-seed, same-fold pairs (60 fits) are
balanced over four folds (8, 8, 7, 7). Per-fit provenance must show equal
initial parameter hashes within each pair. Every run records training loss,
learning rate, gradient norms, updates, time and fixed endpoint predictions
with group IDs and hashes. Targets and masks are read from the hash-bound shared
`valid_targets.npz`, not duplicated for each run.

The first resource attempt failed argument validation because it omitted
`--no-eval-during-fit`; it performed no optimizer updates and is preserved as a
preflight failure. The corrected five-batch profile passed: 60/60 finite
updates, 2.71 GB cgroup peak under the 5,000,000,000-byte cap, 0.91 GB CUDA
allocated and 1.29 GB reserved peaks, 1.53 seconds per batch of eight full
sequences, and no new OOM events. The separate predictor smoke exported finite
FP32 outputs for all 468 fold-0 evaluation groups and verified their order and
split identity. This fixes a prior-loading omission in `predict_run.py` for the
new feature mode. The current runtime estimate is 113.8 GPU-hours for training,
plus setup and evaluation exports; runs will execute one at a time with one
CPU core and no swap.

The primary effect is the equally weighted mean of the four fold-level paired
`global_wp_clip2_v1` deltas. Report all paired deltas, a fold-stratified seed
interval, and component-cluster uncertainty on fixed predictions separately.
An interval that includes zero means uncertainty at this evidence level, not a
reason to drop the family. A +0.001 absolute WP difference is a useful-effect
reference, not a universal decision threshold. This comparison can establish
only a local exposed-CV result; it does not estimate public score transfer or
feature value in a teacher/student blend.

## Links

- [R4 campaign README and artifacts](../../campaign_2026-10-17/geometry_from_scratch_r4/README.md)
- [R4 frozen plan](../../campaign_2026-10-17/geometry_from_scratch_r4/plan.json)
- [R4 admission R3](../../campaign_2026-10-17/geometry_from_scratch_r4/admission_r3.json)
- [Superseded R2 admission](../../campaign_2026-10-17/geometry_from_scratch_r4/admission_r2.json) — input hash rejected before any fit.
- [R4 independent runner/admission audit](../reviews/R4_INDEPENDENT_ADMISSION_AUDIT_20261018_R1.md)
- [R4 durable resource ledger](../../campaign_2026-10-17/geometry_from_scratch_r4/runtime/README.md)
- [Local monitor quota adjustment](../operations/R4_LOCAL_ADMISSION_AND_MONITOR_BUDGET_20261018_R1.md)
- [R4 profile evidence](../../campaign_2026-10-17/geometry_from_scratch_r4/profile/profile_success_001.json)
- [R4 matrix runner](../../../scripts/run_geometry_r4_matrix.py)
- [R4 progress and resource ledgers](../../campaign_2026-10-17/geometry_from_scratch_r4/README.md)
- [R3 preceding results](../../campaign_2026-10-17/geometry_continuation_r3/README.md)
- [Broader sequence-moment EDA and earlier pilots](SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md)
