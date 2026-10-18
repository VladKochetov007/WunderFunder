# From-scratch geometry/discrete feature comparison R4

[The frozen plan](plan.json) registers 30 paired seed/fold comparisons between
the same H512×1 GRU trained from scratch with and without the 60 causal
geometry/discrete channels. Each pair uses one of the four component-complete
CampaignSplit folds, the same seed, and identical model width; the control
zeros the feature suffix. This tests whether the causal summaries help a
stronger recurrent model after the earlier tiny H128×2 screen left the
standalone dictionary arm near control and showed a conditional fixed-mixture
signal.

The four official split manifests provide 10,533–10,539 fitting train
sequences, 1,368–1,430 fitting validation sequences, and 141–149 evaluation
components per fold. Evaluation membership is held out at the connected
component level. Folds 0 and 1 were used in earlier geometry experiments;
all folds are treated as exposed or potentially exposed development data.

The planned recipe is FP32 AdamW + OneCycle, batch 8, 2,000-row chunks, three
epochs, and fixed epoch 2. CampaignSplit runs disable evaluation during fitting;
after training, `scripts/predict_run.py` exports only that fold's held-out
evaluation groups, which are graded with the official metric. This keeps the
primary endpoint separate from the trainer's fitting loop. Logs capture training
loss, gradient norms, learning rate, update counts, timing and resource peaks,
and preserve checkpoints and continuation state. Runs share one local GPU
worker, one CPU core and a 5,000,000,000-byte hard RAM cap, with zero data-loader
workers and scratch on the physical `research_tmp` filesystem. The bounded
training and prediction profiles have passed without new OOM events. The
measured five-batch profile peaked at 2.71 GB cgroup RAM, 0.91 GB CUDA allocated
and 1.29 GB reserved; it measured 1.53 seconds per eight-sequence batch and
5.24 sequences/s, with 60/60 finite updates and unchanged OOM counters. The
full 60-fit matrix is estimated at about 114 GPU-hours before evaluation and
setup overhead. Attempt R4 passed the durable resource controller with the
5 GB / one-core / no-swap reservation. The runner verified frozen source, input
and split identities, acquired the shared GPU lock, and started pair 0's fold-0
seed-42000 control fit at 2026-10-17 21:27:24 UTC (00:27:24 EEST on 18 October).
No R4 score exists yet. Earlier attempts R1 and R2 stopped before fitting due
to entrypoint and input-hash errors; R3 was rejected while the periodic monitor
held a full-core reservation. All three have zero fits and zero OOM events.
[Admission R3](admission_r3.json) binds the corrected input identities and
current code.

The primary result is the equally weighted mean of the four fold-level paired
WP deltas. The plan separates a fold-stratified training-seed interval from
component-cluster evaluation uncertainty. A +0.001 absolute WP change is a
practical reference, not a stop rule; all 30 pairs are intended to run. These
exposed folds cannot establish pristine confirmation or public transfer. The
custom mode is not yet submission-ready because serving parity has not been
implemented.

- [Frozen experiment plan](plan.json) — question, endpoint, membership, seed/fold
  allocation, compute limits, significance analysis and stopping rule.
- [Initial admission receipt](admission.json) — source and split hashes plus the
  original profile command; the attempt failed preflight because CampaignSplit
  requires `--no-eval-during-fit`, so it performed zero optimizer updates.
- [Admission R3](admission_r3.json) — corrected streamed hashes for all three
  inputs, current source and split hashes, successful profile evidence, runtime
  estimate, and the 5 GB/one-core/no-swap resource contract.
- [Superseded admission R2](admission_r2.json) — preserved unchanged; its
  training-file hash was mistyped and the runner rejected it before fitting.
- [Independent runner and admission audit](../../unified/reviews/R4_INDEPENDENT_ADMISSION_AUDIT_20261018_R1.md)
  — records seven operational/measurement defects found by a read-only critic,
  the actor's code corrections, and the remaining profile limits. It is not a
  scientific result or a post-correction code approval.
- [Durable resource ledger](runtime/README.md) — dedicated R4 admission state;
  the older campaign barrier remains closed for unrelated work.
- [Local monitor quota adjustment](../../unified/operations/R4_LOCAL_ADMISSION_AND_MONITOR_BUDGET_20261018_R1.md)
  — preserves five-minute read-only checks while reducing the support CPU
  reservation to fit the current two-core aggregate.
- [Profile receipts](profile/README.md) — bounded training and exact-group
  prediction checks, logs and the recorded preflight failure.
- [Matrix runner](../../../scripts/run_geometry_r4_matrix.py) — sequential,
  resumable execution with per-fit command logs, matched-ancestry checks and
  resource telemetry.
- [Training logs](logs/README.md) — epoch loss, gradient diagnostics, learning
  rates, update counts and fit durations for each endpoint.
- [Evaluation predictions](predictions/README.md) — fixed epoch-2 outputs for
  exactly the held-out evaluation groups, with row order and split hashes.
- [Official score receipts](scores/README.md) — official metric identity,
  per-endpoint grades and paired comparison outputs.
- [All-eligible membership feasibility manifest](all_eligible_membership_20261017.json)
  — metadata-only component-count check; it is not the R4 fit membership and no
  multi-hundred-GB feature array was staged.
- [R3 precursor](../geometry_continuation_r3/README.md) — tiny H128×2 results
  that motivated this larger from-scratch test.
- [Unified validation note](../../../unified/validation/GEOMETRY_FROM_SCRATCH_R4_20261017.md)
  — canonical status and result summary; will link all run receipts and analysis.

## October 18 scope change

The latest user cancels this local 30-pair allocation. The first control was stopped before completing its first epoch, with no quality score; all logs remain. The admission barrier is closed. [One quick local warm probe](../../campaign_2026-10-18/quick_geometry_probe_r1/README.md) and an exploratory submission ZIP replace local matrix execution. Deep confirmation moves to teammate 8×B300. This does not reject the feature mechanism.
