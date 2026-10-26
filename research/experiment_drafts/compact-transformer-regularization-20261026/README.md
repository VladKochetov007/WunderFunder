# Compact Transformer: matched regularization screen on eight B300 GPUs

User-authorized execution on October26: four fresh label-only teacher fits,
control versus a joint dropout/weight-decay package on F0/F2. Co-residency with
the separate ARC-AGI Ray/Megatron job is explicitly permitted. Its processes
must never be stopped or attributed to this experiment's throughput.

| Cell | Outer fold | Seed | Dropout | Weight decay | Assigned GPU indices |
|---|---:|---:|---:|---:|---|
| ctl_f0 | 0 | 201001 | 0 | 0.01 | 0,1 |
| reg_f0 | 0 | 201001 | 0.1 | 0.1 | 2,3 |
| ctl_f2 | 2 | 201002 | 0 | 0.01 | 4,5 |
| reg_f2 | 2 | 201002 | 0.1 | 0.1 | 6,7 |

All cells use hidden768, 16 layers, 12 heads, FF multiplier4, exact causal
context1024, raw112 inputs, chunk512/accumulation8, five complete epochs,
Muon/auxiliary AdamW, peak LR1e-4, warmup5% of this five-epoch update budget,
cosine floor0.1, clipping1, BF16/BF16x9 with FP32 master weights and states.
This is a new short recipe, not the first five epochs of the old30-epoch schedule.
The treatment tests the joint regularization package, not its separate factors.

Concurrent real-shape profiles compared batches64/128/256 per rank. The selected
batch is128/rank, global256: about98.85% of the best warmed bottleneck throughput,
with81.25GiB peak reserved versus159.86GiB at batch256. The latter nearly exhausts
its160GiB allocator cap for only about1% extra throughput. Each fit has1150updates,
230per epoch. Profiles do not count as scientific fits and their trained weights
are discarded. Final config, exact update budget, measured memory and ETA are
frozen in the receiver's selected-plan and estimate receipts.
No throughput or quality equivalence to the original1.54B recipe is assumed.

## Data and evaluation

The canonical outer manifests remain historical exposed reporting folds. Inner
splits use whole dependency components and 20% of eligible valid-source fitting
sequences (selection seeds41001/41002). Actual prepared memberships:

| Fold | Fit train-source | Fit valid-source | Inner validation |
|---|---:|---:|---:|
| F0 | 10523 | 1121 | 284 |
| F2 | 10500 | 1143 | 287 |

Linked fitting components are removed in both sources, and normalization uses
fitting members only. Each paired arm shares initialization seed, membership,
batch, schedule, source and precision. A deterministic64-sequence fitting probe
is frozen by membership hash before validation results. Comparable train/inner
WMSE uses all post-warmup rows; official inner WP uses its scored mask. Training
probe WP is labeled in-sample, not official validation.

Keep every epoch checkpoint. An external observer evaluates each immutable safe
epoch immediately while fits continue, with per-job serial evaluation and four
parallel job lanes. It retains predictions, targets, masks, scores and hashes.
For each arm separately, choose a common epoch maximizing mean inner WP across
its F0/F2 cells over all five retained epochs; exact ties choose the earlier epoch.
Report both paired fold deltas and the full curves; any later outer
report or student-transfer comparison is separately frozen. Four fits cannot
establish training-variance significance, and already exposed folds are not fresh.

## Runtime, receipts and safety

Canonical source is pinned to
`0a70b4ab96c1810484544f382df4bcf0c08e8fd4`, based on merged PR42.
Its only trainer change explicitly admits `dropout_policy=explicit_regularization_v1`;
omitted policy retains the historical dropout0 guard. This was required because
the original regularized profile was rejected before any updates. Failed profiles
are preserved. The corrected source passed55 bounded CPU tests on the receiver.

`cuda_budget.py` caps only each owned rank's allocator. `profile_panel.py` measures
the actual concurrent cohort. `run_panel.py` verifies source/config/data/profile
identity, launches four disjoint two-GPU groups, preserves checkpoints and
restricts timeout/cancellation cleanup to owned process sessions. It reuses
the selected profile's compiler caches. `launch_once.py` runs the admitted job
once under native cron, independently of SSH, without scientific retries.

`monitor.py`, `evaluate.py`, `metrics.py` provide per-epoch inner validation;
`telemetry.py` records shared-node GPU/PID memory, CPU load, RAM and disk;
`watchdog.py` stops only this cohort if the validation observer fails or loses its heartbeat;
`report.py` updates GitHub on new complete validation points or hourly progress.
Canonical rank logs retain LR, preclip gradient norm, clipping, loss, optimizer
timing, row throughput and peak memory. Failures and partial outputs are retained.

Receiver root: `/data/epanov_s/tabdata/runs/compact-reg-20261026/`.
Runtime operators and immutable scientific source are separate. Raw arrays,
datasets and weights stay off Git. No paid resource or competition upload is part
of this authorization. Full receiver profiles and actual fit/evaluation startup
must be distinguished from prepared code and completed five-epoch results.

## Actual startup and measured estimate

All four full fits started under a native cron PAM session at02:56UTC October26,
after exact source/data/config/profile admission. Actual successful updates were
observed in every fit. Validation observer and hardware sampling are independently
owned cron processes. The five-epoch fitting estimate from warmed steps plus
measured outside-window overhead is63–67minutes/control and74–78minutes/regularized
fit. Allow approximately1.4–1.8hours for the concurrent panel including validation
and startup, subject to shared-node load. These are estimates, not completed fits.
First inner validation is expected about15–25minutes after startup.

GitHub handoff: [draft PR43](https://github.com/VladKochetov007/WunderFunder/pull/43).
