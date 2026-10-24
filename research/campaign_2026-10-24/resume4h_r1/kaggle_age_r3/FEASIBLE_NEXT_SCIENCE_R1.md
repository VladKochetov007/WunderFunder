# Feasibility of a bounded PR26/27 follow-up

## Decision

A faithful new paired panel that compares innovation at 10% and 25% fitting
data does **not** fit under a two-GPU-hour cap. The minimum cross-fidelity
panel is four new fits: innovation and moments at one matched fold/seed at
each fidelity. The historical B300 profiles extrapolate about **6.18
single-GPU hours if the four fits run serially** at the profiled per-fit wall
rates. Reusing both historical control scores leaves about **3.02 projected
fit-process hours** for the two candidates, still over the cap if serial. These
are profile-derived B300 wall estimates, not measured GPU utilization or a
Kaggle throughput guarantee. In the original paired schedule, the p25 F0
candidate/control forecasts are 2.21/2.31 process-hours while sharing one
B300; that cell alone is already above the two-hour wall cap.

The 10% F0/seed1201 matched pair alone sums to **1.66 projected
fit-process-hours** in the original schedule, where both arms shared one B300.
It is the
largest same-recipe unit that might fit the cap, but it repeats only the
positive 10% endpoint and cannot test whether the near-zero 25% result was a
data-fidelity effect. It also cannot test a teacher/readout interaction: this
short screen has no teacher, teacher targets or teacher checkpoints. Kaggle
speed for this recipe has not been measured. Treat even the 1.66-hour pair as
conditional on a bounded profile and available evaluation/export time.

## What #26/#27 actually ran

The 22 October short-screen packet is shared by both PR branches and is a
separate cold-start successor, **not completion of either original PR's
full-data warm/teacher proposal**. The PR26 and PR27 heads are respectively
`56a81f4aacaecea05df52133c9d8ccf19e055908` and
`b8ae611da1bb80aaf6dceb73dee4b6149e8f633d`; both contain the same short-screen
packet. Its exact frozen base trainer is source commit
`a266b036b02e8e012c6eb31072291dad618a169d`, trainer source identity
`fce5257e02b8d4fa701f987af1b662bf4190050f9167b0e66a9ed935da375189`, and
grader SHA256
`56ffc693da52668726bab1f7008e31753a76aed3ff5d79089167785079d9eb31`. The
deployed 147-file source tree is pinned by archive SHA256
`420395e9fd1206ec0466c8304b80dc095e9d416b97752ebfe39ac53a78e5b2b1`, manifest
SHA256 `6f2d3188ae3c547976d6948044bfd97fe9036b12092ce675c5c779ea033791bd` and
tree identity SHA256
`f4ef9ca8450a3e76d6b6f8295259e70aec5024d3a4f3296558bb0323add74130`.

The immutable packet plan SHA256 is
`b74d5c34412b3afa8bc17bf098e945969fb8bda7de62b50a4d931be98ddb62d6`. Its
packet manifest pins `recipe.py` (`e39ac9982771f766d34191ed5cc4e5401d422982c0be6dbc2ba7d7eac61843bd`),
`run_cohort.py` (`24a2de5c19c20ca4bb6dee53c401acc6128524463d7f1b9e958c32989cef8a3b`),
`validate_fit.py` (`3fda3f091b9f6b99fdcbcc51e61d643a0f7396a2834934746b1853363cf6b817`),
`bootstrap.py` (`113e57b5c0e9bb18d85896051d81018deb92ca367ea7e7514998f8c38222b24c`),
and `publish.py` (`fd64ceb349a4d7713a13ba6d0b3779259917243dcf0f1a57de1d5e1e88934fcc`).
The runtime dependencies are hash-pinned in `PLAN.json`: `feature_common.py`
`9ad8a5b9cee6da98d21530d065a75affff6fea575d6d2fadf8f262f276add4a9`,
`feature_worker.py`
`2b5af9189156a5ca60e3b4939d78d807404231852f3ce128f0490a17f963b931`,
`feature_outputs.py`
`8b516c81b43a930e1fd100114fb4b004c0e2ec580beaa3c5224f7445c8b07ac9`,
`feature_validation.py`
`7a88d29a43158161e08b2411c2ff7c48b5d90c548ddb73e008f3a727f8177541`,
`feature_controller.py`
`71cb8cb56443daa7e7a08f957d06d1b66be88e8d551c18312a85c17a9aeb1870`,
`panel.py`
`25a4bce2a9c3d8217cefa0545d0dbe901407b61f7327160d739957cb3401e587`, and
the GitHub reporter `github_api.py`
`d02a48a3656ffb7941e885eb9e78f7295cc0c10d0ff1c69358687e0aa62edecc`. The
per-fit request contents themselves were held under the private state
directory; the public plan has their hashes, not the request bodies.

Both mechanisms used cold H512×1 GRUs, label-only loss, batch 8, 2,000-row
chunks, six strict-FP32 epochs and fixed epoch 5. Autocast, scaler and TF32
were off; AdamW used LR 0.0001, beta2 0.999, weight decay 0.01, OneCycle
warmup 0.05, clip 1 and dropout 0. The fixed full 20,000-row sequences and
official held-out membership were retained. No teacher, warm initializer,
validation checkpoint selection or distillation was used.

| Panel | Comparison | Inputs / parameters | Components and groups per fold |
|---|---|---:|---|
| Age, p10/p25 | `field` vs `group`, same 323-input width | 2,267,650 parameters | p10: 315 components; p25: 786–787 |
| Innovation, p10/p25 | `innovations` vs `moments`, same 291-input width | 2,251,266 parameters | p10: 315 components; p25: 786–787 |

F0/1201, F2/1202 and F3/1203 were the three cells. Component selection was
label-independent and nested by a fold-keyed SHA ordering. The p10 subsets had
937–1,047 train and 125–153 valid groups; p25 had 2,537–2,691 train and
342–372 valid groups. Each subset records the exact split-manifest SHA,
selected-member/component hashes and evaluation membership. The four-fold
evaluation was already exposed. The packet's source inputs were
`ROOT/data`, `ROOT/splits/fold{0,2,3}.json`, and the source-qualified
`a266b...` runtime; fitting subsets were written under the private
`/data/epanov_s/tabdata/runtime/short4h-20261022-r1/subsets/` tree. The public
packet does not bind the raw train/valid file SHA or publish the per-run
`INPUTS.json` contents.

The PR27 comment reports innovation-minus-moments deltas of `+0.00085285`,
`+0.00064007`, `+0.00134467` at p10 (equal-cell mean `+0.000946`), and
`−0.00101477`, `+0.00045897`, `+0.00046894` at p25 (mean `−0.000029`). The
shared cells and nested fidelities are exposed evidence, not independent
replications or a clean holdout.

The operator launched 24 profile requests then 24 fit requests in a four-hour
owned systemd service. The exact outer command is in the [launch receipt at
the PR26 head](https://github.com/VladKochetov007/WunderFunder/blob/56a81f4aacaecea05df52133c9d8ccf19e055908/research/campaign_2026-10-22/tabdata_short4h_r1/receipts/LAUNCH.json):

```text
/data/epanov_s/tabdata/.venv/bin/python -B -u /data/epanov_s/tabdata/runtime/short4h-20261022-r1/code/run_cohort.py --start-unix 1792638131.2087376
```

The service was limited to 14,400 seconds, with fit cutoff T0+11,100s,
export cutoff T0+12,900s and grade cutoff T0+13,140s; the 32-GPU-hour ceiling
came from eight B300s. Each job used the source `runtime/pr15-host-v2` GPU
entrypoint with its private `fit_request.json`. `recipe.py` generates the
canonical trainer arguments and appends the pinned complete-component subset
manifest. Profiles used 8 steps, 2 warmup steps and at most 10 batches.
Profile data admitted exact paired requests and owned-process limits. All 24
profiles passed; all 24 fits were admitted. The later PR comments say all 12
attempts per mechanism were integrity-accepted, exported and officially
scored. Profiles are engineering estimates; the published PR comments do not
give per-fit observed wall time. The result comments are [PR26](https://github.com/VladKochetov007/WunderFunder/pull/26#issuecomment-6331586358)
and [PR27](https://github.com/VladKochetov007/WunderFunder/pull/27#issuecomment-6331586689).

## Cost bound and what can be reused

The recipe limits profiling to 8 profile steps, 2 warmup steps and 10 batches.
The runner predicts six-epoch fit wall from each profile receipt as
`elapsed + 1.5 * 6 * ((train + valid) // 8) * step_wall / 1000`. It explicitly
left `measured_subset_runtime_seconds` null. Summing its innovation forecasts:

| New panel | B300 forecasted fit-process wall; sum is serial single-GPU equivalent |
|---|---:|
| One F0/1201 p10 pair: 2 fits | 5,985 s / 1.66 h |
| One F0/1201 p25 pair: 2 fits | 16,261 s / 4.52 h |
| Both fidelities at F0/1201: 4 fits | 22,246 s / 6.18 h |
| All three cells, p10 only: 6 fits | 19,112 s / 5.31 h |
| All three cells, p25 only: 6 fits | 56,069 s / 15.57 h |

Reusing the exact historical controls for the same F0/1201 cells would still
sum to about 3.02 projected fit-process hours for the two new candidate arms across p10
and p25. It could save the control fits because the official per-cell scores
are in the PR comments, but it would not make the minimum cross-fidelity panel
fit the cap. Reuse is restricted to those exact F0/1201/source/recipe cells;
the old controls are not new replications. No old predictions or weights are
available to regrade or inspect.

The single p10 pair is the only naturally paired, original-endpoint unit
whose B300 forecast is under two GPU-hours. On Kaggle, the matching GPU type
and measured throughput are unknown; setup, data/cache reconstruction,
official inference and artifact export could consume the remaining 20
minutes or push it over. The published `vladkochetov07/wunder-funder` dataset
is referenced by an existing Kaggle kernel and can supply the raw competition
data, but the PR26/27 packet does not pin its Kaggle version or public raw-file
hashes. The private exact caches, `INPUTS.json`, scored predictions and model
weights are absent from the public branch. The separate 1.14-GB R2 age-profile
bundle omits raw Parquet and target/full-cache arrays, uses a different
profile-only purpose and cannot stand in for the full `a266b...` training
inputs.

## Teacher/readout mismatch

The 24-fit panel is a cold label-only student feature comparison. It contains
no teacher fit, teacher checkpoint, teacher targets, or teacher/readout
interaction. The PR bodies explicitly say their short-screen successor is not
completion of the original warm/teacher/12-epoch proposals. The PR32 engineering
archive recovered later contains engineering receipts only, zero scientific
fits and no checkpoint/prediction payload; the separate private PR32 teacher
runs remain incomplete. Consequently, no existing public artifact can supply
the teacher side for a matched readout comparison. The cost of a
teacher-matched readout panel is unknown from these receipts; it adds new fits
and cannot be forecast or admitted under the two-hour cap without a compatible
teacher checkpoint/target and a relevant profile.

## Recommendation and prerequisites

Do not label a one-pair p10 repeat as resolving the p10/p25 discrepancy or the
teacher/readout mismatch. Under a hard two-GPU-hour ceiling, either:

- run one **new p10 innovation/moments pair** on one predeclared cell only if a
  Kaggle profile demonstrates room for fit, full evaluation and export; state
  that it only checks the p10 mechanism and does not test the p25 effect; or
- defer the faithful cross-fidelity follow-up until at least the forecasted
  6.18 B300-equivalent GPU-hours are available for the minimal four-fit F0
  panel, with extra budget for Kaggle throughput and grading uncertainty.

Before even the two-fit option, an operator needs a source-qualified checkout
or verified deployed tree matching `a266b...` and the 147-file/tree hashes;
the feature runtime dependency pins and packet code hashes; exact raw train /
valid data identities and split-manifest versions; the six-component-subset
generator and the chosen subset receipt; the recipe/fit requests rebuilt from
the pinned source; a Kaggle GPU profile for this exact model/data path; and
root admission for precisely the stated fits, output collection and cost cap.
No fitting, imports, provider calls, downloads, or source mutations were made
for this feasibility review.

## Evidence links

- [PR26/27 24-slot short-screen packet](../pr_audit_25_40/PR26_SHORT4H_README.md)
- [PR26 source comment and per-cell age results](https://github.com/VladKochetov007/WunderFunder/pull/26#issuecomment-6331586358)
- [PR27 source comment and per-cell innovation results](https://github.com/VladKochetov007/WunderFunder/pull/27#issuecomment-6331586689)
- Exact captured PR JSON and full changed-path metadata: [`PR26_GH.json`](../pr_audit_25_40/PR26_GH.json), [`PR27_GH.json`](../pr_audit_25_40/PR27_GH.json), [`PR26_FILES_METADATA_GH.json`](../pr_audit_25_40/PR26_FILES_METADATA_GH.json), [`PR27_FILES_METADATA_GH.json`](../pr_audit_25_40/PR27_FILES_METADATA_GH.json).
- [R3 source-closure handoff](README.md) — separate age-profile prep; not a trained artifact or substitute for this recipe.
