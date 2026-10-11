# H382 preregistration

## Question

Does the fold-1 GRU256x2 student's Muon-all treatment retain useful behavior on
fold 0, and do propensity power or a longer fixed OneCycle schedule alter that
crossfold behavior? This is a confirmation/interaction screen in the current
teacher/student portfolio. It does not tune served-blend weights or claim a new
blend endpoint by itself.

## Frozen comparisons

All three jobs are fresh seed-3 fits from the same frozen V8 source snapshot,
fold-0 split, propensity model, and recipe. Shared settings are GRU-256x2,
Muon scope `all`, batch 32, chunk 1,000, WMSE chunk normalization, LR 0.001,
WD 0.01, OneCycle, fp16, 0.3 next-row and structural auxiliary weights,
since-weight 32, and fixed checkpoint selection. No arm resumes another arm or
uses a teacher initialization.

The `prop2/e3` job is the all-scope fold-0 anchor. `prop0/e3` changes only the
propensity power. `prop0/e4` uses a fresh four-epoch OneCycle run and checkpoint
at epoch 3; relative to `prop0/e3`, it changes the schedule horizon and selected
checkpoint. It is not an extra epoch appended to the three-epoch run. The
historical fold-0 prop0/prop2 recipes used `Muon scope rnn`, so those runs are
context rather than matched controls for this all-scope contrast.

## Data, exposure, and outputs

The immutable split SHA is
`09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`. Fitting
membership is 10,535 train and 1,405 validation sequences (11,940 total); the
separate fold-0 evaluation membership contains 468 valid sequences. The worker
exports the complete 1,873-row valid prediction arrays so the controller can
apply the frozen grader on exactly the appropriate evaluation membership.
This fold has already been exposed in the campaign and will be reported as
exposed-fold evidence. A different seed does not make it a fresh holdout.

The only checkpoint ancestry used as an input is the project's own
`c20260929_prop_f0` propensity artifact and its matching cached arrays. The
fit models themselves initialize from scratch. The packet binds these assets
to the provider asset manifest by content SHA and verifies their sidecars and
array headers before any fit. The split's own fitting/evaluation component
separation is checked at packet build time and the exact ordered fit membership
is checked again against every output provenance record.

## Resource bound and analysis

Each fit is capped at 5,400 seconds, each export at 1,800 seconds, and the
two-GPU cohort at 14,400 wall seconds (28,800 maximum GPU-seconds for two
occupied slots). The prior completed V6 cohort recorded actual fit durations
between 2,899 and 3,545 seconds and exports between 97 and 142 seconds for the
same model shape and batch/chunk on two Kaggle T4s. Those measurements support
the bounds but do not substitute for current quota admission or a fold-0
profile.

The controller/grader should compare the three outputs on the frozen fold-0
metric and evaluate each fixed mixture against its matched local endpoint.
Interpret propensity and schedule contrasts separately; do not add gains,
choose blend weights from public feedback, or label any one seed as a
multi-seed mean. Muon scope `all` may use AdamW fallback groups at the shared
learning rate, so results are scoped to this optimizer configuration rather
than a pure-Muon claim.
