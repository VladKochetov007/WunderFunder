# Objective mechanism: conditional tilt, chunk ratios and the actual distilled recipe

**The 5% normalizer-variation gate survives, but measured estimator shifts are
modest and do not justify a training launch.** On the frozen 256-sequence TRAIN
subset, main-denominator CV is15.16% at chunk1000 and12.87% at chunk2000. Variability
is partly predictable from inputs, with additional label dependence. Replacing
self-normalization changes a teacher-prediction gradient proxy by5.11% in aggregate
norm-relative difference, while keeping its direction very close (cosine.99876).
The coarse input-bin conditional combo-target shift is.00502 RMS. None of these is
a validation/public performance gain or a demonstrated bias of the full-history
Bayes predictor.

**`norm_ema` is not used by the cached-teacher branch.** Setting `--loss-norm running`
would leave the recorded distilled recipes numerically unchanged. Labels already
clip before weighting; there is no alleged raw-label-weight bug here.

## Active recipe and source identity

The completed `c20261003_combo_s3_f0/provenance.json`, rather than defaults alone,
fixes Muon, batch32, chunk1000,4 epochs, clipped-weight MSE, propensity power2+
floor.03, since-weight32/floor.25, two same-fold teacher caches, alpha.5, stopped
bootstrap horizon8, EMA consistency1/decay.999, structural and next-change heads
.3 each, full TRAIN+allowed-validation fit, and fixed EMA epoch3 selection. The
queued additional-fold jobs inherit this recipe. Width320 uses chunk2000 and
bootstrap0 but otherwise the same objective weights. Only metadata/source/cache
slices were read; no strong student was run or trained here.

`DERIVATION.md` gives the exact objective. Main MSE divides by both-target clipped
label mass $W$. EMA and structural heads divide by row mass $R$; next-change
auxiliaries divide by eligible future-difference-change mass $N$. EMA weights
broadcast over both outputs without doubling its row denominator. The main
normalizer therefore also changes its relative weighting against the auxiliary
and EMA terms. The true EMA target is an evolving un-clipped model output; cached
teachers are separate clipped targets. That evolving coupled objective has no
single frozen-target population optimum claimed here.

Source/checkpoint/sidecar hashes are in `source_hashes.txt`; selected ordered
parquet columns and teacher/propensity slices are independently hashed in
`sample_artifacts.json`. Complete multi-GB prediction-file hashes are recorded
from provenance, not recomputed by reading all predictions. Completed combo,
propensity and teacher checkpoints were hashed as small files. No outside literature
was used: the metric is the current repository specification, and the conditional
optimum and counterexamples are derived and executed here. Prior literature
recollections in RESET are not presented as verified theorems.

## Frozen protocol and exact exposure

`PREREG.json` was written before synthetic or actual scoring; its immutable hash
matches the hash recorded in `results.json`. Sample is
`default_rng(173101).permutation(fold0.fit.train)[:256]`, with no label/class-based
selection. Every selected row group has20000 rows; rows0–98 are warmup. Group IDs
are listed in `sample_ids.json`, actual `seq_ix` values in `sample_artifacts.json`.
All5,120,000 rows were streamed once from TRAIN, using only8 parquet columns; no
validation/evaluation rows were read. The first permutation partitions256
sequences into8 batches of32; two prespecified re-pairings check composition
sensitivity.20 chunk positions at1000 rows and10 at2000 are not independent units.

We reconstruct the actual TRAIN weight
`valid*(prop_cache**2+.03)*max(.25,1/(1+since_i0px/32))`, then multiply by
`abs(clip(y,-2,2))` separately for each target. Since is reconstructed from the
trainer's AND of bid/ask price-change flags, cap10000, float32 log1p/expm1. A CPU
test compares it against the actual torch feature implementation. Teacher averages
use the row-group lookup from `train_targets.npz`, whose `row_group` member alone
is read; target tensors from that archive are not loaded. Stopped-bootstrap uses
raw-pair changes and strict `next_change > t+8`, including sequence-end handling.
The selected raw teacher slices are averaged before clipping. Float64 accumulation
is intentional diagnostic precision, not claimed bit parity with GPU reductions.

TRAIN-only sampling omits the11.8% allowed-validation part of the completed fit
(1405 of11940 sequences). Its true-mask weights use a batch-dependent replacement
scalar and could change the full-mixture CV. We do not estimate that mixture here.
Target-derived market classes were not used for sample selection. Chunk-summary
input predictors observe the whole chunk, not just causal information available
at its first row. Held-out-batch predictive R² is observational, not a deployed
normalizer predictor or independent component-level confirmation.

## Measured actual normalizers

| TRAIN diagnostic | chunk1000 | chunk2000 |
|---|---:|---:|
| CV(main W), primary order |15.16%|12.87%|
| CV(main W), two re-pairings |16.24%,16.44%|14.88%,14.97%|
| Input-summary prediction of log W, leave-one-batch-out R² |.549|.555|
| Inputs+exact row mass+labels, R² |.766|.714|
| Position-only R² |-.126|-.178|
| Label-only R² |-.016|-.296|

For chunk1000, removing the warmup chunk leaves CV15.14%; within-position CV
ranges9.07%–19.75%. This is not merely the99-row warmup deficit. Row-mass CV is
12.84%; label-only mass CV is7.95%. Both input weighting and conditional label
magnitude matter. Input summaries alone explain substantial variation; labels add
information *conditional on inputs*. Label-only predictors failing held-out-batch
R² does not mean labels cause none of the denominator variation.8 batch folds and
correlated chunk positions limit precision, and no confidence interval is claimed.

`mean(W)/W` ranges.706–1.558, with1%,median,99% quantiles.742,1.014,1.467. The
ratio W/R, which changes the main/EMA-relative quadratic coefficients, has10.57%
CV. Training mass shares are49.754% target0 and50.246% target1; this reproduces the
balanced-target diagnostic relevant to E-204, not a new head-imbalance rationale.

## Gradient and conditional-bin effects, including negatives

The diagnostic uses exact reconstructed targets/weights, but student predictions
are either zero or the clipped cached-teacher average. Gradients are output
intercepts plus two simple input head coordinates (log-since/10 and propensity),
not full GRU/trunk/EMA/auxiliary parameter gradients. They are evaluated at a fixed
surrogate across chunks, so aggregate gradient differences isolate estimator
reweighting rather than optimization history.

At chunk1000:

| Fixed prediction proxy | cosine(SN,pooled mean gradient) | relative mean-gradient difference | trace variance SN / pooled |
|---|---:|---:|---:|
| zero |.999974|1.29%|.10224/.13230|
| cached teacher |.998761|5.11%|.002798/.003055|

The pooled correction increases proxy gradient variance, rather than furnishing
a free variance reduction. At chunk2000 the teacher-proxy mean cosine is.999912.
Large CV by itself therefore overstates the evidence for a useful new training arm.

Coarse conditional input bins are fixed propensity quartiles crossed with since
thresholds16/64/256. Weighted-RMS shifts of the bin mean when weighting chunks
equally versus pooling row weights are.00862 for clipped labels,.00502 for combo
bootstrap targets and.00639 for nonbootstrap distilled targets. Maximum nonempty
bin shifts are.06568,.02528,.04568 respectively, and rare bins should not be read
as robust subgroup effects. These are finite-sample weighted conditional-bin
ratios, not identified changes in the conditional optimum at full causal history.
`results.json` retains bin masses and all means to expose the difference.

## Clustered regression controls and mathematical claims

The exact population positive control uses240 clusters,16 identical rows per
cluster and three observed input states. A saturated weighted regression under
chunk normalization cancels outcome magnitude, fitting ordinary means. Pooled
regression fits the clipped-|y|-tilted means. Their population metric is.797979
versus.843079. Four preregistered seed fits favor pooled regression in every case;
finite-sample fits approach the analytic tilted optimum. This is a counterexample
to universal “normalization only changes variance,” not evidence that the real
training population is that example.

Negative controls: constant chunk mass yields exact fit equality, and deterministic
outcomes given observed inputs retain their conditional optimum despite large mass
variation (max fitted difference5.8e-15). A separate restricted constant-model
counterexample shows that even input-only weights can move a capacity-limited
optimum. A biased teacher reversing the tilted mean makes alpha.5 distillation
fit constant0/WP0. The correct official optimum is the conditional mean under
score-selection and clipped-label magnitude tilt, up to a feasible positive global
affine transform of the **clipped** prediction. Raw affine rescaling followed by
saturation is not invariant. Details and derivations are in `DERIVATION.md`.

The identity behind normalization bias is a *tilted conditional covariance* of
blended target and inverse chunk mass, divided by tilted conditional mean inverse
mass. Correlated sequence rows and the other target's shared mass matter. We do
not claim a bias rate of O(1/raw-row-count) from iid importance-sampling folklore.

## Estimator-alignment sham: distinct exploratory follow-up

After seeing the primary CV/proxy results, `PREREG_SHAM.json` froze a follow-up
with seeds9000–9031 before running it. It is explicitly exploratory. Set
`s=W/mean(W)`: aligned `s*main_SN` equals a fixed pooled main denominator, while
`permute(s)*main_SN` keeps the identical finite scalar-multiplier multiset but
breaks own-chunk mass alignment. `estimator_control.py` implements it; tests check
multiset equality, mean1, pooled-gradient equality and the constant-mass negative.

In the clustered positive control, aligned population WP is.841961, compared with
ordinary.828204 and shuffled-multiplier mean.829359. Shuffling thus mostly retains
the ordinary conditional mean, distinguishing alignment from scalar distribution.
On the TRAIN teacher proxy, sham-to-aligned relative mean-gradient distance has
10/50/90% quantiles4.00/6.81/9.69%; the sham average stays near the original SN
mean. This remains a proxy and finite-multiset experiment, not a training update
or matched full-network gradient-norm study. It supplies a reusable control rather
than a claim that pooled normalization will improve WP.

Matching every individual main-only gradient norm would cancel the intervention:
main gradients are collinear inside a chunk. Exact preservation of all parameter
updates would preserve the trajectory and cannot test a changed objective. The
honest matched condition is identical update **count**, clock, membership,
TBPTT/schedule, teachers and auxiliary/EMA terms, plus identical multiplier
multiset; realized update lengths must still be measured.

## Ranked follow-up and alternatives, nearest prior experiments

1. **Novel follow-up: mass-alignment versus scale-distribution control on the
   actual distilled multi-term objective.** First recover fixed-student main,
   structural and next-change gradient vectors on a newly frozen small TRAIN set;
   current exports contain EMA-selected student weights, not paired raw/lagged-EMA
   training states, so the EMA component remains unidentified without additional
   controller instrumentation. If full gradients show a material alignment effect,
   propose exactly one narrow full-schedule contrast: aligned main `H/D` versus
   permuted-multiset-scaled main `s_perm H/W`, leaving EMA/aux normalizers unchanged.
   Same batches/order,chunk1000,update count,seed,teacher caches,bootstrap8,clip,
   Muon and OneCycle; fixed TRAIN-estimated D, frozen permutation, no validation
   calibration. Match scalar multiset, measure real pre/post-clip and parameter
   update norms. Primary endpoint fixed EMA epoch3 global WP, paired components;
   full-fold confirmation if the screen survives. **No launch recommended yet.**
   Nearest priors E-160/E-161 (mismatch algebra) and E-180 (nondistilled running-
   denominator null); novelty is distillation-branch implementation, own-mass
   alignment sham and relative multi-loss coefficients. E-180's null makes a
   blind fixed-denominator repeat low value. Rough future cost2 full fits for one
   paired fold ~1.5–2.5 GPU-hours, not measured here. No nonexistent CLI advertised.
2. **Conditional-coefficient variance reduction.** Cross-fit an input predictor of
   target-specific magnitude and replace only the quadratic coefficient with its
   conditional expectation while keeping the `|Y|*blended_target` linear term.
   With a correct input conditional mean and fixed pooled denominator this
   preserves the expected gradient; replacing only the loss weight and leaving
   target unchanged does not. Freeze inputs/cluster splits, distinguish learned
   predictor bias from variance. Nearest E-160 and RESET's RB-weight critique;
   applies specifically to the stopped-distilled quadratic, not a discovered gain.
3. **Weight–distillation/boot interaction, not optional-stopping folklore.** Audit
   tilted conditional teacher errors near raw-label transitions, with bootstrap0/8
   and identical |clip(y)| weights. Alpha0 with bootstrap8 is not a raw-label
   negative. Nearest E-352 (since×alpha), E-353 (weak stopped-bootstrap gain),
   E-356/E-359 (combo), and E-317 (mature-label leakage warning). Preserve explicit
   causal/future target distinction; do not turn future labels into inputs.
4. **Full TRAIN+validation-in-fit mass reconstruction.** Extend the frozen sample
   only with allowed fit-validation sequences and execute the batch true-mask
   replacement scalar before since weighting. Re-measure mixture W/R/N and
   gradient alignment with training-only metadata; this can falsify the current
   TRAIN-subset extrapolation. Nearest E-182 (since weights), E-180 (normalizer)
   and E-152 (additional data/weighting changed the population). No evaluation
   scoring or validation-calibrated denominator should enter the intervention.
5. **Close target-head denominator imbalance for now.** Mass share49.75/50.25 gives
   no obvious target allocation rationale. Nearest E-204 found the same balance;
   revisit only if actual parameter-gradient/trunk capacity diagnostics contradict
   both checks. This is a negative, not another tuneable flag.

## Reproduction, costs and scope

`run_all.sh` executes all existing frozen probes/tests using
`continuation/run_cpu.sh round2-objective`, one core,2GiB,900-second scope. Final
CPU tests pass, including an AST execution of the live trainer branch, clipped
weights, actual bootstrap parity, denominator bypass, since reconstruction,
main+EMA gradient factorization and affine clipping counterexample. Existing Ruff
formats/checks pass; logs retain initial import-order fixes and the missing guessed
source path. No experiment, scoring, OOM or resource-limit failure occurred.

Actual stream:4.96seconds, peak RSS1198912KiB (~1.14GiB),8 parquet columns and
selected mapped cache slices only; analysis and shams were subsecond. Test timing
and RSS are in `logs/tests_final.log`. Operator/build/review time is within the
~40-minute pass; no GPU job, install, service, live trainer/optimizer edit, queue
mutation, evaluator edit, submission, commit or recursive agent. All new files
remain in this objective scope. This provides executable estimator understanding
and falsifiers; no public-score forecast or strong-baseline efficacy is claimed.
