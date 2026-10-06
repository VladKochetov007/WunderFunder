# Metric-aware error budget: design before new outcomes

**Primary question:** where does the fixed student4's remaining official-score deficit live, and which parts differ from strong4/fixed3? The decomposition below is descriptive and exact for the fixed evaluation population. It does not estimate a Bayes error, fit calibration parameters, select coefficients, or authorize another model.

Source-only review: [canonical frozen evaluator](../../../scripts/fold_eval.py), lines30–53; [metric helper](../../../src/wnn/gmetric.py), lines1–59; [old calibration EDA](../../../scripts/eda_scale.py); [CSTAT1 implementation](../2026-10-01-eda/ensemble_statistics.py), especially `class_decomposition`; and [native student evaluator](../2026-10-01-continuation/student/native-quality-01/evaluate.py). No new prediction/label array or metric was read/computed in preparing this memo.

## Bind the same estimand before execution

Use the already defined pure held-out chains in folds0/1, the canonical row order, `need_prediction AND is_scored`, and warmup rows0–98 excluded. The intended population is the previously used860 sequences/250 chains; assert this against metadata. Pure chains exclude adjacent training members but have been exposed repeatedly during research.

Native student4 and strong4 have matching guarded INT8 cache provenance in the completed native-quality evaluation. The existing `student/evaluate_student.py` constructs all three recipes, including fixed3, from GPU/member caches. **Root's subsequent pre-result decision: use coherent GPU/AMP OOF caches for student4, strong4 and fixed3, plus the four individual student4 members.** Native scores are context only; do not combine a native profile with GPU fixed3 and attribute differences to composition. No fresh inference is required. This is not the historical one-epoch CSTAT1 baseline.

For target t and selected row i, let `Y_it = clip(y_it,−2,2)`, `P_mit = clip(raw fixed-profile prediction,−2,2)`, `w_it=|Y_it|`, and `W_t=sum_i w_it`. Preserve the actual registered profile's accumulation order and clip **after** mixing members. Target t0 and t1 have separate weights, weighted means and variances; final WP is their arithmetic mean. Unselected rows carry zero weight. Raw MSE and an average of group/sequence correlations do not implement this objective.

## Exact nonnegative deficit and group contributions

Within the complete scope being described, define weighted global means/SDs for Y and each P_m, then

`u_it = (Y_it−mu_Yt)/sigma_Yt`,

`z_mit = (P_mit−mu_mt)/sigma_mt`,

`e_mit = u_it−z_mit`.

For nondegenerate targets/predictions, `E_w[u²]=E_w[z_m²]=1` and `E_w[u*z_m]=rho_mt`, giving

**`L_m = 1−WP_m = (1/4) sum_t E_wt[e_mt²]`.**

For any exhaustive, mutually exclusive grouping g, its nonnegative score-deficit budget is

**`B_mg = (1/4) sum_t [sum_{i in g} w_it e_mit² / W_t]`.**

Then `sum_g B_mg=L_m`, and `sum_g(B_reference,g−B_student,g)=WP_student−WP_reference`. Keep target-specific contributions too. These are allocations of a fixed globally normalized deficit, **not** the score gain obtainable by correcting a group. Models each use their own global prediction normalization, so group delta attribution also contains changes in global centering/scale. The total delta is exact; local causal attribution is not.

For group rows, report selected-row count, target-weight fraction `pi_gt=W_gt/W_t`, deficit amount and share, and conditional error intensity `E_wt[e²|g]`. A large error amount with ordinary intensity can simply reflect large metric mass. Equal-chain summaries are labeled separately and cannot replace the pooled budget. Per-target weight concentration `(sum W_chain)²/sum W_chain²` is a concentration index, not an inferential ESS.

Global standardization must be retained when partitioning by class/chain/regime. If presenting a separate fold's own WP, recompute that fold's normalization and make a separate exact fold budget. Contributions of fold0/1 to the combined pooled budget use combined normalization and must not be confused with the two standalone fold scores.

The canonical helper returns zero for total weight<1e−8 or SD<=1e−8. Do not divide by a floor and pretend the standardized identity still holds. If a global side is degenerate, explicitly report the target's conventional score0 and its unallocated deficit contribution1/2; mark standardized target/group decomposition unavailable. For ordinary scopes require both targets nondegenerate. Numerical final rho clipping may contribute machine-rounding differences; assert reconstruction to1e−10 and record the residual rather than silently adjusting one group.

## Separate relative levels/scales from within-group fluctuations

For each target and group, using the same globally standardized u,z, define conditional means `mu_ug, mu_zg`, SDs `s_ug, s_zg` and covariance `c_g`. Then

`E[e²|g] = (mu_ug−mu_zg)² + (s_ug−s_zg)² + 2*(s_ug*s_zg−c_g)`.

Multiply each term by `pi_gt/4` and sum targets/groups. The three nonnegative components exactly reconstruct the same total deficit:

1. **Relative group-level mismatch.** Group means disagree after global centering/scale.
2. **Relative within-group amplitude mismatch.** Conditional target/prediction SDs disagree after global normalization.
3. **Mismatch of within-group fluctuations.** The remaining covariance shortfall; compute it directly without dividing by a zero subgroup SD.

Do not call the third term irreducible noise. Do not call the first two an achievable calibration gain. A group-specific affine transformation changes the global normalizers and may cross prediction clipping boundaries; correcting these terms while holding the rest fixed is not an evaluated deployable intervention. Full-chain target means/SDs use future labels and are diagnostic only. Class means depend on the population, weighting and target law; between-class covariance is a legitimate part of pooled WP, not automatically spurious signal.

The earlier `eda_scale.py` is useful historical context, not a reusable oracle-gain estimator: its class-scale search transforms already clipped sufficient statistics without reapplying clipping, and its sequence oracle uses the same sequence's labels. Neither its fitted scale nor its oracle gap is an identified attainable gain under the current clipped metric. No calibration/grid search is included here.

## Consensus residual versus disagreement: one exact identity

For the three fixed comparable profile outputs, standardized as above, define `zbar=(z_student+z_strong+z_fixed3)/3`. Per target/row,

`mean_m (u−z_m)² = (u−zbar)² + mean_m (z_m−zbar)²`.

After the same `sum_t E_w/4` weighting, **consensus error plus dispersion equals the mean of the three official-score deficits**. For student-specific attribution, with `rbar=u−zbar` and `d_student=zbar−z_student`,

`L_student = [sum_t E_w rbar²/4] + [sum_t E_w d_student²/4] + [sum_t E_w rbar*d_student/2]`.

The signed cross-term is necessary; dropping it misattributes student error. Both identities also partition by the fixed groups above. zbar is an arithmetic mean of separately standardized, clipped profiles; **it is not a new served ensemble or a predictor whose WP equals1 minus its residual energy**. No score for zbar is requested.

Student4 and strong4 share three members, and fixed3 consists of those shared members. Their agreement is therefore strongly dependent by construction. Dominant consensus error says only that these current forecasts share the miss. It cannot distinguish missing causal information, common training bias, feature insufficiency, target stochasticity or a distribution shift. Small disagreement neither proves irreducibility nor guarantees that another seed is useless. Root also includes the four individual student4 members: apply the same identity separately to that member set if used. Do not average all seven outputs together and present it as seven independent votes; the three profiles are combinations of overlapping members.

## Minimal readout: three linked tables, not a slice search

**A. Class/target budget.** For all three fixed profiles show the4 classes ×2 targets, counts/mass, total deficit budget and the three level/amplitude/fluctuation components. Give pooled and per-fold reconstruction checks, plus student-reference signed budget differences. Alongside it, give chain concentration (top10/top25 shares of mass and deficit, largest share) and the singleton-chain versus multi-sequence-chain totals. Chain length is an explanatory retrospective property, not an inference feature known at sequence start. Do not turn ranked hardest chains into a validation-selected training set.

**B. One causal position axis.** Use predetermined sequence positions `[99,2000)`, `[2000,8000)`, `[8000,20000)`; show their class-conditional mass/intensity and budgets rather than an unadjusted average that confounds class mix. This tests whether misses concentrate near reset/context acquisition versus later rows. Row position is causal; any mechanism still requires a future controlled intervention. No horizon, event-clock, threshold or crossing-grid search is implied. If root has a separate preregistered causal-regime hypothesis, substitute it before execution rather than accumulate dozens of overlapping tables.

**C. Shared miss versus disagreement.** Show the exact consensus/dispersion identity overall and by class, including the student cross-term. Add only the existing profiles' prediction-clipping weight/row fractions and target boundary mass at|Y|=2 as context. Label those boundary groups as diagnostic; target-derived categories are unavailable at inference. Do not optimize clipping, coefficients or calibrations.

Useful decision patterns are conditional. Large stable class-relative level/amplitude mismatch can motivate a matched causal calibration/training investigation, subject to the earlier negative calibration history. Excess early-row intensity within classes can motivate context/reset diagnosis. A broad shared fluctuation error with little profile dispersion weakens an immediate ensemble-expansion rationale but does not establish a Bayes ceiling. Error concentration tracking weight mass alone does not justify reweighting training. The EDA should end with at most one or two concrete follow-up hypotheses, or explicitly no supported intervention.

## Efficient computation, uncertainty and checks

All single-profile budgets and group components can be reconstructed from the six weighted sums already used by `fold_eval`: W, WY, WP, WYY, WPP, WYP. Expand the globally standardized quadratic within each group's sums. Consensus accounting additionally needs three pairwise weighted prediction cross-products. Thus one streaming pass can collect per-chain/group statistics; no fitted residual model, perturbed forecast or second pass over labels is required.

If uncertainty is allocated, resample whole chains within outer fold, paired across profiles, with the already established sample/seed convention. **Recompute global means/SDs and all derived budgets inside each draw.** Resampling precomputed row errors with fixed normalizers estimates a different conditional functional. Keep empty/degenerate scopes explicit. A small number of preregistered broad contrasts can receive intervals; every exploratory cell need not receive a significance claim. Bootstrap remains conditional on fitted checkpoints and exposed chains, excluding training-seed variation, adaptive search and hidden-domain shift.

Generated tests before real execution should verify: exact overall/group/delta reconstruction; constant/all-zero/masked targets/predictions; subgroup zero variance with nondegenerate global variance; clipping before normalization and after member mixing; positive affine invariance only when clipping is inactive; covariance/consensus identities including the student cross-term; and shuffled chunk/merge order of sufficient statistics. Bind prediction provenance/backend/mask/row IDs before the readout. These are correctness checks, not new empirical hypotheses or model-promotion gates.

## Added root-allocated ridge diagnostic: strongest traps and minimal checks

Root additionally proposes one fixed diagnostic with three nested feature blocks: prediction/confidence; then causal activity/history; then member deviations. The target is `q=u−rho*z` for the fixed student4. This is different from standardized residual `e=u−z`: for a nondegenerate scope, `E_w[q]=0`, `E_w[z*q]=0`, and `E_w[q²]=1−rho²`. Its relation to the actual post-clipping Pearson derivative is

`d rho / d P_i = w_i/(W*sigma_P) * q_i`.

The raw-output derivative additionally multiplies by `1{|raw_P_i|<2}`, with a declared convention at exactly±2. A probe of q alone asks about errors in post-clip prediction space; it does not automatically identify a useful raw-output correction for saturated rows. Keep the target definition explicit and report clipping mass, rather than silently changing the label after looking at results.

**The main leakage trap is normalization, including rho itself.** Inside each outer fold, divide whole chains into two fixed halves before outcome inspection. The outer-fold model has seen neither half. For each direction, compute weighted target/prediction means/SDs, rho, q centering/scaling, and all X weighted means/scales **only on the fitting half**. Target-dependent weights in X normalization must obey the same restriction. Transform held-out Y/P/X with those fitting constants. This makes held-out q the extension of a fit-defined gradient component; it is not exactly the held-out population's Pearson gradient. Using full-outer-fold rho or label-weighted moments before inner splitting would transmit held-out labels into the training target. Keep rows/chains and q definition identical across the three nested feature blocks.

Use `W_fit`-normalized weighted least squares: with fitting-standardized X and q, minimize `sum(w/W_fit)*(q−b−X beta)² + .01*||beta||²`; intercept b is unpenalized. This is a fixed lambda on the normalized Gram matrix, not `.01` added to an unnormalized sum whose strength changes with row count. Explicitly record/drop fit-constant columns, and define missing/unknown features from the fitting half without looking at held-out outcomes. No hyperparameter or feature-column search is implied.

Primary diagnostic reports held-out weighted q-MSE and squared-error skill relative to the **fitting-mean constant**. A conventional held-out-mean-denominator R² is also mathematically legal as an evaluation statistic, but must be separately named: its oracle held-out intercept is not the deployable baseline. Aggregate error numerators and their registered baseline denominators before forming an overall skill score; do not average unequal-half R² values. Keep per-target and per-outer-fold results so opposite signs remain visible. q's coordinate/scale differs between fitting halves; call the aggregation a cross-fitted normalized-gradient diagnostic, not a new official score.

Causal X may include current/past inputs and frozen model outputs. Do not include future event timing, actual target magnitude/sign, residuals whose target has not demonstrably matured, or whole-sequence/whole-chain target statistics. A retrospective regime label suitable for an error table is not automatically a legal inference feature. If a feature block uses an additional expensive reference model's prediction, label its serving cost; causal availability alone does not make it free.

Positive held-out skill would show that this fixed feature map predicts some variation in the fit-defined q beyond the baseline block. It does not establish a WP gain, an optimal correction size, independent confirmation, or a causal mechanism. Null skill does not imply irreducibility: this is one ridge model, finite features, a selected family of base models, and repeatedly exposed folds. The exact four-member overlap is particularly important when interpreting member-deviation features.

Minimal generated checks before real execution:

1. For a nondegenerate synthetic scope, verify weighted q mean0, weighted covariance with z0, variance `1−rho²`, and the analytic derivative against finite differences with the appropriate clipping mask. Handle constant/all-zero scopes explicitly.
2. Change only held-out labels and verify that every fitted q/X normalizer, rho, ridge coefficient and held-out **prediction** stays identical. Held-out q targets, evaluation weights and errors may change. The same test should fail for an intentionally full-scope-normalized control, proving sensitivity to the leakage route.
3. On a small synthetic weighted linear problem, verify the intercept/no-penalty solution and lambda/W convention against a separate direct least-squares calculation; confirm that nested feature arms share the same target, row mask and split. Retain a deterministic nuisance-only control as a correctness check, not as an empirical extra null search.

If ridge intervals are requested, whole-chain loss resampling with the fitted ridge models and fit-defined coordinate systems held fixed gives conditional diagnostic uncertainty. It does not include uncertainty of fitting those normalizers/ridge coefficients; refitting would be a different, more costly protocol. Keep this distinct from the exact metric-budget bootstrap above, which must recompute the global official-metric normalizers per draw. No new empirical execution is performed by this design document.
