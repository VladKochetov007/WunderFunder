# R5: one conditional-moment decoder probe for idea12

Objective_mode: competition allocation through bounded discovery, tierC external validity with tierA algebra/source/metric checks. The user authorizes EDA and simple CPU models. This is one small analytic fit on the existing sample, not a new GPU campaign, learned recurrent model, ridge-grid search or confirmation study.

## Fixed mechanism and selection history

Idea H-NF19-12 proposes estimating coherent label moments instead of multiplying separately estimated sign and magnitude. R3/R4 have not measured this mechanism. R5 tests only whether a row-dependent estimated denominator helps a simple moment predictor relative to a constant denominator, holding fitted heads exactly fixed. It does not test end-to-end joint WMSE optimization or prove an advantage of sign-head capacity.

Use only raw+signed-delta EMA(HL32),224base channels, selected as a development allocation because R3 showed a history signal. Reuse its exact causal construction, sequence reset and whole-history-before-stride semantics. No raw/tanh/width/seed alternatives. The entire F0 panel is exposed; there is no fresh holdout.

Reuse B/research_tmp/nf19-sample-r3-v2, SAMPLE SHA b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64, canonical F0c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7.64fit components, one train sequence each,31808scored stride40rows;32evaluation components, one valid sequence each,59436original scored rows. No resampling, row exclusion, changed scored mask, new feature fit or dataset scan.

For each clipped target y=clip(Y,−2,2), define fitting targets A+=y²·1(y>0), A−=y²·1(y<0), B=abs(y). Fit all six columns (three moments times two targets) jointly by unweighted analytic ridge on the same fit-stride rows. Base mean/std are unweighted fit-only; standard deviation below1e-12 becomes1. Intercepts are unpenalized: center the standardized features and moment targets, solve (Xc'Xc+100I) coefficients=Xc'Zc, reconstruct intercepts. Float64 analysis. Fixed ridge alpha100, no CV or hyperparameter selection. This alpha is fixed from the completed R3 development protocol, not a claim of equal effective regularization between different target objectives.

The three moment heads share the exact fit data, normalizer and Gram matrix. Signed numerator Dhat=Ahat+−Ahat−. Linear ridge on the signed target abs(y)*y is algebraically equivalent with the same linear estimator, but it is not fit separately on actual data. These unconstrained linear moment estimates can violate true conditional-moment bounds; measure that failure rather than claiming probabilistic consistency.

Let muB=mean_fit(abs(y)) per target and epsilon=0.05*muB. Require both muB>0, otherwise fail the recipe without excluding rows/targets. Two fixed decoders use identical fitted heads:

1. conditional_ratio: Dhat / maximum(Bhat,epsilon).
2. constant_denominator: Dhat / muB.

No positivity clipping of A+/A−, no extra prediction clipping/calibration or outcome-based floor selection. All raw predictions must be finite. The unchanged official scorer alone clips predictions/labels±2 and uses its original abs-clipped-label weights/mask. The denominator floor is a fixed engineering choice, not a tuned or guaranteed optimal treatment.

Under a representative common conditional population, the ideal ratio E[abs(y)*y|H]/E[abs(y)|H] minimizes abs(y)-weighted squared error. Sparse hidden evaluation selection/source changes can invalidate transfer of the training moments; this assumption is unproven. R5 does not estimate hidden selection S or redefine the official evaluation measure.

## Outcomes, matched control and limits

Primary: conditional_ratio minus constant_denominator canonical t0/t1/WP on the full original evaluation masks, plus all32fixed-prediction drop-one contrasts. Same fitted functions/head count/data/normalizer/regularizer makes the decoder the only primary treatment. No independence or significance claim from32drop-one values.

Secondary: compare each decoder descriptively against the fixed R3 full64_raw_signed_ema_hl32 selected alpha100 endpoint (WP.53600373890), using only frozen R3 RESULT SHA a7e19f6358fbd1f61496efd7c1712b80cb7d582f22445a1e3c12988851b323dc and its component moments. Do not rerun R3 fit/inference/grade. Direct WMSE and unweighted moment regression differ in objective/functional class; these are recipe comparisons, not an isolated causal objective effect.

Report per-target muB/epsilon; actual coefficient norms; Bhat<epsilon and Bhat<0 fractions; Ahat+<0/Ahat−<0 fractions; abs(Dhat)>2*maximum(Bhat,epsilon) fraction; decoder raw abs(pred)>2 fractions and denominator distributions. Compute these descriptively on full eligible fit masks and original eval masks, with component counts and pooled totals. No diagnostic chooses rows, floor, alpha or decoder. Retain all outcomes even if denominator noise is severe.

Save canonical additive component six-sums for both decoder endpoints, per-component/full/drop-one metrics, reference deltas, all counts and source/sample identities; numeric NPZ with shared mean/std/moment coefficients/intercepts/muB/epsilon on B. NPZ producerSHA/stat is not independent payload verification. Cost/progress/START/DONE/FAILURE receipts remain exact bytes.

## Bounded execution

One serial B CPU job:0.5CPU/MemoryHigh1.4GB/MemoryMax1.5GB/swap0/600s, CUDA hidden, BLAS1, physical B/research_tmp, create-only source/output/unit. Reuse canonical data/scorer and existing analytic predictor/data helpers; do not modify R1–R4 or frozen B source. Before arrays: freeze complete import closure and exact source pins, independent design/implementation review, isolated package/data-read-sentinel smoke and tiny algebra/decoder/canonical-pooling/report tests. Tiny synthetic checks may run in existing maintenance.venv on Mac; no arrays/real fits/install there.

Fresh existing F3/current-step/ALL-populated-descendant/resource/ownership/headroom/disk/sample/reference/source/outputabsence gates before launch. Parent2CPU/18GBhigh/20GBmax/no-swap/17GBcommitted target; current fit1CPU/8GB,controller.1CPU/512MB. No foreign GPU/process actions, new cells or permission/budget question. If caps/gates fail, diagnose rather than expand them or change the sample. Stop after these two decoders or a concrete validity/resource failure; no automatic grid/retry. Capture exact exit/status/raw receipts before stopping only the owned exited unit. Root owns admission/capture/publication; separate actor/critic roles own implementation and audit.
