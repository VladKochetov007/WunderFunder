# QRENEWAL-01 input-only execution specification v1

Status at preparation: no actual data execution. This specification resolves implementation details left open in `../quant-reframe.md`; it authorizes no target association, forecast fit, WP query, feature deployment, or submission. The controller alone may allocate the command after independent review. SAM sources are untouched.

## Scope and immutable sample

`selected-chains.json` contains exactly the 256 ordered `{chain,row_group,class,half}` records from `research/iterations/2026-10-01-quant-controls/prior-inputs.json`, which copied the original QMARK selection. There are 32 chains in every half × class cell, one 20,000-row training sequence per chain. No reselection or replacement is allowed. These chains were selected previously to exclude chains containing validation neighbors; this run reuses that historical manifest and does not reopen masks or fold caches. Reuse is exposed train diagnostic evidence, not fresh competition evidence.

The sole Parquet read is `read_row_group(row_group, columns=ALLOWED_COLUMNS, use_threads=False)` with this exact ordered whitelist: `a5,a6,i0_dv0,i0_dv1,i0_dv2,i0_dv3,i0_dp0,i0_dp1,i0_dp2,i0_dp3`. The controller clarified that causal `i0_dp` inputs are permitted for the original miss strata. No `t0`, `t1`, scoring flag, prediction mask, OOF cache, model, or validation file is opened. Footer/schema metadata are read to verify sequence lengths and source identity. Source byte count, mtime_ns, footer length, and SHA256 of footer+trailer must match the previous receipt; full parquet SHA256 is not claimed. Full stat/footer fingerprint must also remain unchanged after execution.

The 10 input columns are loaded once in manifest order, preserving float32 dv/dp comparisons. Ages are copied to float64 before reset comparisons and fitting. This deliberately uses a bounded in-memory input table (about287 MB with age copies) rather than `SequenceFile`, which reads unauthorized columns. All calibration in one direction uses only its fitting half. Directions run once, fitting half0/auditing half1, then fitting half1/auditing half0. Neither direction is selected over the other.

## Causal resets and mark states

`R[0]=D[0]=R*[0]=False`. For later rows, R requires finite adjacent a5 and `a5[t] < a5[t-1]-1e-6`. Negative a5 values and the ±5.19 extremes are not excluded; anonymous codes are not physical ages or seconds. R* further requires all current dv finite, abs strictly below5.19, and current `ptp(dv)>1e-6`. D uses precisely the existing payload detector: both adjacent dv vectors pass finite/abs<5.19, any coordinate differs exactly, and current ptp exceeds1e-6. Invalid marks hold state; no old or future payload is borrowed. Every competition sequence starts from zero state and no preceding age. Resets at an unchanged zero code are unobservable by this recipe; the test concerns observed downward resets only.

Fit mark median and IQR on all fitting-half rows admitted by the union R* OR D, counting a row once. Use NumPy linear quantiles; scale=max(IQR,1e-3). There is no per-audit-sequence centering. The state update uses `u=clip((dv-median)/scale,-4,4)` and `lambda=2**(-1/16)`; update S^R only at R* and S^D only at D. All four coordinates otherwise hold. Serving prefix causality therefore follows from fixed fitting-half constants, current/previous input, and preceding state.

Report every R* not D. An intended repeated-dv miss requires previous dv validity and exact current/previous dv equality. Split into (a) valid dp endpoints with changed dp, (b) valid dp endpoints with unchanged dp, and (c) invalid dp endpoints; dp validity is all finite/abs<5.19. Previous-invalid dv misses and any unclassified miss are separate. To conservatively implement the proposal's intended cases(a)+(b), only their sum can satisfy the repeated-mark gate; invalid-dp and previous-invalid-dv cases cannot make it pass. All R* remain in its denominator. This pre-data specificity is stricter than simply counting R* not D.

## Inverse-clock fit: exact optimization

For each age channel j, concatenate finite fitting-half values, then obtain 32 equally spaced quantiles from0 through1 with NumPy `method='linear'`; deduplicate repeated knots. Reject fewer than4 distinct knots, 95th-minus5th percentile <=1e-12, or IQR<=1e-12. Quantiles and IQR use fitting rows only. Unequal valid-row counts may affect quantiles; the loss, separately, gives each chain exactly equal weight. No audit knot or scaling statistic is used.

Let raw knots be k_jb. Variables d_jb are nonnegative *output increments* in adjacent knot bins; this is equivalent to nonnegative slopes. Define `B_b(x)=clip((x-k_b)/(k_(b+1)-k_b),0,1)` and `g_j(x)=sum_b B_b(x)*d_jb`. Each leftmost knot has g=0, removing the arbitrary offsets from optimization. This interpolation clamps beyond support, but clamped pairs cannot count as audit successes.

On each fitting chain, a pair is eligible only when all four adjacent a5/a6 values are finite, both raw increments are >=0, and both endpoints lie inside their channel's fitting support. Thus positive and zero ties both enter fitting. Microdrops in[-1e-6,0) do not enter the equality fit, but remain in the audit coverage denominator. Keep all eligible indices up to1024; otherwise take `indices[floor(linspace(0,len(indices)-1,1024))]`. No RNG, chain replacement, activity stratification, or further pair search. A chain with no eligible pair stops this recipe.

Let A_ct be the concatenation of the g5 basis increment and negative g6 basis increment. Let r be the median strictly-positive (`>1e-12`) sampled raw a5 increment divided by fitting a5 IQR, pooled over the retained pairs. No positive r stops. This makes the residual unit explicit rather than allowing the raw input scale to choose the strength of the ridge.

In standardized input coordinates `(a_j-median_j)/IQR_j`, first slopes are `s_jb = d_jb * IQR_j / (k_j,b+1-k_jb)`. Minimize the convex quadratic

`L(d) = mean_over_chains(mean_over_sampled_pairs((A_ct d / r)^2)) + .001/2 * [mean_b((s_5,b+1-s_5,b)^2) + mean_b((s_6,b+1-s_6,b)^2)]`.

This is the declared second-difference smoothness penalty: adjacent first-slope differences on the unequal quantile grid, with no additional bin-spacing divisor or hidden identity ridge. It has no intercept penalty. Enforce the exact linear equality `g5(q95_fit5)-g5(q05_fit5)=1`. A constant joint map therefore cannot solve the problem. No g6 range equality is imposed: shared increments determine its scale insofar as the input equation supports it.

Solve with SciPy SLSQP, exact quadratic gradient, nonnegative bounds, and one equality; maxiter1000, ftol1e-13. Start each map as linear between its own5th/95th percentiles. Divide the entire Hessian/objective by `max(spectral_norm(H),1e-12)` solely for numerical scale. Stop on optimizer failure, nonfinite coefficients, coefficient min<-1e-10, or range error>1e-7. Do not repair a failed solution, restart, increase iterations, change solver, add knots, alter the ridge, search lags, or try decreasing maps.

Additionally verify KKT stationarity and complementarity for the scaled quadratic. Free coefficients are x>1e-8; estimate the equality multiplier by least squares on free coordinates. Divide residuals by `max(1,max(abs(gradient)),abs(multiplier)*max(abs(range_constraint)))`. Require free-coordinate stationarity, negative active-coordinate dual violation, and maximum absolute coefficient×Lagrangian-gradient complementarity all<=1e-5. Lack of a free positive-range coordinate or any residual violation stops, without solver retries.

Report the regularized Hessian eigenvalues on the equality-constraint tangent space (orthonormal SciPy null_space basis). Reject smallest scaled eigenvalue<=1e-12 as a fixed numerical stability rule. Separately report the **unregularized weighted increment design** tangent rank, full Gram eigenvalues, retained-subspace condition, and full condition if full-rank, using relative Gram-eigenvalue threshold1e-10. Ridge-induced uniqueness is not claimed to establish data-only clock identification. These rank reports are descriptive; there is no post-hoc rank-based gate change.

After fitting, evaluate all fitting-half eligible pairs, not just sampled pairs. Let c be the median positive common mapped increment `(delta_g5+delta_g6)/2`, restricted to both mapped increments >=-1e-12 and c>1e-12. Divide both maps by this one c; absence of c stops. Then fit a5 reset origin as the median mapped a5 over all supported fitting R endpoints. Fit the excursion scale from complete consecutive-R runs: `[start_reset, next_reset)` must have every a5 value finite/in support, every internal raw increment >=0, and positive mapped `(age[next_reset-1]-age[start_reset])`. Use the median excursion. No origins or no complete positive runs stops. Report every complete-run admission/rejection. The origin is a stored offset for comparisons; subtracting it from all g5 values would leave increments unchanged.

## Held-half accounting and fixed gates

For every chain report all N-1 pairs, finite/invalid, either/both reset indicators, finite non-reset, microdrop non-reset, nonnegative, both-raw-zero ties, one-raw-zero ties, outside-support non-reset, mapped-zero common increment, and usable counts. A finite non-reset pair is finite with neither raw increment <-1e-6; it includes microdrops, ties, and out-of-support pairs. Usable pairs are nonnegative raw increments, both endpoints inside both supports, finite nonnegative mapped increments (tolerance1e-12), and positive common mapped increment>1e-12. Counts have intentionally overlapping diagnostic strata; finite versus invalid is the disjoint all-pair partition.

At the floating-point threshold, every reset calculation uses the exact same expression `current < previous - 1e-6`, including the pair audit. It does not replace that expression by `(current-previous)<-1e-6`, whose rounding can differ at the boundary.

Both directions must pass all five fixed gates:

1. Usable pairs / **all finite non-reset pairs** >=.50. No denominator removing zeros, microdrops, or outside-support rows.
2. On usable held pairs, pooled median normalized increment discrepancy<=.05 and p90<=.20, with discrepancy `abs(d5-d6)/(abs(d5)+abs(d6)+1e-8)` after the common unit normalization. Report all per-chain counts/errors/coverage and the equally counted distribution of per-chain medians, with class metadata; pooled output alone is insufficient for interpretation.
3. At least90% of **all observed held R endpoints** are supported and within `.05 * fit_median_complete_excursion` of the fixed fitting origin. Unsupported reset endpoints remain failures in that denominator. Report sentinel/non-sentinel and invalid-current-mark strata separately.
4. Qualified miss cases(a)+(b) / all R* >=.10, with at least20 held chains having at least one qualified miss. Report all validity misses and dp strata even if gate fails.
5. At fixed rows `99+64*k<20000` (311 per sequence), compute S^R-S^D. Its four RMS values average chain means equally and divide by `max(IQR_fit(S^D_on_same_fixed_grid),1e-3)` coordinatewise. At least two coordinates must have RMS>=.10. This resolves “fitting-IQR RMS” as the old-clock state's fitting grid IQR; mark scaling and this reference scale are distinct and both saved.

Zero denominators fail, not pass. All diagnostics are emitted even if a previous gate fails, unless an explicit fit/resource/input validity stop occurs. No alternative sample or target phase follows automatically. A pass only supports a reusable anonymous shared-clock/reset structure and a material changed causal mark history. Shared preprocessing could also pass. It establishes neither physical time, trade executions, sign, physical prices, nor price-target relevance.

There is no additional per-chain pass-count gate beyond the declared coverage and20-chain repeated-mark gate. Thus a pooled pass alone is not described as uniformly shared support across chains. The saved per-chain distribution is required evidence for any later allocation judgment; it cannot be retrospectively converted into a new cutoff for this run.

## Resource envelope and receipts

Linux runner enforces affinity to the first2 currently permitted CPUs, all numeric thread environment variables=2, hidden GPUs, RLIMIT_AS=4GiB, RLIMIT_CPU=480 seconds, and a480-second wall alarm including imports/read/fit/report. It checks source hashes and caps before any input read. It creates `renewal/result/` exclusively; an existing directory refuses another run. A start receipt precedes dataset access; errors create `failure.json`, and a failed attempt is retained/counts. SIGKILL/OOM may leave only the start receipt. There is no automatic retry or fallback. Two directions write separate immutable JSON before final aggregation. All source fingerprints, source/plan hashes, versions, resource use, and read time are recorded. Budget is an enforced upper bound, not an empirical benchmark already obtained.

The one controller command, after review/allocation, is:

```sh
cd /home/grinchenkoeg/WunderFunder-creative
.venv/bin/python research/iterations/2026-10-01-overnight/renewal/run_inputs.py --plan research/iterations/2026-10-01-overnight/renewal/execution-plan.v1.json --execute-inputs
```

Synthetic validation only uses generated arrays: known shared linear and nonlinear clocks and their true inverse oracle; independent positive-increment false clocks; constant support rejection; repeated same-payload histories with identical old states but different renewal states; every-prefix parity; independent sequence reset; missing/neutral/sentinel marks; all-row denominator edge cases; and the actual mark audit including changed/unchanged dp strata. No empirical resource profile or data-shaped pilot is authorized during preparation. Exact latest receipt and code hashes are frozen in the execution plan.
