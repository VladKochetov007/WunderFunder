# QRENEWAL-01 — count observed renewals, not only changes in their payload

**One proposed mechanism:** the four trade-volume codes can repeat while a separate age-like channel resets. The existing trade detector observes payload changes, not every possible occurrence of a repeated payload. If `a5` really has an age/reset structure, recover those anonymous renewal occurrences and use them as the update clock for a small continuous-mark history. This is a measurement/partial-observability hypothesis, not another categorical transition model or an extra neural layer. It can fail cheaply before forecasting.

Status: **proposed, no empirical run**. Only this document was written; frozen `sam/` files were not changed. This scout has now read the historical deep/EDA/quant-control reports and source, plus the overnight 68-channel F/T implementation and the controller's null-outcome summary. No dataset, checkpoint or prediction array was read for this proposal. No nuisance fit, price-target fit or WP query ran.

## Why this survives the duplicate check

`transition/features.py` emits an event only when the current four `i0_dv` codes differ from the immediately previous row, both endpoints pass its validity rule, and the current vector is nonneutral. Its current/previous categories, cumulative F/T counts, probability, residual and surprise all use that clock. Equal successive raw payloads cannot produce two emissions, even if another input records a fresh occurrence. The broad `heuristics.py` trade counter and old `quant.py` trade-volume-event flags also key off payload inequality. Conversely, old `micro_features.py` detects non-median rows and can treat held payload rows as trades. Neither rule identifies every occurrence of a repeated observation.

E-091 reported Spearman association between `a5` and rows since a trade (~.614) and between `a6` and rows since a book change (~.79). The ledger then called these real-time clocks. **That semantic conclusion was not established by the reported correlations.** The new proposal explicitly tests the stronger age-clock structure rather than inheriting the name. The official feature description leaves `a0..a7` anonymous. A successful audit would identify an additive observable clock candidate up to scale, not certify seconds, executions, aggressor side or a physical exchange price.

The earlier EDA already noted that the detector *could* omit repeated same-code trades, but no reset-channel reconstruction/audit was found in the searched experiment ledger, hypothesis ledger, quant journal, deep/EDA/quant-control dossiers or relevant feature sources. The exact proposed intervention is therefore not a rerun of an evaluated arm. If another unpublished reset-based test exists, inspect it before allocation and reject this proposal if equivalent.

| Prior result | Constraint and actual distinction |
| --- | --- |
| Ordered marks beat marginal frequencies by ~.79 nats; raw-packet surrogate erases that advantage | Predictability belongs to the emitted code sequence. It does not establish useful missing occurrences or target relevance. No new mark-log-loss competition is proposed. |
| Overnight 68-channel T/F forecasting null | Both arms use the same payload-change clock and discard most continuous amplitudes from their added block. The original GRU still sees raw amplitudes, so changing the clock must demonstrate a concrete new history, not claim new raw information. |
| E088, raw-a probes, trade EMAs and `quant` sums negative | Those make another generic raw-flow feature low prior. This proposal changes which observations advance the summary, with an equal-dimensional old-clock control. It does not merely add another half-life. |
| E112 price-impact MO0/MO1 split negative | That detector/side taxonomy differs and uses a next-quote-derived side table. This proposal fits no price-side map, applies no delayed impact taxonomy and claims no economic sign. |
| Hazard/age output rescaling negative | No prediction rescaling, probability-of-imminent-event head, or future event-time oracle. The candidate records repeated input occurrences before predicting the original target. |
| Dual clock / attention / contextual-state memory | No additional recurrent network, attention, decoded quote clock or stored GRU token. The new state is four explicit continuous-mark averages on a separately tested observation clock. |
| Global sparse-volume dictionary failed | No inverse queue sizes, sparse factor geometry or best-queue identification is assumed. |

## A precise counterexample and representation

Consider two causal prefixes ending in identical current raw trade payload, identical last payload-change row, identical final `a5`, and the same original F/T state. In one prefix `a5` rose and reset three times while that payload was held; in the other it reset once. These prefixes are indistinguishable to the F/T summary. If resets are genuine observable renewals, they represent different repetition histories. The full row GRU can in principle distinguish the prefixes because it saw `a5`; the claim is an accessible representation/inductive bias, not information absent from the112 inputs.

Freeze the candidate observed reset indicator before data:

`R_t = 1[finite(a5_(t-1), a5_t) AND a5_t < a5_(t-1) - 1e-6]`.

Set `R_0=R*_0=D_0=0`. Do not exclude age values merely because they equal the common extreme ~−5.199: that may be the encoded zero of a timer rather than missingness. Report extreme/flat/reset fractions separately. For the **mark-bearing** reset `R*_t`, also require the current four volume codes to meet the existing finite, abs<5.19, nonneutral criterion. Reset rows without an admitted current mark are explicit missing-mark cases; they must not silently borrow a future or old payload. `D_t` is the unchanged existing transition-feature detector, including its previous-endpoint validity condition. All state resets at the competition sequence boundary.

For each of four current volume coordinates, use a single training-only robust scale, common to both clocks:

`u_tj = clip((dv_tj - median_j) / max(IQR_j, 1e-3), -4, 4)`.

Compute two four-dimensional states, initialized at zero:

`S^R_t = λ*S^R_(t-1) + (1−λ)*u_t` when `R*_t=1`, else hold;

`S^D_t = λ*S^D_(t-1) + (1−λ)*u_t` when `D_t=1`, else hold;

with **λ=2^(−1/16)**, fixed from the old trade-EMA horizon rather than selected in this audit. The proposed forecasting treatment would supply `S^R` instead of `S^D` to an identical four-channel zero-initialized head extension. It does not combine a new radius, age filter, side decoder or two-instrument model. No i1-specific clock is available from the present evidence, so do not quietly extrapolate `a5` to i1.

Repeated equal payloads matter because they replace older contrary marks in the last16-occurrence history even when the payload-change process holds. These continuous coordinates remain anonymous; positive/negative code values are not identified buy/sell or physical volume.

## Cheap input-only falsifier: does a5 actually behave like an age clock?

An age representation makes a restrictive prediction beyond correlation with row counts. Suppose

`a5_t = f5(τ_t − τ_last5)` and `a6_t = f6(τ_t − τ_last6)`,

with fixed increasing encodings. On rows where neither clock resets, inverse maps `g5,g6` must obey

`g5(a5_t) − g5(a5_(t-1)) ≈ g6(a6_t) − g6(a6_(t-1)) ≥ 0`.

This follows algebraically from a shared elapsed clock. A correlation with rows-since-event does not imply this identity. On an a5 reset, `g5(a5_t)` should return to a stable origin even when the four trade-volume codes are unchanged. These are the two discriminating expected observables: **shared positive increments between resets and a stable reset endpoint**.

Use the same frozen256 train-only chains previously used by QMARK/PRIOR02/DETECTOR02, one sequence per chain, excluding every chain with a validation neighbour. Preserve the exact two halves and class metadata for sampling/reporting only. No validation data, scoring masks, prediction caches or new sample search. Reuse does not make the evidence fresh.

Fit/audit once in each direction. Fit two monotone piecewise-linear inverse maps with32 knots each at fitting-half input quantiles. Parameterize them by nonnegative slopes. Minimize the chain-equally-weighted squared difference of the two mapped increments on finite, nondecreasing, non-reset endpoint pairs, with fixed second-difference ridge .001 after standardization. Enforce a nonzero range for g5 between its fitting5th/95th percentiles; otherwise the constant solution solves the objective trivially. After fitting, rescale both maps so the fitting median positive common increment is1. Set the g5 origin from the median mapped a5 at fitting resets. Clamp outside fitted knots and report boundary coverage. No targets participate in these fits.

The nonlinear maps are diagnostic only; this candidate's serving reset is the raw downward step. Their purpose is to falsify the interpretation that makes repeated resets interesting. Their shared scale is arbitrary. Synchronized encoding machinery could also satisfy the identity, so passing does not identify physical time.

Freeze these conservative allocation cutoffs, not significance claims:

1. At least50% of all finite non-reset row pairs are inside both fitted supports and yield usable positive common increments. Report all-row denominators and zero-time ties separately; do not throw them away from coverage.
2. On those held-chain pairs, median `|Δg5−Δg6|/(|Δg5|+|Δg6|+1e−8)` ≤.05 and90th percentile ≤.20 in both fit/audit directions. Also report a per-chain distribution; a pooled count dominated by one active chain is insufficient.
3. At least90% of held-chain a5 reset endpoints lie within5% of the fitting median complete-run mapped age excursion around the fixed reset origin. Large origin drift falsifies this fixed-clock recipe. Report finite/invalid/sentinel strata rather than redefine reset states afterward.
4. Mark-bearing resets not emitted by D comprise at least10% of R* occurrences in each half, with at least20 chains per half contributing such occurrences. Split these misses into (a) unchanged dv but changed dp, (b) unchanged complete dp/dv payload, and (c) neutral/validity-rule differences. Cases(a) and(b) are the intended observable distinction; a result entirely due to changing validity filters is not a pass.
5. `S^R−S^D` has at least .1 fitting-IQR RMS on fixed audit rows for at least two of four coordinates in each half. A changed event count with effectively unchanged candidate state does not justify forecasting.

These thresholds deliberately make a false positive harder and can reject an approximately valid timer that this32-knot implementation represents poorly. Record that as failure of this fixed practical recipe, not a theorem that clocks are unrecoverable. Do not add knots, try decreasing transforms, search lags or move cutoffs after outcomes. A nonzero cross-lag alignment peak may be a descriptive failure explanation, but does not authorize delaying/reassigning events in this recipe.

## Separate price relevance from input predictability

Passing the input audit still does not show useful price information. If the controller allocates the following target-reading stage, use a fixed **association audit**, not another code NLL and not a competition WP query. It fits nuisance relationships among causal inputs only; no model of price targets is fitted.

At fixed rows `99 + 64*k < 20000` in every audit sequence, construct `Z = S^R−S^D`. This row rule uses neither completed interval length nor future targets. Construct the nuisance design X from the existing211 current inputs, the old-clock S^D, current R*/D flags, their causal ages and cumulative counts, and previous a5/a6. Including reset/count/age controls asks whether the **mark history** matters beyond merely discovering another activity indicator. Do not select only reset rows or high-error intervals after seeing labels.

On the fitting half, standardize X/Z from inputs and fit the four-output input-only ridge `Z≈XB`, with fixed penalty .01 on standardized covariance and an unpenalized intercept. No price target enters B. On the opposite half evaluate `U=Z−XB`. This residualization removes linear redundancy with existing input/statistic coordinates; it does **not** condition on the full nonlinear GRU or prove conditional independence. That limitation is material.

Only now read the sampled train targets and define `Dtarget=(clip(t0)−clip(t1))/2`, `w=(|clip(t0)|+|clip(t1)|)/2`, clipping each target to[-2,2]. Compute the four within-chain weighted correlations of U with Dtarget, then combine paired chain covariances/variances with equal chain weight. Centering is a retrospective diagnostic calculation, never a feature. Also report unweighted versions and all four historical classes so class or weight composition cannot silently drive the conclusion.

Predeclare one vector-valued evidence summary per half: c contains those four correlations. Require `sqrt(mean(c²)) ≥.05` in **both** directions and cosine(c_half0,c_half1) ≥.5. Do not pick the largest coordinate or flip signs after seeing labels. Bootstrap whole chains within half×class, paired across old/new states, solely to describe conditional uncertainty; require the point-gate and report intervals without converting the gate into a p-value claim. Inputs/feature/scaler fitting and this target calculation must be separately timed and hashed.

As a fixed nuisance control, permute whole Z histories between audit chains **within class and half** using seed20261031 and rerun the same association calculation on the identical311-row grid without refitting B or changing X. This intentionally breaks input-history alignment and has imperfect covariate matching; it is a negative control, not a calibrated conditional randomization test. Require original correlation RMS to be at least2× the permuted RMS in each half in addition to the .05/cosine gates; otherwise stop. Preserve all chains; do not search another permutation seed.

If the clock passes but this direct target association fails, close the proposed marked-renewal representation instead of returning to richer code predictors. If association passes, it warrants at most a separately registered small matched forecasting test against the old-clock control **and the ordinary V1-like ensemble**, with paired seeds and fixed serving cost. It still does not establish incremental signal beyond the current GRU: nonlinear nuisance dependence, train sampling, target overlap and hidden-period shift remain explanations. Do not market an association gate as an expected WP gain.

## Cost, artifacts and implementation boundary

One future script can reuse `SequenceFile(...).read(..., targets=False, mask=False)`, the fixed chain manifest, `event_features` and the immutable old `transition.features.detect`. Stream sequences; retain at most1024 deterministic equally spaced eligible increment pairs per chain for the64-variable clock fit and the fixed311-row grid per sequence for the association audit. If explicitly allocated, the second stage reads the selected train row groups' target columns and uses only the fixed grid for association; do not claim Parquet physically reads only311 target rows. Sampling complete intervals for reset-origin diagnostics may use their endpoint retrospectively; those durations never enter an online feature or select the price-association rows.

Concrete proposed entry point: `quant_renewal_eda.py --plan quant-renewal-plan.json --stage inputs`, followed only on its fixed gate by `--stage target-association --input-result <immutable result>`. This is a minimal implementation path, not an existing or authorized command. The plan should freeze chain IDs,32 knots, ridge penalties, the64-row grid, λ, all gates and one permutation before execution. Output all reset/miss categories, support, per-chain clock errors, state divergence, input-only nuisance hashes and (if allocated) all four correlation coordinates—never just the favorable aggregate.

Planning cost: ≤480 seconds,2 CPU threads,4GiB RAM, no GPU, no neural/model forecast fit, no validation WP. The256-sequence raw read is about5.12M rows; fit matrices are small and audit-row storage is around80k×~220 floats, so streaming should remain well within4GiB. This is a bound to measure, not a benchmark already run. Abort at the cap; no automatic fallback sample. Any later forecasting experiment consumes its own fit/WP budget.

If promoted to a feature, serving adds four state floats, previous a5, a few flags, and roughly4×128 extra head multiply-adds per modified h128 member. It needs no inverse-clock evaluator at inference and no extra model. Native/reset/chunk/prefix parity and complete-ensemble runtime remain mandatory. The potential is a materially different event history of repeated marks, especially if the old detector misses many same-payload renewals; the prior remains modest because the ordinary GRU already sees a5 and has absorbed many explicit-flow alternatives.

**Decision:** investigate the observation clock before another market-process model. The earliest useful result may be a clean rejection: if a5 does not satisfy the age/reset constraints, or if recovered repeated-mark history has no stable train-target association, this specific mechanism has no allocation case.
