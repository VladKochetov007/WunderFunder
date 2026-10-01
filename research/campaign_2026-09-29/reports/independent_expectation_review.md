# Independent public-expectation audit

Verdict: the existing artifact is useful exploratory evidence, but BLOCK any
claim that it supports an honest expected public score of at least 0.6660.
No listed candidate supports that claim. No calibration computation was launched
by this reviewer, as requested; this review inspected the script, artifact,
historical helper implementations and failed submission-package source.

## What the existing artifact establishes

`artifacts/nested_public_calibration.json` contains 11 user-reported portal
observations, including both EMA6 failures (0.6616 and 0.6612), and 21 proxy
variants. Its nested LOSO RMSE is 0.00179541, MAE 0.00137904, mean prediction-minus-
public error +0.00012043, observed error range [-0.00222555,+0.00435818]. EMA6
held-out forecasts overestimate their portal scores by +0.00129355/+0.00233446.
The best listed exploratory forecast is `pair_base15` at 0.66210225, below both
0.6660 and the incumbent's user-reported 0.6622. These are reported artifact
values, not an independently recomputed result or a prediction interval.

## Prioritized findings and remedies

1. **Two public-predictor associations are CPU-dependent.**
   `submissions/final_ema6_shared/pkg/solution.py:42` and
   `submissions/final_mixed_ema6_shared/pkg/solution.py:42` replace the two-member
   manifest with member 0 on v2. `campaign_public_calibration.py:27` and
   `lb_calibration.py:36` assign the failures to two-member predictions without
   portal CPU-level evidence. Keep both failures; mark their association uncertain
   and report both v2/v3 mappings or archived-package predictions under a proven
   portal variant. Prediction hashes alone do not resolve this ambiguity.
   The author subsequently confirmed identical EMA6 member-0 NPZ and v2 library
   bytes in both packages; their differing portal scores would contradict a
   common deterministic v2 path if those associations and evaluation period are
   exact. That is useful consistency evidence, not direct observation of the
   portal CPU path. The author added pair/single mapping scenarios; their rerun
   remains deferred, so no revised numerical result is approved here.

2. **Nested selection is correct only for the current fixed proxy family.**
   `campaign_public_calibration.py:101` excludes the outer package before inner
   proxy selection and refitting; `inner_error` excludes each inner package.
   This correctly measures that declared selection procedure conditional on the
   fixed local-score matrix. It does not repeat upstream proxy-family invention,
   architecture/checkpoint search, candidate-blend selection, or historical use
   of these portal scores. Related packages remain in training when their sibling
   is held out. Add a preregistered model/ancestor-family holdout sensitivity;
   state that few families make its uncertainty large. Do not call submission
   LOSO a selection-free estimate for a new mechanism.

3. **Eleven observations are not eleven independent errors.**
   Packages reuse members/ancestors and the same public period, while all proxies
   reuse the inspected, chain-linked 368-sequence panel. Residuals and candidate
   forecasts are correlated. RMSE is neither a confidence interval for an expected
   score nor the standard error of its mean. The 0.823529 top-pair accuracy is
   exploratory: its pairs share endpoints and its proxy was selected using all
   public outcomes (`campaign_public_calibration.py:126`, `:167`). Report that
   ranking without binomial independence claims.

4. **The selected calibration understates selection error if used alone.**
   Full selected-variant LOSO RMSE is 0.00120366, versus nested 0.00179541
   (approximately 49% larger). The script correctly labels candidate forecasts
   exploratory (`:169`), but those forecasts still use all 11 outcomes to select
   one variant and then rank previously searched blends (`:118`). Use the nested
   procedure's outer errors for descriptive sensitivity. Winner's-curse credit
   requires replaying the actual candidate-selection rule with correlated
   component errors; an invented number of independent trials or an IID
   `sigma*sqrt(2*log(K))` deduction is not justified here.

5. **New-mechanism transport is not identified by this historical regression.**
   The fitted slope 0.762773 concerns historical final-model scores on this
   holdout, not gains from component-excluded OOF bilinear/retrieval/adapter fits.
   New clean controls are recipe analogues, not the public weights. Do not
   multiply a new clean-fold improvement by that slope and label it an expected
   public improvement. Require a stated, independently audited transport model,
   paired bridge controls where available, and sensitivity to EMA-like reversal.
   Frozen confirmation can establish local effect and component uncertainty; it
   does not itself establish public-period transport.

6. **Reproducibility is improved but the new artifact is pending.**
   The updated script now saves the 21x11 matrix, proxy/submission specifications,
   row identities, source/prediction hashes (`:133`–`:156`). The reviewed existing
   JSON predates those fields. After rerun, independently reproduce proxy choice,
   all outer predictions and summary errors from the small matrix; preserve the
   old artifact and source revision. Add archive/member/native-variant identity
   alongside prediction hashes. The historical `blend_sums`/`decide.wp` helpers
   still use raw moments and clip before raw finiteness checks; either validate
   their actual inputs and compare against the audited centered reference, or
   move calibration scoring to that reference before calling the output official.

## Expected-score reporting rule

A conditional bridge model could be written as

`E[public_candidate | stated bridge assumptions] = 0.6622
 + tau * (clean_local_gain - estimated_selection_optimism) + transport_bias`.

Here selection optimism must come from a replay of the declared search using
correlated component errors, and `tau`/transport bias must be supported for this
mechanism and fitting domain. Posterior conditioning that already corrects
selection must not be charged for it a second time. Subtracting one or two RMSEs
would produce a risk bound under additional assumptions, not an expected mean.

This audit does not identify those bridge parameters or a numeric corrected
expectation from eleven historical submissions. The defensible current default
is **no expected-public credit for unsupported novel-mechanism gains**, explicitly
an approval/evidence rule, not a claim that the candidate's true mean equals the
incumbent. Report the target as unsupported until the evidence and bridge
assumptions support it; keep the verified incumbent/public association separate.
