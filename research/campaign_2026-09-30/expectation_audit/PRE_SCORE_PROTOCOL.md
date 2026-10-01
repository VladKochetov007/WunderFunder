# Public expectation protocol frozen before new candidate scores

## Verdict at freeze

The required improvement is **0.0038** over the user-reported incumbent score
0.6622. No new candidate quality has been read by this auditor. No current
artifact establishes a public forecast mean of at least 0.6660. This is a
measurement limit to address while research continues; it is not a reason to
stop safe experiments.

This protocol separates three questions: whether the mechanism improves the
clean local panel, how much the selection procedure exaggerates that improvement,
and how the remaining effect might transfer to the public market. A chain
bootstrap answers the first question's sampling uncertainty. It does not by
itself answer the other two.

## Source and archive associations

The metadata-only script `inventory.py` directly checked CRC integrity and hashed
all 11 currently available archives assigned to the historical calibration
observations. Their embedded manifests, solution source and payload hashes are
saved in `source_inventory.json`. It did not read targets, masks, predictions or
checkpoints. All three previously recorded archive hashes agree with current
bytes. The incumbent SHA-256 is
`7990bafd721b09bde2670bc9281fdb0913456a7e2caf046d894fed31faae10f7`;
its two embedded members have coefficients 0.5 and 0.5.

The portal score-to-archive association remains user-reported filename evidence,
not independently observed uploaded-byte evidence. The early baseline, ab18,
train+valid two-epoch and their blend have no embedded `members.json`; their
payloads and static solution coefficients are inspectable, but their cached
prediction-to-archive equivalence has not been reproduced here. Later baseline
blends embed coefficient 0.0875 for raw baseline output, consistent with the
calibration's 0.35 share of baseline scaled by 0.25. Do not normalize those raw
manifest coefficients to sum to one.

Both EMA6 archives must remain in the evidence. On v2 both collapse to identical
member 0; their member-0 weight and v2 library hashes are identical. If the portal
used the same deterministic v2 path and identical panel, their reported 0.6616
and 0.6612 could not both arise. The distinct-pair path is a plausible explanation,
but portal CPU path/panel identity/upload association remain unobserved. Report
both pair and single scenarios. Choosing the single scenario because it has
lower calibration error is prohibited.

## What the existing calibration measures

| Scenario | Nested submission LOSO RMSE | Post hoc labeled-family holdout RMSE | Maximum listed exploratory forecast |
| --- | ---: | ---: | ---: |
| Pair | 0.00179541 | 0.00257672 | 0.66210225 |
| Single | 0.00077825 | 0.00167112 | 0.66266487 |

These are values in previously independently replayed small matrices, not a new
fit or fresh prediction score. The older selected artifact's first variant has
LOO RMSE 0.00099737 and maximum forecast 0.66382040. Its candidate inventory
differs from the current `lb_calibration.py` inventory, so it cannot be treated
as a reproducible output of today's script without its original source revision.
Nested pair LOSO explicitly includes inner selection among 21 fixed proxies;
its full selected-variant LOO RMSE is 0.00120366, below the outer 0.00179541.
The earlier aggregate replay exited 1 after its matrix and ZIP phases passed,
because its extra reviewer metric fixture had an incorrect expected answer.
Its preserved log supports those completed phases; no whole-command pass is
claimed and that fixture was not rerun here.

The five family labels are post hoc and do not provide ancestor independence.
The baseline appears in several nominal families. `ab21_final_tv_e2` appears both
alone and in its baseline blend. The incumbent's seed-4 raw parent appears in
the mixed EMA6 archive; EMA6 archives share member 0. Holding out one nominal
family therefore leaves related models in calibration training. A graph that
connects literal shared models/parents leaves at most three groups among these
11 observations: baseline/two-epoch/s4/EMA4 blend lineage, ab18, and since/EMA6
lineage. This is a conservative dependency sensitivity, not a claim that groups
have independent market errors.

The original proxy panel has 368 repeatedly inspected sequences and only 48
declared isolated sequences. Raw nonfinite predictions are clipped before
validation in historical helper code, so finite saved moments do not establish
raw predictor validity. The prior arithmetic replay did not regenerate the
dataset scores. Any new bridge must use the frozen centered metric and reject
raw nonfinite required predictions before clipping.

Neither RMSE nor an empirical error range is a predictive confidence interval.
Even under IID exchangeability, 11 calibration errors cannot provide a finite
distribution-free 95% upper residual bound: the split-conformal rank is
ceil((11+1)*0.95)=12. Ancestor and public-period dependence further weaken that
assumption. `1.96 * RMSE`, RMSE divided by sqrt(11), and observed minima/maxima
must not be presented as a calibrated 95% interval for a novel family.

## Local evidence and selection protocol

1. Before any quality score, freeze each arm, initialization, epoch, final fitting
   recipe, candidate weights, controls, masks, metric, split manifests and source
   closure. Log every quality-visible variant and every rejected result. A
   code/parity failure can be repaired before quality is observed; a quality-led
   revision is a new candidate with explicit history.
2. Screen on the declared screening fold only. Require at least +0.0005 paired
   Global WP versus the unchanged reference and a positive mechanism-specific
   matched-control difference, with the campaign's class, parity and validity
   gates. One screen success does not imply expected public 0.6660.
3. Freeze the chosen candidate before scoring confirmation folds 2 and 3. They
   were inspected historically and must never be described as untouched. Use
   the pooled confirmation estimate as the primary new effect; report screening
   and all-four-fold estimates separately. Require positive effects on each
   confirmation fold, the declared component-bootstrap interval and sensitivity
   gates. Do not use the maximum of confirmation/all-fold/soft-mask effects.
4. If several candidates are selected using the same scoring rows, save their
   joint component bootstrap replicates and replay the complete frozen selection
   rule on each resample. A descriptive local optimism estimate is
   mean(delta_bootstrap[selected_bootstrap] - delta_observed[selected_bootstrap]).
   Resample the same components for every candidate and retain shared-component
   dependence. This plug-in bootstrap assumes the observed candidate effect
   vector approximates the population vector; it is a diagnostic, not a proof
   of total historical search correction. No IID trial-count penalty is allowed.
5. A frozen separate confirmation estimate reduces the current screening
   winner's curse; it does not erase earlier feature, fold or family inspection.
   If a hierarchical posterior already conditions on selection, do not subtract
   a second selection penalty. Where replay is impossible, label the residual
   optimism unidentified and show sensitivity instead of setting it to zero by
   assertion.

For spread-label continuation, compare the candidate to both its matched
proxy-known continuation and the unchanged pair. Current-versus-proxy labels
can identify a local supervision mechanism; they do not identify public
transport. For SSM, compare multi-scale versus single-scale under identical
updates; both rates are learned, so this tests initialization/optimization.
For event memory, compare content versus position and endpoint versus raw-code
clock. This tests the declared content/clock policies, including both writes and
decay; it is not a proof of physical quote reconstruction.

## Conditional forecast, without invented calibration

Write the conditional bridge as

`public = 0.6622 + tau * (delta_confirmation - optimism) + bridge_bias + error`.

Here `delta_confirmation` is one complete fixed candidate's gain versus the
clean since/since recipe analog. It is not the public fitted archive's gain.
`tau` includes market and fitting-domain transport; `bridge_bias` covers additive
recipe/panel effects not captured by the paired gain. No gain from two families
is added unless the complete combined prediction is fitted and evaluated under
the same frozen contract. A positive matched-control contrast is additional
mechanism evidence, not an extra score term.

No empirical `tau` distribution for these three families is currently
identified. The historical absolute-score regression slope is not a new-family
gain-transfer coefficient. Use these fixed diagnostic scenarios before attaching
any scientific prior: tau = -0.5, 0, 0.25, 0.5, 0.762772753143306, 1, 1.5;
optimism = 0, one local bootstrap SD, two local bootstrap SDs; bridge_bias = 0,
-0.0009, -0.0023344573995395246. These are sensitivity coordinates, not weights
of a posterior mixture and not mandatory deductions. The slope coordinate is
the historical pair calibration coefficient. Negative transfer is included
because EMA-like locally positive effects failed publicly; the two negative
bias coordinates illustrate recorded forecast overstatement. None is a
calibrated new-family bound. Keep a further -0.006999271855804956 historical
family-heldout overprediction stress case outside the main grid.

With zero optimism and zero bridge bias, reaching the target requires the
following *conditional* clean gain:

| Assumed tau | Required gain |
| ---: | ---: |
| 0 | No positive gain suffices |
| 0.25 | 0.0152 |
| 0.5 | 0.0076 |
| 0.7627727531 | 0.00498181 |
| 1 | 0.0038 |
| 1.5 | 0.00253333 |

In general the threshold is `optimism + (0.0038 - bridge_bias) / tau` for
tau > 0. A +0.0005 screen would forecast just 0.6627 even at tau=1 with all
other terms zero. Larger clean gains and broad component/class stability can
justify updating a scientific transfer assessment, but cannot mechanically
turn the grid into an empirical posterior mean.

## Evidence sufficient for a forecast decision

The final independent audit must be able to trace a numeric forecast mean to
declared, family-specific assumptions and evidence frozen before confirmation
outcomes. Acceptable supporting evidence includes matched historical treatment
pairs scored on both this clean panel and a public-associated proxy panel,
archived predictor parity, compatible training regimes, multiple fixed
confirmation components/classes, and evidence that the mechanism targets a
relationship stable across the observed domains. Models with assumptions
contradicted by EMA6 transfer must be revised or rejected, not excluded from the
inventory. A future user-reported public probe is especially informative, but
the current task prohibits agent upload and none is requested by this protocol.

A paired bridge must specify precisely which mappings are being learned:
historical final-fit holdout score to public score, clean OOF gain to historical
holdout gain, and clean recipe to final-fit recipe. One baseline level anchor
does not estimate gain transport. In-sample final-model predictions on development
rows cannot serve as that bridge. Fitting 11 historical final scores supplies
only the first mapping, on an inspected chain-linked panel.

If the only evidence remains clean OOF gains for entirely new families, the
audit may publish explicit conditional forecasts and a reasoned scientific
forecast with openly declared priors. It must also report how the mean changes
under credible alternative transport assumptions and distinguish predictive
uncertainty from local sampling intervals. It must not call that uncertainty
empirically calibrated. An expectation claim is withheld if a numeric mean
>=0.6660 relies on an unsupported favorable transport coefficient or hides
material selection/recipe effects. Mathematical nonidentifiability alone does
not reject research: the obligation is an honest account of which assumptions
make the forecast credible and which evidence might falsify them.

## Reporting fields for every survivor

- Complete frozen candidate identity and the public archive association tier.
- Screen, matched-control and confirmation gains on exact identical rows.
- Component bootstrap SD and 95% local interval, folds/classes and mask checks.
- Search history, selection rule/replay scope and uncorrected historical exposure.
- Separate transfer, recipe, fitting, public-score rounding and predictive errors.
- Conditional forecast grid and any independently justified scientific prior.
- Mean, predictive uncertainty, probability of exceeding target if a probability
  model exists, and model sensitivity; no nominal 95% claim without its assumptions.
- Audit decision: expectation supported, conditional only, or unsupported.

The reported incumbent has four decimal places. If rounded conventionally its
underlying score differs by up to 0.00005; preserve this sensitivity separately
from market uncertainty. Archive/package parity and CPU runtime gates remain
mandatory even if a forecast mean passes.
