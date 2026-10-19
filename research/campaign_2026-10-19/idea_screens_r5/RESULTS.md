# R5: conditional moments on the fixed R3 panel

A row-dependent amplitude denominator hurt the fixed moment predictor. The same fitted numerator with a constant denominator scored **0.549662 WP**, versus **0.524107** with the estimated denominator. The primary ratio-minus-constant contrast is **−0.025555 WP**; both targets lose, and all 32 fixed-prediction drop-one contrasts remain negative (−0.028686 to −0.020505). These are sensitivity checks, not 32 independent experiments.

| Fixed endpoint | t0 | t1 | WP |
|---|---:|---:|---:|
| Conditional ratio | 0.5342623962 | 0.5139511763 | 0.5241067862 |
| Constant denominator | 0.5604658166 | 0.5388572205 | 0.5496615186 |
| Stored R3 signed-history WMSE reference | 0.5484853440 | 0.5235221338 | 0.5360037389 |

The constant endpoint is +0.01365778 above the stored R3 reference; the ratio endpoint is −0.01189695 below it. This secondary comparison changes the fitting objective and functional form. It is a useful development lead, not an isolated proof that moment estimation improves a full model. No R3 fit, inference or grade was repeated.

## What was actually compared

[The pre-code contract](CONTRACT.md) fixed raw112 plus causal signed-delta EMA(HL32), 224 inputs, ridge alpha100 and one shared six-column fit. For each clipped target, the heads estimate positive-square, negative-square and absolute-label moments. Their difference forms the numerator. Both decoders use precisely the same heads; only the denominator changes: predicted absolute amplitude with a fixed 5%-of-fit-mean floor, or the constant fit mean. There was no CV, alpha/floor tuning, replacement sample or endpoint selection.

The unchanged, exposed F0 component panel has 64 fitting components, 31,808 stride40 fit rows and 32 evaluation components / 59,436 scored rows. Whole-sequence causal features precede downsampling. Full train-source masks contain 1,273,664 rows; their diagnostic WP is 0.403533 ratio / 0.425594 constant. They include rows outside the fitting stride and are not a separate validation panel. Different source/mask composition prevents interpreting their absolute difference from evaluation scores as a generalization effect.

## Descriptive moment diagnostics

| Evaluation row fraction | t0 | t1 |
|---|---:|---:|
| Predicted denominator below fixed floor | 0.2423% | 0.0522% |
| Negative positive-square estimate | 4.9751% | 8.1516% |
| Negative negative-square estimate | 9.3798% | 7.8370% |
| Raw ratio prediction outside ±2 | 3.4945% | 3.0705% |
| Raw constant prediction outside ±2 | 3.6291% | 2.4228% |

The unconstrained estimates violate moment bounds on some rows. The floor activates rarely, so this screen does not establish that floor events cause the loss; the denominator may change the ranking well above the floor too. These are row fractions, not label-weight shares. No diagnostics were used to tune or remove observations. The ideal conditional-moment identity does not guarantee useful finite ridge estimates or transfer across sparse evaluation selection.

## Evidence and operational outcome

[Raw RESULT](records/probe_v1/terminal_capture/RESULT.json), [independent compact-moment audit](RESULT_AUDIT.json), [source manifest](records/probe_v1/terminal_capture/SOURCE_MANIFEST.json), [implementation review](IMPLEMENTATION_REVIEW.json), [seven tiny checks](TEST_RECEIPT.json), [resource reconciliation](RESOURCE_RECONCILIATION.json) and [execution receipts](records/README.md) bind the result. The audit uses compact component sums and metadata, not arrays or a repeated fit.

The single 0.5CPU / 1.5GB / no-swap / 600s unit exited0 after 90.647s; helper wall90.0848s / CPU44.9839s / maxRSS992,872KiB. Unit CPU45.3477s / MemoryPeak268,558,336B are separate measurements. Exact raw receipts were captured before stopping only this exited invocation. Parent and F3 child memory events did not change. Arrays and the 15,962B model archive remain on Beleriand; its contents were not independently replayed.

Idea12 now has one narrow decoder test. Learned moment heads, joint end-to-end objectives and recurrent models remain untested. All F0 data here are exposed, and this adaptive exploratory panel is neither a fresh holdout nor full-fold/public evidence. No new GPU fit or follow-up sweep is admitted by this result.
