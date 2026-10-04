# E428 synthetic gain-assay calibration

All three preregistered calibration checks passed: planted state signal recovered; planted input-summary signal recovered; cue-free null did not show significant gain-MSE recovery or later-WP gain. Strong and weak state conditions achieved mean gain R² of about 0.99555 and 0.99067. This confirms sensitivity for these synthetic cues and noise levels; it does not establish real sequence-gain predictability, test the 48-event maturation implementation or close the representation hypothesis.

The full four-condition reports and paired component intervals are in `eda/gain_probe/synthetic/e428/report.json`; the figure is `eda/gain_probe/synthetic/e428/e428_sensitivity.png`. Fixed shared nuisance draws, eight-row summary-equivalent order cues, 32 evaluation components, 2000 bootstrap draws. No GPU was used; the CPU attempt completed with verified teardown.
