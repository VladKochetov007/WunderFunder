# Stationary final-parent distillation repair R2

The first final-parent QAT adaptations changed targets to the available F0 teacher cache and reduced the complete fixed blend diagnostic by 0.000502 (S3) and 0.001039 (S4). This explicit successor tests the original final CP5 initializer as a constant FP32 causal teacher, preserving inherited preprocessing and the matched 1440-update budget. All supplied evaluation groups were used by the original final parent; outcomes are fitted-data diagnostics, not held-out CV or public forecasts.

- [Frozen recipe](RECIPE_R2.json): source/checkpoint hashes, matched S3 FP32/W4 conditions, resource bounds and target change.
- [Component subset](fit_components_10pct.json): the same315 complete fitting components/1152 sequences, with successor experiment identity.
- [Profile sequence](PROFILE_SEQUENCE_R2.json): two sequential100-update actual-shape profiles with cache handoff and single-GPU admission.
- [Fit sequence](FIT_SEQUENCE_R2.json): two matched oneepoch fits, pending independent profile acceptance.

- [Profile sequence receipt](PROFILE_SEQUENCE_RECEIPT_R2.json) and [fit admission](PROFILE_ACCEPTANCE_FIT_ALLOCATION_R2.json): both100-updateprofiles finite, host<4GB andVRAM<2.18GB; root admits matched1440updatefits afterindependent sourceGO.
