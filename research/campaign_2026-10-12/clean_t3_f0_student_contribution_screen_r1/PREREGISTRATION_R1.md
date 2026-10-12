# Preregistered clean-T3 F0 conditional-contribution screen

**Status: preregistered, not run.** One root-frozen candidate is the equal-weight arithmetic mean of the fixed clean-T3 CP5 endpoint and the exact old Z4 student CP1 endpoint. The sole reference is that same old endpoint at weight 1. The blend weight will not be changed after observing the score.

The evaluation is the repeatedly exposed 468-group / 141-component fold-0 valid-source panel, scored by `global_wp_clip2_v1`; both endpoints must pass the existing sidecar/checkpoint/row-order/hash and fit-membership validation in `scripts/score_fold0_panel.py`. The new endpoint’s prior official WP is 0.6811553603 and the old endpoint’s is 0.6831418134. Old inference precision is FP16 and new inference precision is FP32; this is an endpoint comparison with distinct target/initializer recipes, not an architecture-only test.

The only planned contrast is the fixed 50:50 blend versus the old endpoint, with the existing whole-component paired bootstrap over 141 components (800 replicates, seed 271828). Any interval summarizes component resampling on this exposed panel; it does not capture fit-seed, recipe, precision, or transfer variation and cannot establish a recipe winner or public forecast.

The exact command, resource cap, endpoint hashes and receipt pins are in `PREREGISTRATION_R1.json`. No score, blend, or component-bootstrap result was inspected before this file was frozen.
