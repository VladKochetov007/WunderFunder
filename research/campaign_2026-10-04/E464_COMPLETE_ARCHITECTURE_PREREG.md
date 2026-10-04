# E464: complete predictors for the eight E424 architectures

Registered after inspecting E424's parent-only results and before fitting any
E464 head. This is a follow-up protocol, not a newly independent test set.

Question: does the incumbent's fixed head-fine-tuning and equal parent/head
procedure change the architectural comparisons? The eight E424 parents remain
at their already-fixed epoch 3. Fit all eight heads, including every routing
and adapter control; do not select only the most favorable parent.

## Fixed procedure

- Parent runs: `c20261004_h093_{learned,hash,universal,gru_budget}_s3_final`
  and `c20261004_h094_{conditional,unconditional,diagonal,gru_budget}_s3_final`.
- One head epoch, fixed checkpoint 0; FT seed 3, AdamW lr 0.0005,
  weight decay 0.01, batch 16, chunk 2000, `loss=gcorr`, mask weighting,
  inherited preprocessing and the incumbent's inherited auxiliary targets.
- Fit only the 1,505 historical validation fitting sequences. The 368 excluded
  validation sequences remain excluded. Suppress evaluation during these fits.
- Freeze the complete representation and routing machinery. Specialist freeze
  prefixes are `inp,gru,experts,prefix_proj,router`; only `universal_head` and
  `expert_heads` parameters may move. Adapter freeze prefixes are
  `inp,gru,adapter_,prefix_prior`; only `head` parameters may move. Ordinary GRU
  freeze prefixes are `inp,gru`. Check actual parameter names before fitting.
- Require every non-head model tensor, including normalization and fixed hash
  buffers, to be bitwise identical before and after fitting. Preserve parent
  checkpoint hashes. A violation invalidates that fit rather than broadening
  the meaning of "head-only".
- Export every complete validation prediction in FP32, then evaluate fixed
  equal parent/head predictors using the existing frozen paired scorer.

## Comparisons and interpretation

Report all E424 comparisons under the complete procedure: each architecture
versus its budget-matched GRU and the fixed public-confirmed combo; learned
versus uniform and hash routing; conditional versus diagonal and unconditional
adapters. Preserve the eight parent-only results alongside the new complete
results. Use 2,000 paired component-bootstrap replicates and class diagnostics.

The existing panel is historically exposed. Bootstrap intervals condition on
these fitted models and do not cover seed or research-selection uncertainty.
Normalization differs between some different architectures because initialization
consumes RNG before normalization fitting; inherit each parent unchanged and
report that limitation. Never silently refit normalization to improve matching.

The new head fits need an estimated 0.5–1 local GPU hour including prediction
exports; reserve at most three job-hours before admission. These eight fits
extend the approximately 82-fit review portfolio to approximately 90 fits,
without adding eight distinct research directions. The cumulative 80-device-
GPU-hour budget remains in force. Admit only after the existing one-GPU queue
has completed and remaining budget is reconciled. Source preparation and CPU
checks are authorized now; scientific fitting still requires root review.

No forecasting gain or submission readiness follows from this preregistration.
