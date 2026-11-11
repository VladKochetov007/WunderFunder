# Loss/objective ideation and red team — 2026-10-25 R1

User-requested review of current state, PR updates and custom-loss research, with
external Grok/OpenCode ideation and red teaming. Zero fits, zero uploads.

- [Synthesis](SYNTHESIS_R1.md) — why distance-function losses failed (WMSE is already
  Bayes-aligned with pooled |y|-weighted WP; NLL test was clipping-confounded), why
  measure/weighting and distillation succeeded, PR32 B300 teachers memorize (val WP
  0.16–0.41), verified/rejected external claims, eight ranked next ideas.
- [Zero-fit target-mass receipt](zero_fit_target_mass_valid_R1.json) from
  [script](zero_fit_target_mass.py): scored-valid weighted variance 1.79/1.82, zero-
  predictor WMSE 1.94, t0 mass share 0.496 (per-target normalization is a no-op).
- [External brief](external/BRIEF_R1.md) and outputs: [Grok 4.7](external/grok_4.7_R1.md),
  [DeepSeek](external/oc_deepseek_R1.md), [GLM](external/oc_glm_R1.md),
  [Nemotron](external/oc_nemotron_R1.md); [invocation receipt](INVOCATIONS_R1.json).
