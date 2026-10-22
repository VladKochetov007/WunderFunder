# Run the fixed numerical-input Transformer comparison on B300

This implements the six-fit experiment from #35. All three arms retain raw112,
causal event99 and missing112 inputs; uniform16 and quantile16 add equal-sized
piecewise-linear branches. The primary question is whether fitting-distribution
knot placement helps more than evenly spaced knots. It targets the teacher's
input representation, building on #8/#31; no teacher/student link is tested here.

## Expected mechanism and decision

A shared linear map may represent nonlinear threshold effects inefficiently.
If knot placement is the useful mechanism, quantile16 should beat uniform16 on
both fixed cells without simply benefiting from extra parameters. Raw contrasts
are secondary and capacity-confounded. Report F0/1001 and F2/1002, both targets,
all arms, and paired whole-component uncertainty. Mean +0.0003 with consistent
cells supports a larger independent confirmation; conflicting cells or an
interval crossing zero are inconclusive. A clear practically negative effect
rejects this bounded recipe, not every numerical representation. All folds have
been exposed; this screen cannot establish generalization or statistical significance.

## Included implementation

- Reuses the pinned #32 trainer, merged into this branch for a self-contained checkout.
- Shared fitting-only normalization and immutable, source-bound knot artifact;
  bounded seeded reservoir quantiles with full-fit endpoints (explicit prospective
  amendment before any outcomes, detailed in README).
- Matched common initialization, zero-initialized basis, FP32 RoPE Transformer,
  WSD schedule, one-GPU/global-batch32 enforcement and fixed epoch11 endpoint.
- Six consumable configs and single-cell prepare/profile/fit/evaluate commands,
  checkpoint/resume support, causal cached inference and canonical pooled grading.
- Panel grader verifies shared data/preprocessing/source/runtime/initialization,
  reports all contrasts, and computes equal-cell CIs from independent fold draws.
- Receiver tests for missing inputs, fitting-only artifact membership, tied bins,
  prefix causality, full/chunk/row parity, serialization and common initialization.

The full fitting set is used after warmup; the train parquet has no scored mask.
Evaluation alone uses the true scored mask. Calibration never uses labels or
held-out features. More GPU spend can scale teacher capacity/confirmation later;
this screen fixes six complete fits and does not measure a pay-to-win scaling law.
CPU student inference, 20-MB packaging and quantization remain separate.

## Validation and handoff

Static formatting/lint, AST structure checks and source review only. **No local
tests, training, inference or GPU jobs were run.** Device fit, performance and
quality are unmeasured. The receiver must bind the actual source/split assets,
run the written acceptance checks and approve successful real-shape B300 profiles
before fitting. A failed profile requires an explicit recipe amendment, not a
silent batch/precision change.

See [receiver commands](README.md), [frozen recipe](recipe.json),
[static validation](VALIDATION.json), and [original proposal](PROPOSAL_20261022.md).
