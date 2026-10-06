# Quant-probes after the user refreshed Git

Verified ref: `origin/quant-probes@b7fd5d2fff8dbf6645f34de9835c4f634caa8273`,
committed October6 03:09:59Kyiv. The previous cached ref was `0927b1e6`.
This ref now includes a completed Transformer result, class-scale diagnostics,
updated adversarial-validation figures and routing-loader updates. No friend
implementation was executed, copied into our implementation or merged.

Primary-source bytes and source hashes are in [manifest.json](manifest.json).
[branch_inventory.json](branch_inventory.json) contains every available branch's
last commit, date and subject after the user's fetch.

## What the branch actually reports

- Causal Transformer teacher:6layers,d256,8heads,block250,ALiBi,4.9M parameters,
  lr0.0005,4epochs,fold0. Parent final score0.7001 versus GRU2560.7012.
- After head-only fine-tuning, pure-chain fold0 Transformer0.71328 versus
  GRU2560.71323: delta+0.00005, reported bootstrap sd0.0022. As a fifth
  member of the branch's existing four-model mixture, delta+0.00028,
  reported sd0.00039. These are branch-reported validation results, not a
  reproduced run, student-transfer test, or public submission result.
- Earlier ordinary GRU256 ensemble distillation: student versus GRU256 seed3
  delta+0.0016 (folds+0.0015/+0.0014), class3+0.0038. Replacing a member
  in the complete blend added+0.0004. This is a stronger transfer example
  than the newly completed Transformer comparison, but uses the friend's old
  recipe/comparator and does not establish a gain over our current0.6660 endpoint.
- Privileged teacher score0.813 did not automatically translate to a larger
  causal-student gain: branch reports+0.0007/+0.0011 versus causal-distillation
  +0.0010 for its tested h128 one-fold recipe. This motivates testing transfer
  learnability separately; it does not rule out different privileged mechanisms.
- Best public result reported in this note:0.6641 for V1. C reported0.6640;
  V2 validation-upweighting reported0.6611; V3 train-only parents0.6629.
  No significantly higher public Transformer result is documented here.
- Oracle per-sequence recalibration reports0.74406 versus mixture0.70082,
  scale-only oracle0.71851. These oracles use future labels and are diagnostic
  headroom, not deployable candidates or public predictions. Cross-fitted
  class scales actually scored0.70069, below0.70082.

Sources: [research note](quant-probes-2026-09-29.md),
[scale diagnostics](eda_scale.json), [shift diagnostics](adversarial_validation.json).
All folds are historically exposed. Reported bootstrap sd describes sampling
of fixed predictions, not fresh training seeds or public-transfer uncertainty.

## Brief that can inspire an independent agent

1. **Make the teacher useful before increasing its size.** Revisit causal attention
   with a convergence-aware schedule and verified event/quiet context, matched
   against the GRU teacher. The four-epoch run is one recipe, not a capacity ceiling.
   Compare complete endpoint contributions as well as teacher scores.
2. **Transfer the causally learnable part.** Cross-fit teacher predictions by chain;
   compare ordinary prediction distillation with residual/confidence-weighted or
   temporal-summary supervision. For privileged teachers, explicitly separate
   predictable structure from irreducible future innovations. Keep a fixed causal
   student and a paired unchanged-recipe control.
3. **Optimize complementarity.** Train an independent specialist or student, inspect
   its aligned residuals against our champion, then test a fixed blend. A causal
   regime gate is a separate arm with its own selection/evaluation accounting.
   Higher teacher WP alone is not the optimization target.

Deeper trials remain open despite the branch's pessimistic ceiling verdict.
Every child must specify what changes and the paired test that can distinguish it.
No new training or public submission was performed for this audit.
