# Task: propose new training ideas for a causal sequence regressor

Setting: 10k sequences x 20,000 rows, 112 anonymized standardized market features (two instruments, order-book
like), two regression targets per row, metric = |y|-weighted Pearson with labels clipped to +-2, pooled over rows.
Model: 2-layer GRU, hidden 256 (about 1M parameters), served causally on one CPU core. Validation is a
regime-shifted block (adversarial AUC 0.98 vs train).

Known facts:
- Labels follow an event clock: the target is the rounded mean mid price over the next 4 book-price events.
- Down-weighting loss on rows far from the last price event (weight max(0.25, 1/(1+since/32))) gives +0.003.
- Distillation from a bigger GRU teacher gives +0.002. Longer BPTT, depth 1/3, dropout, SWA, EMA, mixup-like
  masks, implicit ensembles, bilinear inputs: no gain.
- Muon (orthogonalized updates) on recurrent matrices at LR 1e-3 gives +0.003 vs AdamW; rapid overfit after 4-6 epochs.
- Seed ensembles correlate 0.99; blend gain ~ 0.168*(1-rho).

Give 8 distinct, concrete, cheap-to-test ideas (each: mechanism, exact change, why it could beat the list above,
falsifier). Prefer ideas about loss shaping, target construction, noise-robust objectives, event-clock structure
and regime-shift robustness. No generic advice. Write the answer to ideas.md in this directory.
