# Mechanism sources and limits

Primary sources checked 2026-09-30 campaign session. These establish algorithm
ideas, not transfer or score improvements for anonymised market sequences.

- [Linear Transformers](https://arxiv.org/abs/2006.16236): kernel-feature attention
  admits an iterative recurrent form. [Author implementation](https://github.com/idiap/fast-transformers).
  D1 uses a bounded normalised decayed memory, an added GRU and an observed input
  event clock; it is a task-specific variant with separately graded arithmetic.
- [Structured State Spaces](https://arxiv.org/abs/2111.00396): efficient state-space
  sequence modelling is an alternative for long dependencies.
  [Author repository](https://github.com/state-spaces/s4). The campaign's scalar
  plus real-rotation model is simpler than S4 and does not reproduce its specific
  low-rank/Cauchy parameterisation. Multi versus single tests initialisation and
  optimisation of the same learned half-life family.
- [Switch Transformers](https://arxiv.org/abs/2101.03961): sparse conditional
  parameter selection motivates a possible mixture direction; router gradients,
  matched active work and arithmetic near ties remain unresolved for the current
  proposed small recurrent mixture. This card is deferred, not implemented.
- Supplied-data current-code spread reconstruction motivates the label-only
  experiment without an external generator assumption. The old proxy includes
  legal future-tail information and is the matched supervision control.

Detailed proposals and independent criticisms are in ../proposals/architecture_cards.md,
../proposals/dataderived_cards.md and ../audit/*critique*.md. Recurrent content
memory, structured dynamics and causal supervision are distinct mechanisms.
