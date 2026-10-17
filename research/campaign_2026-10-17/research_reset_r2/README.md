# Research reset R2: practical exception impact and prediction dynamics

Date: 2026-10-17. Successor to [R1](../research_reset_r1/README.md), prompted by
the user's objection that 1,253 exceptions among 249.6 million rows may have
little practical importance. **No model fitting, optimizer steps, calibrator
fitting, new model inference, GPU use or submissions.**

- [Russian synthesis](REPORT_RU.md) — actual findings, the correction to the
  earlier emphasis on rare exceptions, exact output-policy effects, alternative
  explanations and revised research decisions. E-RR17-R2-EXCEPTIONS,
  E-RR17-R2-OUTPUTS, E-RR17-R2-DECOMPOSITION and E-RR17-R2-RESIDUALS.
- [Contract](CONTRACT.json) — resource/authority boundary, fixed comparison
  matrix and retrospective-oracle restrictions.
- [Full exception census](exception_audit/README.md) — all 249.6M rows on the
  selected price/target/mask columns; 1,253 complete exception records, clipped
  weights, concentration, timing, recall/precision and short label-reconstruction
  mismatch episodes. Not a full112-feature scan.
- [Frozen prediction audit](prediction_audit/README.md) — 14 policies on three
  arrays over a common467-sequence exposed panel, exact target/mask alignment,
  conditional143-chain bootstrap, age/class and cross-model diagnostics. Raw
  outputs and append-only reporting corrections are both retained. Prediction
  producer/checkpoint linkage remains unverified.
- [Mathematical interpretation and independent pooling check](theory/README.md)
  — label-run variance identity, within/between-sequence covariance, prior loss
  overlap, scoped primary-literature transfer and reproducible numerical checks.
- [Iterative reflection and independent review](reflection/README.md) — an
  initial framework, old-code novelty attack, implementation review and revised
  dispositions for all25 R1 cards. These are public research assessments, not
  private reasoning transcripts or fitted-model results.
- [Figure](plots/README.md) — observed output-policy deltas with conditional
  intervals; oracle shown separately because it is unavailable online.
- [Integration verification](VERIFICATION.json) — input/output receipt hashes,
  independent numerical parity, no-fit checks and documented limitations.
- [Publication](PUBLICATION.md) — branch integration and scope; R1 artifacts and
  existing proposal branches remain preserved.

The approximate price clock misses very few target transitions, yet does not
justify holding emitted predictions. All36 causal policy comparisons are
negative on these fixed arrays; this does not establish a hidden-state-skipping
effect or a future leaderboard result. All local data remain exposed.
