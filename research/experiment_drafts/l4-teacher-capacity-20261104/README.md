# L4 teacher capacity and student transfer — 4 November2026

The accepted allocation is eight fresh teachers plus four compact students,
with official train-only fitting and complete official development validation.
Use [the frozen contract](CONTRACT_R1.json) and [PR description](PR.md) for recipes,
scope and bounds; live progress is published in [PR61](https://github.com/VladKochetov007/WunderFunder/pull/61).

- [Completed GRU results](RESULTS_GRU_20261104_R1.md),
  [native curves](GRU_CURVES_COMPLETE_R1.json), and linked plots: six4-epoch fits
  completed; wider teachers lose at the fixed endpoint. Observed bestep2 remains
  separate from the fixed primaryep3. The terminal-transcription error was corrected.
- [Transformer packet](transformer/README.md): separate pinned causal Transformer
  implementation, FP32 runtime profiles and teacher comparison; teacher-only
  scores cannot establish useful student transfer.
- [Stage2 handoffR1](STAGE2_HANDOFF_R1.md) and
  [mapping correctionR2](STAGE2_HANDOFF_R2.md): one-bank streaming target construction,
  exact10607-source to10325-fit mapping, original auxiliary arrays and disk bound.
- [Clipping and objective audit](CLIPPING_AND_OBJECTIVE_AUDIT_R1.md): hybrid
  optimizer scale behavior, auxiliary-loss reductions, independent review,
  [synthetic CPU replay](CLIPPING_MECHANISM_PROBE_R1.json), and proposed diagnostics.
  The replay is mechanism evidence only; future quality comparisons are unlaunched.

Raw checkpoints, owned process receipts and failed attempts remain in the pinned
receiving runtime. Current serving/submission all-corpus weights are ineligible
for independent validation here. No public-score improvement is claimed.
