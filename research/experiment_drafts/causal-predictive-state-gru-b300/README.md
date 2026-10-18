# Causal predictive-state GRU

**Status:** implemented draft for B300 review; no B300 hardware profile, fit,
independent grade, or leaderboard result has been produced.

This packet proposes a paired full-precision test of whether training a causal
GRU to predict its next input row improves the supervised competition task.
The same-size head is present in both arms and is discarded for serving. The
candidate's auxiliary loss reads only the next row of canonical model inputs
from fitting sequences; it never reads future competition targets or scoring
masks. This is training-time self-supervision, not test-time label learning.

## Files and implementation

- [PR.md](PR.md) answers the eleven experiment-draft questions, explains the
  mechanism and uncertainty, and gives B300 profile/fit/collection instructions.
- [recipe.json](recipe.json) freezes the 30 paired fold/seed cells, exact model
  and optimizer settings, source/split identities, and evidence to return.
- [run_cell.sh](run_cell.sh) runs a matched control/candidate pair on one GPU;
  `profile` runs a bounded 50-update engineering profile, and `pair` runs both
  six-epoch fits with alternating arm order.
- The implementation adds `--aux-next-input-head` and
  `--aux-next-input-weight` to the existing trainer. The next-row objective and
  finite-element telemetry live in
  [next_input.py](../../../src/wnn/training/next_input.py); output layout,
  validation, and per-epoch fit summaries reuse the modular trainer.

The [unified idea note](../../unified/ideas/CAUSAL_PREDICTIVE_STATE_GRU_B300_20261018_R1.md)
records the hypothesis and the exact limits. The [source note](../../unified/sources/CAUSAL_PREDICTIVE_STATE_SOURCES_20261018_R1.md)
separates primary literature from the user-supplied competition anecdote.

## Provenance and handoff

- Shared experiment base: `research-team` at
  `a0355ed6b08f87661522744a5be57d265bc45bbc`.
- Shared-policy main snapshot observed during preparation:
  `24e9001570de158db9a5031e632dbb0b4477145f`.
- Code includes no dataset, weight, or credential payload. Input/output paths
  must be bound to the B300 worker's approved private project root.
- This is a draft proposal, not an admitted cluster job. A posted issue needs a
  worker/cell claim and actual GPU/environment proof before work can be called
  running. The first 50-update profile is an engineering gate, not a quality
  replicate.

Published handoff: [draft PR #18](https://github.com/VladKochetov007/WunderFunder/pull/18) and [cluster issue #19](https://github.com/VladKochetov007/WunderFunder/issues/19). Neither publication means a worker has claimed or started the profile.
