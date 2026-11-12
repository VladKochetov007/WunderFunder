# L4 recurrent follow-ups, EMA, and auxiliary-head screen

This publication packet records the ROOT-run L4 follow-ups and their exact execution/readout provenance. Source work remains on separate refs so this packet does not claim a newly merged runtime was executed:

- `experiment/l4-adapters-20261112` at `390c846cbdacd8e0e06e6079dd3b04312fa6bcc5`: recurrent Muon and causal since-event source.
- `experiment/l4-ema-20261112` at `73a2f64bcb4df20327225f045f51791fd607b69a` (implementation ancestor `bc02d541d`): per-update EMA source and configs.
- `experiment/l4-auxheads-20261112` at `3f9a8dc8db22f4c9a0cb381562d71ac82bedf378` (implementation ancestor `a780cad8e`): auxiliary-head source and configs; this publication branch is rooted at this ref.

The source contracts are [Muon/since-event](../sequence-gru-muon-20261112/README.md), [EMA](../sequence-gru-ema-20261112/README.md), and [auxiliary heads](../l4-auxiliary-heads-20261112/README.md). The EMA runtime remains a separate source line; no combined EMA-plus-auxiliary runtime was qualified or fit.

## Chronological validation contract

All fitted arms use the supplied official chronological split SHA-256 `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`: 10,325 whole official-train sequences for fitting and train-only normalization, zero official-valid fit members, and all 1,873 later official-valid sequences / 3,528,273 selected rows for full-valid epoch evaluation. `seq_ix` is not treated as time. This validation endpoint has been exposed during prior research and is not an untouched holdout. No teacher or initializer entered these arms.

The executed follow-up source SHA-256 is `29fb45164667e69a71d4be2c696731d9824884d2d51674f37d2a633bd17d9511`; EMA source SHA-256 is `33574f496a2640e1a4c14690c8858eeaad5075e21d5b96042b746d35ad55810a`. The fixed campaign clock is [CLOCK_R1](execution/pinned/CLOCK_R1.json), SHA-256 `0181e9889a45d0636e3b9981ddb15e17fb5cd26429068bb0ee37f78d09d71270`. Native admissions and original paths are retained in the exact [receipt bundle](execution/receipts-r1/l4-publication-admission-receipts-r1/RECEIPTS_MANIFEST.json).

## Fit status

- All four Muon/since-event fits completed the six-epoch endpoint.
- The two same-source raw K16 EMA controls completed E6 and reproduced the historical monitor scores to about `1e-9`.
- Both EMA treatment runs reached 38,760 updates but timed out during final E6 evaluation. E6 checkpoint weights were not committed; only E1-E5 checkpoints are retained. They are partial fits, not E6-complete.
- Auxiliary-head code and matched configs are present, but no native profile or fit ran in this wave.

## Canonical readout status

The unchanged canonical R3 receiver completed all eight first-cohort full-valid readouts. Each readout covered 3,528,273 rows and agreed with the full-valid epoch monitor within the receiver's `1e-5` tolerance. The R3 attempt initially rejected four completed follow-up cells and two completed raw EMA controls because their fit roots lacked a `project` path; this was a layout/provenance preflight failure, not a training or data mismatch. The two partial EMA treatments remained ineligible for E6 readout.

ROOT linked the follow-up and EMA fit roots' `project` paths to the same pinned project data source and launched the unchanged R3 receiver into fresh R4 output roots. The R4 launch receipt confirms the source hash is unchanged and all eight validated first-cohort result bytes were preseeded with matching result/model hashes. R4 follow-up/EMA canonical readouts were still pending at the latest included receipt; the rejected R3 attempts remain preserved. See [execution evidence](execution/README.md) and the R3/R4 snapshots in [receipts-r2](execution/receipts-r2).

These are exposed development-validation results, not public/private leaderboard evidence. No fit was resumed or extended after the fixed cutoff.