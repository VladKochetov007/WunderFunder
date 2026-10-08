# Experiment count and replication audit

Snapshot: 2026-10-08 approximately07:43UTC. The reproducible cutoff is the first authenticated Vast access, 2026-10-07T20:44:34Z, recorded in ACCESS_FORK_20261007_V2.json. The exact user message timestamp is not established by this audit. Earlier credential-present/access-failed records and earlier E511/E521 work are excluded from this count, rather than silently equated with the access cutoff.

| Execution venue | Completed training runs | Distinct training recipe cells | Additional scored no-fit hypotheses |
| --- | ---: | ---: | ---: |
| Local | 7 | 4 | 2 |
| Kaggle GPU | 3 | 2 | 0 |
| Vast | 1 | 1 | 0 |
| Total | 11 | 7 | 2 |

Local seven runs: H230 clean-fold heterogeneous candidate and matched family-off control; own all-data parent; source-target all-data students seed3 and seed4; normalized-target all-data student seed3; H218 block-memory attention. The parent is an upstream fit, not another hypothesis; the seed pair is replication, not two hypotheses. Four recipe cells are the family-head contrast, all-data source endpoint, normalized target child and block-memory architecture. Additional no-fit checks are E535 output hold and H264 fixed attention/TCN interaction. Completed all-data fits have no held-out score and their ZIPs are awaiting human public upload/results.

Kaggle three runs: H232 scratch full-feature GRU384; E540 linear control and PLE candidate (one paired embedding hypothesis). Both E540 fits completed72996finiteFP32updates despite postfit validator/export failures. H232 FP32 predictions have now landed; grading is pending at this snapshot. PLE/control predictions remain undelivered. Do not count completed fits as completed score comparisons.

Vast one run: H217 raw-only TCN128x12,823.97seconds fitting, complete predictions and exposed-fold grade. Its proposed matched GRU control was not admitted/fitted because its projection exceeded that recipe's old time cap. The ambiguous V5R2 create and all transport/validation failures are execution events, not scientific failures or tested hypotheses.

The current history-swap student, TPU synthetic numerical/device probe, profiles, cache builds, exports, repeated pushes, seed repeats and code-only proposals are not additional completed training hypotheses. The TPU probe is successful engineering evidence, not a predictive hypothesis test. Thus the count is seven distinct training recipe cells plus two no-fit checks, not eleven independent hypotheses and not dozens of experiments successfully scored.

## Why teammate public parity is unproven

Root fetched origin on2026-10-08. Available teammate references remain quant-probes b7fd5d2f and research/creative-iteration-20260930 833aaf54, last commits October6. No newer remote teammate commits were obtained. Unpushed work cannot be determined from remote Git. The already available wave18/wave20 records nevertheless describe the reported0.6674 endpoint: equal GRU256x2 heterogeneous-distilled students, with seed-specific fbc1k parents; a Transformer teacher is not established by that ZIP.

Our public-scored H230 analogue is0.6649. It uses different own teacher histories and warm initialization, including clean-fold teacher families and an independently trained baseline substitute. It does not reproduce the complete final all-data teacher/parent recipe. Our newer all-data students still use clean-fold family predictions and share one common own parent, while the reference uses seed-specific parents. Matching alpha, coefficients and head shape does not make those trained targets equivalent.

The earlier accidental family-target clipping discrepancy was fixed: current family targets are raw. No claim that this fixed discrepancy explains the public gap is supported. The exposed matched family-head comparison gained only0.0000610; its uncertainty is not a universal reject rule. It does not establish public parity.

The all-data source/normalized ZIPs are delivered but not publicly scored. The history-swap student is currently fitting. We therefore have not established failure of a faithful final recipe; we have measured independent analogues and still owe closer all-data teacher-family/seed-specific-parent evidence. Execution errors and broad exploration consumed meaningful time before that closer endpoint landed; this is a delivery/prioritization limitation, not evidence that the teammate mechanism cannot transfer.
