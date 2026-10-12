---
kind: review
status: complete
created_utc: 2026-10-12T23:47:32Z
updated_utc: 2026-10-12T23:47:32Z
hypothesis_ids: [H-018, H-039, H-332, H-333, H-336]
experiment_ids: [E-023, E-051, E-071, E-STRATEGY-HISTORICAL-AUX-AUDIT-20261013-R1]
evidence_paths: [HISTORICAL_AUXILIARY_AUDIT_R1.json]
---

# Historical forecasting auxiliaries and chunk correlation

## Question or claim

S25 asks whether the proposed next-feature and chunk-correlation mechanisms already have useful negative evidence. They do, at specific recipe scopes. This audit locates terminal histories and recorded scores; it does not recompute predictions or train another model.

## Mechanism and method

The [machine receipt](HISTORICAL_AUXILIARY_AUDIT_R1.json) hashes seven actual histories, records their completed epochs and optimizer counters, and distinguishes withheld fitting-time evaluation from subsequent score reports. These runs include future-feature horizons2/8 and8/32, forecast weights0.1/0.3/0.5/1.0, a next-change interaction and a stronger next-change head.

## Evidence and interpretation

The [October10 loop log, passes43/48/61/64](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/LOOP_LOG_20261010.md) reports the following exposed F1 forecast scores: weight0.1 =0.66532; weight0.3 with horizons2/8 =0.66544; weight0.3 with horizons8/32 =0.66564; weight0.5 =0.66551; weight1.0 =0.66555. The recorded reference seeds were0.66605/0.66578. These screens show no useful mean lead for the tested forecast settings. They do not measure a universal upper bound on feature-based self-supervision.

All seven located histories reach epoch2 after three recorded epochs and withhold evaluation during fitting. The receipt preserves their actual cumulative update counts; the commonly observed22,380 attempted updates are not all successful updates because of AMP skips. Histories establish that fitting occurred; the quoted quality values still rest on historical score reports. This audit does not assert grades for the next-change interaction or stronger-next runs without a located score receipt.

The loss evidence is protocol-dependent. E-023 (September15,3,000 training sequences,dev600,seed1) records pure correlation0.2937/0.2575 versus mixed loss0.6036/0.5983 over two epochs. E-051 reports WMSE0.6272 below the older mixed-loss reference0.6310 on its devB764 recipe. E-071 (September25,global WP,H128,one epoch,seed3) instead reports WMSE0.6651 versus mixed chunk-correlation/WMSE0.6616. The later global-metric comparison supports retaining WMSE as that control; the three studies are not interchangeable matched replications.

## Limits and validity

These are small, exposed historical recipe comparisons, mostly one fitting seed. A complete-endpoint complementary effect was not established here. Differences in evaluation protocol, training coverage, precision, auxiliaries and ancestry prevent using their numbers as deltas against the current student. The log's broad statement that the forecast family is closed exceeds what these experiments establish.

## Decision and next step

S25's historical audit is complete. Avoid repeating these exact forecast weight/horizon screens without a distinguishing mechanism. A future auxiliary arm must specify what changes, why the existing negative does not test it, and how the resulting causal student or fixed endpoint will be compared. New causal moments, path-state features, cross-fitted teachers and controlled gradient-credit arms remain eligible under their separate hypotheses.

## Source records

- [Actual terminal-history audit](HISTORICAL_AUXILIARY_AUDIT_R1.json), with paths and hashes to seven run histories.
- [Original loop log](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/LOOP_LOG_20261010.md), historical reported scores and recipe context.
- [Hypothesis ledger](../../hypotheses.jsonl): H-018/H-039/H-332/H-333/H-336.
- [Experiment ledger](../../experiments.jsonl): E-023/E-051/E-071 and the audit record above.
