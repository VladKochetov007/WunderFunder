---
kind: result
status: complete
created_utc: 2026-10-24T13:50:00Z
updated_utc: 2026-10-24T13:50:00Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: [E-20261024-fit-batch-mass-r1]
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/BATCH_MASS_RESULT_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/EXECUTION_R1.json
---

# The online fitter changes the effective row-weight measure by batch

## Question or claim

The canonical online-head fitter divides each batch's weighted squared error by that batch's total weight. On both actual fitting caches, this substantially changes the relative row weights compared with pooling the same numerators and denominators over the whole dataset. It also makes dependency-component weight shares less concentrated. Neither observation establishes that the existing objective is harmful or that pooled normalization would improve official WP.

## Mechanism and method

The [frozen recipe](BATCH_MASS_RECIPE_R1.json) follows the exact source order: complete selected train sequences, then selected fit-valid sequences, batched across sequence boundaries in groups of 8,192 rows with the final partial batch retained. It streams the five-column `(y0, y1, row_weight, parent0, parent1)` suffix of each original cache. Both payload SHA-256 values match their transport/export records; no raw features, evaluation labels, model or Torch import is needed.

Targets are clipped to ±2 and weights are `abs(clipped target) × row_weight`, matching `primary_pair_wmse`. Predictions remain raw. Each source batch divides by `max(batch_weight_mass, 1e-8)`. The relative weight multiplier against the pooled measure is `(total_weight_mass / batch_count) / batch_weight_mass`. This is the multiplier for averaging fixed-model batch losses; it is not an exact reconstruction of AdamW's trajectory as weights change between updates.

## Evidence and interpretation

| Measure | F0 / seed 1001 | F1 / seed 1002 |
|---|---:|---:|
| Fitting sequences | 1,236 | 1,243 |
| Source-qualified components | 315 | 314 |
| Rows | 24,720,000 | 24,860,000 |
| Batches | 3,018 | 3,035 |
| Final batch rows | 4,736 | 5,472 |
| Relative multiplier, minimum–maximum | 0.109–9.743 | 0.104–8.291 |
| Relative multiplier, 5th–95th percentile | 0.395–3.459 | 0.360–3.235 |
| Relative multiplier, median | 1.446 | 1.528 |
| Fixed-parent pooled weighted MSE | 1.211032 | 1.298912 |
| Fixed-parent equal-batch weighted MSE | 1.058217 | 1.100586 |
| Component concentration: pooled inverse HHI | 118.94 | 125.41 |
| Component concentration: equal-batch inverse HHI | 175.69 | 177.11 |

All batches have positive mass well above epsilon: minimum joint masses are 127.37 and 125.41. The denominator clamp is therefore inactive on these actual batches. The observed weighting difference is ordinary batch normalization, not a zero-denominator bug. F0 uses 1,118 train plus 118 fit-valid groups; F1 uses 1,071 train plus 172 fit-valid groups. Their evaluation groups are excluded under the pinned source-qualified membership contract.

The same frozen parent's loss is 12.62% lower under the equal-batch measure in F0 and 15.27% lower in F1. These are values of different descriptive measures, not training progress or model-quality improvement. The top 10% of components carry 38.94%/37.57% of pooled weight versus 30.33%/30.53% under equal-batch weighting. The largest component's share falls from 3.90%/3.34% to 2.10%/2.08%. Equal-batch normalization may therefore provide useful balancing; the audit does not choose which measure generalizes better. Inverse HHI is a concentration statistic, not a count of independent training replications.

Global target-0 mass shares are 49.74% and 49.11%; the corresponding means of within-batch target-0 shares are 50.03% and 49.43%. Large variation between batches does not imply a large overall imbalance between the two targets in these caches.

## Limits and validity

This is a full descriptive scan of the two fitting suffixes, not a sample uncertainty estimate or official endpoint evaluation. The two fold/seed pairs are not independently varied. Float64 reductions from stored float32 values do not establish Torch FP32 reduction parity, exact gradients, optimizer behavior, or causation for the earlier online-head score losses. Both fitting datasets are exposed; no fresh held-out evidence is created.

The broader population argument that properly weighted MSE can target the correlation-optimal direction does not remove these finite-capacity, batching, optimization and sampling differences. Conversely, finding those differences does not show that weighted MSE is the wrong loss family.

## Decision and next step

Keep batch normalization as a concrete experimental variable, behind the missing equal-capacity Z-versus-A control in the next allocation. A future normalization comparison must hold fitting membership, complete-sequence batch order, optimizer update count, initialization and LR progress fixed, and report both per-target official scores and complete endpoints. No normalization change or new fit is authorized by this diagnostic alone; there is no expected public-score increment.

## Source records

- [Result JSON](BATCH_MASS_RESULT_R1.json), SHA-256 `a26a92e60ff97a710b94d85a7c1402dc3210948aab659a4de98347ef77684e8d`, includes all distribution summaries and every component's two weight shares.
- [Execution receipt](EXECUTION_R1.json): 6.942 seconds, one CPU core, 399,998,976-byte peak, zero OOM and swap, 1,354 memory-max reclaim events. The [successful log](EXECUTION_R1.log) is empty. Both suffix hashes were verified during the complete byte streams before publishing the result.
- [Root admission](ADMISSION_R1.json), [source manifest](SOURCE_MANIFEST_R1.json), [producer/header receipt](SOURCE_RECEIPT_R1.json), [synthetic fixture](SYNTHETIC_CHECK_R1.json), and [independent source review](CODE_REVIEW_R1.md) establish the frozen measurement contract. The original scratch result and repository copy have identical bytes.
- [Independent result review](RESULT_REVIEW_R1.md) verifies saved payload identities and reconstructs mass/loss/component aggregates from JSON; no suffix rescan or optimizer/quality claim.
- [Astra fundamentals review](../../astra_fundamentals_r1/FUNDAMENTALS_AND_MISSING_TESTS_R1.md) motivated checking the actual optimization measure rather than assuming a mismatch of loss families.
