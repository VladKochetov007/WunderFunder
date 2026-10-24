---
kind: result
status: complete
created_utc: 2026-10-24T11:08:56Z
updated_utc: 2026-10-24T11:08:56Z
hypothesis_ids: [H-20261024-age-clock-censoring-r1]
experiment_ids: [E-20261024-age-clock-eda-r2]
evidence_paths: [EDA_PLAN_R2.json, CENSORING_SAMPLE_R2.json]
---

# Observed field age is rarely censored on scored validation rows

## Question or claim

Does sequence-start censoring affect enough scored observations to plausibly explain a broad age-feature effect? On this fixed validation sample, censoring is concentrated near sequence starts and is rare overall. This describes coverage; it does not establish predictive usefulness or rule out large effects in rare observations.

## Mechanism and method

The sample contains 96 uniformly selected complete validation sequences, chosen with seed 20261024 before this scan: 1,920,000 rows, including 192,763 rows satisfying `need_prediction AND is_scored`. For each of 112 fields, a change is an exact difference from the previous row. Row zero is not an observed change. Age is the number of rows since the last observed change; a field remains censored until its first change. Label-derived summaries use absolute clipped-label magnitude, with clipping at ±2. They are descriptive weights, never model inputs.

The original train/valid plan encountered its 750 MiB worker memory ceiling before producing a training-source result. R2 preserves the same preselected validation sequences and ran within a 600 MiB worker ceiling and 1 GiB aggregate ceiling. The successful scan took 12.584 seconds and reported peak kernel RSS of 571,440 KiB. The raw Parquet input was identified by path, size, mtime and footer; its entire payload was not rehashed.

## Evidence and interpretation

There are 16,059 censored field–row pairs, or **0.07438%** of the 192,763 × 112 eligible pairs. The maximum single-field rate is **0.30089%**, shared by several `i0_p` fields. These figures are not the fraction of rows with any censored field; that union was not collected.

The largest single-field shares of total clipped-label magnitude are **0.1871% for t0** and **0.2251% for t1**. No censored field contributes mass at observed age ≥10,000 in this sample.

| Observed age threshold | Median field's scored-row coverage |
|---|---:|
| ≥1 row | 64.89% |
| ≥16 rows | 22.50% |
| ≥64 rows | 5.70% |
| ≥512 rows | 0.252% |
| ≥4,096 rows | 0% |
| ≥10,000 rows | 0% |

Mean field censoring within position bins is 2.806% at rows 99–499, 0.400% at 500–1,999, 0.00223% at 2,000–9,999 and 0% at 10,000–19,999. There are no scored rows before position 99. Short ages cover substantially more data than boundary censoring.

## Limits and validity

All validation is exposed. This is a sequence sample, with no new fitting, prediction evaluation, fold comparison or training-source result. Aggregates do not support sequence-bootstrap confidence intervals; none are claimed. The overview's shaded band is the 10th–90th percentile across fields, not uncertainty about population rates. Exact float changes are a definition of observed activity, not proof of an underlying market event or elapsed physical time.

## Decision and next step

Deprioritize sequence-start censoring as the sole broad explanation for an age mechanism. Retain it as a possible rare-case interaction. Next inspect how much per-field clocks duplicate shared update events and whether short-age behavior is explained by group activity or sequence position. A predictive claim still requires a matched, source-qualified experiment; the new reported PR26 age shortscreen is mixed and does not settle these representation questions.

## Source records

See the [EDA index](README.md), [original plan](EDA_PLAN_R1.json), [R2 amendment](EDA_PLAN_R2.json), [raw aggregate result](CENSORING_SAMPLE_R2.json), [memory incident](EDA_MEMORY_INCIDENT_R1.json), [live notebook](research_frontier.py) and [PR25–40 audit](../pr_audit_25_40/REPORT.md). The campaign's append-only H/E ledgers use the IDs above.
