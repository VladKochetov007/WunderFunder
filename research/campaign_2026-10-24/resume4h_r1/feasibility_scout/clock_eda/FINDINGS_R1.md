---
kind: result
status: complete
created_utc: 2026-10-24T11:33:35Z
updated_utc: 2026-10-24T11:33:35Z
hypothesis_ids: [H-20261024-age-clock-censoring-r1]
experiment_ids: [E-20261024-clock-redundancy-r1, E-20261024-clock-geometry-r1]
evidence_paths: [CLOCK_STATISTICS_R1.json, CLOCK_GEOMETRY_RESULT_R1.json]
---

# The age comparison includes a consequential grouping choice

## Question or claim

How much do 112 field clocks duplicate common activity, and what does the existing seven-group control discard? The sample has substantial within-family overlap, but grouped and per-field ages differ on many scored observations. Their contrast includes a particular grouping decision, not just the presence of independent clocks.

## Mechanism and method

Both no-target scans reuse the same 96 fixed validation sequences: 1.92 million complete rows and 192,763 scored rows. Events are exact adjacent float32 changes; row zero is not an event. Only raw features, sequence IDs and masks are read. The first scan measures cochanges divided by event unions (Jaccard) on scored rows, while deriving ages from all rows.

The follow-up was planned after inspecting those overlaps. It hashes each field's event bitstream across every row, then compares individual ages with seven-group and nine-family clocks. Group age is the minimum member age. Ages are capped at 10,000 and transformed by `log1p(age)/log1p(10000)`, matching the current representation.

## Evidence and interpretation

| Family | Mean within-family scored-row event Jaccard |
|---|---:|
| i0 price | 0.9353 |
| i1 price | 0.9448 |
| i0 volume | 0.7910 |
| i1 volume | 0.8422 |
| Each four-field dp or dv block | 1.0000 |
| a fields | 0.8653 |

Cross-block overlap is not universal: i0 dp versus dv averages 0.5589 and i1 dp versus dv 0.7732. The scientific G arm combines each dp/dv pair into seven groups; the EDA's nine families separate them.

The all-row follow-up finds **65 distinct event-pattern classes among 112 fields**. Several eleven-field price blocks share complete event bitstreams. Some apparent four-field equalities disappear outside scored rows: i0_dp1, i1_dp2 and i1_dv2 each have one fewer total change than their block peers. Unit scored-row Jaccard therefore does not certify identical full clocks.

| Grouping | Scored field–row ages changed | Mean normalized log-age difference |
|---|---:|---:|
| Existing seven groups | 14.3923% | 0.0189424 |
| Nine named families | 11.5332% | 0.0144376 |

The scans took 10.914 and 6.974 seconds under a 400,000,000-byte worker cap, with zero OOM or swap. Peaks were 399,998,976 and 399,986,688 bytes; memory-limit events show reclaim pressure. The first scan's censoring counts exactly agree with the independent notebook implementation.

## Limits and validity

This is exposed descriptive evidence from one sequence sample. Event hashes certify only observed strings, not future equality, equality of raw field values or an optimal representation. The first overlap conditions on scored rows; the second event check includes every sampled row. No population interval is estimated. Age measures row count, not verified physical time. The nine-family alternative is not silently added to the frozen six-fit experiment.

## Decision and next step

Interpret F-versus-G as a comparison with the specific seven-group control. A nine-group follow-up or activity-clock interaction remains a separate hypothesis requiring matched capacity, initialization and source-qualified evaluations. Do not infer that a larger count of ages is the causal ingredient. The [literature review](../../pr_audit_25_40/clock_literature/README.md) supplies related mechanisms and marks the already implemented log-age transform as a duplicate idea.

## Source records

See the [clock plan](CLOCK_EDA_PLAN_R1.json), [statistics](CLOCK_STATISTICS_R1.json), [execution](EXECUTION_R1.json), [geometry plan](CLOCK_GEOMETRY_PLAN_R1.json), [geometry result](CLOCK_GEOMETRY_RESULT_R1.json), [geometry execution](GEOMETRY_EXECUTION_R1.json) and [live notebook plots](../../eda/README.md). Scientific scores remain separate in the [F0 correction grade](../RESULTS_R2.md).
