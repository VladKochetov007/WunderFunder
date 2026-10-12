# Validation and public transfer

This domain distinguishes clean held-out evidence, exposed validation, all-data fitting, and public leaderboard observations.

## Interactive results dashboard

Open the [October 12 results dashboard](RESULTS_DASHBOARD_20261012.html) for a self-contained, searchable view of the public-score climb, local/public comparisons, runtime plot, recorded confidence intervals, and all 798 experiment and 483 hypothesis entries. It was built by [build_results_dashboard.py](build_results_dashboard.py) from the append-only [experiment ledger](../../experiments.jsonl) and [hypothesis ledger](../../hypotheses.jsonl), the [preserved portal roster](../../campaign_2026-10-12/public_score_regression_r1/raw_portal_roster.tsv), its [endpoint map](../../campaign_2026-10-12/public_score_regression_r1/endpoint_coverage_r1.csv), and the [October 12 A/B/C score record](../../campaign_2026-10-12/public_runtime_quality_r1/PUBLIC_RUNTIME_QUALITY_R1.csv). Its local/public and runtime figures embed the frozen [protocol-separated R2 plot](../../campaign_2026-10-12/public_validation_plot_r2/PUBLIC_VS_LOCAL_BY_PROTOCOL_R2.svg) and [runtime/score plot](../../campaign_2026-10-12/public_runtime_quality_r1/PUBLIC_RUNTIME_VS_SCORE_R1.svg). The HTML embeds its data for offline viewing; receipt links open the source files in the repository.

This is a dated snapshot, not a live view. Rebuild it after the ledgers or score records change; the command below reads their current contents while the frozen source plots remain as cited.

The visible public climb has **41 scored outcomes in 45 detailed portal records**. Three older attempts are absent from the preserved roster, so the page explicitly reports 48 known total attempts and does not plot or infer the missing outcomes. The highest recorded score is C, **0.6671**, reported by the user; A/B/C each have one public outcome and no public confidence interval. Their packages change blend composition, so the runtime plot is descriptive rather than a causal runtime or model-count comparison.

For local/public transfer, the dashboard plots only the 11 source rows with both metrics and separates their validation protocols. It does not fit a pooled score-conversion line. One key warning is the exposed fold-1 pair: its local blend scored 0.67329 versus 0.67103 for the student, yet public scores were 0.6650 versus 0.6652. For fixed blending, the equal teacher trio scored 0.680902 against a distinct-seed mean of 0.679018; adding that trio at 50% to the stronger student scored 0.683167 versus 0.683142 for the student alone. The conditional gain was +0.00002562 (paired 95% component interval [−0.00138485, +0.00139836]; 141 exposed components, 800 bootstrap draws). The interval reflects resampling of exposed components, not variation across training seeds or public-transfer uncertainty. Across the ledgers, 53 experiment records contain explicitly named CI fields; missing intervals are labeled as unrecorded.

Use the findings cards for the current interpretation, then filter the ledgers by model, hypothesis, result, status, or CI and follow the linked receipt/report paths. The archive remains descriptive: it does not promote candidates, forecast public scores, or turn local deltas into public points. Related records include the qualified public-gap hypotheses [H-382–H-386](../../hypotheses.jsonl#L426-L430), blend-quality hypotheses [H-387](../../hypotheses.jsonl#L438) and [H-390](../../hypotheses.jsonl#L441), and experiments [E-PUBLIC-REG-MODELS-20261012-R1-R2](../../experiments.jsonl#L755), [E-CLEAN-T3-F0-EQUAL-TRIO-FIXED-CONDITIONAL-20261012-R1](../../experiments.jsonl#L778), and [E-CLEAN-T3-F0-CONTROL-S3-OLDZ4-FIXED-HALF-20261012-R1](../../experiments.jsonl#L796). H-381 occurs twice in the preserved ledger after a recorded ID collision; see the [original row](../../hypotheses.jsonl#L425) and [corrected seed-count qualification](../../hypotheses.jsonl#L431). The underlying package and method assessment is in the [SOTA submission reflection](../../campaign_2026-10-12/SOTA_SUBMISSION_REFLECTION_20261012_R2.md).

Rebuild the HTML with the repository's bounded command:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python research/unified/validation/build_results_dashboard.py
```

## Read these sources

- [Data, metric, and validation protocol](../archive/2026-10-12/pages/01-data-and-validation.md)
- [Public calibration and plots](../archive/2026-10-12/pages/05-public-transfer.md)
- [Protocol-separated public/local plot](../../campaign_2026-10-12/public_validation_plot_r2/README.md)
- [Runtime versus public score plot](../../campaign_2026-10-12/public_runtime_quality_r1/README_R1.md)
- [Current competition state](../../../specs/current-state.md)

Keep fold, fitting membership, ancestry, selection exposure, and score uncertainty attached to every point. Public results are transfer observations; they are not a replacement for reliable cross-validation or a basis for adding local deltas to public scores.

## Resumed strategy execution — 2026-10-13

[Strategy execution campaign](../../campaign_2026-10-13/strategy_execution_r1/README.md) implements the full October12 review under the user's subsequent explicit resume. It links the approved plan, full proposal coverage, resource authority, frozen comparisons and live frontier. All Kaggle allowance and actual local GPU capacity are authorized; CPU/RAM protections and no-paid-spend remain. Initial status is preparation, with no new fitted result. Existing folds remain exposed.

- [Evaluator precheck R1](../../campaign_2026-10-13/strategy_execution_r1/EVALUATOR_PRECHECK_R1.json) records 43 passing scorer, bootstrap, mask and component-crossfit tests, with source hashes frozen before new outcomes; it establishes engineering correctness only.
- [Domain-summary producer audit](../../campaign_2026-10-13/strategy_execution_r1/DOMAIN_CACHE_SCHEMA_OBSERVATION_R1.json) finds partial training coverage and target-derived fields in existing sequence summaries. Reuse requires observable-column whitelisting and fitting-only membership; no stress result is claimed.
- [Initial frozen strategy stress panels](../../campaign_2026-10-13/strategy_execution_r1/DOMAIN_PANELS_R1.md) and [machine-readable cutoffs/membership](../../campaign_2026-10-13/strategy_execution_r1/DOMAIN_PANELS_R1.json) implement S30 activity, instrument-1 frequency and feature-support panels with valid-fit-only cutpoints, at least 60 components per stratum and fixed 25/75 component-frequency mixtures. Source-qualified coverage and input-column whitelisting are explicit; three correctness tests passed. Remaining panel families and independent review are pending; this is preparation evidence, not a fitted improvement.
