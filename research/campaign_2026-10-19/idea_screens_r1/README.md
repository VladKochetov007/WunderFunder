# Idea screens R1: bounded EDA and simple models

The user authorized trying EDA and simple models for the25discussion cards. Start with [measured findings in Russian](RESULTS.md) and the [status of all25](STATUS_25.md). This wave completed48whole-sequence data materialization, fit-only operator diagnostics, nine fixed weighted-ridge views and matched-row context from existing current F0 predictions. No model GPU, quantization, historical weight recovery or candidate submission.

## Protocol and review

- [Contract](CONTRACT.md): canonical exposed F0, target-blind component selection seed19019, ≤32fit/16evalgroups, stride40 only after full causal features, fixed ridge1.0, no evaluation tuning. [Original screen map](SCREEN_MAP.md) declares what each idea can and cannot establish.
- [Implementation review](REVIEW.md) preserves initial blockers and their pre-execution fixes: complete eligible memberships, global component separation, original mask/warmup checks, physical scratch, accurate coverage labels and component rather than sequence pooling. [Result review](RESULT_REVIEW.md) independently checks compact source/output bindings, all9scores, component heterogeneity and same-panel reference limits.
- [EDA design](DESIGN.md), [sample materializer](sample_data.py), [causal feature views](feature_views.py), [fixed ridge](ridge_probe.py), [EDA entry](eda_mechanisms.py), [input summaries](input_diagnostics.py), [fixed target operators](target_operator.py), [constants](constants.py), [analysis entry](run_analysis.py). Five EDA invariant cases and four direct probe helper cases were exercised by authors; independent unittest discovery executed five exposed test cases. Exact coverage is in the test files, not inferred from the count.
- [EDA tests](test_eda_mechanisms.py), [probe tests](test_probe_helpers.py), [reference alignment](reference_panel.py), [plot source](plot_results.py), [report entry](run_report.py). Root independently called all four plain-function probe checks after noting that unittest discovery covers only the five EDA class methods; all four passed. Reporting was a documented amendment after initial ridge progress; it added no candidate/setting.

## Actual outputs and execution

[Sample manifest](records/SAMPLE.json) hashes to0b03a4e2a6aab4eca36bdbb721d30e94b675c3425c0e0a7deca5ae562c2bbaa5; [pre-array membership](records/MEMBERSHIP.json) records the frozen group/component/source choice.32fitgroups/14components and16evalgroups/6components;35467evalscoredrows. Source Parquet stats/footers were freshly checked; full payload SHAs are registered prior identities, not a new35GBrehash. Every selected sample file was content-hashed. Arrays remain in `B/research_tmp/nf19-sample-v1`, B=`/home/grinchenkoeg/computer-vision-homework-20261018`.

| Phase | Launch | Raw capture | Result / timing |
|---|---|---|---|
| Sample | [launch](SAMPLE_LAUNCH.json) | [capture](records/SAMPLE_CAPTURE_RAW.json) | [sample](records/SAMPLE.json),12.09s |
| EDA | [launch](EDA_LAUNCH.json) | [capture](records/EDA_CAPTURE_RAW.json) | [result](records/EDA_RESULT.json),[start](records/EDA_START.json),[done](records/EDA_DONE.json),27.40s |
| Nine ridge views | [launch](RIDGE_LAUNCH.json) | [capture](records/RIDGE_CAPTURE_RAW.json) | [result](records/RIDGE_RESULT.json),[start](records/RIDGE_START.json),[done](records/RIDGE_DONE.json),[progress](records/RIDGE_incremental_progress.jsonl),76.73s |
| Fixed reference and figures | [launch](REPORT_LAUNCH.json) | [capture](records/REPORT_CAPTURE_RAW.json) | [reference](records/REPORT_REFERENCE.json),[done](records/REPORT_DONE.json),5.90s |

All four own units exited0, were captured as exact raw bytes and stopped; no model/array execution on Mac. CPU50%, max1.5GB/swap0, physical scratch, serialized with existing F3. EDA source used remote `B/support-source/idea_screens_r1_v1`; ridge used separately stagedv2after the component-pooling correction. V1 ridge code was never executed. The active EDA dependencies and corrected ridge code are retained here; per-launch complete source inventories remain in their launch/START records and remote immutable folders. Report source is `B/support-source/eda19-report-v1`.

The sample phase coincided with580new hierarchical memory.high events, max561/OOM0 unchanged and parent-local0; EDA/ridge/report added0events afterward. No cap was increased. The missing local `marimo-pair` skill was searched at the instructed path and installed skill catalog but unavailable; these are standalone bounded scripts/figures, not a live notebook session. No dependency installation.

## Figures and next step

[PNG](plots/screens.png) / [SVG](plots/screens.svg) were rendered on B and visually inspected locally. Raw JSON, per-component outcomes and limits are in [RESULTS](RESULTS.md). [Next bounded wave](NEXT_WAVE.md) fixes fit-only regularization selection and separate EMA controls before any new result; it is not executed yet. Full recipe claims retain30pairedcells/arm, strictFP32, clean ancestry and official evaluation. This wave is exploratory, not a complete test of every mechanism or a public-score forecast.

[Completed R2 follow-up](../idea_screens_r2/RESULTS.md) separates fit-selected regularization from alpha1 sensitivity and slices signed/absolute EMA. The initial lag loss largely vanishes in the pooled score, while the joint EMA gain becomes essentially zero under fit-only selection; component concentration and ranking reversals remain material. No full-recipe gain is claimed.
