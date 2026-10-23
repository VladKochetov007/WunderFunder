# Groupwise fast/slow train-proxy grader review R1

**Disposition: PASS for the bounded train-row proxy grade.** This was an independent static review of the grader, its addendum, the frozen fit recipe and receipt, the data overview, the sequence reader and the metric implementation. The reviewer did not execute Python, open Parquet payloads, read labels or launch a worker.

## Findings

- The data contract says `train.parquet` contains `t0`/`t1` but no `is_scored`; the explicit scored-row mask belongs to validation. The original official-mask grader stopped at this missing column before producing a metric result.
- The new grader explicitly selects every target-bearing train row from step 99 through 19,999 and labels this as a development-only proxy. It does not claim that these rows reproduce the official validation mask or score.
- The addendum binds the exact fit-recipe SHA, fit/prediction-receipt SHA, grader SHA and data-contract SHA. Before reading train targets it checks the receipt pin, frozen recipe source hashes against the receipt, component membership, member-manifest bytes, model archive, and every OOF prediction array.
- The candidate/control deltas and component bootstrap use paired resampling over the same 32 component moments. The uncertainty remains conditional on four fixed cross-fit models and does not include seed, sample-selection, official-mask or transfer uncertainty.
- The reader handles one sequence at a time and the output is restricted to physical research scratch. Prior exact feature/fit and first-row grade resource receipts stayed under the admitted 2 GB cap with no new OOM/max events; this review does not itself profile the completed grade loop.

## Scope limits

This is a code/provenance review, not a metric rerun, result review, clean validation claim, recurrent endpoint comparison or B300 recommendation. Any proxy result must retain its nonofficial mask label and exposed-sample limitations.

## Provenance

Reviewed the [grader](../../campaign_2026-10-18/online_features_r1/group_fast_slow_r1/grade_train_proxy.py), [grade addendum](../../campaign_2026-10-18/online_features_r1/group_fast_slow_r1/grade_train_proxy_r1.json), frozen [fit recipe](../../campaign_2026-10-18/online_features_r1/group_fast_slow_r1/recipe.json), [fit receipt](../../campaign_2026-10-18/online_features_r1/group_fast_slow_r1/fit_predict_receipt_r1.json), [data contract](../../../docs/data_overview.md), `src/wnn/data.py` and `src/wnn/gmetric.py`. The reviewed grader SHA-256 is `1d685314dd86a147ee8cc05979825be5ff6fc94bda1b59a87b06b0df59e422da`; the pinned fit receipt SHA-256 is `a7f6c65aa623834907d60bcf23b13cee92a66fdbde7eff4d8479218e16c07205`.
