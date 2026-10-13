# Independent FP32-continuation and W4 compression grade R1 — 2026-10-13

This grade separates two questions using existing frozen endpoint sums: how 10% and 30% FP32 continuation endpoints score after native-Q8 deployment versus the unchanged parent, and what W4 post-training quantization costs relative to native-Q8 deployment of the **same** FP32 checkpoint. No predictions were reread, no fits were run, and no evaluator was changed.

## Identity checks

I ran the existing paired component-bootstrap grader (`grade_qat_component_bootstrap.py`, SHA-256 `19ae775c4e4402a9185a21e24ccf7fc5577f0ecfdf3b8b4a2fdd772d4c79e95d`) in the campaign's one-CPU/500-MB bounded wrapper. It returned `graded` for all four pairs against the admitted unchanged-parent anchor `deployment_frozen_r2/native_q8`, under manifest SHA-256 `8894af708dcba1500fc8c1b431d30468a29a9999531dd7133ec982c3b1d19556`.

The consumer verified each endpoint's own manifest and source-file hashes, exact ordered 468-group F0 membership, the 141-component mapping, 808,863 selected rows, official `global_wp_clip2_v1` score recomputation and full-sequence replay. Each candidate also matched the anchor's per-component target-only sufficient statistics (`W`, `WY`, `WYY`) for both targets. These checks establish a common full-fold panel and target mask fingerprint; the six raw sums do not prove row-level mask identity against every possible collision.

The unchanged parent is checkpoint `e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda`, run `clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1@ep5`, with native-Q8 WP `0.6811353992845013`. The 10% FP32 continuation manifest (`a2c01d7296cc6627563c2e1c4c75c9d0e3fa625de86f39a032980586c24cf7a2`) binds checkpoint `a8d99b018165ff144c7cc2d2b2e2e3e5ff476f3f583ac1df8ecb85354284fb3e`; the 30% manifest (`ebfbe18ce78cfdac0360c8cd8caaacd1d982051cc4451d81cc87604570854e2c`) binds checkpoint `7922c61955d0084d10725cb9ca5d95aa0671dce84cddf77cafa73ed93c7320cf`. Within each FP32 manifest, `native_q8` and `recurrent_w4` cases share that one parent checkpoint, so their paired difference isolates the deployment quantization change for that checkpoint. The 10% and 30% evaluations use deployment source R2 and R3 respectively; within-pair source identity is exact, but cross-scale comparisons are not a matched source/fit contrast.

## Paired results

All deltas below are **candidate minus reference** WP. Bootstrap intervals use 10,000 paired resamples of the same 141 components, seed `20261013`, and the frozen raw-six-sum scoring approximation.

| Contrast | Reference WP | Candidate WP | Paired ΔWP | 95% component interval |
|---|---:|---:|---:|---:|
| 10% continuation native-Q8 vs unchanged-parent native-Q8 | 0.6811354 | 0.6818185 | +0.0006831 | [−0.0001200, +0.0015783] |
| 30% continuation native-Q8 vs unchanged-parent native-Q8 | 0.6811354 | 0.6813920 | +0.0002566 | [−0.0009159, +0.0014795] |
| 10% same-checkpoint W4 PTQ vs native-Q8 | 0.6818185 | 0.6814132 | −0.0004053 | [−0.0012775, +0.0004708] |
| 30% same-checkpoint W4 PTQ vs native-Q8 | 0.6813920 | 0.6806602 | −0.0007318 | [−0.0013993, −0.0000997] |

The native-Q8 point-score order on this exposed F0 panel is 10% continuation, 30% continuation, unchanged parent. Both continuation-minus-parent intervals include zero. The same-checkpoint W4 contrast is uncertain at 10%; at 30%, its point cost is about `0.00073` WP and the descriptive raw-sum component interval is below zero. Treat this as conditional within-panel compression evidence for one seed, not a universal effect or a stop rule.

## Limits and record

These results are full-fold fixed-prediction endpoint comparisons, not pristine cross-validation. The continuation cases use different subset coverage and fit budgets, so the 10% versus 30% absolute ranking does not isolate scale. All comparisons use one fitting seed and no public score. The bootstrap applies `wp_from_sums` to resampled raw six-sum aggregates because the producer does not save the centered moments used by `GlobalAccumulator.result`; intervals are descriptive approximations. They describe component resampling only, not training-seed or public-transfer uncertainty.

The full machine receipt, including every result/component-sum hash, score audit and interval, is [QAT_FLOAT_CONTINUATION_COMPRESSION_GRADE_R1.json](receipts/QAT_FLOAT_CONTINUATION_COMPRESSION_GRADE_R1.json), SHA-256 `cb0149f8d39cf2bb8927742a4fa6747101ec71ce5dc21737c485d3e39208bc36`. Method details are in the [component-bootstrap specification](QAT_DEPLOYMENT_COMPONENT_BOOTSTRAP_METHOD.md). `E-QAT-DEPLOYMENT-20261013-R1`; no package promotion or public/winner claim follows.
