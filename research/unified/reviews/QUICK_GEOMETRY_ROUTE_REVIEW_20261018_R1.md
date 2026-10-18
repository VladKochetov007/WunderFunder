---
kind: independent-review
status: pre-result-route-review
created_utc: 2026-10-17T21:38:00Z
experiment_ids: [E-GEOMETRY-QUICK-PROBE-20261018-R1]
---

# Independent quick-submission route review

Reviewer: native `quick_submission_route`, gpt-6-luna / max, fresh context and read-only assignment. Root records the returned public findings. This is pre-result route criticism, not approval of an unseen package.

The [frozen recipe](../../campaign_2026-10-18/quick_geometry_probe_r1/recipe.json) matches parent, seed 3, F0 membership, 4,096 replacement draws of complete sequences, update budget, fresh optimizer/schedule and fixed epoch between feature and same-width zero-suffix control. The initializer expansion zeroes 60 new input-weight columns and sets mean zero/std one, preserving the parent function apart from FP rounding. The geometry prior is deterministically sampled from 64 source-qualified fitting groups and excludes the outer evaluation components. Its prefix cumulants, exact-level categories/occupancies and dwell statistics are causal.

This supports an exploratory interaction probe. It does not establish 4,096 unique fitting sequences, seed robustness, fresh holdout evidence, significance or teacher/student transfer. Parent-versus-child includes an optimizer continuation; feature-versus-control isolates this added block at the stated short recipe. R3 small-panel/mixed results are motivation only. No blend is evidence-backed before paired predictions exist.

The material release risk is serving. The existing shared streaming class rejects geometry and native EnsemblePredictor rejects its 271-channel input. A custom streaming adapter must match the checkpoint's 60-column feature order/prior plus 99 event features, update on unscored warmup and reset completely at each new sequence. The generic native ModelStepper.run accepts arbitrary input width but allocates on every one-row call, making full callback runtime a real risk. Require actual feature parity, reset A–B–A, checkpoint/native FP32 prediction parity, ZIP size/ABI/import and callback-throughput measurements. A short synthetic callback proxy is not a full target-host runtime guarantee.

The actual profile establishes 60 finite updates, 3.33 GB cgroup peak and 1.11 GB VRAM reserve, median 0.920 seconds per B8 batch. It supports fitting admission only; no quality or serving feasibility result existed at review time. The root retains execution and packaging ownership.

## Follow-up: current-best integration review

A second bounded read-only review checked the user-supplied tl3 reference. Its audited identity is `pkgwork/tl3.zip` (12,538,394 bytes; SHA256 `d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`) on Beleriand. It is not present in this local checkout/known scratch path, and the configured SSH route timed out. Nearby local C/E archives have different members and are not replacements for tl3. The inventory reports `wnn_cpp` with 211-input distilled and 263-input tiedlead members, while the new geometry engine is `wnn_native`/271 inputs. An exact combined callback needs the original tl3 source/native payload; compatibility is unverified. No blend is delivered merely from matching names.

The 39m12s public tl3 runtime leaves 20m48s to a 60-minute limit. A small coefficient does not reduce new-member compute. Exact final ZIP size, dual-runtime compatibility and complete callback measurements are required; arithmetic addition of timing or local/public score deltas is not evidence of a valid stronger endpoint. Meanwhile the serving actor measured synthetic Python/FP32 callback parity but found its cost too high; optimized/native or quantized serving is being evaluated separately. Those are engineering measurements, not model quality results.

## Native implementation source review

The same independent reviewer inspected serving source commits `bdf996f9d` and `ce5f8f63c`, before native build. No concrete static mismatch was found. Training and native order are `[raw112|event99|geometry60]`; geometry offsets are mean0:4,std4:8,residual8:12,atom occupancy12:16,normalized16:20,current levels20:36,occupancy36:52,dominant-event52:56,dwell56:60, selecting raw0,3,52,55. Accumulators are sequence-localFP64. First-row geometry treats a change as true while events have zero delta. Reset clears all state; every output slot is overwritten before model execution. Native recurrence/features advance even when need_prediction=False. The production callback intentionally requires one271-input member and an actual GeometryPredictor API; it has no slow fallback and is not a TL3 callback. This is source review only; actual-weight parity, ABI and measured runtime remain required.

## Native execution found a defect missed by source review

The first native implementation indexed the first 20 geometry features by field instead of statistic. Independent source review had not detected it. Full 20,000-row execution parity against the Python producer exposed the mismatch before packaging. Correction `03e6db635977d9514f46ec990738112075c588de` uses statistic offsets plus field index, and retains FP64 division until the final dominant-event fraction cast. The corrected library SHA-256 is `ebba8fe295175bf1c2ab57dbe76dd764c5c893941020af774be95cfe8a83043d`. Synthetic event/geometry/prediction parity, resets, ABI and Python 3.11 import subsequently passed. Actual trained-checkpoint quality and exact-ZIP checks remain separate release work. This demonstrates a concrete limit of the earlier static review.

## Independent R2 adapter source review

The `quick_submission_route` Luna/max critic inspected adapter setup, initializer expansion, normalization inheritance and optimizer construction without running a fit or loading tensors. For this declared recipe, adapter configuration occurs before optimizer construction; every parameter is frozen except the masked `inp.weight`, AdamW is fresh and decay is zero. No source-level preservation blocker was found. Actual preservation remains subject to the post-fit bitwise checkpoint check.

Two reusable-API risks were identified: adapter setup can override a separately supplied generic `--freeze` flag, and the suffix boundary is inferred from `added_columns` without independently checking feature-name positions. R2 uses no generic freeze and the audited append-only 211→271 layout, so these do not invalidate this frozen run. Add explicit guards before general reuse. Do not modify the running source snapshot. The scientific result remains an adaptive one-fold followup, with no clean-CV, fitting-seed or public-transfer claim.

## Independent completed R2 and initial ZIP review

The same independent critic checked the completed recipe, score/comparison/progress receipts, frozen-parent proof, exact ZIP inventory and hash-bound build receipt. The result agrees across receipts: adapter 0.6804499040 versus parent 0.6812265132, fixed-fit component interval [−0.00135282, −0.00027913]; 5,120 updates in 693 seconds; old 16 state keys and 211 input columns preserved, all 15,360 appended weights learned. The ZIP is 5,323,202 bytes with eight entries and one coefficient-1 q8 member. Static ELF/GLIBC checks passed.

At this review point, the exact offline callback/streaming gate was still pending and remained a concrete release blocker. Static code/order/reset checks did not remove that requirement. Full actual-data replay, actual-trained-weight parity and final package receipts must be checked before calling the ZIP ready.

## Final independent release interpretation

The critic independently rechecked the selected 5,323,200-byte q16 submission hardlink/SHA, exact-ZIP Python 3.11 smoke, same-precision native-cache panel, and full 468-group F0 replay. It found no concrete blocker to the explicitly requested exploratory release. The trained recipe remains negative versus its parent, and the native endpoint has its own directly measured WP 0.6804519697.

The predeclared Torch/native FP32 elementwise tolerance **failed**: maximum absolute difference 0.0007533, mean 0.0000555. Native construction uses `Activations::Fast`; the cause of the discrepancy has not been isolated. No threshold was relaxed, and Torch-level numerical equivalence is not claimed. This does not invalidate the independently evaluated deterministic native q16 callback. Runtime remains a local 40.4-minute callback / roughly 44.4-minute full-replay-cost extrapolation, not an observed competition run or guarantee. See the [hash-bound release record](../../campaign_2026-10-18/geometry_adapter_probe_r2/RELEASE.json) for the exact delivered artifact and gate scopes.

## Prospective remaining-hour R3 review

The independent critic inspected the already-reviewed B300 mask implementation and frozen-adapter rules. `moments_only` retains0:20 and zeroes20:60; `dictionary_only` zeroes0:20 and retains20:60. With zero initialization and AdamW decay0, corresponding input weights remain exactly zero, allowing the same full60-feature native callback without a new mask implementation. Root synchronized the audited feature/CLI/provenance code before profiling and admission. Actual dictionary-mask profile completed60finiteupdates withoutOOM before the two fits started.

Different active widths can alter gradient norms/clipping; retain preclip norm and clip counts, interpret results as exact recipe rankings rather than general feature-family effects. The sourceparent/preprocessing/sampler/seed/update budget must match, and old211 columns/backbone plus inactivecolumns must be checked after fitting. No new zero-feature fit is necessary because the fixed parent is the mathematical zero-adapter control.

The separately predeclared alpha grid scales only the existing R2 input correction. It creates no fitting replication and is nonlinear in output through GELU/GRU; sign reversal cannot by itself identify auxiliary-loss conflict. Ranking is adaptive on the same468exposedF0sequences. Conditional component intervals do not cover alpha selection, trainingseed or public transfer. The original ZIP remains frozen. A critic's runtime sentence was corrected: actual fullF0 native replay took633seconds;40.43minutes is a39.4million-row extrapolation. Four new complete-F0 GPU forwards and one selected native replay are bounded by the remaining-hour contract.
