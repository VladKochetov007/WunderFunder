# Independent static review — F1 mean-shrink R1

## Finding

**PASS for root admission as a fixed-endpoint exploratory grade; no static contract blocker found.** The review supports the integrity of the declared protocol and its source identities. It does not establish a score, validate the actor's unrun preflight, or approve target access. Root owns admission and execution.

## Frozen identities and scope reviewed

- Recipe: [F1_MEAN_SHRINK_RECIPE_R1.json](F1_MEAN_SHRINK_RECIPE_R1.json), SHA-256 `78537f892ba969a6f6b61be00a557c93e88de8bf61339a8ef52d9e06f490c35b`.
- Source manifest: [F1_MEAN_SHRINK_SOURCE_MANIFEST_R1.json](F1_MEAN_SHRINK_SOURCE_MANIFEST_R1.json), SHA-256 `1b4f4f7c21aedea95e5bbc287438f230994b9b470dc502604d54b0d03e88b87f`.
- Reviewed implementation: [preflight](f1_grade_preflight.py), [scorer](f1_grade_scoring.py), and [runner](grade_f1_mean_shrink_r1.py).
- The recipe and manifest hashes match. A bounded standard-library hash audit found all 85 manifest pins present and matching, with zero missing or mismatched files. This checked source and metadata identities; it did not open prediction/target arrays or run preflight/scoring.
- Scope is one saved F1, seed-1002, A/lag-1 readout endpoint; no fit. The fixed correction is `parent + float32(0.1) * (raw A - parent)`. The primary contrast is shrink minus parent; secondary contrasts are raw minus parent and shrink minus raw. No alpha search is present.

## Contract checks

| Area | Review result |
|---|---|
| Evaluation identity | Recipe and preflight source require the exact 457 ordered valid row groups and derive 149 connected components from the pinned fold-1 split manifest. Candidate and parent sidecars bind the exact source-qualified order, `[457, 20000, 2]` shape, and float32 dtype. |
| Fit separation | The candidate readout uses 1,243 source-qualified groups across 314 components (1,071 train and 172 valid). The code checks its selected components are disjoint from all 149 evaluation components and its ordered group list matches the subset manifest and fit-cache refs. |
| Parent and ancestry | Parent is `oldz4_h230_parent_f1_s3_r4`, epoch 3. The F1 verifier follows the declared 14-run parent/teacher/initializer/propensity/resume graph; it hashes checkpoint/provenance inputs, enforces expected per-run ordered memberships within fold-1's matching train/valid namespaces, and checks evaluation overlap. Parent history used FP16 AMP; the saved endpoint used for this comparison is CPU strict FP32, which the recipe states. |
| Teacher-cache sidecars | The pinned cache metadata links to the parent provenance and source coefficients. Teacher sidecars bind run, split, epoch, checkpoint, prediction digest, split manifest and ordered row groups. Their producer schema has no explicit `scope` or no-target-read field, and the historical teacher prediction payload arrays are not rehashed by this adapter. The code records that limitation. It is nonblocking for this grade of already-saved, hash-pinned parent and candidate endpoints; it should not be described as a newly verified teacher-payload audit. |
| Target-read gate | Label-free preparation hashes the frozen target archive and checks its size without opening target members. Grade mode requires the exact recipe-bound root admission and preflight receipt before calling the scorer. The scorer hashes the target archive immediately before member access and again after all endpoint scoring, requiring both to match the recipe SHA. |
| Metric and formula | Scorer uses the repository's `GlobalAccumulator` (`global_wp_clip2_v1`): clipped predictions/targets at ±2, official prediction mask plus row index ≥99, and absolute clipped-target weights. Shrink arithmetic is float32. There is no changed metric or additional fitting-target read. |
| Contrasts and uncertainty | The runner emits exactly the three recipe contrasts. Each paired interval resamples the 149 connected components using the canonical centered-moment bootstrap (2,000 draws, fixed seed 271828). Intervals are conditional on these fixed predictions; they do not estimate fitting-seed or public-transfer uncertainty and are not a kill gate. |
| Resources and writes | Recipe and admission checks cap the worker at 400,000,000 bytes, one CPU core, 300 seconds and zero swap, with physical-disk scratch. Output and preflight paths are recipe-bound and create-only. Actual peak memory/runtime remain unmeasured until root runs the limited command. |

## Claim limits

This is a post-F0-designed, single exposed-fold/seed follow-up, not independent confirmation and not completion of the historical 12-fit panel. Arm A fixes the relation slope at zero but retains its 211 raw/event-base, 208 matured/forecast/error/slope channels, running means, and matured-increment history; this comparison cannot support a generic no-history or raw-only claim. No target result or public-score forecast is reviewed here.

## Checks performed and not performed

Performed: static recipe/code/reference inspection; SHA-256 verification of the recipe, source manifest, and all 85 manifest entries; contract review of membership, ancestry, metric, contrast, target gate, and resource checks.

Not performed: label-free preflight, any prediction or target-array read, score, synthetic test, fit, runtime profile, or execution of the grader. The producer-side teacher sidecars' absent scope field remains the recorded metadata limitation above.
