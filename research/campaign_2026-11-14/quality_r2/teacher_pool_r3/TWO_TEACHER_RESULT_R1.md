---
kind: result
status: complete
created_utc: 2026-11-14T15:09:00Z
updated_utc: 2026-11-14T15:09:00Z
hypothesis_ids: [H-Q14-TEACHER-POOL-R3]
experiment_ids: [E-Q14-TWO-TEACHER-R1]
evidence_paths: [TWO_TEACHER_PROBE_R1.json, TWO_TEACHER_GRADE_R1.json, IMPORT_R1.json]
---

# Equal FBC and TLead teacher target

## Question or claim

Can two available clean teacher exports improve the current single teacher's target quality? One fixed equal mixture was declared before its CPU grade. The user reserves local GPU for PHCapital; this diagnostic uses only completed predictions and adds zero fits.

## Mechanism and method

Original PR49 FBC seed5 and TLead seed5 use different inputs/representations with clean, source-qualified train-only ancestry. Their fixed epoch3 tensors were imported without numerical changes; all16 ancestor memberships exclude valid. The existing reproduced FBC is the comparison teacher. Score all1,873 official-valid sequences and3,528,273 scored rows with unchanged `global_wp_clip2_v1`; check against the separate metric reference. Coefficients are exactly .5/.5, with no coefficient or checkpoint search.

## Evidence and interpretation

| Fixed endpoint | t0 | t1 | WP |
| --- | ---: | ---: | ---: |
| Original FBC s5 | .687842381 | .694867585 | .691354983 |
| Original TLead s5 | .687837982 | .694992576 | .691415279 |
| Current reproduced FBC | .687875494 | .694965466 | .691420480 |
| Equal original FBC/TLead | .688780566 | .695740331 | **.692260448** |

The fixed pair improves the current teacher by **+.000839968**, with both targets positive. Metric/reference agreement passes. The individually similar teachers contribute useful complementary predictions on this evaluation block. The grade binds complete prediction bytes, selected checkpoint bytes, source-qualified fitting/evaluation membership, provenance and target-cache identity.

## Limits and validity

Full valid is repeatedly exposed and this is one additional fixed-mixture query. There is no new seed replication, uncertainty interval, pristine holdout, student result or public-score forecast. This target remains below the previously measured complete native TL3C endpoint (.692335900); different numerical execution also precludes treating that difference as a clean ablation. No new serving winner or ZIP is established.

The first grading attempt rejected the old teacher's `campaign_export_scope="evaluation"`. The consumer was repaired narrowly to accept that original scope only for the baseline; imported teacher exports still require `full-valid-feature-only`. Exact full-valid membership, source, checkpoint and prediction hashes remain mandatory. The rejected invocation produced no accepted result.

S6 export remains incomplete. `target_quality_trigger=false` means this two-teacher diagnostic cannot satisfy the separate frozen three-teacher admission gate; it does not mean the measured diagnostic delta missed .0005. `diagnostic_delta_at_least_0005=true` records that arithmetic separately.

## Decision and next step

Retain stronger-teacher distillation as the leading prepared follow-up. The three-teacher grade, train prediction caches, canonical dataset target check and measured resource admission remain outstanding before the proposed three matched student fits. Local GPU work stays held for PHCapital; no automatic restart. Existing six-fit Kaggle panel is unchanged and has no outputs at this snapshot.

## Source records

[Frozen diagnostic](TWO_TEACHER_PROBE_R1.json), [grade](TWO_TEACHER_GRADE_R1.json), [import receipt](IMPORT_R1.json), [independent import review](INDEPENDENT_REVIEW_R1.md), [prospective student recipe](RECIPE_R3.json), [GPU hold](../LOCAL_GPU_HOLD_R1.json), [allocation](ALLOCATION_R3.json). H/E records remain in the append-only repository ledgers.

[Independent grader review](TWO_TEACHER_REVIEW_R1.md) supports this narrow diagnostic; it audited code/receipts, not an independent array rescore. Target-cache identity is pinned in code but omitted from the grade JSON along with scorer-source hashes; retain the source and review when reusing this receipt.
