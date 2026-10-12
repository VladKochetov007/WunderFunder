# Reviews and audit receipts

This domain stores independent critiques, strategy reviews, and structured audit receipts. Reviews interpret or challenge evidence; they do not replace the source experiment, score receipt, or append-only ledger row they cite.

## Current structure

- [Receipt index](receipts/README.md) describes machine-readable audit records, including the F1 recipe audit and the strategy plan coverage review.
- [`legacy/`](legacy/) contains historical narrative reviews and critiques moved from the repository root and `writeups/`.
- Operational controller commentary is indexed under [operations](../operations/README.md), and imported external competition writeups are under [sources](../sources/README.md).

## Review files

- [First-principles review plan and verdict (2026-10-12)](FIRST_PRINCIPLES_REVIEW_PLAN_20261012.md) — independent critique of the whole campaign: the productive Sep 15–28 analysis versus the Oct 6–12 compliance-heavy phase, the ignored power analysis (1-fold/1-seed MDE 0.0064), effort spent on sub-noise recipe tuning instead of variance reduction, a mis-specified >0.700 public objective and the private-runtime risk. Lists planned deliverables (full review, ranked idea cards, ledger rows). Proposal and static review of recorded evidence only; no new fits or recomputed scores.
- [October 12 critical review and separate public and future private leaderboard plans](CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md) — audits recipe comparisons, fold exposure, target assumptions and delivery failures; proposes ranked experiments, scaling ladders and separate public and future-private selection criteria. Source-linked retrospective and proposed plan; no new empirical results.
- [Independent strategy review receipt](../../campaign_2026-10-12/strategy_review_r1/INDEPENDENT_REVIEW.json) — records the validation, model and literature reviewers, material corrections and remaining limitations. Static critique by three Luna roles, not an empirical replication.
- [WNN strategy review and agent execution brief](legacy/WNN_2026-10-11_Strategy_Review_and_Agent_Execution_Brief_EN.md)
- [Unified campaign review](legacy/WNN_Unified_Review_EN.md)
- [Independent critic memo](legacy/critic.md)
- [Literature and modeling review](legacy/review_2.txt)
- [WunderFunder source audit](legacy/WunderFunder_audit_evidence.md)
- [F1 recipe audit receipt](receipts/F1_RECIPE_AUDIT_R4.json)

The former root `reviews/` and `writeups/` aliases have been removed. New reviews should be saved here and linked from the relevant domain README and source experiment record.

## Resumed strategy execution — 2026-10-13

[Strategy execution campaign](../../campaign_2026-10-13/strategy_execution_r1/README.md) implements the full October12 review under the user's subsequent explicit resume. It links the approved plan, full proposal coverage, resource authority, frozen comparisons and live frontier. All Kaggle allowance and actual local GPU capacity are authorized; CPU/RAM protections and no-paid-spend remain. Initial status is preparation, with no new fitted result. Existing folds remain exposed.

- [Independent resource-policy review R1](../../campaign_2026-10-13/strategy_execution_r1/reviews/resource_r1/README.md) checks the new GPU allowance against actual launch code and historical records. It confirms the GPU-cap change and identifies unused active-worker RAM reservations missing from admission accounting; a narrow successor is required before concurrent admissions. Its 24 tests are engineering evidence, not scientific gains.
[F0 target-amplitude delivery packet](../../campaign_2026-10-13/strategy_execution_r1/target_amplitude_f0_s3_r1/README.md) binds the exact clean-T3 F0 cache, seed-3 initializer, matched control and FP32 CP5 export path. Its R2 correction receipt documents two launch-blocking findings in R1 and the corrected dynamic allocator and absolute resume paths. R2 awaits independent recipe review and root's fresh fit release; no candidate result exists.

- [R2 recipe correction receipt](../../campaign_2026-10-13/strategy_execution_r1/target_amplitude_f0_s3_r1/RECIPE_REVIEW_CORRECTION_R2.md) records the independent critic's R1 findings, R2 fixes and artifact hashes for H-268; it is preparation evidence, not a fit result.
- [R2 exact fit/export/grade commands](../../campaign_2026-10-13/strategy_execution_r1/target_amplitude_f0_s3_r1/FIT_AND_EXPORT_COMMANDS_R2.json) gives the staged root-controller and worker argv, including pinned live-VRAM safety reserve and absolute resume paths; launch remains disabled pending critic verdict and root release.
- [R2 source/input bindings](../../campaign_2026-10-13/strategy_execution_r1/target_amplitude_f0_s3_r1/SOURCE_AND_INPUT_BINDINGS_R2.json) binds the actual F0 teachers, cache, parent, split, propensity, historical control, profile and two explicit source overrides.
- [R2 source delta](../../campaign_2026-10-13/strategy_execution_r1/target_amplitude_f0_s3_r1/SOURCE_DELTA_R2.patch) separates the target-scale trainer treatment from the operational resource-policy override; it changes no model/optimizer/schedule target semantics beyond the intended amplitude.
- [R2 complete source manifest](../../campaign_2026-10-13/strategy_execution_r1/target_amplitude_f0_s3_r1/source_overlay_r2/SOURCE_MANIFEST_R2.json) hashes all R3-derived source files and identifies the trainer and resource implementation overrides; no model fit has run from it.
