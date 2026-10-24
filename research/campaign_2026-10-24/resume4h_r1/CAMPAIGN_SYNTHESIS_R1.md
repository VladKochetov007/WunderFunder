---
kind: review
status: running
created_utc: 2026-10-24T13:39:35Z
updated_utc: 2026-10-24T13:53:52Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1, H-ASTRA24-FUND-01, H-ASTRA24-FUND-02, H-20261024-tl3-h390-quota-r1]
experiment_ids: [E-20261024-online-f0-exploratory-r2, E-20261024-online-f1-mean-shrink-r2, E-20261024-tl3-h390-quota-r1]
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/FRONTIER.md
  - research/campaign_2026-10-24/resume4h_r1/ALL_PR_RESEARCH_SUMMARY_R1.md
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/RESULTS_R2.md
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/correction_shrink_r1/RESULTS_R3.md
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/f1_mean_shrink_r1/RESULTS_R2.md
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_RESULT_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/RESULT_REVIEW_R2.md
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/RESULTS_R1.md
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/RESULT_REVIEW_R1.md
  - research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/RESULTS_AND_HANDOFF_R1.md
---

# Four-hour continuation synthesis — measured results and open decisions

## Question or claim

What did the October 24 continuation actually learn about the current endpoint, which apparent failures are scientific evidence, and what is still worth buying with the remaining budget? The campaign has produced useful fixed-endpoint diagnostics and two technically checked quota ZIPs, but no new scientific fit or public confirmation. The Kaggle engineering profile may have begun optimizer-bearing microfits; its update count is unknown. The strongest scientific update is narrower: most of the F0 relationship-head correction direction lies in shared calibration/mean-only directions, and that common shrinkage result did not reproduce on F1. This weakens the case for another unqualified B/C head panel while leaving the representation family open.

## Mechanism and method

The user resumed work from 10:38 to 14:38 UTC on October 24, with new admission ending at 14:23. The later marimo request raised the aggregate CPU ceiling to ten cores, but the 93°C report led root to operate at one core. The one-GiB no-swap cap and protected production reserve remain. See the [campaign frontier](FRONTIER.md) and [resume contract](RESUME_R1.json).

The new F0 and F1 score analyses reuse frozen predictions and parents. The original six-fit online panel remains incomplete; these grades add **zero fits**. The quota candidates reuse existing H390 student checkpoints and TL3 weights; package construction, native replay and grading also add zero fits. The clock EDA is target-free and adds zero fits. A separate Kaggle profile was authorized as engineering work, with three optimizer-bearing A/G/F microfit arms, 16 fitting sequences, three epochs, and at most 360 updates; training labels were permitted while evaluation targets were not. Its observed status is reported below. Earlier PR26/27 short-screen fits and current teammate teacher work are not newly run by this campaign.

## Evidence and interpretation

### Saved online heads: the trained corrections lose, and a small common shrink is not stable across cells

The F0 grade covers 468 evaluation sequences, 141 dependency components and 808,863 scored rows. The frozen parent scores 0.6811541494. All six saved head endpoints are lower; the best is C/lag1 at 0.6806097840, a −0.0005444 difference. The A predictions at lags 1 and 16 are byte-identical. The B−A and C−B point changes are much smaller than the common parent loss, and their conditional component intervals span zero. This is evidence against those exact trained F0 corrections on this exposed cell, not evidence that online adaptation cannot help. See the [F0 result](feasibility_scout/RESULTS_R2.md) and its [independent scalar review](f0_grade_review/RESULT_REVIEW_R1.md).

The head architecture already freezes the parent and adds a zero-initialized residual, `parent + 0.1 * residual`; missing parent preservation or initialization is not the explanation. A is a slope-zero online estimator, but it still updates means and retains matured movements, forecasts and errors alongside 211 raw/event inputs. It is not a no-history or no-adaptation control. The [control interpretation](feasibility_scout/CONTROL_INTERPRETATION_R1.md) distinguishes A/B/C from an unfitted same-width zero-state control.

After inspecting the original grade, the same extra float32 multiplier 0.1 was applied to each of five unique F0 correction directions. All shrunken endpoints score about +0.000129 to +0.000136 above the same parent; A gains +0.0001314, almost as much as C/lag1. Every interval crosses zero. This is consistent with a correction that is too large at full magnitude, but does not prove optimizer overshoot, an optimal coefficient, or a relationship-specific gain. The [fixed-shrink result](feasibility_scout/correction_shrink_r1/RESULTS_R3.md) is an exposed post-grade diagnostic, not a preregistered replication.

F1 supplies the important counterexample. The parent scores 0.6699228637, raw mean-only A scores 0.6683554506 (−0.0015674), and the fixed extra-0.1 correction scores 0.6699181675 (−0.0000047). Its conditional interval is [−0.0001129, +0.0001095]. Shrinking removes nearly all of A's observed loss, but does not reproduce F0's positive point change. F0/F1 differ in both fold and fitting seed, and the F1 diagnostic was designed after F0. Neither cell is a fresh holdout; both bootstrap intervals condition on fixed predictions. See the [F1 result](feasibility_scout/f1_mean_shrink_r1/RESULTS_R2.md) and [independent review](feasibility_scout/f1_mean_shrink_r1/RESULT_REVIEW_R2.md).

### Direction probe R2: high geometric overlap does not mean high score contribution

The repaired fixed-prediction probe completed in 90.434 seconds under one core and a 400,000,000-byte cap; peak memory was 399,998,976 bytes, with no OOM or swap. It reproduced every historical score, added no fit, searched no coefficient, selected no endpoint and exported no serving coefficient. The official fixed-alpha operation and metric were retained. The independent [result review](feasibility_scout/direction_probe_r1/RESULT_REVIEW_R2.md) validates point and projection arithmetic; it could not independently reconstruct the bootstrap intervals.

On F0, calibration directions account for 67.0–70.7% of A/B/C weighted correction variance, but their projected mean score derivative is only about 0.00000127 versus a full direction derivative near 0.00158—less than 0.1%. A large norm share therefore does not mean calibration explains score movement. For B/C, calibration plus common A captures 98.79–99.58% of weighted direction variance. Residual alpha-zero derivative means are −0.000002714 (B/lag1), −0.000014617 (B/lag16), +0.000051219 (C/lag1), and −0.000010559 (C/lag16). The reported actor-produced conditional intervals cross zero; the independent reviewer did not reproduce them. The point geometry leaves little resolved B/C direction after shared controls, but establishes neither equivalence nor a broad failure of online state.

F1 A's real-arithmetic derivative is +0.0001511 at alpha zero, with actor-produced interval [−0.000934, +0.001371], and −0.00024195 at alpha 0.1, interval [−0.001313, +0.000958]. These are local path derivatives, not finite score gains. The independent review confirms point derivatives and projection ranks from aggregate moments, but did not reread row predictions/targets and could not reconstruct the bootstrap after reviewer-script and output-capture errors. Intervals remain actor-produced, not independently verified. The [direction geometry PNG](eda/direction_geometry_r2.png) and [SVG](eda/direction_geometry_r2.svg) show the variance-share/score-slope distinction and B/C residual directions.

### Quota packages: ranked on two exposed samples, not forecast for the public leaderboard

The exact ZIPs, byte hashes, coefficients and sample were frozen before grading. Both retain TL3 teachers TL5/TL6 at 0.33 each. H390 is an experiment identifier; the students are H256×2. They were trained on all supplied train and validation data, and the retained TL3 teachers have unresolved ancestry. The fixed packaged native callbacks passed ABI, finite-output and A–B–A reset checks on 64 complete sequences, 32 components from each of F0 and F2. All ordered input/feature/need-mask hashes matched across fresh replays.

| Rank | ZIP | Bytes · SHA-256 | F0 delta vs TL3 | F2 delta vs TL3 | Replay time |
|---|---|---|---:|---:|---:|
| 1 | [TL3 + H390 S3/S4](../../../submissions/quota_20261024/tl3_h390_s3_s4_20261024.zip) | 16,161,940 · `8bf614168d97d50b58557db3a1f5befa16f4912b5b8b1fbf609b087f62f96be2` | +0.0037422 | +0.0002054 | 176.348 s |
| 2 | [TL3 + H390 S3](../../../submissions/quota_20261024/tl3_h390_s3_20261024.zip) | 12,537,972 · `56ebf18d26187541e539a31078a50dcbeb698e395ca0ecde5e1174aa2e929ded` | +0.0035131 | −0.0000293 | 150.550 s |

The pair has the higher point score on both strata, but pair-minus-S3 intervals span zero in F0 and F2, and the pair callback costs 24.1% more than TL3 on this host/sample. Equal-cell mean deltas are +0.0019738 for the pair and +0.0017419 for S3; these are not pooled significance tests. F0 components and F2 components are exposed sampling strata, not clean holdouts. The [quota handoff](quota_submission_r1/RESULTS_AND_HANDOFF_R1.md) supports this exploratory ranking only. Neither ZIP has been uploaded, and no expected public gain is assigned; these local deltas must not be added to the historical public TL3 score of 0.66832.

### PR evidence, EDA and remote execution

The refreshed GitHub inventory contains 38 PRs. Heads and states were unchanged from the initial refresh; only an existing PR32 progress comment changed. It moved from 17,773 to 21,745 of 47,100 updates (46.17%); all four 630M/1.54B × F0/F2 teacher endpoints still had zero official evaluations, with no new artifact link. The BF16/BF16x9 amendment is an engineering/precision variant, not strict-FP32 identity or quality evidence. Its projection is roughly 34.85 fit-hours before setup and evaluation. PR40's ten real-data batch profiles selected throughput settings but produced no scientific score. See the [all-PR summary](ALL_PR_RESEARCH_SUMMARY_R1.md), [refresh interpretation](PR_REFRESH_INTERPRETATION_R2.md), and [PR reports for #1–24](pr_audit_01_24/REPORT.md) and [#25–40](pr_audit_25_40/REPORT.md).

Earlier PR26/27 short-screen results remain a promising external signal: 24 reported cold H512×1 strict-FP32 fits, six epochs, nested 10%/25% component subsets on F0/1201, F2/1202 and F3/1203. Innovation-minus-moments deltas were +0.000853/+0.000640/+0.001345 at 10% (mean +0.000946), versus −0.001015/+0.000459/+0.000469 at 25% (mean ≈−0.000029); age averaged slightly negative at both fidelities. Private prediction/weight payloads are inaccessible. Cells reuse fold/seed, while fidelity changes components, examples, update count and OneCycle progress, so the contrast is not causal. A faithful four-fit comparison is estimated at 6.18 serial B300-hours; a p10 pair costs 1.66 hours but cannot resolve the reversal. Kaggle throughput/artifacts remain unavailable. See [source/command feasibility](kaggle_age_r3/FEASIBLE_NEXT_SCIENCE_R1.md).

The completed target-free EDA scanned 96 preselected validation sequences (1.92 million rows, 192,763 scored rows). Field-row boundary censoring averaged 0.07438%; the maximum field rate was 0.30089%. Seven-group and nine-family clocks differ from per-field ages on 14.392% and 11.533% of sampled scored field-row pairs, with 65 event-pattern classes observed. These are row-count/float-change geometry, not physical time, predictive evidence, or evidence that age is absent. The parent already uses 99 event features. See the [EDA record](eda/README.md) and [clock analysis](feasibility_scout/clock_eda/README.md).

Kaggle R3's first attempt failed before extraction on an admission/packet checksum-copy error; a later terminal-receipt checksum typo was corrected separately and did not indicate a bad remote overlay. Version 3 passed archive/source/raw checks but failed before opening the portable cache because the consumer compared `subset_members_sha256` with a nonexistent/mismatched subset field; it completed zero microfits and optimizer steps. The independently reviewed membership repair was included in R4. Version 4 passed archive, overlay, source-map and raw-source checks, then timed out after 855.52 seconds while waiting on the profile child. The receipt has no profile output, no child exit code and no reliable cache-open telemetry. It is unknown whether cache reconstruction completed or how many optimizer updates ran. It produced no quality grade. This is an execution/telemetry failure, not a negative model result; no retry is admitted. See [R3/R4 history](kaggle_age_r3/README.md), [R4 terminal receipt](kaggle_age_r4/TERMINAL_R4.json), and [cost/telemetry analysis](kaggle_age_r4/CACHE_RECONSTRUCTION_AND_PANEL_FEASIBILITY_R4.md).

### The fit cache has a different per-row measure under equal-batch normalization

The admitted batch-mass audit streamed both original five-column fitting suffixes (991,600,256 bytes total; F0: 1,236 sequences; F1: 1,243) in 6.942 seconds, verifying both full payload hashes. Peak memory was 399,998,976 bytes under the 400,000,000-byte cap, with zero OOM and swap. It read no evaluation labels, imported no model/Torch code, and added no fit. There were 3,018 and 3,035 8,192-row batches; all had positive mass above the denominator epsilon.

The source's equal-normalized-batch measure changes each batch's relative multiplier versus pooled rows from 0.109 to 9.743 in F0 and 0.104 to 8.291 in F1. The fixed parent’s descriptive pooled/equal-batch weighted MSEs are 1.211032/1.058217 (F0) and 1.298912/1.100586 (F1). Component mass is less concentrated under equal-batch weighting: inverse-HHI effective counts rise from 118.94 to 175.69 and 125.41 to 177.11, respectively. The [batch-weight geometry PNG](eda/batch_weight_geometry_r1.png) and [SVG](eda/batch_weight_geometry_r1.svg) visualize these measures. This is an in-sample objective audit, not proof that batch normalization harms quality or pooled normalization would improve WP. Float64 reconstruction does not establish Torch FP32 reduction or optimizer-trajectory parity. The [result review](feasibility_scout/batch_mass_audit_r1/RESULT_REVIEW_R1.md) checks saved identities and aggregate arithmetic; it did not rescan payloads.

## Limits and validity

All local folds are exposed. Fixed-prediction bootstraps condition on endpoints and omit training, selection and public-transfer variation; F0/F1 change fold and seed together, and F0 follow-ups were chosen after F0 grades. Quota students use all supplied data and teacher ancestry remains unresolved. PR comments/manifests are not private payloads, nor are engineering metrics quality grades. The timeout is not scientific evidence. Public 0.66832 is dated context, not a baseline for adding local deltas.

The exact saved F0 head endpoints and their small fixed shrink are the only online corrections directly assessed here; neither result falsifies online adaptation or the B/C state family. The direction probe weakens the interpretation that calibration's large geometric overlap explains the score movement, while leaving the causes of the original fit-side residual degradation unresolved. The Kaggle checksum and membership failures and the final timeout are execution evidence, not negative model comparisons. R4's optimizer-step count remains unknown, so no engineering throughput or microfit count is claimed.

The October 23 [ten-reviewer Astra synthesis](../../unified/reviews/astra_ultra_audit_20261023_r1/SYNTHESIS.md) emphasized conditional endpoint value, fit-side parent-residual mismatch, optimizer exposure, exact-control design and delivery cost. The one-Astra [fundamentals reframe](astra_fundamentals_r1/FUNDAMENTALS_AND_MISSING_TESTS_R1.md) motivated the completed score-direction decomposition and identifies a prospective equal-width zero-state control. Its [independent claim review](astra_fundamentals_r1/INDEPENDENT_CLAIM_REVIEW_R1.md) found the central source argument accurate but limited: weighted-MSE/weighted-Pearson alignment is a population result under a fixed measure; the finite implementation has row weights, per-batch normalization, a final tail and an AdamW update per batch. That does not make the finite optimizer equivalent to official global WP. A future fidelity panel should match total optimizer steps and LR progress; source-qualified train/evaluation separation is not fresh validation.

## Decision and next step

1. Treat direction R2 as a bounded point/projection audit. Its independent review passes those calculations but not the bootstrap reconstruction; keep intervals explicitly actor-produced. Do not treat local slopes as observed gains, coefficient choices, or public forecasts.
2. Retain batch normalization as a measured optimization variable. The full-suffix audit documents a large shift in batch measure and lower component concentration under the source measure, but neither proves harm. Any later normalization comparison must hold membership, sequence order, update count, initialization and LR progress fixed, and needs a separate admitted quality grade.
3. If the root review still finds a decision-changing feature-access question, the Astra proposal is a prospective four-head Z-versus-A control (two cells, two fits per arm, zero new parent/GRU/teacher/student fits). It is exposed and unadmitted; verify new seeds, ancestry, cache identity and total execution cost first. Do not restart the original incomplete 12-fit panel by silently changing its recipe.
4. Preserve the two quota ZIPs and their pre-grade identities. Any future portal score is transfer evidence only; no coefficient tuning or expected public boost follows from this sample.
5. For PR26/27, defer a fidelity-wide follow-up until the complete matched panel and evaluation/export fit inside measured wall time. The standalone p10 pair would not answer the p25 reversal. Keep PR32 teacher progress separate from the local quota candidates and await official quality receipts.

### Root completion fields

- **Direction R2 independent result review:** complete with bounded pass — point derivatives, moments and projection ranks independently checked; bootstrap interval reconstruction incomplete. Numeric intervals remain actor-produced; see [review](feasibility_scout/direction_probe_r1/RESULT_REVIEW_R2.md).
- **Batch-mass audit:** complete with bounded pass — both suffix hashes and aggregate arithmetic reviewed; measured training-objective description only, no quality or harm claim. See [review](feasibility_scout/batch_mass_audit_r1/RESULT_REVIEW_R1.md).
- **Final campaign terminal/budget, public upload status and stop:** pending — root to fill from final campaign collection; the ZIPs remain unuploaded in this record.

## Source records

- [Campaign frontier](FRONTIER.md), [campaign README](README.md), and [PR refresh](PR_REFRESH_INTERPRETATION_R2.md).
- [F0 head grade](feasibility_scout/RESULTS_R2.md), [F0 shrink](feasibility_scout/correction_shrink_r1/RESULTS_R3.md), and [F1 shrink](feasibility_scout/f1_mean_shrink_r1/RESULTS_R2.md).
- [Direction-probe R2 narrative](feasibility_scout/direction_probe_r1/RESULTS_R2.md), [independent result review](feasibility_scout/direction_probe_r1/RESULT_REVIEW_R2.md), [result JSON](feasibility_scout/direction_probe_r1/DIRECTION_PROBE_RESULT_R2.json), and [execution receipt](feasibility_scout/direction_probe_r1/EXECUTION_R2.json).
- [Quota ZIP inventory](../../../submissions/quota_20261024/README.md) and [grade handoff](quota_submission_r1/RESULTS_AND_HANDOFF_R1.md).
- [EDA README](eda/README.md), [Kaggle feasibility](kaggle_age_r4/CACHE_RECONSTRUCTION_AND_PANEL_FEASIBILITY_R4.md), and [batch-mass result](feasibility_scout/batch_mass_audit_r1/RESULTS_R1.md) with its [independent review](feasibility_scout/batch_mass_audit_r1/RESULT_REVIEW_R1.md).
- [Direction geometry PNG](eda/direction_geometry_r2.png) and [SVG](eda/direction_geometry_r2.svg); [batch-weight geometry PNG](eda/batch_weight_geometry_r1.png) and [SVG](eda/batch_weight_geometry_r1.svg).
- [Comprehensive PR comparison](ALL_PR_RESEARCH_SUMMARY_R1.md), [PR #1–24 report](pr_audit_01_24/REPORT.md), and [PR #25–40 report](pr_audit_25_40/REPORT.md).
- [Prior ten-reviewer synthesis](../../unified/reviews/astra_ultra_audit_20261023_r1/SYNTHESIS.md), [new fundamentals report](astra_fundamentals_r1/FUNDAMENTALS_AND_MISSING_TESTS_R1.md), and [independent claim review](astra_fundamentals_r1/INDEPENDENT_CLAIM_REVIEW_R1.md).

## Later remote refresh

The [third all-PR refresh](PR_REFRESH_INTERPRETATION_R3.md) again finds 38 PRs, unchanged source heads and states. PR32's existing comment advances to 23,725/47,100 updates (50.37%): t1280 at 21/30 epochs and t1792 at 12/30, still zero of four evaluated endpoints. No new PR-text artifact destination appears. Separate [Actions/release queries](ARTIFACT_CHANNEL_REFRESH_R3.json) return zero artifacts and no releases. These are access and progress observations, not new quality evidence.
