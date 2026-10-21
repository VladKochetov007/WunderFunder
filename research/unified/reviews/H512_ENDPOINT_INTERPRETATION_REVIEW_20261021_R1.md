---
kind: review
status: complete
created_utc: 2026-10-21
updated_utc: 2026-10-21
hypothesis_ids: [H-H512-ENDPOINT-GAP-20261021-R1]
experiment_ids: [E-H512-ENDPOINT-DIAGNOSIS-20261021-R1, E-PLATEAU-I1-20261021-R1]
evidence_paths:
  - research/campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_GRADE_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/TL3_STUDENT_EXPOSURE_RECOVERY_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/endpoint_diagnosis/RESULT_R3.json
  - research/campaign_2026-10-21/next_cycle_r1/i1/interim_f0_abc/RESULT_R1.json
  - artifacts/runs/r01_h512_f0_s1003_local_cp5_r1/provenance.json
  - research/campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1/FIT_TERMINAL_ACCEPTANCE_R1.json
  - research/campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1/H512_FIT_AND_PRODUCTION_PLAN_R1.json
---

# Interpreting the H512 endpoint gap and first i1 cell

## Question or claim

The current-host replay makes the H512 replacement deficit reproducible, but the exposed F0 comparison still does not isolate architecture. The completed i1 cell also separates the value of ordinary i1 inputs from the incremental history channels; its standalone scores and fixed mixture give different answers.

## Mechanism and method

This is a source and JSON review of the current host grade, TL3 lineage recovery, streamed prediction geometry and interim F0 A/B/C grade. I did not open prediction or target arrays, run code, or inspect fit tensors. Bootstrap intervals below condition on the same fixed predictions and source-qualified F0 components; they are not fit-seed or fold uncertainty.

## Evidence and interpretation

Correction to the first draft: it attributed the separate i1 screen’s 10% fitting subset to the CP5 H512 endpoint. The H512 endpoint is the full fold-specific fit described below.

The highest-ranked explanation for the H512 endpoint gap is a comparison mismatch in exposure and training objective. The recovered identity receipt links the TL3 seed-5 student payload to its preserved final-training script: final all-data heterogeneous distillation, validation included, alpha 0.7, and in-sample final metrics; its initializer is `fbc1k_s5@ep3`. Exact final row-group receipts and recursive teacher/initializer hashes remain unavailable. By contrast, the CP5 H512 member is a cold WMSE model trained on the full F0-fold fitting membership: its provenance records 10,535 train-source and 1,405 valid-source groups, excluding all 468 F0 evaluation groups, and the terminal receipt records six epochs with fixed epoch 5 selected. It is not the 10%/three-epoch subset used in the separate i1 screen. Thus the fixed-prediction replacement loss is real, but it is not evidence that this architecture or history representation loses under matched training and exposure. The two retained TL3 members also have unknown ancestry.

The second explanation is prediction geometry under the frozen 34/33/33 serving mixture. In the endpoint diagnosis, standalone H512 scores 0.67377 and SWA 0.673996, while the CP5 and SWA replacements score 0.694455 and 0.694449 against 0.701818 for the current-host reference. For CP5, the target-weighted clipped covariance ratio is 0.9982 on t0 and 1.0061 on t1, while the variance factor is 0.9894 and 0.9854; their products reproduce correlation ratios of 0.9876 and 0.9914. This describes a fixed-output dispersion cost that outweighs nearly unchanged or slightly higher covariance on this panel. All four fixed position bins lose for both replacements; the earliest bin loses more, but it contains only 17,198 of 808,863 scored rows. These facts explain the observed endpoint geometry, not a corrective scaling rule or an architectural defect.

A separate, modest i1 result is that raw/event i1 inputs matter more clearly than this history addition on the first cell. A−B is −0.015901 WP [−0.019737, −0.011972], where A removes raw/event inputs and history. C−B, the history contrast, is −0.003537 [−0.007283, −0.000066]. Yet the frozen 90%B/10%C mixture is +0.000360 [−0.0000119, +0.0007089]. This is consistent with a weak complementary contribution despite worse standalone C, but the interval does not establish a positive gain. It is not evidence that history is useless or that the readout is the bottleneck: a 91.35% head gradient-square share and 97.27% clip rate are optimization descriptions, not causal tests.

## Limits and validity

The host replay now scores the cached legacy endpoint at 0.7018181213 versus 0.7018175337 for the current-host reference, a −0.000000588 difference with a 95% component interval spanning zero. CP5 minus that same-host reference remains −0.007362916 [−0.009617579, −0.005574297]. This supersedes the earlier interim review's statement that a full current-host comparison was still pending, and falsifies a current-host runtime/replay mismatch as the material source of the measured replacement gap. It does not identify a dependency-specific cause for the earlier Docker/host prediction discrepancy, nor remove TL3 exposure or ancestry limitations.

F0 was previously inspected. The native grade covers 468 sequences and 141 components; its conditional intervals do not measure training variation, clean-CV generalization or public transfer. The i1 grade is one F0/seed1001 cell from the predeclared six-fit screen, also on exposed development data. The near-zero lower endpoint of the 90/10 interval must not be presented as established improvement. The i1 labels A/B/C are unrelated to the endpoint-diagnosis symbols B/R/S.

## Decision and next step

For i1, the cheapest decision-changing evidence is the already-authorized F2/seed1002 A/B/C cell and the frozen full-panel grade. It adds no experiment beyond the six-fit panel and can show whether raw-input and fixed-mixture patterns recur across the second fold/seed. If those prediction artifacts have not landed, the current source and JSON cannot resolve that uncertainty; wait for them rather than add fits or tune coefficients.

For H512, the existing same-host replay and cached-prediction geometry exhaust the useful no-fit checks here. Keep this result labeled as an exposed, exposure-asymmetric fixed-endpoint diagnostic. It cannot support an architecture ranking, a public-score prediction, or a claim that rescaling or another blend would improve the submission.

## Source records

- [Interim H512 endpoint-gap review](H512_ENDPOINT_GAP_INTERIM_REVIEW_20261021_R1.md) (see the canonical link in the unified review index).
- [Same-runtime full native grade](../../campaign_2026-10-21/next_cycle_r1/NATIVE_HOST_GRADE_R1.json) and [TL3 exposure recovery](../../campaign_2026-10-21/next_cycle_r1/TL3_STUDENT_EXPOSURE_RECOVERY_R1.json).
- [Endpoint prediction geometry](../../campaign_2026-10-21/next_cycle_r1/endpoint_diagnosis/RESULT_R3.json).
- [F0 A/B/C interim grade](../../campaign_2026-10-21/next_cycle_r1/i1/interim_f0_abc/RESULT_R1.json) and [frozen i1 panel](../../campaign_2026-10-21/next_cycle_r1/i1/README.md).
- [CP5 H512 provenance](../../../artifacts/runs/r01_h512_f0_s1003_local_cp5_r1/provenance.json), [fit terminal](../../campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1/FIT_TERMINAL_ACCEPTANCE_R1.json), and [frozen full-fold recipe](../../campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1/H512_FIT_AND_PRODUCTION_PLAN_R1.json).
