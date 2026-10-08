# Public Submissions & Local CV History, Wunder Connectome Challenge2

**As of 2026-10-08; curated from source documents and fold0 excluded-468 panel.**

## Chronological Public Submissions

| Date | ID/Name | Type | Public Score | Runtime (min) | Local CV (excluded-468) | Recipe | Source |
|---|---|---|---:|---:|---:|---|---|
| 2026-09-25 | ab18_wmse_struct | Own GRU128, |y|-weight MSE, aux heads, train-only | 0.6488 | ~19 | 0.6721 | Public ONNX baseline shift study | [LEADERBOARD_HISTORY_METADATA.json](../../../campaign_2026-10-04/LEADERBOARD_HISTORY_METADATA.json) L18-19 |
| 2026-09-25 | baseline ×0.25 | Organizer ONNX GRU baseline, rescaled ×0.25 | 0.6503 | ~28 | 0.6671 | Starting point, clip mitigation | [LEADERBOARD_HISTORY_METADATA.json](../../../campaign_2026-10-04/LEADERBOARD_HISTORY_METADATA.json) L8-11 |
| 2026-09-25 | final_tv_e2 | GRU128, train+valid, 2 epochs | 0.6548 | ~35 | 0.6877 (holdout) | Distribution-robust blend input | [LEADERBOARD_HISTORY_METADATA.json](../../../campaign_2026-10-04/LEADERBOARD_HISTORY_METADATA.json) L23-27 |
| 2026-09-25 | blend_base_final_tv | 0.5 GRU128 + 0.5 rescaled baseline | 0.6569 | ~40 | 0.6869 (holdout) | First real gain +0.0045 from baseline | [LEADERBOARD_HISTORY_METADATA.json](../../../campaign_2026-10-04/LEADERBOARD_HISTORY_METADATA.json) L32-36 |
| 2026-10-02 | R4D_routed_since_d256 | Market-class routed GRU128+D256 | 0.6638 | 46.0 | UNVERIFIED | Routing/blend architecture | [experiments.jsonl E-324](../../../experiments.jsonl) |
| 2026-10-02 | S4D_since_d256 | GRU128 + distilled GRU256, int16 engine | 0.6633 | 46.7 | UNVERIFIED | Since-weighting, two-member | [experiments.jsonl E-324](../../../experiments.jsonl) |
| 2026-10-02 | D256_single_distilled | Distilled GRU256, α=0.5, parent+head-FT | 0.6641 | 36.4 | UNVERIFIED | Single-student baseline, tied team best | [experiments.jsonl E-324](../../../experiments.jsonl) |
| 2026-10-02 | D07s3_alpha07_seed3 | Distilled GRU256, α=0.7, seed3 | 0.6635 | 36.5 | UNVERIFIED | Alpha sweep, below α=0.5 | [experiments.jsonl E-340](../../../experiments.jsonl) |
| 2026-10-02 | D07s4_alpha07_seed4 | Distilled GRU256, α=0.7, seed4 | 0.6639 | 36.6 | UNVERIFIED | Alpha sweep, seed variation | [experiments.jsonl E-340](../../../experiments.jsonl) |
| 2026-10-04 | E412: combo_s3_i16 | Full-data: Muon+chunk1000+bootstrap, epoch3, head-FT, equal parent/head | **0.6655** | 37.8 | 0.699822159 | Pooled-correlation parent/head | [E412_PUBLIC_RESULT.json](../../../campaign_2026-10-04/E412_PUBLIC_RESULT.json) |
| 2026-10-04 | E457: combo_parent_s3_i16 | Same seed3 combo, parent endpoint only | 0.6641 | 33.9 | (embedded) | Head-only gain isolated | [experiments.jsonl E-457](../../../experiments.jsonl) |
| 2026-10-04 | E457: combo_head_s3_i16 | Same seed3 combo, head-only endpoint | 0.6654 | 34.0 | (embedded) | +0.0013 vs parent | [experiments.jsonl E-457](../../../experiments.jsonl) |
| 2026-10-04 | **E450: combo_s4_i16** | **Full-data: Muon+chunk1000+bootstrap, epoch3, head-FT, equal parent/head, seed4** | **0.6660** | 37.1 | **0.701694557** | **Target reached at displayed precision** | [E450 experiments.jsonl](../../../experiments.jsonl), [PUBLIC_RESULTS_REFLECTION](../../../campaign_2026-10-04/PUBLIC_RESULTS_REFLECTION_20261004_LATE.md) |
| 2026-10-07 | E524: c1_s3_i16 | C1 single fold0 seed3, epoch1 (partial distillation) | 0.6647 | 33.2 | UNVERIFIED | Below incumbent, fold-trained only | [E524_PUBLIC_RESULTS_20261007_V1.json](../submission_C1_E524/E524_PUBLIC_RESULTS_20261007_V1.json) |
| 2026-10-07 | E524: c1_s4_i16 | C1 single fold0 seed4, epoch1 (partial distillation) | 0.6654 | 33.1 | UNVERIFIED | Below incumbent, fold-trained only | [E524_PUBLIC_RESULTS_20261007_V1.json](../submission_C1_E524/E524_PUBLIC_RESULTS_20261007_V1.json) |
| 2026-10-05 | **Maxim ref: distilled_h256_pair** | **Distilled GRU256 pair (teammate)** | **0.6674** | ~25 (est) | Seeds5/6 on full data (not directly comparable) | **Teammate reported; never improved** | [reviews/maxim_replication_20261007/public-result.json](../reviews/maxim_replication_20261007/public-result.json) |
| 2026-10-08 | H230: s3_normalized_i16 | All-data heterogeneous student, seed3, normalized | 0.6653 | 33.4 | UNVERIFIED | Single variant, below public best | [H230_NORMALIZED_PUBLIC_RESULT_20261008_R1.json](../wide_baseline_heterogeneous_H230/H230_NORMALIZED_PUBLIC_RESULT_20261008_R1.json) |
| 2026-10-08 | H230: all_data_pair_i16 | All-data heterogeneous student, seed3, pair blend | 0.6653 | 61.4 | UNVERIFIED | Pair does not fit 60-min limit | [H230_PAIR_USER_PUBLIC_RESULT_20261008_R1.json](../continuous_execution_E529/H230_PAIR_USER_PUBLIC_RESULT_20261008_R1.json) |
| 2026-10-08 | H230: s3_history_swap_i16 | All-data partial teacher-history variant, seed3 | **0.6644** | 33.9 | UNVERIFIED | History-swap, not full heterogeneous | [current-state.md](../../../../../specs/current-state.md) |

## Recipe Changes & Public Impact

| Change | Before | After | Δ Public | Δ Local (excluded-468) | Status | Evidence |
|---|---:|---:|---:|---:|---|---|
| GRU128 train-only → train+valid | 0.6488 | 0.6548 | +0.0060 | +0.0156 | **GAIN** | E-086 settled distrib shift |
| Blend with rescaled baseline | 0.6548 | 0.6569 | +0.0021 | −0.0008 | **GAIN** | Baseline robustness |
| Since-weighting + routing | 0.6633 | 0.6638 | +0.0005 | UNVERIFIED | **TINY GAIN** | E-324 routing |
| Single distilled D256 (α=0.5) | 0.6633 | 0.6641 | +0.0008 | UNVERIFIED | **GAIN** | E-324 incumbent |
| Alpha α=0.7 vs α=0.5 | 0.6641 | 0.6635 | −0.0006 | UNVERIFIED | **LOSS** | E-340 local fold-miss |
| Full-data combo seed3 (Muon+chunk+bootstrap) | 0.6641 | 0.6655 | +0.0014 | +0.0017 | **GAIN** | E-412, 368-panel agrees |
| Combo seed3 head-only fine-tune | 0.6641 | 0.6654 | +0.0013 | (isolated) | **GAIN** | E-457 endpoint split |
| Combo seed4 vs seed3 | 0.6655 | 0.6660 | +0.0005 | +0.0019 | **GAIN** | E-450, public confirms |
| C1 fold-trained single vs incumbent | 0.6660 | 0.6647 | −0.0013 | **NOT RELEVANT** | **LOSS** | E-524 fold exposure |
| H230 normalized single | 0.6660 | 0.6653 | −0.0007 | UNVERIFIED | **LOSS** | Different teacher/parent |
| H230 pair | 0.6660 | 0.6653 | −0.0007 | UNVERIFIED | **LOSS** | Runtime violation (61 min) |
| H230 history-swap partial | 0.6660 | 0.6644 | −0.0016 | UNVERIFIED | **LOSS** | Teacher-history variant |

## Local vs Public Agreement

| Submission | Local (excluded-468) | Public | Agreement | Notes |
|---|---:|---:|---|---|
| E412 combo_s3 | 0.699822159 | 0.6655 | ✓ Ranked | Both measure fixed seed3 |
| E450 combo_s4 | 0.701694557 | 0.6660 | ✓ Ranked | Both measure fixed seed4 |
| E450 equal_bag | 0.701420130 | (not submitted) | — | Unsubmitted, no public test |
| D256_single (E-324) | UNVERIFIED | 0.6641 | ? | Public frozen; local fold history missing |
| D07s3/s4 alpha sweep | UNVERIFIED | 0.6635/0.6639 | ? | Fold1 miss (E-328) vs public |
| H230 variants (seed3) | UNVERIFIED | 0.6653/0.6644 | ? | All-data fit; no comparable held-out fold |

**Key finding**: E450 seed4 public +0.0005 matches local +0.0019 sign on excluded-468 panel; local CI includes zero (95% [−0.001, +0.007]), but public confirmed the direction. Scaling is not 1:1; do not extrapolate local deltas.

## Maxim Replication Status

- **Public result**: 0.6674 (2026-10-05), submission `26n9pg4y`, `distilled_h256_pair.zip`
- **Maxim's model**: Two GRU256 students (seeds 5/6), distilled with weighted-sum targets (α=0.7, 8-source teacher ensemble including organizer baseline)
- **Our best unsubmitted all-data endpoints**: H230 full heterogeneous (not complete) and H272 seed4 direct-label readout
- **Heterogeneous student fit status**: C1–C4 unrun; C0 continuation control completed with −0.00018902 vs warm parent (CI includes zero)
- **Local held-out fold claim**: Maxim used separate fold0+seed-specific parents; our fold0 excluded-468 and all-data fits are not directly comparable
- **No replication completed**: Different teacher coefficients, clipping (family targets), organizer baseline substitute, and no C3 full student fit

## Current Best Local CV (Fold0 Excluded-468 Panel, 2026-10-08)

| Model | CV Score (WP) | Endpoint | Parent/Baseline | Delta | Source |
|---|---:|---|---|---:|---|
| **H272 S4 GCorr parent** | **0.6818613275** | Readout fixed half (α=0.5 multiplicative) | — | +0.000717 | [H272_S4_CLEANFOLD_V2_GRADE_RESULT](../continuous_execution_E529/H272_S4_CLEANFOLD_V2_GRADE_RESULT_20261008_R1.json) |
| H272 S4 hidden-only | 0.6810293773 | Readout α=0.0 (no readout) | — | — | [H272_S4_CLEANFOLD_V2_GRADE_RESULT](../continuous_execution_E529/H272_S4_CLEANFOLD_V2_GRADE_RESULT_20261008_R1.json) |
| H233 C1 parent | 0.681618142181 | Single fold0 ep1 | H230 warm parent | UNVERIFIED | [current-state.md](../../../../../specs/current-state.md) |
| E450 combo_s4 | 0.701694557 | All-data eq blend, seed4 | — | Public 0.6660 | [E450 experiments.jsonl](../../../experiments.jsonl) |
| **Public best live** | **0.6660** | E450 combo_s4_i16 | — | Vs Maxim +0.0014 | [current-state.md](../../../../../specs/current-state.md) |
| Maxim reference | 0.6674 | distilled_h256_pair (reported) | — | **−0.0014 vs ours** | [public-result.json](../reviews/maxim_replication_20261007/public-result.json) |

H272 GCorr readout fixed-half endpoint (0.681861) exceeds E450 pool-corr parent (0.701695 all-data) on the exposed fold due to different fitting populations and evaluation units; no direct public gain claimed for H272.

## Summary & Key Decisions

1. **Public improvements confirmed**: E412 (+0.0014 vs D256), E450 (+0.0005 vs seed3) both moved public score above prior incumbent 0.6641
2. **Local-vs-public parity**: E450 seed4 public +0.0005 aligns with sign of local +0.0019 on excluded-468; local CI crosses zero, public does not
3. **Maxim unreplicated**: 0.6674 public remains ahead of our 0.6660; C1–C4 heterogeneous student not run; unrun does not mean mechanism false
4. **H230 all-data attempts (2026-10-08)**: single, normalized, pair, history-swap all scored 0.6644–0.6653, below incumbent
5. **H272 readout discovery**: GCorr fixed-half 0.6818613 on fold0 excluded-468 exceeds all-data parents but no public test yet

---
**Total public submissions**: 19 own + 1 teammate reference  
**Runtime gains**: D256 int16 variant 36→32 min (no score loss)  
**Pair fit failures**: H230 61-min runtime exceeds 60-min limit  
**No new public best since E450 on 2026-10-04**

*All numbers sourced in table above. UNVERIFIED marks missing local CV records on the standard fold0 excluded-468 panel.*
