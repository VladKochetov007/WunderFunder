# H-274/H-275/H-276 blend lab: fold0 excluded-468 panel

**Caveat.** The 468 evaluation groups are a repeatedly exposed development panel (see the grade JSONs' `status` fields). Nothing here is a public forecast, a fresh confirmation, or a promotion. Component bootstrap is conditional on fixed predictions and does not estimate training variance or public transfer.

Sources: `blend_lab_analysis.py` (this directory), `blend_lab_results_20261008.json` (all numbers below, keys named in brackets), `plots/*.png`. Metric: `global_wp_clip2_v1`, via `campaign_score.moments`/`pooled` (`research/campaign_2026-09-30/audit/campaign_score.py:85,119`) and `global_wp` (`src/wnn/gmetric.py:159`). Bootstrap: `wnn.evaluation.paired_component_bootstrap` (`src/wnn/evaluation.py:89`), 2000 draws, seed 271828.

## 1. Inventory (16 prediction sets kept, 0 dropped)

Membership is taken from each grade/result JSON (train 10535 + valid 1405 fit groups, source-qualified overlap 0, component overlap 0, 141 evaluation components). I did not re-run the graders' membership validators. Arrays are read only at the 468 evaluation rows, with `cache/valid_targets.npz` masks (step >= 99).

| Name | Trunk (distinct checkpoint assumed unless noted) | Reported WP source |
|---|---|---|
| R0_c20261003_GRU256x2 | c20261003 GRU256x2 (H218 reference lineage) | `H218_FOLD0_REFERENCE_BLEND_20261008_R1.json` reference |
| H272S4_parent | H272 S4 parent, ckpt sha 0203b109 | `H272_S4_CLEANFOLD_V2_GRADE_RESULT_20261008_R1.json` |
| H272S4_hidden_only / _multiplicative | same trunk as parent (heads) | same, weight 1.0 |
| H233_hidden_only, _additive, _conditional_residual, _multiplicative | shared trunk T_C1 (parent ckpt 7d34ee56 = bd26_c1v2_s3_f0 model_ep1) | `H233_FIXED_EP1_RESULT_20261008_R1.json` |
| H272d_hidden_only, _additive, _conditional_residual, _multiplicative | same shared trunk T_C1 (same parent ckpt 7d34ee56, verified in `recipes/H272_C1S3_DIRECT_LABEL_READOUT_V1.json` and `recipes/H233_C1S3_CONDITIONAL_READOUT_V2.json`) | `H272_FIXED_EP1_COMPARISON_RESULT_20261008_R2.json` |
| H232_GRU384_v7b | GRU384 | `H232_FIXED_REFERENCE_BLEND_RESULT_20261008_R2.json`, alpha 1.0 |
| H218_block_memory_s7 | block-memory (468-row export) | `H218_FOLD0...` candidate, alpha 1.0 |
| E540_v8_linear_GRU128, E540_v9_PLE_GRU128 | GRU128 ep0 subset | `E540_FIXED_PAIR_RESULT_20261008_R1.md` table |

Recomputed WP matches every reported value (16/16 within 1e-8; largest gap 3.9e-11, from the 10-decimal E540 table) [`inventory`].

Excluded or not included: `cache/pred_baseline_valid.npy`, `legacy_cache/*`, `local/E511_*` exports (no clean-membership receipt located in this pass). H264 attention and TCN member arrays were not found on disk, so H264 is cited only from its blend table (`H264_FIXED_EQUAL_MIXTURE_RESULT_20261008_R1.json`): all nonzero equal-mixture weights reduce WP (0.680941 to 0.670350 at weight 1). The H232 GRU384 and H218 candidate are included as members; neither was selected in any fold.

Official single-model WP on the full panel (exposed, descriptive only) [`full_panel_single_wp_exposed_descriptive`]: H272d_multiplicative 0.681732, H233_conditional_residual 0.681618, H233_hidden_only 0.681618, H272S4_hidden_only 0.681363, H272S4_parent 0.681029, R0 0.680941, H232 0.668596, H218 0.662404, E540 v8 0.636186, v9 0.625410.

## 2. Correlations (metric-weighted, selected rows N = 808,863) [`correlation`]

Plots: `plots/prediction_correlation_t0_t1.png`, `plots/residual_correlation_t0_t1.png`.

- Heads on one trunk are near-duplicates: the four H233 heads correlate at 0.996 to 1.000 with each other on predictions, and the four H272d heads at 0.994 to 1.000 (t0).
- Across trunks, H272S4_parent vs H272d_multiplicative: prediction 0.996 (t0), residual 0.997 (t0). H272S4_parent vs H233_hidden_only residual 0.991 (t0).
- The weakest trunks are the most decorrelated: H218 vs R0 residual 0.982 (t0); E540_v9_PLE vs H233 residual 0.948 (t0).

## 3. Stacking, 5 folds by dependence component [`fold_picks`, `methods`]

Five folds over 141 components (seed 271828; fold sizes 89/92/107/99/81 sequences). Single and equal-pair choices use training-fold official WP only. Stackers are fit on training-fold selected rows with metric weights |clip(y,-2,2)| and clipped inputs. Held-out predictions are pooled and scored with the official metric. Bootstrap compares each method with `best_single_train_selected`.

| Method | Pooled held-out WP | Delta vs best single | 95% CI (component bootstrap) | P(delta <= 0) |
|---|---:|---:|---|---:|
| best single (train-selected) | 0.681389 | 0 | - | - |
| best equal pair (train-selected) | 0.682231 | +0.000843 | [-0.000125, +0.001899] | 0.046 |
| NNLS (Cholesky-NNLS on weighted normal equations) | 0.680706 | -0.000683 | [-0.002192, +0.000828] | 0.814 |
| ridge (lambda = 1e-3 mean diag) | 0.680614 | -0.000774 | [-0.002798, +0.001085] | 0.787 |
| tiny MLP (16 tanh, early stop) | 0.678281 | -0.003108 | [-0.006328, -0.000338] | 0.987 |
| residual booster on best single | 0.671562 | -0.009827 | [-0.015575, -0.004918] | 1.000 |
| diagnostic: a*best single, no booster | 0.680969 | -0.000420 | [-0.000900, -0.000026] | 0.986 |

Per-fold picks. The best single was H272d_multiplicative in folds 0, 1, 2 and 4, and H272S4_hidden_only in fold 3. The best equal pair was H272S4_hidden_only + H272d_multiplicative in all five folds. Its pooled-WP lead over the single is the only positive result, and its CI includes zero.

Reading. Learned linear stackers do not beat the best single on this panel. The MLP and the residual booster lose, and the booster loses even though the no-booster diagnostic is only slightly below the single. This points to overfitting or cross-component shift in the residual, not to a scaling artifact. The booster result is a weak negative for this recipe, not evidence against residual models in general.

## 4. Servability [`inventory`, section 2]

- Heads on one trunk are nearly free. The H233 and H272d heads all sit on T_C1 (eight heads, one trunk). The H272S4 hidden-only head sits on T_S4 (a second trunk).
- The best combination within two trunks is the fixed equal blend 0.5 * H272S4_hidden_only + 0.5 * H272d_multiplicative, using two GRU256x2-class trunks, roughly 60 min of the 1-vCPU budget by the lead's estimate. Its fold-picked advantage over the train-selected single is +0.00084 (CI [-0.00012, +0.00190]).
- If one trunk is preferred: T_C1 with its multiplicative head (full-panel 0.681732, exposed). It is one trunk, but its selection used the panel.
- Trunk identity is now verified. T_S4 has parent ckpt 0203b109 and T_C1 has parent ckpt 7d34ee56, so the pair uses two distinct trunks.

## 5. Recommendation

1. Treat the 2-trunk equal blend (H272S4 hidden-only + H272d multiplicative) as a candidate for fresh confirmation, not as a result. Expected gain is small (+0.0008 on an exposed panel, CI crossing 0), and I cannot state a public-score expectation from this.
2. Do not spend on learned stackers or a residual booster for this recipe family on this panel. Each is negative or non-significant.
3. Before any claim, rerun the pair on fresh valid folds. Trunk identity is already settled (see section 4).

## 6. Per-architecture breakdown (H-280) [`blend_lab_architecture_20261008.json`, script `blend_lab_architecture.py`]

Best GRU256 is chosen per fold among all twelve GRU256-trunk members on training folds (picks: H272d_multiplicative in folds 0, 1, 2, 4; H272S4_hidden_only in fold 3). Each family's representative is chosen per fold on training folds. The blend weight w on top of the best GRU256 is chosen per fold from {0, 0.1, 0.25, 0.5} on training folds. Correlation is metric-weighted over selected rows, against H272d_multiplicative (the full-panel best GRU256, descriptive).

| Family | Trunk / serving cost class | Standalone full-panel WP (best member) | Corr with best GRU256 (t0 / t1) | Conditional gain vs best GRU256 (pooled held-out) | 95% CI | CV-chosen w per fold |
|---|---|---:|---|---:|---|---|
| GRU256 best (reference) | T_C1 (H272d_multiplicative) | 0.681732 | 1.000 / 1.000 | 0 | - | - |
| GRU256 second trunk (S4 parent/readouts, R0) | T_S4, full extra GRU256x2 trunk | 0.681363 (H272S4_hidden_only); R0 0.680941 | 0.996 / 0.996 (S4 hidden-only) | +0.000314 | [-0.000292, +0.000928] | 0.5, 0.5, 0.5, 0, 0.5 |
| readout heads on C1 trunk (non-best) | T_C1, shares trunk (no extra cost) | 0.681618 (H233 conditional residual) | 0.9997 / 0.9997 | -0.000154 | [-0.000974, +0.000658] | 0, 0.25, 0.5, 0.5, 0.1 |
| GRU384 (H232) | full extra GRU384 trunk | 0.668596 | 0.982 / 0.984 | 0 (CV chose w=0 in all folds) | - | 0 x5 |
| block-memory attention (H218) | full extra attention trunk (hidden 128, 2 layers, 4 heads) | 0.662404 | 0.975 / 0.977 | 0 (CV chose w=0 in all folds) | - | 0 x5 |
| TCN (H264) | not included: member arrays not on disk | - | - | - | - | - |
| E540 GRU128 linear | small extra GRU128 trunk | 0.636186 | 0.924 / 0.933 | -0.000560 | [-0.000957, -0.000159] | 0, 0.1, 0.1, 0, 0.1 |
| E540 GRU128 PLE | small extra GRU128 trunk | 0.625410 | 0.910 / 0.920 | 0 (CV chose w=0 in all folds) | - | 0 x5 |

Notes.
- The GRU256 second-trunk gain is positive but not significant (CI crosses zero). Its CV-chosen weight is 0.5 in four of five folds and 0 in fold 3.
- Readout heads on T_C1 add nothing once the best head is present. Their correlation with the best GRU256 is 0.9997 (heads on one trunk are near-duplicates).
- GRU384, block-memory attention and E540 PLE get zero CV weight in every fold, so their conditional gain is zero by construction. This is a statement about the CV rule and the exposed panel, not a rejection of those architectures.
- E540 linear is significantly negative when blended (CI below zero), so it is not a useful served component on this panel.
- No TCN (H264) row: its members were not located on disk, so the table cannot cover it.
- All conditional numbers come from the same exposed 468-group development panel. They are not a public forecast.

