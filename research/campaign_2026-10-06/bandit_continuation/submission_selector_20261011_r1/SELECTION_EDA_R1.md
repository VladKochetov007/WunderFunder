# Submission selector and prediction EDA (R1)

## Decision-relevant result

The strongest fixed blend on the repeatedly exposed fold-1 panel is `H256 + mean(H384 seed3/4/5)`, grouped 50/50, at **0.6726401**. That is only **+0.0000985** over the three-H384 mean (0.6725416); the existing paired component bootstrap for this gain has a 95% interval of `[-0.000659, +0.000891]`. The extra H256 member therefore has weak measured marginal value over this teacher trio, not enough evidence to justify a larger package by score alone.

For a compact two-model recipe, equal `H256 + H384 seed5` scores **0.6725743** on fold 1 and **0.6110276** on the deterministic 138-group component complement. Its complement gain over H256 alone is **+0.0010039**, with an 800-resample component-bootstrap interval of `[-0.003324, +0.005882]`. Treat this as an exposed, noisy blend diagnostic, not a winner claim.

The selector found no reason to tune blend weights. On the 138-group complement, targetwise inverse-output-SD weighting scored 0.6111638, residual-covariance weighting shrunk 80% toward equal weights scored 0.6111952, and equal grouped weights scored 0.6111906. The requested calibration-only, targetwise overall-scale fit used multipliers `[1.05281, 1.07072]` and scored 0.6111906429 versus 0.6111906285 for the unscaled equal blend: paired bootstrap mean difference `1.44e-8`, 95% interval `[-1.04e-7, +1.67e-7]`. Official weighted Pearson is scale-invariant away from clipping, and clipping did not change this result materially.

For final full-data packaging, the two distinct endpoints worth prioritizing are the existing **old Z1c/Z4 equal pair** and the **H390 seed3/seed4 equal student pair**. This is a provisional packaging recommendation across two different lineages, not an assertion that either has an exact same-recipe clean-fold grade. The H390 pair has a strong recipe-family relationship to the F1 H256/strong-T3 evidence, but its all-data parents and `since_weight` differ from the evaluated F1 student. Preserve the old pair as the second family rather than substituting a higher-scoring exposed fold-1 blend without exact all-data validation. The measured callback profiles below are for fold-1 checkpoints; they establish capacity only, not Q8 score fidelity of the final all-data replacements.

## Evaluation frame and exposure

The analysis scored only five source-qualified fold-1 prediction arrays, each aligned to the same 457 evaluation row groups and official `global_wp_clip2_v1` mask. The split SHA is `31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae`; evaluation groups form 149 connected components. The scorer clips labels and predictions to ±2, uses `abs(clipped_label)` weights, and excludes rows before 99. Predictions were memory-mapped and scored in chunks. No all-data prediction was scored.

All of fold 1 has prior model and blend selection exposure. A deterministic component split held out 52 components (138 groups) from blend fitting, using the other 97 components (319 groups) for calibration. This complement is still from an exposed development fold; it is an internal fixed-prediction blend diagnostic, not a pristine holdout. The bounded pool contains **12** mixtures: nine predeclared equal-weight candidates, two calibration-only targetwise group-weight candidates, and one calibration-only overall-scale candidate. The complete score and bootstrap records are in `generated_r6/F1_SELECTOR_METRICS_R1.json`.

## Fold-1 results

| Candidate | Full exposed fold-1 WP | Component-complement WP | Comparison |
|---|---:|---:|---|
| H256 + H384 s3/s4/s5, grouped 50/50 | **0.6726401** (`t0 .6674206`, `t1 .6778596`) | **0.6111906** | Best fixed score; only +0.0000985 vs H384 trio |
| H256 + H384 s5, 50/50 | 0.6725743 (`.6667829`, `.6783657`) | 0.6110276 | Best compact two-family blend among fixed candidates |
| H384 s3/s4/s5, equal | 0.6725416 (`.6669449`, `.6781384`) | 0.6107323 | Same-recipe, three-seed ensemble; +0.0010447 vs best single seed on full fold |
| H384 s4/s5, 50/50 | 0.6723361 (`.6660953`, `.6785769`) | 0.6104683 | Two-seed same-recipe blend; complement gain +0.0011304 vs s5, CI `[-0.003488,+0.005914]` |
| H384 s3/s5, 50/50 | 0.6722754 (`.6667703`, `.6777804`) | 0.6103974 | Two-seed same-recipe blend |
| H256 + H391 H384 student, 50/50 | 0.6717665 (`.6663477`, `.6771854`) | **0.6115447** | Complement point score is highest; gain vs H256 +0.0015211, CI `[-0.001149,+0.004491]` |
| H256 + H384 s4, 50/50 | 0.6718943 (`.6665299`, `.6772586`) | 0.6104471 | Fixed two-member blend |
| H256 + H384 s3, 50/50 | 0.6717683 (`.6671936`, `.6763430`) | 0.6102646 | Fixed two-member blend |
| H384 s3/s4, 50/50 | 0.6716132 (`.6665093`, `.6767171`) | 0.6098072 | Fixed same-recipe seed blend |

Single scores on the full exposed fold are H256 `0.6710315`, H384 s3 `0.6702074`, H384 s4 `0.6703370`, H384 s5 `0.6714970`, and the H391 H384 student `0.6702361`. The H384 seed trio mean of singles is `0.6706805`; its equal ensemble reaches `0.6725416` (**+0.0018612 vs mean**, **+0.0010447 vs best seed5**). The average of the three tested two-seed pair scores is `0.6720749`, so pairwise seed averaging accounts for `+0.0013944` over the single mean, and the third seed adds `+0.0004667` beyond the pair average. These three pairs reuse the same seed models and are not independent replications. Same-recipe seed diversity has the clearest measured incremental value. The H256 addition to that trio contributes only `+0.0000985` on the full panel and is not supported as a required fourth member.

Among the two fitted weight schemes, targetwise group weights remained near equal: inverse-SD `[0.5474,0.4526]` / `[0.5475,0.4525]`; covariance weights after 80% equal shrinkage `[0.4909,0.5091]` / `[0.4901,0.5099]` for targets 0/1. Neither adds a meaningful complement gain. Keep equal coefficients for operational simplicity.

## Prediction scale and error dependence

Weighted prediction correlations are high across all five models (roughly 0.991–0.998). Unscaled residual correlations appear lower across H256/H384 (about 0.980–0.982) than among H384 seeds (about 0.992–0.993), but H256 output variation is smaller. After fitting a zero-intercept scale per model and target on calibration components, then measuring residual correlations on the disjoint complement, cross-family values are about **0.993–0.995**, overlapping same-family seed values. Therefore the raw residual-correlation gap is scale-confounded and does not establish structurally distinct errors. The complement correlations are descriptive and remain exposed.

The H256/H384-trio weighted output SDs on calibration groups were approximately `[0.7865,0.9513]` for target 0 and `[0.8153,0.9866]` for target 1. That scale difference motivates testing equal model-group weights, but it does not justify a learned weight advantage; the held-back complement result was effectively unchanged.

Selected pairwise weighted correlations (`t0/t1`) show the scale effect directly:

| Pair | Prediction correlation | Raw residual correlation | After calibration-scale alignment, on complement |
|---|---:|---:|---:|
| H256 / H384 s5 | `.99197 / .99256` | `.98027 / .98020` | `.99426 / .99431` |
| H384 s3 / s5 | `.99127 / .99180` | `.99241 / .99235` | `.99336 / .99353` |
| H256 / H391 H384 student | `.99297 / .99355` | `.99592 / .99594` | `.99504 / .99516` |

The complete weighted covariance and correlation matrices for all five predictors and both targets, plus the targetwise single-model scores, are preserved in the machine-readable metrics artifact.

## Exact endpoint and ancestry map

| Evaluated fold-1 model | Fitting / teacher ancestry | Eligible full-data counterpart or endpoint | Match qualification |
|---|---|---|---|
| `vt_f1_muonall_h384_e3@ep2` seed3 | H384 2-layer GRU; Muon-all; e3; same F1 membership; propensity `c20260929_prop_f1` power2/eps .03; batch32/chunk1000; one-cycle; `since_weight=32`; own independent fit | `vt_all_h388_muonall_h384_e3_s3@ep2`, CP `f61c1e72…074a6b5` | Same teacher recipe/seed and checkpoint epoch; full-data valid outputs are in-sample, not held-out grades |
| `vt_f1_muonall_h384_e3_s4@ep2` seed4 | Same recipe as seed3, distinct fitting seed | `vt_all_h388_muonall_h384_e3_s4@ep2`, CP `592c51fb…f9f0fcc` | Same recipe/seed and epoch; all-data counterpart only |
| `vt_f1_muonall_h384_e3_s5@ep2` seed5 | Same recipe as seed3, distinct fitting seed | `vt_all_h388_muonall_h384_e3_s5@ep2`, CP `7feefb8a…dc61943` | Same recipe/seed and epoch; all-data counterpart only |
| `vt_f1_dsq_dm_t3_a9_e6@ep5` seed3 | H256 2-layer GRU, AdamW/OneCycle, alpha .9/e6, `since_weight=0`; parent `c20261003_combo_s3_f1@ep3`; target metadata `a2f82d8d…0856779` selects ordered mean of the three strong H384 teacher outputs then multiplies by .8125; no fit/eval overlap | H390 `vt_all_h390_ds_homogeneous_s3_r3@ep5`, CP `04675f82…d361e0`; paired s4 `cefee5b5…3018419` | Same main target family and strong-T3 teacher source family, but H390 parent is seed-matched all-data `vt_all_fin_mu256p0e4_s3/s4@ep3`, and H390 uses `since_weight=32`; these are material student-recipe/initializer differences. No exact F1 counterpart score claim. |
| `vt_f1_h391_dsq_h384_t3_a9_e6_s3_r2@ep5` | H384 student, one seed; H384 parent seed4 ep2; width/parent interaction | No exact all-data counterpart identified | Keep as a diagnostic family/width interaction; no full-data grade mapping |
| Old public Z1c/Z4 package | Z1c: own H230 seed4 all-data H256 student core ep1 → HFT ep0. Z4: own faithful H256 seed99 Muon-parent core ep1 → HFT ep0. | Exact historical package SHA `ff498873…1705327`, `members.json` SHA `ee2f33b4…a72ec9f`, equal 0.5/0.5 | No exact clean-fold counterpart found. The cleanF0 H230 heterogeneous 31-head teacher→23-head crop→HFT child is only a family analogue, not Z1c/Z4 certification. |

The H390 cache metadata `da986019…419126e` binds the target as the float32 ordered mean of three own cleanfold H384 strong-teacher epoch-2 predictions, multiplied by 0.8125, unclipped. Both final H390 students use their seed-matched all-data H256 parents at epoch 3. The H390 packet explicitly labels all 12,479 train+valid groups as fitting data and has zero evaluation groups; its valid prediction arrays cannot be used as validation evidence.

## Cross-fold and submission/runtime limits

The existing fold-0/fold-1 comparison receipt contains 29 other paired recipes, not this exact H384 Muon-all seed trio. The H382 folds 0/2/3 receipt is nearby propensity/epoch evidence, with one seed per fold and some confounded control contrasts; it does not supply an exact cross-fold grade for this recipe. No recipe winner claim should be based on merging those nearby observations.

Earlier local headroom record `HEADROOM_BLEND_SCORES_R1.txt` gives 0.67326 for H256 plus two prop0/e4 H384 students on fold 1, but the two prediction arrays are not present in the local bank, so no correlation, conditional gain, bootstrap, or reconstruction was made. Its member and teacher/parent lineage differs from the H384 raw-teacher ensembles above. An earlier public transfer had H256 + prop0 s4 at 0.6650 versus H256 alone 0.6652; this is a cautionary distinct result, not an arithmetic adjustment to any public score.

The callback capacity receipt profiles the **fold-1** H256 + H384 s3 Q8/VNNI pair at 12,913,680-byte ZIP / 113.1419 µs per row and H384 s3+s4 at 17,271,543 bytes / 158.9771 µs per row. Those numbers are callback capacity measurements on a specific CPU, not quantized score measurements or final all-data package measurements. Final all-data H390 and old-Z package quantized quality and runtime should be checked from their exact bytes before nomination. No expected-public score is inferred.

## Reproduction and pinned artifacts

The 12-mixture result was produced by `analyze_f1_pool.py` with 800 component-bootstrap replicates, reading only the five pinned fold-1 prediction arrays and `cache/valid_targets.npz`. The successful invocation was:

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=700000000 WNN_RUNTIME_MAX=120s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
/home/vlad/development/wunder/wnn_submission_selector_20261011_r1/research/campaign_2026-10-06/bandit_continuation/submission_selector_20261011_r1/analyze_f1_pool.py \
--main-root /home/vlad/development/wunder/wnn_connectome_starterpack \
--output /home/vlad/development/wunder/wnn_submission_selector_20261011_r1/research/campaign_2026-10-06/bandit_continuation/submission_selector_20261011_r1/generated_r6 \
--bootstrap 800
```

The final metrics SHA-256 is `e030f8914aafa7bb172824577552d059a6e101ea7fc5c1ab97b1c2b9d33ff1f5`. `SOURCE_ENDPOINT_MAP_R1.json` carries the candidate identities and exact checkpoint/prediction digests. `EVIDENCE_MANIFEST_R1.json` pins the metric, split, targets, callback profiles, old pair and relevant source receipts.
