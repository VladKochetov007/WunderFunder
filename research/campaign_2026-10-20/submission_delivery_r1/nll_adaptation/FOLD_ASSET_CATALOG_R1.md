# Fold and production asset catalog

Metadata-only inventory for a possible later matched WMSE/NLL continuation. No tensor checkpoints or arrays were loaded. The local NLL screen currently uses F0; these assets are candidates for follow-up only if its scored endpoints justify the compute. F1/F2 are previously exposed development folds, so they add robustness evidence rather than a fresh holdout.

## F1: closest cached continuation

- Parent checkpoint: `artifacts/runs/oldz4_h230_parent_f1_s3_r4/model_ep3.pt`, SHA-256 `be2273cc96027295c19133504354a262cd1a2066fa5ae3df7d6be64ed0a1ad8e`; provenance is `artifacts/runs/oldz4_h230_parent_f1_s3_r4/provenance.json`. This is an H256×2, seed-3, fold-1 WMSE student with the family head disabled. Its historical precision is FP16/AMP; a new strict-FP32 continuation would be an adaptation from an AMP ancestor, not a clean-FP32 rerun or replication.
- Split: fold-1 SHA-256 `31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae`; 10,533 train and 1,416 valid fitting groups, with 457 evaluation groups excluded from fitting.
- Reusable targets: `cache/h230_four_family_fold1_source_20261009_r1/metadata.json`, SHA-256 `8bd3854d867c47eacbeed56528ea9712f7598cc7eb21a46f5c85e7a0f2fba703`. Main-source and family train/valid arrays are present. The valid arrays cover all 1,416 fit-valid groups. Metadata review found zero overlap between the 457 evaluation groups and all eight teacher sources plus their recorded recursive initializer chains. This is only the recorded lineage check; F1 itself remains exposed.
- Reusable propensity: `artifacts/runs/c20260929_prop_f1/provenance.json`, model SHA-256 `1fbf412df8823c703fdf79263830f341269a68e769bc4a7258d93cdf8e40dc5c`; fold identity matches and 457 evaluation groups are excluded. Both `cache/propcache_c20260929_prop_f1_train.npy` and `_valid.npy` are present with sidecars.
- The no-family parent has 23 output rows; adding the two variance rows for the NLL head gives a 23→25 continuation. Do not use `artifacts/runs/oldz4_h230_student_f1_s3_r4/model_ep1.pt` for that recipe: it adds eight family rows (31 total), so the matching NLL extension would be 31→33, not 31→25.

## F2: available parent, incomplete distillation cache

- Parent checkpoint: `artifacts/runs/c20261001_d256_s3_f2/model_ep3.pt`, SHA-256 `8b94fdfc5c06cb33dc1247387c96d4e63bdf45ada2b2ecbc2bbd80735856ca58`; H256×2, seed 3, fold 2, WMSE, historical FP16. Split SHA-256 is `a9973e5798d3109362571c299b2d08eca71d01e92008da39b0d0ddde54c93aa9`, with 10,535 train and 1,430 valid fitting groups and 443 evaluation groups excluded.
- Propensity provenance: `artifacts/runs/c20260929_prop_f2/provenance.json`, model SHA-256 `f0b9eb7ba76b66c1710579082f434e66e9f8d3b4ec230d052a0cd466375307d6`; train/valid propensity arrays and sidecars are present.
- The supervised no-family parent is a 23→25 NLL initializer candidate. A full F2 train+valid teacher-target cache matching the F1 cache was not found; available prediction metadata is partial (including one F3 teacher). A matched distillation continuation therefore needs target-cache work first. Without it, an F2 pair would be supervised-only and would answer a different question from the F0 cached-target pair.

## All-data production lineage

- Existing H230 all-data student: `artifacts/runs/bd26_h230_all_data_student_gru256_seed3_history_swap_v2/model_ep1.pt`, SHA-256 `a9d5a6c528326f9879ae8c3301a6b83328c082f3693ad65cde7881facfe6fe33`; parent CP3 SHA-256 `478773bd3d0754d261f8f3c9d46a135d56c3596c269d2a856b760caef05b93c7`. It fits all 10,606 train and 1,873 valid groups, has `include_heldout_valid=true`, and no excluded evaluation groups. Its historical precision is AMP and it has the eight family rows (31 total).
- Cache metadata: `cache/h230_all_data_history_swap_v2/metadata.json`, SHA-256 `1f76c0b97e411f4302438d81e72f004812982015da1fc852967c04672a5fa10f`; main train/valid target arrays are present. This is a usable production lineage candidate, but its all-data membership makes it invalid as a clean-fold comparator.
- The official package `submissions/team_tl3_original_20261018.zip` (SHA-256 `d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`) has an exposed F0 diagnostic of 0.701818 (`research/campaign_2026-10-18/online_features_r1/tl3_diagnostics_r1/result.json`). The cached F2 prediction sidecar `research/campaign_2026-10-18/online_features_r1/tl3_fixed_blend_f2_r1/tl3_f2_sidecar.json` covers 443 groups with the fold-2 split SHA above. Available package/sidecar metadata does not establish exact teacher and initializer membership, so neither is a clean fold comparator on this evidence.

All inventory statements above are based on provenance, recipe, cache and prediction metadata. No new ancestry was inferred where those records were incomplete; no quality claim is made for the unfit F1/F2 continuation candidates.
