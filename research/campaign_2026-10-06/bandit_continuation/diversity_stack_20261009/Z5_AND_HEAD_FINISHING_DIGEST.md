# Z5 and head-finishing digest (read-only mining, 2026-10-10)

Scope: read-only search of reports, provenance, packages and scripts. No training, no file changes except this digest. Paths are relative to the repository root.

## Key corrections to the premise

- **Z5 has no recorded public score.** The only source for "Z5 = 0.6664" is `reviews/WNN_Unified_Review_EN.md` (untracked, dated 2026-10-10). The release audit and package receipt state no public claim. The portal records list no 0.6664 row.
- **Public-best receipts are 0.6660 (seed-4 combo) and 0.6655/0.6654/0.6641 (seed-3 rows).** No row is recorded for `c20261004_combo_head_s3_i16` at 0.6660.
- **Z5 is a single merged-weight model, not a bag of checkpoints.** Its ZIP holds one member with coefficient 1.0. The member is a 0.5/0.5 tensor merge of a core student and its head fine-tune.
- **The head stage is `--freeze inp,gru` on the head only.** It is not the two-stage full fine-tune used by `ft_gclp.sh`.

## Table

| item | value | source path:line |
|---|---|---|
| Z5 ZIP | `submissions/c20261009_z5_combo_i16/submission.zip`, SHA-256 `60834d3190b76832f038abbdc505c6ac526d44b8f3a191c2e980e2f14f1ee1cc`, 5,172,638 bytes, precision i16 | `scratch/z4z5_independent_package_20261009/RELEASE_AUDIT_20261009.md:8`; `submissions/c20261009_z5_combo_i16/TENSOR_PROVENANCE_AUDIT.json` (`zip_sha256`, `precision`) |
| Z5 members (declared) | one member, `weights_0.npz`, coef 1.0, source `bd26_h230_faithful_student_seed9_muonparent_v1_core_identity_v2@ep1 \| z5_seed9_student_hft_alldata_v2@ep0` | `submissions/c20261009_z5_combo_i16/pkg/members.json` |
| Z5 core checkpoint | `artifacts/runs/bd26_h230_faithful_student_seed9_muonparent_v1_core_identity_v2/model_ep1.pt`, SHA-256 `8ba49427...052d2b`, Muon-parent student, epoch 1, fit groups train 10606 / valid 1873 | `submissions/c20261009_z5_combo_i16/TENSOR_PROVENANCE_AUDIT.json` (sources[0]) |
| Z5 head checkpoint | `artifacts/runs/z5_seed9_student_hft_alldata_v2/model_ep0.pt`, SHA-256 `12dda9c8...1ac03`, init = core checkpoint (hash matches), fit groups valid 1505 | `TENSOR_PROVENANCE_AUDIT.json` (sources[1]) |
| Z5 merge | `source_head_merge_coefficients [0.5, 0.5]`; trunk tensors bitwise equal to the source pair; `head_finetune_freezes_input_and_recurrent_trunk true` | `TENSOR_PROVENANCE_AUDIT.json` (merge fields) |
| Z5 equal-weight meaning | equal 0.5/0.5 average of head tensors in weight space (shared trunk). Matches the "equal-weight shared-trunk" wording used for the seed-3 and seed-4 combos | `research/campaign_2026-10-04/E412_PUBLIC_RESULT.md:5` |
| Z5 seed naming | name says `seed9`, but head provenance records `seed: 3` and `hidden: 256`, `optim: adamw`, `lr: 0.0005`, `epochs: 1`, `loss: gcorr`, `freeze: inp,gru`, `eval_source: dev`, `train_source: valid`, `full_valid: true`, `include_heldout_valid: false` | `artifacts/runs/z5_seed9_student_hft_alldata_v2/provenance.json` (args) |
| Z5 public score | **not found** in portal records. Only `reviews/WNN_Unified_Review_EN.md` states 0.6664 | `reviews/WNN_Unified_Review_EN.md:3, 43`; `RELEASE_AUDIT_20261009.md` (no public claim); `submissions/c20261009_z5_combo_i16/RELEASE_CHECK.json` (`public_score_claim: none`, `score_evaluated: false`) |
| Z5 status | exploratory, unpromoted, technical gates passed, no held-out claim (fits include validation groups) | `submissions/c20261009_z5_combo_i16/RELEASE_CHECK.json` (scientific_status) |
| Z5 runtime | matched local replay estimate 35.05 min (0.944 ratio vs incumbent). No portal runtime | `RELEASE_AUDIT_20261009.md:17` |
| Z4 ZIP | `submissions/c20261009_z4_combo_i16/`, SHA-256 `29e6b89b...28667dd`, 5,172,422 bytes. Head source `z4_seed99_student_hft_alldata_v2`, core `bd26_h230_faithful_student_seed99_muonparent_v1_core_identity_v2@ep1`; same HFT args as Z5 | `RELEASE_AUDIT_20261009.md:7`; `artifacts/runs/z4_seed99_student_hft_alldata_v2/provenance.json` |
| Z1c / Z2c | `z1c_h230s4_student_hft_alldata_v3@ep0` on core `bd26_h230_all_data_student_gru256_seed4_retry1_v1_core@ep1`; `z2c_h230s3_student_hft_alldata_v3@ep0` on `bd26_h230_all_data_student_gru256_seed3_v1_core@ep1` | `reports/Z1C_PACKAGE_TENSOR_AUDIT_20261008_R1.json:46`; `reports/Z2C_PACKAGE_TENSOR_AUDIT_20261008_R1.json:46` |
| Public seed-4 combo | `c20261004_combo_s4_i16`, submission `2e5j6bnt`, **0.6660**, 37m07s / 626 MB, runtime 2227 s, ZIP SHA-256 `c4dc258b...4f94a`. Member `c20261004_combo_s4_final@ep3 \| c20261004_combo_head_s4_final@ep0`, coef 1.0 | `research/campaign_2026-10-04/PORTAL_RESULTS_20261004_LATE.json:5-20`; `submissions/c20261004_combo_s4_i16/pkg/members.json`; `research/campaign_2026-10-05/NEW_SUBMISSION_AUDIT_2026-10-05.md:13` |
| Public seed-3 equal combo | `c20261004_combo_s3_i16`, submission `10tq248b`, **0.6655**, 37m47s / 625 MB, runtime 2267 s, ZIP SHA-256 `21f9b736...dafb` | `PORTAL_RESULTS_20261004_LATE.json:76-80`; `NEW_SUBMISSION_AUDIT_2026-10-05.md:14` |
| Public seed-3 head-only | `c20261004_combo_head_s3_i16`, submission `0np7c273`, **0.6654**, 34m01s / 624 MB, member `c20261004_combo_head_s3_final@ep0` | `PORTAL_RESULTS_20261004_LATE.json:51-57`; `NEW_SUBMISSION_AUDIT_2026-10-05.md:15` |
| Public seed-3 parent-only | `c20261004_combo_parent_s3_i16`, submission `0cawau9i`, **0.6641**, 33m57s / 624 MB, member `c20261004_combo_s3_final_r1@ep3` | `PORTAL_RESULTS_20261004_LATE.json:28-34`; `NEW_SUBMISSION_AUDIT_2026-10-05.md:16` |
| c20261004_combo_head_s3 public 0.6660 | **not found** (the seed-3 head scored 0.6654) | `PORTAL_RESULTS_20261004_LATE.json:57` |
| Head-finishing script (combo) | `scripts/continuation_recipes.py`, function `head_args` (argv builder), trial `combo_head` | `scripts/continuation_recipes.py:150-185`, `:257-273` |
| Head-finishing executor | `scripts/train_rnn.py` (called as `.venv/bin/python scripts/train_rnn.py`) | `scripts/continuation_recipes.py:153-156` |
| Head-finishing command (reconstructed from `head_args`) | `train_rnn.py --name <OUT> --init <PARENT>@ep3 --train-source valid --full-valid --mask-weight --seed 3 --epochs 1 --batch 16 --chunk 2000 --fast --inherit-init-preprocessing --loss gcorr --optim adamw --freeze inp,gru --lr 0.0005 --wd 0.01 --checkpoint-policy fixed --checkpoint-epoch 0 --save-training-state --eval-source holdout` (final); fold runs use `--split-manifest <fold manifest> --no-eval-during-fit` instead of `--eval-source holdout` | `scripts/continuation_recipes.py:150-185` |
| Head-finishing parent init | seed-4: `c20261004_combo_s4_final@ep3`; seed-3: `c20261004_combo_s3_final_r1@ep3` (args `init`) | `artifacts/runs/c20261004_combo_head_s4_final/provenance.json` (args.init); `artifacts/runs/c20261004_combo_head_s3_final/provenance.json` (args.init) |
| Head-finishing epoch / checkpoint | 1 epoch, `checkpoint_policy fixed`, `checkpoint_epoch 0`, output `model_ep0.pt` | same provenance files (args) |
| Head-finishing optimizer | AdamW, lr 5e-4, wd 0.01, onecycle schedule, batch 16, chunk 2000, clip 1.0 | same provenance files (args: `optim`, `lr`, `wd`, `sched`, `batch`, `chunk`, `clip`) |
| Head-finishing trained parameters | everything except `inp.*` and `gru.*` (trunk frozen); i.e. head only | `src/wnn/nn.py:217` (`--freeze inp,gru`); `scripts/continuation_recipes.py:177-178` |
| Head-finishing loss | `gcorr` (pooled weighted correlation), `mask_weight` on, `gcorr_weight 1.0` | provenance args (`loss`, `mask_weight`, `gcorr_weight`) |
| Head-finishing data | `train_source valid`, `full_valid true`, `include_heldout_valid false`. Historical holdout excluded from fitting for finals (`eval_source holdout` only for evaluation) | provenance args; `reviews`/`E412_PUBLIC_RESULT.md:5` |
| Head-finishing seed | 3 for both seed-3 and seed-4 heads (provenance `seed: 3`) | `artifacts/runs/c20261004_combo_head_s4_final/provenance.json` (seed) |
| Head-finishing evaluation | seed-4 head: holdout368 `t0 0.6921`, `t1 0.7112`, `weighted_pearson 0.70163` (exposed sequences; not held out) | `artifacts/runs/c20261004_combo_head_s4_final/history.json` |
| Two-stage LP-FT alternative (not the combo head) | `scripts/ft_gclp.sh` runs head stage (frozen trunk, lr 5e-4) and then full FT (lr 2e-4); `FT_STAGES=head` stops after head stage | `scripts/ft_gclp.sh:1-8, 27-34` |
| Ensemble-aware head FT (not used for combo head per provenance) | `scripts/joint_head_ft.py` (`--mode joint\|solo`) | `scripts/joint_head_ft.py:1-8` |
| Local gain, seed-4 (exposed 368 sequences) | equal combo 0.701694557; parent alone 0.700363274; head alone 0.701577995. Combo minus parent = **+0.001331** (computed from those table values; no interval printed in this row) | `research/campaign_2026-10-04/E450_SEED4_PAIRED_REPORT.md:22-25` |
| Local gain, seed-4 paired interval | seed-4 minus seed-3 combo: +0.001872 [-0.000975, +0.006931]. Combo-minus-parent paired interval for seed 4: **not found** | `E450_SEED4_PAIRED_REPORT.md:28-29` |
| Local gain, seed-3 (E410 complete predictor) | combo minus incumbent +0.002414, paired interval [+0.001566, +0.003449] (E410 is the pre-public local screen, not a seed-4 head-only paired check) | `research/campaign_2026-10-04/E410_complete_predictor.md:3-5` |
| Public gain, seed-3 head finishing | parent-only 0.6641 -> head-only 0.6654 (+0.0013); equal combo 0.6655 (+0.0014 vs parent-only). Single seed, single portal reading each | `NEW_SUBMISSION_AUDIT_2026-10-05.md:14-16`; `research/campaign_2026-10-04/E412_PUBLIC_RESULT.md:13` |
| Public gain, seed-4 head finishing | parent-only public **not found**; seed-4 combo 0.6660 only | `E450_SEED4_PAIRED_REPORT.md:1-3` |
| Seed-4 public vs seed-3 public | +0.0005 public (0.6660 - 0.6655) | `research/campaign_2026-10-04/E450_SEED4_PAIRED_REPORT.md:1-3`; `NEW_SUBMISSION_AUDIT_2026-10-05.md:13-14` |
| Combo ZIP runtime / memory | seed-4 combo 2227 s / 626 MB; seed-3 combo 2267 s / 625 MB; seed-3 head 34m01s / 624 MB; seed-3 parent 33m57s / 624 MB | `PORTAL_RESULTS_20261004_LATE.json`; `NEW_SUBMISSION_AUDIT_2026-10-05.md:13-16` |
| Packaging rule for combos | single member, coef 1.0, source `parent@ep3\|head@ep0`; the 0.5/0.5 is inside the weight merge, not a multi-member bag | `submissions/c20261004_combo_s4_i16/pkg/members.json`; `submissions/c20261009_z5_combo_i16/pkg/members.json` |
| H-320 note on Muon-parent public levels | ledger text says Muon parents give "0.6662-0.6664" (non-Z5 H230 students, not a receipt for Z5) | `research/hypotheses.jsonl:356` |

## Notes on limits

- The E457 parent/head public split (`research/campaign_2026-10-04/E457_HEAD_TRANSFER_PREREG.md`) is a planned test. Its verified outcomes are the seed-3 rows above.
- Head-finishing gains are measured on exposed validation sequences and single seeds. No held-out claim exists for Z5 or the combo heads.
- `research/experiments.jsonl` and `research/hypotheses.jsonl` do not contain `c20261009_z5_combo_i16`, `combo_head`-named ledger rows with "Z5" wording, or a Z5 public receipt (searched with `rg -i z5`).
