# Model and submission catalog

Snapshot: October 17, 2026. Names identify specific artifacts and ancestry; a research checkpoint, final all-data model, FP32 prediction cache and native ZIP are separate objects. See the [validation protocol table](../validation/VALIDATION_PROTOCOLS.md) before comparing scores.

## Current team packages

The [official public snapshot](../reviews/audit-2026-10-17/official_leaderboard_20261017.json) records kotiki at **.66832, rank 22, 97 submissions**. The server journal records tl3/tl4 at rounded .6683. An exact current-best portal-ID-to-ZIP-hash binding is not available.

| Package | Fixed members | Recorded public / runtime | Interpretation |
|---|---|---|---|
| `tl3.zip` | .34 `w18final_distilled_s5@ep1` + .33 `fttall_tlead_s5@ep3` + .33 `fttall_tlead_s6@ep3` | .6683 / 39:12 | Latest recorded creative/rs1013 best; 12,538,394 bytes |
| `tl4.zip` | Distilled s5/s6 and tiedlead s5/s6, each .25 | .6683 / 50:07 | 16,173,182 bytes; same rounded public result does not establish exact equality |
| `pair_ftt2.zip` | Distilled/FTT production combination in its frozen manifest | .6682 | Earlier team reference |
| `pair_lowlr.zip` | Frozen low-LR continuation package | .6680 / 45:07 | Its local improvement did not transfer to higher public score |
| `xteam_h4.zip` | Distilled s5, FTT s5, merged Z4 and H390 s3, each .25 | .6681 / 50:16 | Cross-team source mixture, original CPU policy A |
| Cross-team seed6 | Frozen seed6 release, otherwise source-qualified team combination | .6679 / 46:51 | Different weights; timing is not an engine-only comparison |
| QAT learned-W4 | xteam_h4 with its fourth member replaced; coefficients unchanged | .6680 / 54:26 | 17.46% smaller ZIP; no observed public/CPU/RAM advantage |

The [ZIP inventory](../reviews/audit-2026-10-17/submission_inventory.json) contains all 11 server ZIP manifests, sizes and SHA256 values, including diagnostic probes. `tl3` SHA256 is `d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`; `tl4` is `659ad2e6c984579708ef95ec6bfc08d622e4838b518aa26628dc58153d26669b`. These hashes identify files, not private portal uploads.

## Training lineage

| Family | Inputs and training | Evaluation caveat |
|---|---|---|
| `w18final_distilled_s5/s6` | 211 inputs = 112 raw + 99 causal event features; GRU256×2; warm continuation of fbc1k parents, two epochs AdamW, distillation alpha .7 | Final train+valid; own valid scores are fitted diagnostics |
| `fttall_tlead_s5/s6` | 263 inputs = 211 + 52 lead features; shared instrument encoder, GRU256×2; four-epoch scratch parents, Muon/AdamW, EMA, alpha .5 | Final train+valid; teacher treatment and fitting weights differ from distilled family |
| Main H390 / Z4 | Separate Main teacher/student and merged-model ancestry, contributing to xteam | H390 is a lineage identifier, not a hidden width; Main F0 is not creative strict425 |
| Clean T3 Main student | Warm H256×2 seed3/4, fixed six-epoch endpoints | Same teacher cohort, repeatedly exposed F0; seed mean is not an ensemble score |

The w18 teacher's stored coefficients sum to .8125. Tiedlead uses a different teacher and bootstrapped target treatment. Neither can be described faithfully by an unqualified “ensemble distillation” label. Historical source models/checkpoints remain in their original storage; the [storage map](../operations/STORAGE_MAP.md) identifies the locations.

## Keep serving policies separate from model names

Main October 12 A/B/C/D/E are model combinations. The later CPU A/B/C/D are engines for the **same xteam_h4 weights**. CPU D (fused GRU projections) recorded .6681 in 47:27 versus original A's 50:16; C took 57:27 and B timed out. These were different single runs, so the 169-second gap is an observation, not a stable capacity guarantee. [CPU result and exact IDs](../../campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1/PUBLIC_CPU_POLICY_RESULT_20261015_R1.md).

Main E later received .6675 twice; its earlier “unscored” label belongs to the October 12 snapshot. [Late Main/team results](../../campaign_2026-10-13/team_models_r1/PUBLIC_LB_UPDATE_20261013_R2.md), [QAT transfer](../../campaign_2026-10-13/qat_renewable_r1/final_self_distill_r2/PUBLIC_TRANSFER_REVIEW_20261014_R1.md), [70 preserved portal IDs](../reviews/audit-2026-10-17/public_receipts.json).

## Research endpoints are not serving candidates by default

The [CPU/student frontier](../../campaign_2026-10-14/cpu_student_frontier_r1/FRONTIER.md) contains five completed short architecture screens. H512×1 leads that specific one-seed comparison; it is not a measured replacement for the mature production ensemble. PLE/GDN research results belong to their original baseline and protocol. Fourteen `experiment/*` draft packets are implementation handoffs, not fourteen fitted models. [Current state](../../../specs/current-state.md) distinguishes preparation, engineering checks, fitting, evaluation and public results.
