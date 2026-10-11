# Old Z1c/Z4 clean-fold reconstruction

Prepared 2026-10-11 in isolated writer worktree `agent/old-pipeline-reconstruct-20261011-r1`. This is a source and recipe handoff only. No fit, prediction, export, upload, queue mutation, budget/grader edit, or main-worktree source edit was performed.

## Result

The frozen public-best ZIP is a 50/50 prediction blend of two completed all-data endpoints. It establishes the historical recipe and package lineage; its checkpoint weights fit rows from both local folds, so it supplies no clean-fold score. The member manifest and package hashes are in [`stage_inventory.json`](stage_inventory.json).

The exact clean-fold Z4-like counterpart is still missing on both folds. The fold-specific four-family target caches and all their teacher checkpoints are already present and valid, so neither teachers nor caches need rebuilding. Both exact fold-matched Z1c combo @ep3 initializers are also already present; only the continuation and descendants are missing. Fold 0 and fold 1 already have clean H230 family-student analogues, but both start directly from `c20261003_combo_s3_f{fold}@ep3`, not from a Z4-method Muon parent. Fold 0 also has a core crop and HFT for that analogue. Fold 1 has another head-finetune, but it starts from a different student and is not eligible. The source-qualified lineage is the deciding difference, not the HFT command itself.

## What the public pair actually did

| Member | Own-weight lineage | Final stage |
|---|---|---|
| Z1c | `c20261003_combo_s3_f0@ep3` → `bd26_h230_all_data_parent_b16_ep3_retry3_v1@ep3` (4-epoch AdamW continuation) → seed-4 H230 family student `bd26_h230_all_data_student_gru256_seed4_retry1_v1@ep1` → 23-output crop | `z1c_h230s4_student_hft_alldata_v3@ep0` |
| Z4 | Fresh seed-99 H230 hidden-Muon parent `bd26_h230_strong_parent_seed99_muon_local_v1@ep3` → 2-epoch AdamW family student `bd26_h230_faithful_student_seed99_muonparent_v1@ep1` → 23-output crop `bd26_h230_faithful_student_seed99_muonparent_v1_core_identity_v2@ep1` | `z4_seed99_student_hft_alldata_v2@ep0` |

The ZIP combines the two HFT predictions with equal coefficients. It does not merge weights. The Z4 parent itself uses the H230 main teacher cache (`alpha=.5`, bootstrap horizon 8), EMA-teacher evaluation, and the since auxiliary (`weight=32`, `floor=.25`) in addition to hidden-scope Muon. The student uses main-cache `alpha=.7`, four raw family targets at family-head weight `.1`, and a new seed-specific eight-row family head. These fields are frozen in [`recipe.json`](recipe.json), alongside exact command vectors and source hashes.

The H230 manifests record an independently written project implementation and no copied teammate code or weights. The reference commit in the manifest identifies the inspected method, not an imported implementation or an exact historical-weight identity. A clean rerun therefore tests recipe fidelity with new fold-specific teachers and parent/student seeds; it does not claim bitwise recovery of seed 99.

## Completed and missing stages

| Fold | Clean split | Reusable target ancestry | Completed H230 family analogue | Missing for the Z4-like chain |
|---|---|---|---|---|
| 0 | 468 evaluation groups; 1,405 eligible inner-valid groups; split SHA `09263ad4…` | 8/8 source checkpoints and all cache arrays present; metadata SHA `398d7753…`; source split fingerprints match; zero source-fit/evaluation overlap | `bd26_h230_hetero_s3_f0@ep1` → audited core → `f0_h230_hetero_s3_hft_v1@ep0` | compatible H230 Muon parent, student initialized from it, its crop, and its HFT |
| 1 | 457 evaluation groups; 1,416 eligible inner-valid groups; split SHA `31b9dc04…` | 8/8 source checkpoints and all cache arrays present; metadata SHA `8bd3854d…`; source split fingerprints match; zero source-fit/evaluation overlap | `bd26_h230_hetero_s3_f1@ep1`; `r3_headft_h230_f1@ep0` is another student’s head-finetune | compatible H230 Muon parent, student initialized from it, its crop, and its HFT |

The caches retain the same four family names and source-coefficient sum as the old pipeline: strong `.30`, full-finetuned `.20`, raw-WMSE `.25`, same-fold baseline `.0625`, total `.8125`. Each fold has two own sources per family. The propensity models are also fold-specific (`c20260929_prop_f0`, SHA `b243b9d4…`; `c20260929_prop_f1`, SHA `1fbf412d…`) and their fitted valid groups have zero overlap with their evaluation groups. `stage_inventory.json` carries every source run, epoch, checkpoint SHA, membership count, split SHA, cache file-presence flag, and overlap result.

Existing fold normalization hashes are available and are frozen as expectations: fold 0 mean/std `bb85c72b…` / `b7306008…`; fold 1 `c2735f3c…` / `d3d87f51…`. Parent and student recompute from fold-eligible fitting rows. HFT inherits the core student’s normalization and must match it exactly.

A provenance search found no reusable exact Z4 parent. It checked fold-specific GRU-256×2 Muon runs against the material H230 parent fields, including hidden-only Muon geometry, LR multiplier 3, momentum `.95`, OneCycle/warmup `.05`, B16/C1000, main-cache alpha `.5`/boot 8, EMA/since settings, and fold-specific membership. None matched on either fold. The closest models differ in optimizer scope, batch geometry, or teacher/EMA recipe. They remain useful experiments but cannot be relabeled as the Z4 parent.

## First runnable comparison cohort

The first minimal cross-fold screen is fixed at fold 0 / seed 3 and fold 1 / seed 3, with seed 3 carried through parent, student, and HFT in both cells. This is n=1 per fold and is diagnostic only; it cannot estimate seed variance or rank seed-averaged endpoints. Per cell, the dependency graph is:

```text
fold-specific H230 cache + fold propensity
                  │
                  ▼
4-epoch H230 hidden-Muon parent @ep3
                  │
                  ▼
2-epoch H230 family student @ep1
                  │
                  ├── audited 31→23 output crop @ep1
                  │                 │
                  │                 ▼
                  │       1-epoch masked GCorr HFT @ep0
                  │
                  └──────── compare HFT endpoint on that fold
```

This is six model fits plus two crop transformations, with zero teacher fits or cache rebuilds. The matching T3 control must use the same excluded components and exact fold/seed cell. The specific strong-T3 checkpoint IDs were not in this reconstruction assignment; root must bind those identities before admitting the comparison.

After the seed-3 diagnostic, the smallest seed-variance confirmation adds seed 4 on both folds: 6 model fits plus 2 crops. The resulting fully crossed fold `{0,1}` × seed `{3,4}` design totals four parents, four students, four HFTs, and four crops (12 fits + 4 transforms). Use matched controls for exactly the same fold/seed cells. The first screen is not sufficient for seed-averaged ranking; this confirmation is a concrete successor, not an ETA or launch authorization.

The H382 R3 kernel metadata attaches private dataset `vladkochetov07/wnn-c20261004-fold0-assets`; its local manifest lists 23 files totaling 7,243,689,021 bytes and contains no family/distill cache paths. The separate R3 bundle dataset status is ready at version 1. The older H230 parent cache asset is explicitly main-source-only and omits normalized, homogeneous, and family arrays. These are metadata-only findings: this packet neither proposes nor authorizes an upload, and root must bind a cache payload or mounted input before assuming family targets are remote.

For eventual same-method final transfer, use a fresh full-permitted-data parent → family student → audited crop → GCorr HFT chain, with a separately bound own-teacher family cache and propensity fitted on declared permitted training membership. Keep the fold caches out of final training, inherit HFT normalization from the final core checkpoint, and freeze a seed or fixed seed ensemble only after root reviews the cross-fold/two-seed evidence. This transfer contract is recorded in `recipe.json`; its full-data source hashes remain a root-owned packet dependency.

The separate Z1c continuation is now frozen as a second recipe in `recipe.json`; the stage inventory confirms both exact combo initializers are present, while the extra parent and its student/crop/HFT descendants are absent on each fold. It reuses the exact fold-matched combo @ep3 initializers (F0 SHA `d4a1e50b…`, F1 SHA `b0dcc761…`), then runs the four-epoch AdamW continuation, family student, crop and masked GCorr HFT. The clean recipe uses seed 3 across both folds and stages to match the Z4 diagnostic cell; the public Z1c student used seed 4, so this compares method rather than historical seed-specific weights. This is six fits plus two crops, with no teacher fit/cache rebuild. Root should admit it as a separate parent interaction after the Z4 first queue; the existing hetero student/HFT skip this continuation parent.

## Observed timing and exact remaining blockers

Historical all-data Z4 telemetry records 6,019.2 s for the Muon parent, 2,324.0 s for its student, and 85.7 s for HFT. Nearby clean H230 analogue telemetry records 1,983.5/2,119.3 s for fold-0/fold-1 students and 68.5 s for the fold-0 HFT. Their recipes and loads differ, so these measurements are not scaled into a fold-fit ETA.

The outstanding dependencies are specific: root must bind the matched strong-T3 IDs, check the final packet’s actual checkpoint producer and memberships, and recheck current GPU/provider/quota and queue state. The root’s supplied prepaid headroom/active recovery snapshot is not an admission receipt. The prepared vectors do not start jobs and must be wrapped in the existing root-owned resource/packet admission process.
