# H230 preparation audit v1

Captured 2026-10-07 13:52 UTC from the source and artifacts present at that time. This is a recipe/data/cache audit and a conditional command draft only. No export, cache build, profile, fit, or grading was run.

## Findings first

The H230 source-family draft has three stale fields that should be reconciled before freezing any cache recipe: `missing_exports`, `new_baseline_recipe`, and `cache_budget_estimate_new_bytes`. Its source rows and intentional-difference entries describe the GRU384 seed pair, but `new_baseline_recipe` still says GRU128. The missing-export list names six old train exports that now exist, and it asks for both W384 valid exports even though the seed-4 sequence already schedules that valid export after its fit.

At the captured state, the six legacy teachers have both train and valid prediction `.npy` files and sidecars present (12 pairs total). I checked path presence only; I did not read or hash those large prediction arrays. W384 seed3 has its fixed epoch-3 checkpoint, provenance, valid prediction and sidecar; its train prediction is missing. W384 seed4's fit is still running, so its final checkpoint and train prediction are absent; its current sequence already schedules the valid prediction after the fit. Cache construction must wait for that sequence to finish and for the seed4 fixed-epoch checkpoint/provenance hashes to be sealed.

The pinned cache builder's estimate does not match the draft. With fold0's 10,535 train and 1,405 valid fit groups, 20,000 rows per sequence, and 14 float32 output channels (three 2-channel main variants plus eight family channels), its formula gives `11,940 × 20,000 × 4 × 14 = 13,372,800,000` bytes. The draft says `24,200,000,000` bytes. The builder enforces a 30-GiB output budget and a 25-GiB free-space floor; with its computed size it needs at least 40,216,345,600 bytes free at build time. Preserve the larger draft estimate as the conservative planning reservation until the controller reconciles the discrepancy. This excludes the two missing wide train exports, about 1,696,960,000 bytes each at the expected full train-target shape; refresh disk headroom after those exports and before the cache build.

The raw-family treatment is available only in the persistent source worktree at commit `dde32d0f5e39efe65c0497fec06449d38e18b369`. Its `scripts/train_rnn.py` accepts `--distill-family-target-mode raw`; the current main-tree trainer does not. The worktree's `.venv`, `artifacts`, `cache`, and `datasets` are symlinks to the main project. Run code from the worktree source and use unique `bd26_h230_*` run/cache names, because output writes land in the shared main artifact/cache directories. Every command below sets `PYTHONPATH` explicitly; without it, both command-line help checks failed to import `wnn`.

## Identity and ancestry checked

The fold manifest is `research/campaign_2026-09-29/artifacts/splits/fold0.json`, SHA256 `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`. It declares 10,535 fitting train groups, 1,405 fitting valid groups, 468 evaluation valid groups, and one corrupt train group excluded from fit. The fit-group provenance walk checked each currently available source plus its initializer/propensity ancestors: their split hashes match this manifest, each recorded fit set stays within the corresponding fit pool, and source checkpoint/provenance hashes match `SOURCE_FAMILIES_DRAFT_V1.json`. Some valid-only ancestors correctly record zero train groups and 1,405 valid fit groups. The W384 seed4 closure remains pending. The full pinned builder must repeat its stronger checks, including checkpoint-internal metadata, fixed epoch/policy, prediction bytes, row order, shape, dtype and finiteness; this lightweight walk is not a substitute.

| Family | Fixed source | Draft hashes | Prediction files currently present |
|---|---|---|---|
| strong | `c20261003_combo_s3_f0@ep3`, `c20261003_combo_s4_f0@ep3` | Both checkpoint and provenance hashes match draft | train+valid for both |
| full_finetuned | `c20260929_ft_since_s3_f0@ep0`, `c20260929_ft_since_s4_f0@ep0` | Both checkpoint and provenance hashes match draft | train+valid for both |
| raw_wmse | `c20260929_since_s3_f0@ep3`, `c20260929_since_s4_f0@ep3` | Both checkpoint and provenance hashes match draft | train+valid for both |
| samefold_baseline | `bd26_w384raw_s3_f0@ep3`, `bd26_w384raw_s4_f0@ep3` | Seed3 checkpoint `90b7357c…54eb08f` and provenance `3301c303…4423d6d` match; seed4 still pending | Seed3 valid only; seed3 train absent; seed4 pending |

The E521 seed3 valid export is bound by `FIT_EXPORT_RESULT_V1.json`: checkpoint SHA256 `90b7357c8d5cf01219764ae58373099710f11d041a01ef5400e683bf854eb08f`, prediction SHA256 `0dae84be777f8bf4ef80bb7c292e1466de5e326c6471cd2e28f22ef9e075ee2b`, sidecar SHA256 `a45e6c5224e5c09216e199120f75261676416a9d8c8b9b5d0129f4d437b7b6c3`. Its standalone fold0 score is exposed development evidence only; it does not estimate public transfer. The seed4 sequence receipt was `running` at `13:51:08Z`.

The cache builder reads each teacher's full train/valid prediction index, checks it against the corresponding target index, then writes only fold0 fitting groups into the student cache. It recursively checks teacher/initializer/propensity ancestry against the fold. Thus predictions may exist for the already inspected evaluation groups, but those rows must not enter the training cache; the builder's `fit_groups` and row-order checks are the gate. The cache is 8 distinct teacher runs, in family order `strong`, `full_finetuned`, `raw_wmse`, `samefold_baseline`, two runs per family. Its source coefficients are `.15,.15,.10,.10,.125,.125,.03125,.03125`, summing to `.8125`.

## Recipe audit

| Component | H230 treatment | Status against reference / consequence |
|---|---|---|
| Student warm start | Own seed-matched `c20261003_combo_s{3,4}_f0@ep3`; keep the E511 C1 GRU256×2 initialization and training masks | Matches the local control convention; declare checkpoint ancestry. This is a warm-started own student, not scratch initialization. |
| Student updates | GRU256×2, 2 epochs, fixed zero-based epoch 1, no fit-time evaluation, batch 16/chunk 2,000, AdamW + OneCycle, LR `1e-4`, WD `.01`, fp16, chunk-normalized wMSE | Matches the local C1/Maxim analogue where declared. Must profile the actual family-head/cache shape before admission; E521's GRU384 teacher profile does not cover this student. |
| Main target | Weighted raw source sum with the eight coefficients above; `distill-alpha=.7`; source variant retains total scale `.8125` | Preserve `--distill-cache-variant source`; `normalized` would silently multiply the teacher target by `1/.8125`. Trainer clips labels and blended main teacher target to ±2 before its main wMSE; row weights are absolute clipped labels times the declared mask. |
| Four family targets | Equal mean of the two teacher predictions inside each family; canonical order is strong, full_finetuned, raw_wmse, samefold_baseline; `distill-family-weight=.1` | Intentional extra eight-output head. Use `--distill-family-target-mode raw` to preserve teacher outputs; family wMSE reuses the main clipped-label row weights and averages over four families. This is separate from official metric clipping. |
| Baseline family | Replace planned compact GRU128×2 pair with own same-fold GRU384×2 seeds 3/4 at `.03125` each | Intentional H230 capacity/source interaction; E520 compact control remains separate. The changed teacher identity means this is an independent analogue, not exact checkpoint-level replication. |
| Other teachers | Own strong, full-finetuned and raw-wMSE families retain `.30`, `.20`, and `.25` total main coefficients | Checkpoint ancestry and fold membership are present in the draft and passed the lightweight provenance walk for available sources. Builder validation remains required. |
| Comparison | H230 equal seed pair versus frozen E511 C1 pair and warm parents | This tests a joint heterogeneous-source/family-head recipe. C1 is not a family-head ablation. To claim a family-head effect, add a matched H230 source-scale student pair with the same targets/init/updates and family weight zero, then compare that pair to H230. |
| Evaluation | Same exposed fold0 exclusions; official two-target `global_wp_clip2_v1` unchanged | Fold0 has already been inspected. Any H230 score is development evidence, not public forecast or a fresh holdout. An independent grader owns the frozen comparison. |

## Conditional commands

These are the exact command payloads prepared for later root-controlled admission. They were not run. Export and cache commands require current checkpoint/prediction identity gates and live disk checks. Student commands additionally require an admitted actual-shape profile and independent recipe review. Do not run this block as a fit authorization.

```bash
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007

# Missing E521 prediction files only, after the relevant checkpoints are terminal and sealed.
# The current seed-4 sequence already owns its fixed-epoch valid export.
PYTHONPATH="$ROOT/src" "$ROOT/.venv/bin/python" "$ROOT/scripts/predict_run.py" \
  bd26_w384raw_s3_f0 train --epoch 3
PYTHONPATH="$ROOT/src" "$ROOT/.venv/bin/python" "$ROOT/scripts/predict_run.py" \
  bd26_w384raw_s4_f0 train --epoch 3

# Build only after both wide checkpoints, required exports and their sidecars are sealed;
# the builder refuses an existing output directory and validates all input identities.
PYTHONPATH="$WT/src" "$WT/.venv/bin/python" "$WT/scripts/build_distill_cache.py" \
  --data-root "$ROOT" \
  --teacher-cache-root "$ROOT/cache" \
  --checkpoint-root "$ROOT/artifacts/runs" \
  --split-manifest "$ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json" \
  --output "$ROOT/cache/h230_w384_raw_v1" \
  --teacher 'strong=c20261003_combo_s3_f0@ep3*0.15' \
  --teacher 'strong=c20261003_combo_s4_f0@ep3*0.15' \
  --teacher 'full_finetuned=c20260929_ft_since_s3_f0@ep0*0.10' \
  --teacher 'full_finetuned=c20260929_ft_since_s4_f0@ep0*0.10' \
  --teacher 'raw_wmse=c20260929_since_s3_f0@ep3*0.125' \
  --teacher 'raw_wmse=c20260929_since_s4_f0@ep3*0.125' \
  --teacher 'samefold_baseline=bd26_w384raw_s3_f0@ep3*0.03125' \
  --teacher 'samefold_baseline=bd26_w384raw_s4_f0@ep3*0.03125' \
  --homogeneous-family strong \
  --expected-coefficient-sum 0.8125 \
  --chunk-sequences 2 \
  --output-budget-gb 30 \
  --min-free-gb 25

# Seed 3 candidate; send as a serial GPU-job payload through the controller's
# resource/admission tooling after a separate full-shape profile and recipe review.
PYTHONPATH="$WT/src" "$WT/.venv/bin/python" "$WT/scripts/train_rnn.py" \
  --name bd26_h230_hetero_s3_f0 --device cuda --arch gru --cell gru \
  --hidden 256 --layers 2 --dropout 0 --seed 3 \
  --split-manifest "$ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json" \
  --valid-fold 0 --train-source train+valid --train-seqs 10607 \
  --init c20261003_combo_s3_f0@ep3 \
  --epochs 2 --checkpoint-policy fixed --checkpoint-epoch 1 --no-eval-during-fit \
  --batch 16 --chunk 2000 --loss wmse --loss-norm chunk \
  --lr 0.0001 --wd 0.01 --optim adamw --sched onecycle --precision fp16 \
  --propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 --aux-struct 0.3 --aux-next 0.3 \
  --distill-cache-root "$ROOT/cache/h230_w384_raw_v1" \
  --distill-cache-variant source --distill-alpha 0.7 \
  --distill-family-weight 0.1 --distill-family-target-mode raw \
  --distill-boot-h 0 --fast --save-training-state

# Seed 4 is identical except for run name and seed-matched warm start.
PYTHONPATH="$WT/src" "$WT/.venv/bin/python" "$WT/scripts/train_rnn.py" \
  --name bd26_h230_hetero_s4_f0 --device cuda --arch gru --cell gru \
  --hidden 256 --layers 2 --dropout 0 --seed 4 \
  --split-manifest "$ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json" \
  --valid-fold 0 --train-source train+valid --train-seqs 10607 \
  --init c20261003_combo_s4_f0@ep3 \
  --epochs 2 --checkpoint-policy fixed --checkpoint-epoch 1 --no-eval-during-fit \
  --batch 16 --chunk 2000 --loss wmse --loss-norm chunk \
  --lr 0.0001 --wd 0.01 --optim adamw --sched onecycle --precision fp16 \
  --propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 --aux-struct 0.3 --aux-next 0.3 \
  --distill-cache-root "$ROOT/cache/h230_w384_raw_v1" \
  --distill-cache-variant source --distill-alpha 0.7 \
  --distill-family-weight 0.1 --distill-family-target-mode raw \
  --distill-boot-h 0 --fast --save-training-state
```

## Reviewed files and commands

- H230 plan: `PLAN.md`, SHA256 `1407e1c6fa96aecec3506ed8a338982897960668ea97758e96a8320d412a7d94`.
- Source-family draft: `SOURCE_FAMILIES_DRAFT_V1.json`, SHA256 `711c709583525e5e0e1283bcfe81e5664b9a01036937c81adf6226e9a1993085`.
- E521 recipe: SHA256 `3ac30113507f16a555e4e65daac39567a544c395a38233fa56b936deaf6458fd`.
- E521 fixed seed3 fit/export result: SHA256 `d8a252daf4894852292418b655409e1b26d0765115eb8035bb743590bcfd2b31`.
- E521 seed3 fold0 grade: SHA256 `f27c16a7a61224b52bf61eb0a53513c55439989aa20989f303773769dea6a851`.
- Persistent worktree: commit `dde32d0f5e39efe65c0497fec06449d38e18b369`; `scripts/train_rnn.py` SHA256 `fe283da2d70f370bfc82f0faf9f48a75ee4419cb1615024aa6911ec5f0452677`; `scripts/build_distill_cache.py` SHA256 `09f7110f21507195199d1f50fad6d1074fef1cd4d670a91093935be333f9dd26`; `src/wnn/distillation.py` SHA256 `352a0c5b782f9e68fc7153783d20b04825e0c764b94f589cd71993580b7857c7`; `src/wnn/distillation_cache.py` SHA256 `403fd74db91f08bb255db147fb50f7f6b6b9702280350aaa2479c01a7f972f19`.
- Bounded checks executed: `predict_run.py --help` with `PYTHONPATH=$ROOT/src`; `build_distill_cache.py --help` with `PYTHONPATH=$WT/src`; read-only checkpoint/provenance hash and fit-group ancestry walk. No full prediction-array reads, data materialization, cache writes, profile or fit.

Remaining controller gates: wait for E521 seed4 fit+valid export terminal receipt; export and hash only missing wide train predictions; refresh all source/checkpoint/provenance/sidecar hashes and verify full prediction row orders; reconcile cache output-size estimate and live disk headroom; obtain the declared cache/profile/resource admission; independently review the exact cache and student recipe; use a frozen independent grader before any H230 scoring.
