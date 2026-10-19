# Clean Main T3 ancestry: candidate recipe

**Status: proposed for root to freeze; not an admitted command or result.** This note binds a candidate clean parent to the five-arm design and separates historical evidence from the choices needed for a new source-bound run. It reports no teacher, parent, or student fit.

## Proposed ancestry for one fold/seed cell

Use the source-qualified outer split and its fitting groups only, with evaluation disabled throughout fitting. For the current first F0 cell, use student seed **1001**, as declared by the frozen campaign design. Do not substitute historical seed 3 for the new student cell. Teachers use seeds 3, 4, and 5 and are shared within the same fold; a parent is created per fold/seed cell and shared only by its five arms.

1. **Propensity prerequisite.** Use the declared clean F0 fit `fg19_propensity_clean_f0_seed0_e4`: hidden 64, seed 0, four epochs, fixed epoch 3, strict FP32. It is a teacher-loss weighting model, not a student input or an automatic student-loss weight. The prerequisite file still says `declared_not_run`; the live launch/capture receipt owns current status.
2. **Three teachers.** Fit H384×2 linear-encoder GRUs for seeds 3, 4, and 5, each for three epochs and retain fixed epoch 2. The identified archived candidate establishes batch 32, chunk 1000, Muon, one-cycle, WMSE and a propensity dependency, but used FP16 and older fold/cache identities. The clean candidates must use the current outer-fold fit membership and the newly bound propensity checkpoint.
3. **One common target and parent.** Construct the homogeneous arithmetic mean of the three fixed-epoch teacher predictions, scale it by 0.8125, and clip it to ±2 when consumed. Fit one **fresh** H256×2 parent for four epochs from random initialization on the Main T3 target `0.1*clip(y,-2,2) + 0.9*clip(0.8125*mean(T3),-2,2)`, retaining fixed epoch 3. Use the cell seed (1001 for the initial F0 cell) and the bound student target, WMSE, row weights, auxiliaries and optimizer recipe. Keep this warm parent on the established Main T3 base topology: 211 inputs and 23 outputs, with no experimental representation or mean/logvariance flags. Create it once for that fold/seed cell and share its exact checkpoint across all five arms. It is an initializer, not one of the five matched arms. Do not use the historical production-derived parent.
4. **Five six-epoch children.** Initialize every child from the same parent epoch-3 checkpoint, run six epochs, and retain fixed epoch 5. The reviewed warm-load path expands 211 inputs to the common 467-wide layout and 23 outputs to the 25-output mean/logvariance layout. It copies the original input/output rows and auxiliary offsets, and zero-initializes the appended input columns and two logvariance rows so all arms begin with the parent mean function.

| Arm | Representation | Main objective |
|---|---|---|
| control | `base` | WMSE |
| EMA | `ema` | WMSE |
| sparse | `sparse` | WMSE |
| mean/variance | `base` | NLL |
| all | `ema_sparse` | NLL |

For `base` and EMA-only, sparse dictionary/history blocks not used by that arm are exact zeros. The fold-local dictionary and its normalization are fit-only and shared by the five arms in a cell. The 211→467 input and 23→25 output extensions happen on each child warm load, not on the parent. Preserve the original 211 input columns and first 23 output rows, including all auxiliary offsets; zero the added input-projection columns 211:467 and appended logvariance rows. This preserves the parent mean function while a child activates its declared representation. Refit the dictionary per fold as prescribed by the design; do not reuse a panel archive.

The implementation branch's fixed dictionary preparation is a once-per-fold step, not a sixth student arm:

```text
python scripts/fit_input_representation.py \
  --split-manifest FOLD_SPLIT \
  --cache REPRESENTATION_CACHE \
  --output REPRESENTATION_RECEIPT
```

That source binds a train-only sample of 64 fit components (one lowest train row-group each), selection seed 19081, full sequences sampled every 40 rows after the 99-row warmup (the train source has no scored mask), a fit-only float64 normalizer stored/applied in FP32, and one 112→16 top-four signed code plus 112-wide residual dictionary. The dictionary uses seed 19082, 12 AdamW epochs, batch 512, LR 0.001, WD 0.0001, clip 1, and per-epoch order seeds 19083+epoch. These constants belong to the reviewed integration branch, not to the unchanged maintenance CLI.

## Candidate argument binding

These are fragments for the isolated implementation branch `experiment/strong-gru-tweaks-20261019`, **not runnable commands in the unchanged maintenance checkout**. No final source SHA is bound here. Replace `FOLD_SPLIT`, cache/output locations and `CELL_SEED` with the source-bound values for the admitted cell. Bind the launch to strict FP32 (autocast, scaler and TF32 off) and record the actual optimizer/source semantics.

```text
# Teachers, once per fold; SEED in {3,4,5}
--name fg19_t3_h384x2_e3_sSEED --seed SEED --arch gru --cell gru --enc linear
--hidden 384 --layers 2 --head-layers 1 --epochs 3
--checkpoint-policy fixed --checkpoint-epoch 2
--batch 32 --chunk 1000 --optim muon --muon-scope all
--lr 0.001 --wd 0.01 --beta2 0.999 --sched onecycle --warmup-frac 0.05
--clip 1 --loss wmse --precision fp32
--aux-struct 0.3 --aux-next 0.3 --since-weight 32 --since-floor 0.25
--propensity fg19_propensity_clean_f0_seed0_e4
--prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03
--fast --sync-free --fused-adam --pin-memory
--split-manifest FOLD_SPLIT --train-source FIT_SOURCE --no-eval-during-fit

# One fresh shared legacy-topology parent per fold/CELL_SEED (211 inputs/23 outputs)
--name fg19_main_t3_parent_f0_sCELL_SEED --seed CELL_SEED
--device cuda --arch gru --cell gru --enc linear --hidden 256 --layers 2 --head-layers 1
--epochs 4 --checkpoint-policy fixed --checkpoint-epoch 3
--batch 16 --chunk 2000 --optim adamw --lr 0.0001 --wd 0.01 --beta2 0.999
--sched onecycle --warmup-frac 0.05 --clip 1 --loss wmse --loss-norm chunk --precision fp32
--distill-cache-root T3_CACHE --distill-cache-variant homogeneous
--distill-cache-scale 1.0 --distill-alpha 0.9
--aux-struct 0.3 --aux-next 0.3 --aux '' --aux-next-k ''
--distill-family-weight 0
--split-manifest FOLD_SPLIT --train-source FIT_SOURCE --no-eval-during-fit
--save-training-state

# Five children: same parent/shared training settings; warm-load extends 211/23 to 467/25.
--init PARENT@ep3 --seed CELL_SEED --epochs 6
--checkpoint-policy fixed --checkpoint-epoch 5
--representation-mode {base|ema|sparse|ema_sparse}
--representation-cache REPRESENTATION_CACHE
--mean-variance-head 1 --mean-variance-objective {wmse|nll}
--split-manifest FOLD_SPLIT --train-source FIT_SOURCE --no-eval-during-fit
--save-training-state
```

The representation dictionary is a separate fit-only preparation artifact shared by the cell. The T3 cache must be materialized and validated as the un-clipped homogeneous teacher mean scaled by 0.8125; with that value already in the cache, the consumer scale above stays 1.0. This avoids applying 0.8125 twice and satisfies the NLL validator in the implementation branch. Bind cache bytes and target order before launch.

`FIT_SOURCE` above is a placeholder for the exact split-qualified fitting sources admitted for that fold; it must exclude evaluation groups. In particular, retain the known `valid-true-mask` treatment only where the admitted fit sources actually carry a scoring mask, and bind that membership before launch.

The propensity identity and split binding come from [`PROPENSITY_PREREQUISITE.json`](PROPENSITY_PREREQUISITE.json). The six-epoch student values come from [`STUDENT_RECIPE_BINDING.json`](../STUDENT_RECIPE_BINDING.json). The historically identified teacher values come from [`CLEAN_T3_MUON_H384X2_E3_CANDIDATES.md`](CLEAN_T3_MUON_H384X2_E3_CANDIDATES.md). The five-arm science definitions remain in [`DESIGN.json`](../DESIGN.json). The broader 30-cell allocation is a separate plan; this note admits no cells.

## Required departures and admission checks

- The unchanged maintenance optimizer CLI only accepts `--muon-scope rnn|hidden`. The candidate `all` flag requires the separately reviewed implementation-branch port; current `rnn` means recurrent matrices, and `hidden` adds the first hidden readout matrix, with AdamW elsewhere. Do not silently substitute either scope. Bind the port's exact parameter coverage and per-parameter update rules before the teacher fits.
- A prior read of the archived R6 terminal teacher argv records propensity power `2`, propensity epsilon `0.03`, true-mask epsilon `0.03`, auxiliary structure/next weights `0.3/0.3`, since weighting `32` with floor `0.25`, and fast/sync-free/fused-Adam/pinned-memory flags. This note does not yet bind that argv's exact source path and SHA, so the teacher remains non-admitted pending that source receipt. The snippet retains those known settings. The current canonical defaults are LR `0.001`, WD `0.01`, beta2 `0.999`, warmup `0.05`, and clip `1.0`; freeze these explicitly as clean-port choices unless the already-read teacher argv binds exact alternatives. Do not search for old weights or infer missing settings from another teacher.
- The exact four-epoch/fixed-epoch-3 parent duration is retained from the historical parent schedule, but its old parent was seed 3 and trained on older targets. This proposal uses the current cell seed and the current T3 target. It is therefore a **clean rebuild adaptation, not an exact Main T3 lineage reproduction**. Children receive four parent epochs plus six child epochs on the current T3 target; the historic lineage used four epochs on a different target before its six-epoch clean-T3 child.
- The parent intentionally uses the known 211-input/23-output Main T3 base topology; the five children use the experimental 467-input/25-output topology. The zero-extension is a declared warm-start adaptation, not a same-width parent comparison. Validate the existing extension path and initial-function parity before any admitted fit.
- The student binding specifies original clipped-label magnitude times `main_valid`; do not automatically propagate propensity weighting from teachers to the parent or children. Preserve target order: arithmetic teacher mean, cache scale 0.8125 once, clip, 90/10 teacher/label blend, chunk-normalized WMSE for parent/control arms. The NLL arms keep the same blended target and its declared weights; they change the objective only.
- The unchanged maintenance CLI has no `--representation-mode` or mean/logvariance flags. Use the isolated integration branch and bind its exact reviewed source. Confirm that the child loader expands 211→467 inputs and 23→25 outputs, preserves the first 23 outputs and auxiliary offsets, zeros appended input/logvariance rows, and keeps all arm initial mean functions matched.

Until root freezes the source, cache, parent semantics and active propensity receipt, this remains a candidate configuration. It makes no score, quality, parity, runtime, or cross-fold claim.
