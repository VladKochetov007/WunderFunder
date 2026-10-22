# Run handoff

These are canonical CLI templates, not executable node commands yet. The source parser and trainer are exact; the source-qualified fold manifests, project/data mount, runtime environment and independent grader path are external and intentionally remain placeholders. Bind those assets before profiling or fitting.

Set `WNN_PROJECT_ROOT` to the audited project/data root and replace `$SOURCE_ROOT`, `$F0_MANIFEST`, `$F2_MANIFEST`, `$RUNS_ROOT`, and `$GRADER_COMMAND` from the node receipt. Verify each manifest hash and its component roster. Run each command from the committed experiment source root. Keep all four fit processes single-GPU, with distinct output names and one GPU per process. The current source has no distributed trainer.

The pair-cell table is:

| Fold | Seed | Control | Candidate |
|---|---:|---|---|
| F0 | 1201 | `nll_label_f0_s1201_wmse` | `nll_label_f0_s1201_nll` |
| F2 | 1202 | `nll_label_f2_s1202_wmse` | `nll_label_f2_s1202_nll` |

Before fit admission, run the bounded profile on each objective using one exact cell manifest, same data path, full sequence shape, H256x2 model and strict-FP32 settings. It performs two warmup plus four timed batches and stops before evaluation/checkpoint selection. Record the measured time and throughput, extrapolate from the source's exact full-epoch optimizer-attempt count, include startup and final grading/export, and require all work plus cleanup/publication to fit the authorized wall limit. This is a projection until actual endpoint receipts exist.

```bash
export WNN_PROJECT_ROOT=/REPLACE/WITH/AUDITED/PROJECT_AND_DATA_ROOT
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python scripts/train_rnn.py \
  --name nll_label_profile_f0_s1201_wmse \
  --split-manifest "$F0_MANIFEST" --valid-fold 0 --eval-source vfold \
  --train-source train+valid --seed 1201 --train-seqs 10607 \
  --arch gru --cell gru --enc linear --hidden 256 --layers 2 --head-layers 1 \
  --batch 32 --chunk 2000 --epochs 3 --checkpoint-policy fixed --checkpoint-epoch 2 \
  --lr 0.001 --wd 0.01 --beta2 0.999 --sched onecycle --clip 1.0 \
  --precision fp32 --mean-variance-head 1 --mean-variance-objective wmse \
  --profile-warmup 2 --profile-steps 4 --grad-stats --no-eval-during-fit

python scripts/train_rnn.py \
  --name nll_label_profile_f0_s1201_nll \
  --split-manifest "$F0_MANIFEST" --valid-fold 0 --eval-source vfold \
  --train-source train+valid --seed 1201 --train-seqs 10607 \
  --arch gru --cell gru --enc linear --hidden 256 --layers 2 --head-layers 1 \
  --batch 32 --chunk 2000 --epochs 3 --checkpoint-policy fixed --checkpoint-epoch 2 \
  --lr 0.001 --wd 0.01 --beta2 0.999 --sched onecycle --clip 1.0 \
  --precision fp32 --mean-variance-head 1 --mean-variance-objective nll \
  --mean-variance-target-source labels \
  --profile-warmup 2 --profile-steps 4 --grad-stats --no-eval-during-fit
```

After profiles and manifest validation, the four scientific commands use the same arguments, with these substitutions: remove both profile flags and `--grad-stats`; change each `--name` to its table name; set `CUDA_VISIBLE_DEVICES` to the assigned card; select `$F0_MANIFEST` / `--valid-fold 0` or `$F2_MANIFEST` / `--valid-fold 2`; and select WMSE or labels-NLL exactly as listed. Do not add `--init`, `--inherit-init-preprocessing`, `--distill`, `--distill-cache-root`, `--final-distill-initializer`, or `--fit-subset-manifest`. Keep the objective pair matched in seed and all remaining flags.

The independent grader must load the fixed epoch2 artifact and score every source-qualified excluded component under official `global_wp_clip2_v1`. The grader module, invocation, artifact export paths and public repository destination are outside this code checkout; obtain their exact audited paths before launch and include them in the immutable run receipt. The prior 4-hour limit includes fit, grading, actual GPU cleanup and results publication. If resource profiling shows insufficient time, do not start the cohort.
