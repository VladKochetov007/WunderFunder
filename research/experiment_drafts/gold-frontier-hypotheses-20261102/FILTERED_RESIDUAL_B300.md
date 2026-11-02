# A: training-only filtered residual on B300

This uses the existing `scripts/train_rnn.py` standard window loop. The added
loss computes a first difference and a mean of residuals across 16 rows, both
inside eligible consecutive windows in the current sequence and chunk. Any
window containing a zero-weight or nonfinite target is excluded; no window
crosses chunks. Each scale receives equal weight; objective coefficient is 0.05.
The main training loss remains unchanged. Prediction/inference is unchanged.
No observed performance or correctness is claimed; receiver parity and numerical
checks, real data, valid source-qualified fold manifests and GPU profile are
mandatory before long fits. No local tests or training were executed.

The following launches 4 independent fits, one GPU each in sequence, under a
receiver-provided environment with `WNN_PROJECT_ROOT` pointing to a root
containing `datasets/train.parquet` and the required training caches and dependencies. Before use set `F0_MANIFEST` and `F2_MANIFEST` to the
actual component-clean manifests (no full-fit PR54 ancestors); see existing
training data instructions. `CUDA_VISIBLE_DEVICES` must identify an available
owned B300. Outputs are created under `artifacts/runs`. Explicit fixed final
checkpoint avoids selecting on an outer held-out fold. Reserve disjoint run names.
The paired seeds are confounded with folds and give no significance claim. To run
in parallel, allocate separate GPUs and isolate processes/output paths.

```bash
export CUDA_VISIBLE_DEVICES=0
: "${WNN_PROJECT_ROOT:?set the data and output root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${F0_MANIFEST:?set component-clean F0 split manifest}"
: "${F2_MANIFEST:?set component-clean F2 split manifest}"
for CELL in F0 F2; do
  if [ "$CELL" = F0 ]; then MANIFEST="$F0_MANIFEST"; SEED=1001; else MANIFEST="$F2_MANIFEST"; SEED=1002; fi
  for ARM in control filtered; do
    if [ "$ARM" = control ]; then WEIGHT=0; else WEIGHT=0.05; fi
    test ! -e "$WNN_PROJECT_ROOT/artifacts/runs/gold_a_${CELL}_${ARM}"
    python scripts/train_rnn.py --name "gold_a_${CELL}_${ARM}" --device cuda \
      --arch gru --cell gru --enc linear --numeric-map none --raw-only \
      --hidden 256 --layers 2 --batch 32 --optim adamw --lr 0.001 --wd 0.01 \
      --split-manifest "$MANIFEST" --seed "$SEED" --precision fp32 \
      --loss wmse --chunk 2000 --epochs 6 --checkpoint-policy fixed \
      --checkpoint-epoch 5 --no-eval-during-fit --filtered-residual-weight "$WEIGHT"
  done
done
```

The supplied data location, manifest semantics and all remaining defaults, including
the scheduler, must be inspected and frozen with their hashes on the receiving host. The above is a
cold-label baseline, not an exact TL3 recipe or a qualified TL3 comparison. Grade
complete-sequence predictions on matching held-out components with the official
scorer and preserve all cell outcomes. The current implementation rejects
controlled macro loss, compressed row thinning, CE readout and chunks under 16;
this is intentional so the filtered loss cannot silently be skipped. The value
0.05 is a proposal, not tuned or validated. Check target availability, exact loss
mask and post-loss gradient scale on the receiver before any full run.
