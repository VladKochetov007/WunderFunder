#!/usr/bin/env bash
set -euo pipefail
# Run inside the pinned, reviewed repository on a B300 Linux receiver.
: "${WNN_PROJECT_ROOT:?set the data and output root}"
: "${CUDA_VISIBLE_DEVICES:?select exactly one available GPU}"
: "${SPLIT_MANIFEST:?point to source-qualified split manifest}"
: "${DISTILL_CACHE_ROOT:?point to matching fold-qualified teacher cache}"
: "${CELL_SEED:?use the seed declared for this fold}"
: "${RUN_NAME:?use a unique output name for each cell/arm}"
: "${ARM:?choose burnin-reset or burnin-prefix}"
case "$ARM" in burnin-reset|burnin-prefix) ;; *) echo "invalid ARM" >&2; exit 2;; esac
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "select exactly one GPU" >&2; exit 2; }
[[ ! -e "$WNN_PROJECT_ROOT/artifacts/runs/$RUN_NAME" ]] || { echo "output already exists" >&2; exit 2; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec python scripts/train_rnn.py --name "$RUN_NAME" --device cuda \
  --split-manifest "$SPLIT_MANIFEST" --distill-cache-root "$DISTILL_CACHE_ROOT" \
  --distill-cache-variant source --distill-alpha 0.5 --loss wmse --arch gru \
  --hidden 256 --layers 2 --dropout 0 --batch 8 --epochs 6 --seed "$CELL_SEED" \
  --chunk 1000 --update-rows 2000 --detach-rows 1000 --origin-view "$ARM" \
  --burnin-rows 2000 --optim adamw --lr 0.0003 --wd 0.01 --sched wsd \
  --precision fp32 --deterministic --checkpoint-policy fixed --checkpoint-epoch 5 \
  --no-eval-during-fit --grad-stats
