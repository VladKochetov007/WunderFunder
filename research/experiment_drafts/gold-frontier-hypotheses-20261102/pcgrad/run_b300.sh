#!/usr/bin/env bash
set -euo pipefail
: "${WNN_PROJECT_ROOT:?Set WNN_PROJECT_ROOT to the provisioned dataset and output root}"
: "${F0_MANIFEST:?Set F0_MANIFEST to the verified clean F0 split manifest}"
: "${F2_MANIFEST:?Set F2_MANIFEST to the verified clean F2 split manifest}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one available GPU}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo 'PCGrad supports one GPU per process; set exactly one visible GPU' >&2
  exit 2
fi
for manifest in "$F0_MANIFEST" "$F2_MANIFEST"; do
  [[ -f "$manifest" ]] || { echo "Manifest missing: $manifest" >&2; exit 2; }
done
for cell in f0 f2; do
  for arm in control pcgrad; do
    output="$WNN_PROJECT_ROOT/artifacts/runs/gold_pcgrad_${cell}_${arm}"
    [[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 2; }
  done
done
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
for cell in f0 f2; do
  if [[ "$cell" == f0 ]]; then manifest="$F0_MANIFEST"; seed=1001; else manifest="$F2_MANIFEST"; seed=1002; fi
  for arm in control pcgrad; do
    extra=()
    if [[ "$arm" == pcgrad ]]; then extra=(--target-pcgrad); fi
    python scripts/train_rnn.py --name "gold_pcgrad_${cell}_${arm}" --device cuda \
      --arch gru --cell gru --enc linear --numeric-map none --raw-only \
      --hidden 256 --layers 2 --batch 32 --chunk 2000 --epochs 6 \
      --loss wmse --precision fp32 --optim adamw --lr 0.001 --wd 0.01 \
      --seed "$seed" --split-manifest "$manifest" --checkpoint-policy fixed \
      --checkpoint-epoch 5 --no-eval-during-fit --grad-stats "${extra[@]}"
  done
done
