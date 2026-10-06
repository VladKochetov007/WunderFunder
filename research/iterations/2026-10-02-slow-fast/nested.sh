#!/usr/bin/env bash
# nested simulation member: parent excludes valid folds T (pseudo-test) and K, then 1-ep head FT on the remaining valid folds.
#   nested.sh T K GPU
set -euo pipefail
cd ~/WunderFunder-creative
T=$1 K=$2 g=$3
WNN_GPU_FRACTION=0.45 scripts/fold_job.sh nest$K $T $g --extra-exclude-fold $K
bash research/iterations/2026-10-02-slow-fast/ft_sweep128.sh nft$K nest$K $T $g "--freeze inp,gru --epochs 1 --lr 5e-4 --extra-exclude-fold $K"
