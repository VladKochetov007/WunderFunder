#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/research_tmp/overnight-nll-adaptation-20261020
export WNN_PROJECT_ROOT=$ROOT PYTHONPATH=$WT/src TMPDIR=$WT WNN_THREADS=2 WNN_LOADER_WORKERS=0 \
  WNN_LOADER_RAM_GB=3.5 WNN_GPU_MEMORY_BYTES=4500000000 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd $ROOT
ARGV=$(sed "s#{python}#$ROOT/.venv/bin/python#" research/campaign_2026-10-25/class3_public_mix_r1/candidate_f0.argv)
exec $ARGV
