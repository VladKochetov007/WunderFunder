#!/usr/bin/env bash
set -euo pipefail
cd /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-11-14/quality_r2/kaggle_r2
KAGGLE=/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/kaggle
"$KAGGLE" datasets create -p /home/vlad/development/wunder/research_tmp/q14_kaggle_r2/upload_dataset --dir-mode zip
"$KAGGLE" kernels push -p /home/vlad/development/wunder/research_tmp/q14_kaggle_r2/upload_dataset/kernel -t 13500
