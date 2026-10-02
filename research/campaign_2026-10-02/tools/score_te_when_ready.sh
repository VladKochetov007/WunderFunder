#!/usr/bin/env bash
# Wait for the fold-1 teacher-ensemble predictions, then score H-060 on folds 0+1.
cd "$(dirname "$0")/../../.."
until [[ -f cache/pred_c20261002_ft_te07_s3_f1h_valid.npy && -f cache/pred_c20261002_te07_s3_f1_valid.npy ]]; do sleep 20; done
sleep 30
WNN_CPU_PCT=12 WNN_MEM_MAX=6G WNN_SLICE=wnn-campaign.slice CUDA_VISIBLE_DEVICES='' scripts/limited .venv/bin/python \
  scripts/screen_score.py research/campaign_2026-10-02/packages/te_f01.json \
  --output research/campaign_2026-10-02/packages/te_f01_report.json 2>&1 | grep -E '^(pooled|fold|mix:)|wp '
