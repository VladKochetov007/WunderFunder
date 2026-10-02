#!/usr/bin/env bash
# Wait for both LSTM-student folds, score H-064 (dlstm_f01.json), print verdict lines only.
cd "$(dirname "$0")/../../.."
until [[ -f cache/pred_c20261002_ft_dlstm192a05_s3_f1h_valid.npy ]]; do sleep 30; done; sleep 30
WNN_CPU_PCT=12 WNN_MEM_MAX=5G WNN_SLICE=wnn-campaign.slice CUDA_VISIBLE_DEVICES='' scripts/limited .venv/bin/python \
  scripts/screen_score.py research/campaign_2026-10-02/packages/dlstm_f01.json \
  --output research/campaign_2026-10-02/packages/dlstm_f01_report.json 2>&1 | grep -E '^(pooled|fold[01]|mix:)'
