#!/usr/bin/env bash
# Draft only. Do not run until the root admits zero-update preparation.
systemd-run --user --scope --slice=wnn-campaign.slice \
  -p CPUQuota=100% -p MemoryMax=2G -p MemorySwapMax=0B \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  /tmp/wnn-c20261004-native-factors/scripts/export_factorized_control.py \
  /home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/runs/c20261003_w384_s3_f0/model.pt \
  --output /tmp/wnn-c20261004-native-factors/research/campaign_2026-10-04/E420_factorized/zero_update_w384_r128.pt \
  --rank 128
