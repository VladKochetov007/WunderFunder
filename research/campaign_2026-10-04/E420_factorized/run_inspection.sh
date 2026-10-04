#!/usr/bin/env bash
set -u
cd /tmp/wnn-c20261004-native-factors
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log=research/campaign_2026-10-04/E420_factorized/source_inspection.log
set +e
/usr/bin/time -v /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  research/campaign_2026-10-04/E420_factorized/inspect_source.py \
  /home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/runs/c20261003_w384_s3_f0/model.pt \
  research/campaign_2026-10-04/E409_width_runtime/manifest.json \
  research/campaign_2026-10-04/E420_factorized/SOURCE_INSPECTION.json > "$log" 2>&1
rc=$?
printf 'exit_code=%s\n' "$rc" >> "$log"
exit "$rc"
