#!/usr/bin/env bash
# Follow-up to the final teacher-ensemble student (H-060): once its campaign job finishes, score it on the holdout
# and package it (int16 + fp32 reference, all gates). Output: research/campaign_2026-10-02/packages/after_final_te.log
set -uo pipefail
cd "$(dirname "$0")/../../.."
rec=research/campaign_2026-09-29/jobs/c20261002_final_te07_s3.json
until python3 -c "import json,sys;d=json.load(open('$rec'));sys.exit(0 if d['status']!='running' else 1)" 2>/dev/null; do sleep 30; done
python3 -c "import json;d=json.load(open('$rec'));print('JOB', d['status'], d.get('exit_code'), round(d.get('elapsed_seconds') or 0))"
[[ -f cache/pred_c20261002_final_te07_s3_gcorrh_valid.npy ]] || { echo "NO_PREDICTIONS"; exit 1; }
WNN_CPU_PCT=12 WNN_MEM_MAX=6G WNN_SLICE=wnn-campaign.slice CUDA_VISIBLE_DEVICES='' scripts/limited .venv/bin/python \
  scripts/screen_score.py research/campaign_2026-10-02/packages/te_final_holdout.json \
  --output research/campaign_2026-10-02/packages/te_final_holdout_report.json 2>&1 | grep -E '^(pooled|mix:)|wp '
research/campaign_2026-10-02/tools/pack_check.sh c20261002_TE07s3 \
  "c20261002_final_te07_s3@ep3|c20261002_final_te07_s3_gcorrh@ep0:1.0" 2>&1 | grep -E 'i16_vs|bitwise|RESET|kcycles|PACK_CHECK|global_wp' | cut -c1-200
.venv/bin/python - <<'PY'
import numpy as np
a = np.load('research/campaign_2026-10-01/packages/c20261002_TE07s3_i16_image.npz')['pred']
b = np.load('research/campaign_2026-10-01/packages/c20261002_TE07s3_i16_v3.npz')['pred']
print('image_vs_host_bitwise', bool(np.array_equal(np.nan_to_num(a), np.nan_to_num(b))))
PY
echo AFTER_FINAL_TE_DONE
