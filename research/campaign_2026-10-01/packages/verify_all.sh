#!/usr/bin/env bash
# All package gates for one ZIP: host harness (v3/avx2/v2 bitwise), reset/warm-up contract, offline Python 3.11 image,
# size, and a back-to-back runtime ratio against the incumbent in the same harness.
#   verify_all.sh ZIP TAG
set -uo pipefail
cd "$(dirname "$0")/../../.."
zip=$1 tag=$2 out=research/campaign_2026-10-01/packages
L() { WNN_SLICE=wnn-campaign.slice WNN_THREADS=1 WNN_CPU_QUOTA=100% WNN_MEM_MAX=8G CUDA_VISIBLE_DEVICES='' scripts/limited "$@"; }
echo "size $(stat -c %s "$zip") sha256 $(sha256sum "$zip" | cut -c1-64)"
unzip -l "$zip" | awk 'NR>3 && $4=="solution.py"{print "root solution.py ok"}'
for lv in v3 avx2 v2; do
  WNN_NATIVE_LEVEL=$lv L .venv/bin/python scripts/verify_submission.py "$zip" cache/valid_subset20.parquet "$out/${tag}_$lv.npz" 2>&1 | tail -1 | sed "s/^/$lv /"
done
.venv/bin/python - "$out/${tag}" <<'PY'
import sys, numpy as np
b = sys.argv[1]
v3 = np.load(b + "_v3.npz")["pred"]
for lv in ("avx2", "v2"):
    o = np.load(f"{b}_{lv}.npz")["pred"]
    print(lv, "bitwise_equal_v3", bool(np.array_equal(np.nan_to_num(v3), np.nan_to_num(o))))
PY
L .venv/bin/python research/campaign_2026-10-01/packages/reset_check.py "$zip" 2>&1 | tail -1
timeout 1800 research/campaign_2026-10-01/packages/image_test.sh "$zip" "$out/${tag}_image.npz" 2>&1 | tail -2
for z in submissions/final_since2_gclp/final_since2_gclp.zip "$zip"; do
  L taskset -c 15 .venv/bin/python scripts/verify_submission.py "$z" cache/valid_subset20.parquet "$out/${tag}_timing_$(basename "$z" .zip).npz" 2>&1 | tail -1 | grep -o '"us_per_row": [0-9.]*' | sed "s#^#$(basename "$z") #"
done
