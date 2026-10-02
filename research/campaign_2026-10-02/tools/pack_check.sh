#!/usr/bin/env bash
# Package a candidate twice (int16 and fp32 reference), compare the two on 3 valid sequences, time the int16 ZIP in
# cycles, then run every package gate on it.   pack_check.sh NAME MEMBER:COEF [MEMBER:COEF ...]
# Writes submissions/NAME_i16, submissions/NAME_f32 and research/campaign_2026-10-02/packages/NAME_*.
set -uo pipefail
cd "$(dirname "$0")/../../.."
name=$1; shift
out=research/campaign_2026-10-02/packages
PY=research/campaign_2026-10-01/engine/.venv/bin/python
L() { WNN_SLICE=wnn-campaign.slice WNN_THREADS=1 WNN_CPU_QUOTA=100% WNN_MEM_MAX=6G CUDA_VISIBLE_DEVICES='' scripts/limited "$@"; }
for p in i16 f32; do
  L .venv/bin/python scripts/package_submission.py "${name}_$p" "$@" --solution submission/solution_native.py \
    --precision "$p" --native-from research/campaign_2026-10-01/engine/native_build | tail -1 | cut -c1-160
  L taskset -c 15 $PY scripts/engine_fp16_check.py calib "submissions/${name}_$p/submission.zip" --seqs 3 \
    --save "$out/${name}_calib_$p.npy" | tee "$out/${name}_calib_$p.json"
done
.venv/bin/python - "$out/${name}" <<'PY'
import sys, numpy as np
a, b = np.load(sys.argv[1] + "_calib_i16.npy"), np.load(sys.argv[1] + "_calib_f32.npy")
m = np.isfinite(a) & np.isfinite(b)
print("i16_vs_f32_max_abs", float(np.abs(a[m] - b[m]).max()), "finite_rows", int(m.sum()))
PY
research/campaign_2026-10-01/packages/verify_all.sh "submissions/${name}_i16/submission.zip" "${name}_i16" 2>&1 | tee "$out/${name}_verify.log"
echo "PACK_CHECK_DONE $name $(sha256sum "submissions/${name}_i16/submission.zip" | cut -c1-64)"
