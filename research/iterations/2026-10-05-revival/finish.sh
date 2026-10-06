#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-revival"
units=(wunder-revival-onorm3-r2 wunder-revival-onorm4-r2 wunder-revival-tstream4 wunder-revival-rawwmse4-r2)
for ((attempt=0; attempt<250; attempt++)); do
    active=0
    for unit in "${units[@]}"; do
        if systemctl --user is-active --quiet "$unit"; then active=1; fi
    done
    if ((active==0)); then break; fi
    sleep 30
done
for run in cmp_onorm65r2_s3 cmp_onorm65r2_s4 cmp_tstream_s4 cmp_rawwmse65r2_s4; do
    for split in train valid; do
        test -s "cache/pred_${run}_${split}.npy" || { echo "FAILED missing ${run}_${split}"; exit 1; }
    done
done
export WUNDER_REVIVAL_INCLUDE_NEW=1 WUNDER_REVIVAL_OUT="$D/followup"
export WNN_CPU_PCT=3 WNN_MEM_MAX=16G
scripts/limited .venv/bin/python "$D/extract_moments.py" > "$D/followup-extract.log" 2>&1
scripts/limited .venv/bin/python "$D/bootstrap.py" "$D/followup/moments.npz" "$D/followup/report.json" raw_pair > "$D/followup-bootstrap.log" 2>&1
echo "DONE $(date -Is): $D/followup/report.json"
