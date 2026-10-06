#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave2"
units=(wunder-wave2-sentinel3 wunder-wave2-sentinel4 wunder-wave2-complement)
for ((attempt=0; attempt<370; attempt++)); do
    active=0
    for unit in "${units[@]}"; do
        if systemctl --user is-active --quiet "$unit"; then active=1; fi
    done
    if ((active==0)); then break; fi
    sleep 30
done
for name in w2sent_zero w2sent_age w2comp_direct w2comp_residual; do
    for seed in 3 4; do
        for split in train valid; do
            test -s "cache/pred_${name}_s${seed}_${split}.npy" || { echo "FAILED missing $name $seed $split"; exit 1; }
        done
    done
done
export WNN_CPU_PCT=3 WNN_MEM_MAX=16G
scripts/limited .venv/bin/python "$D/evaluate_wave.py" > "$D/wave-evaluate.log" 2>&1
echo "DONE $(date -Is) $D/wave-report.json"
