#!/usr/bin/env bash
# Sequential 12-fit optimizer grid under one-core/900MB/no-swap limits.
# Thermal courtesy: 20s pause between fits. All logs and JSON in results/.
set -u
cd /home/vlad/development/wunder/wnn_connectome_starterpack
EXP=research/experiment_drafts/own-optimizer-muon-20261024/experiments
mkdir -p "$EXP/results/logs" /home/vlad/development/wunder/research_tmp
for optim in adamw muon muon_hybrid muon_ns2stage; do
  for seed in 0 1 2; do
    echo "=== $optim seed$seed start $(date -u +%H:%M:%SZ) ==="
    /usr/bin/time -v env WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=900000000 \
      WNN_RUNTIME_MAX=330s TMPDIR=/home/vlad/development/wunder/research_tmp \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      scripts/limited .venv/bin/python "$EXP/synth_optimizer_screen.py" \
      --optim "$optim" --seed "$seed" --steps 800 \
      --out "$EXP/results/${optim}_s${seed}.json" \
      > "$EXP/results/logs/${optim}_s${seed}.log" 2>&1
    grep -E "wp=" "$EXP/results/logs/${optim}_s${seed}.log" || \
      tail -3 "$EXP/results/logs/${optim}_s${seed}.log"
    grep -E "Maximum resident|Elapsed \(wall" \
      "$EXP/results/logs/${optim}_s${seed}.log" || true
    sleep 20
  done
done
echo "GRID DONE $(date -u +%H:%M:%SZ)"
