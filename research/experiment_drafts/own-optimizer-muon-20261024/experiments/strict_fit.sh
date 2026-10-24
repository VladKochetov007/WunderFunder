#!/usr/bin/env bash
# Strictly gated single fit for the optimizer screen.
# Pre-checks: hottest thermal zone < 84C, MemAvailable >= 8 GB.
# One bounded process (cgroup: 1 core, 700 MB, no swap, 330 s runtime max),
# then a 60 s cool-down with a post-run temperature report.
# Exit 42 = thermal gate, 43 = memory gate.
set -u
cd /home/vlad/development/wunder/wnn_connectome_starterpack
EXP=research/experiment_drafts/own-optimizer-muon-20261024/experiments
optim=${1:?optim}
seed=${2:?seed}

maxtemp() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1; }

t0=$(maxtemp)
if [ "${t0:-0}" -gt 84000 ]; then echo "ABORT-TEMP $t0"; exit 42; fi
avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
if [ "${avail:-0}" -lt 8000000 ]; then echo "ABORT-MEM $avail"; exit 43; fi
echo "GATE-OK temp_millic=$t0 memavail_kB=$avail start=$(date -u +%H:%M:%SZ)"

/usr/bin/time -v env WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=700000000 \
  WNN_RUNTIME_MAX=330s TMPDIR=/home/vlad/development/wunder/research_tmp \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  scripts/limited .venv/bin/python "$EXP/synth_optimizer_screen.py" \
  --optim "$optim" --seed "$seed" --steps 800 \
  --out "$EXP/results/${optim}_s${seed}.json" \
  > "$EXP/results/logs/${optim}_s${seed}.log" 2>&1
rc=$?
grep -E "wp=" "$EXP/results/logs/${optim}_s${seed}.log" || \
  tail -3 "$EXP/results/logs/${optim}_s${seed}.log"
grep -E "Maximum resident|Elapsed \(wall" \
  "$EXP/results/logs/${optim}_s${seed}.log" || true
echo "post-temp=$(maxtemp) rc=$rc end=$(date -u +%H:%M:%SZ)"
sleep 60
echo "cooldown-done temp=$(maxtemp)"
exit $rc
