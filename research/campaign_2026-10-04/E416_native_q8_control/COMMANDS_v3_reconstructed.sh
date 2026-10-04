#!/usr/bin/env bash
# E416: controller-authorized parity. Timing additionally requires a quiet window.
set -euo pipefail
repo=/home/vlad/development/wunder/wnn_connectome_starterpack
source_repo=/tmp/wnn-c20261004-resources
control=$repo/research/campaign_2026-10-04/E416_native_q8_control
out=$repo/research/campaign_2026-10-04/E416_native_q8
build=$repo/research/campaign_2026-10-04/E416_native_q8_build
py=$repo/.venv/bin/python
phase=${1:?stage required: prepare/parity_v3/parity_v2/timing}
cd "$source_repo"
case "$phase" in
  prepare) args=(prepare --repository "$source_repo" --build "$build" --e409 "$repo/research/campaign_2026-10-04/E409_width_runtime" --e411 "$repo/research/campaign_2026-10-04/E411_q8_model");;
  parity_v3) args=(parity --lib "$build/v3");;
  parity_v2) args=(parity --lib "$build/v2");;
  timing) args=(timing --lib "$build/v3");;
  *) exit 2;;
esac
test ! -e "$control/${phase}_started.txt"
group=$(systemctl --user show wnn-campaign.slice -p ControlGroup --value)
date -u +%Y-%m-%dT%H:%M:%SZ > "$control/${phase}_started.txt"
systemctl --user show wnn-campaign.slice -p ControlGroup -p CPUQuotaPerSecUSec -p MemoryMax -p MemorySwapMax -p MemoryCurrent > "$control/${phase}_parent_before.txt"
cat "/sys/fs/cgroup$group/memory.events" > "$control/${phase}_parent_oom_before.txt"
finish() {
  result=$?
  trap - EXIT
  printf '%s\n' "$result" > "$control/${phase}_exit.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$control/${phase}_finished.txt"
  systemctl --user show wnn-campaign.slice -p ControlGroup -p CPUQuotaPerSecUSec -p MemoryMax -p MemorySwapMax -p MemoryCurrent > "$control/${phase}_parent_after.txt"
  cat "/sys/fs/cgroup$group/memory.events" > "$control/${phase}_parent_oom_after.txt"
  exit "$result"
}
trap finish EXIT
env WNN_SLICE=wnn-campaign.slice WNN_SCOPE_UNIT="wnn-e416-$phase" \
  WNN_THREADS=1 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G WNN_RUNTIME_MAX=1800 \
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 POLARS_MAX_THREADS=1 ARROW_NUM_THREADS=1 \
  PYTHONPATH="$source_repo/src" PYTHONDONTWRITEBYTECODE=1 \
  "$repo/scripts/limited" bash "$control/SCOPE_STAGE.sh" "$control" "$phase" \
  /usr/bin/time -v taskset -c 15 "$py" -u "$source_repo/scripts/probe_q8_native.py" \
  "${args[@]}" --out "$out" --cpu 15 > "$control/${phase}.log" 2>&1
