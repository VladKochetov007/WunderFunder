#!/usr/bin/env bash
# Controller-approved E411: two fixed presets, predeclared component sample, CPU only.
set -euo pipefail
repo=/home/vlad/development/wunder/wnn_connectome_starterpack
cd "$repo"
control=$repo/research/campaign_2026-10-04/E411_q8_model_control
out=$repo/research/campaign_2026-10-04/E411_q8_model
build=$repo/research/campaign_2026-10-02/engine/native_build_l2/b2/v3
py=$repo/.venv/bin/python
probe=$repo/scripts/probe_q8_model.py
group=$(systemctl --user show wnn-campaign.slice -p ControlGroup --value)
date -u +%Y-%m-%dT%H:%M:%SZ > "$control/started_utc.txt"
git rev-parse HEAD > "$control/code_commit.txt"
sha256sum "$probe" src/wnn/quantization.py "$control/COMMANDS.sh" > "$control/source_sha256.txt"
systemctl --user show wnn-campaign.slice -p ControlGroup -p CPUQuotaPerSecUSec -p MemoryMax -p MemorySwapMax -p MemoryCurrent > "$control/parent_before.txt"
cat "/sys/fs/cgroup$group/memory.events" > "$control/oom_before.txt"
finish() {
  result=$?
  trap - EXIT
  printf '%s\n' "$result" > "$control/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$control/finished_utc.txt"
  systemctl --user show wnn-campaign.slice -p ControlGroup -p CPUQuotaPerSecUSec -p MemoryMax -p MemorySwapMax -p MemoryCurrent > "$control/parent_after.txt"
  cat "/sys/fs/cgroup$group/memory.events" > "$control/oom_after.txt"
  exit "$result"
}
trap finish EXIT
stage() {
  name=$1
  shift
  date -u +%Y-%m-%dT%H:%M:%SZ > "$control/${name}_started.txt"
  printf '%s\n' "E411 stage $name started"
  env WNN_SLICE=wnn-campaign.slice WNN_SCOPE_UNIT="wnn-e411-$name" \
    WNN_THREADS=1 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G WNN_RUNTIME_MAX=1800 \
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1 POLARS_MAX_THREADS=1 ARROW_NUM_THREADS=1 \
    PYTHONPATH="$repo/src" PYTHONDONTWRITEBYTECODE=1 \
    WNN_PORTABLE_DATA_MANIFEST=research/campaign_2026-10-04/portable_inputs.json \
    scripts/limited /usr/bin/time -v taskset -c 15 "$py" -u "$probe" "$@" \
    > "$control/$name.log" 2>&1
  date -u +%Y-%m-%dT%H:%M:%SZ > "$control/${name}_finished.txt"
  cat "/sys/fs/cgroup$group/memory.events" > "$control/${name}_oom_after.txt"
  printf '%s\n' "E411 stage $name passed"
}
stage prepare prepare --repository "$repo" --lib "$build" \
  --parent "$repo/artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt" \
  --head "$repo/artifacts/runs/c20261004_combo_head_s3_f0/model_ep0.pt" \
  --split "$repo/research/campaign_2026-09-29/artifacts/splits/fold0.json" \
  --data "$repo/datasets/valid.parquet" --out "$out"
stage run run --lib "$build" --out "$out" --cpu 15
