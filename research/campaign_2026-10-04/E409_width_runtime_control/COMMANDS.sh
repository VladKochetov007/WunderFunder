#!/usr/bin/env bash
# Controller approved E409 after source review and CPU unit tests. No GPU use.
set -euo pipefail
cd /tmp/wnn-c20261004-resources
repo=/home/vlad/development/wunder/wnn_connectome_starterpack
control=$repo/research/campaign_2026-10-04/E409_width_runtime_control
out=$repo/research/campaign_2026-10-04/E409_width_runtime
build=$repo/research/campaign_2026-10-02/engine/native_build_l2/b2
py=$repo/.venv/bin/python
probe=/tmp/wnn-c20261004-resources/scripts/probe_width_runtime.py
group=$(systemctl --user show wnn-campaign.slice -p ControlGroup --value)
date -u +%Y-%m-%dT%H:%M:%SZ > "$control/started_utc.txt"
git rev-parse HEAD > "$control/code_commit.txt"
sha256sum "$probe" scripts/engine_fp16_check.py "$control/COMMANDS.sh" > "$control/source_sha256.txt"
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
  printf '%s\n' "E409 stage $name started"
  env WNN_SLICE=wnn-campaign.slice WNN_SCOPE_UNIT="wnn-e409-$name" \
    WNN_THREADS=1 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G WNN_RUNTIME_MAX=900 \
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1 POLARS_MAX_THREADS=1 ARROW_NUM_THREADS=1 \
    PYTHONPATH=/tmp/wnn-c20261004-resources/src PYTHONDONTWRITEBYTECODE=1 \
    scripts/limited /usr/bin/time -v taskset -c 15 "$py" -u "$probe" "$@" \
    > "$control/$name.log" 2>&1
  date -u +%Y-%m-%dT%H:%M:%SZ > "$control/${name}_finished.txt"
  cat "/sys/fs/cgroup$group/memory.events" > "$control/${name}_oom_after.txt"
  printf '%s\n' "E409 stage $name passed"
}
stage prepare prepare --repository "$repo" --build "$build" \
  --d256 "$repo/artifacts/runs/c20261001_final_d256_s3/model_ep3.pt" \
  --w384 "$repo/artifacts/runs/c20261003_w384_s3_f0/model_ep3.pt" \
  --data "$repo/cache/valid_subset20.parquet" --out "$out" --seqs 3
stage parity-v3 parity --lib "$build/v3" --out "$out" --cpu 15
stage parity-v2 parity --lib "$build/v2" --out "$out" --cpu 15
stage timing-v3 timing --lib "$build/v3" --out "$out" --cpu 15 --repeats 5
