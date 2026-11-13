#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo "Use the designated Linux receiver." >&2; exit 2; }
packet=$(cd -- "$(dirname -- "$0")" && pwd -P)
stage=${1:?expected check, probe, profile, approve, fit, predict, score or readout}
cell=${2:-}
case "$stage" in
  check) limit=3600 ;; probe) limit=1200 ;; profile) limit=1800 ;; approve) limit=600 ;;
  fit) limit=25200 ;; predict|score) limit=7200 ;; readout) limit=1800 ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
case "$stage" in
  check|probe|readout) [[ $# == 1 ]] || exit 2 ;;
  *) [[ $# == 2 && "$cell" =~ ^(labels|since32|kd|both)_s(203301|203302)$ ]] || exit 2 ;;
esac
: "${WF_OWNER:?assign the receiver owner}" "${WF_BASE:?pinned native checkout}" "${WF_DATA:?original data root}"
: "${WF_CACHE:?canonical PR49 fit cache}" "${WF_SPLIT:?original chronological split}"
: "${WF_RUNS:?fresh persistent output root}" "${WF_PYTHON:?absolute provisioned Python}"
for path in "$WF_BASE" "$WF_DATA" "$WF_SPLIT" "$WF_CACHE" "$WF_RUNS" "$WF_PYTHON"; do
  [[ "$path" == /* ]] || { echo "Receiver paths must be absolute." >&2; exit 2; }
done
[[ -n "${WF_OWNER//[[:space:]]/}" && -x "$WF_PYTHON" ]]
case "$stage" in
  probe|profile|fit|predict) [[ "${CUDA_VISIBLE_DEVICES:?select one allocated L4}" =~ ^[^,[:space:]]+$ ]] || exit 2 ;;
esac
seconds=${WF_TIMEOUT_SECONDS:-$limit}
[[ "$seconds" =~ ^[1-9][0-9]{0,4}$ && "$seconds" -le "$limit" ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
unset CUBLAS_EMULATE_SINGLE_PRECISION CUBLAS_WORKSPACE_CONFIG WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
dir="$WF_RUNS/stages/${cell:-panel}-$stage"
mkdir -p -- "$dir"
exec 9>"$dir/lock"
flock -n 9 || { echo "Stage already owned: $dir" >&2; exit 1; }
[[ ! -e "$dir/completed" ]] || { echo "Recorded completed stage: $dir" >&2; exit 1; }
attempt=$(mktemp -d "$dir/attempt.XXXXXXXX")
printf 'owner=%s\nstage=%s\ncell=%s\ntimeout_seconds=%s\nstarted_utc=%s\n' \
  "$WF_OWNER" "$stage" "$cell" "$seconds" "$(date -u +%FT%TZ)" > "$attempt/status.txt"
trap 's=$?; printf "exit_code=%s\nfinished_utc=%s\n" "$s" "$(date -u +%FT%TZ)" >> "$attempt/status.txt"; if ((s == 0)); then printf "%s\n" "$attempt" > "$dir/completed"; fi' EXIT
# One deadline includes verification, the native process and all its child workers.
timeout --kill-after=60s "${seconds}s" bash -eu -c '
  packet=$1; stage=$2; cell=$3
  unset PYTHONPATH
  "$WF_PYTHON" "$packet/runtime_source.py" "$stage"
  cd "$WF_RUNS/runtime"
  export PYTHONPATH="$PWD/src:$PWD"
  if [[ "$stage" == readout ]]; then
    exec "$WF_PYTHON" "$packet/readout.py" --run-root "$WF_RUNS" --split-manifest "$WF_SPLIT"
  fi
  exec "$WF_PYTHON" "$packet/receiver.py" "$stage" "$cell"
' -- "$packet" "$stage" "$cell" 2>&1 | tee "$attempt/console.log"
