#!/usr/bin/env bash
# Run one bounded stage on a provisioned Linux receiver.
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo "Use the provisioned Linux receiver" >&2; exit 2; }
packet=$(cd -- "$(dirname -- "$0")" && pwd -P)
mode=${1:?expected check, probe, profile, approve, fit, evaluate, export or score}
cell=${2:-}
if [[ "$mode" == check || "$mode" == probe ]]; then
  [[ $# == 1 ]] || exit 2
else
  [[ $# == 2 ]] || exit 2
  case "$cell" in heads12_s207301|heads12_s207302|heads6_s207301|heads6_s207302) ;; *) exit 2 ;; esac
fi
if [[ "$mode" == evaluate ]]; then
  bash "$packet/run.sh" export "$cell"
  exec bash "$packet/run.sh" score "$cell"
fi
case "$mode" in
  check|approve) limit=600 ;;
  probe) limit=1800 ;;
  profile) limit=3600 ;;
  fit) limit=28800 ;;
  export|score) limit=7200 ;;
  *) echo "Unknown stage: $mode" >&2; exit 2 ;;
esac
seconds=${WF_TIMEOUT_SECONDS:-$limit}
[[ "$seconds" =~ ^[1-9][0-9]{0,4}$ && "$seconds" -le "$limit" ]] || exit 2
: "${WF_OWNER:?assign an owner}" "${WF_RUNTIME:?clean pinned PR75 checkout}" "${WF_DATA:?source root}"
: "${WF_SPLIT:?original split}" "${WF_RUNS:?persistent output directory}" "${WF_PYTHON:?provisioned Python}"
[[ -n "${WF_OWNER//[[:space:]]/}" && "$WF_PYTHON" == /* && -x "$WF_PYTHON" ]]
for path in "$WF_RUNTIME" "$WF_DATA" "$WF_SPLIT" "$WF_RUNS"; do [[ "$path" == /* ]] || exit 2; done
case "$mode" in
  probe) [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[^,[:space:]]+$ ]] ;;
  profile|fit|export)
    : "${CUDA_VISIBLE_DEVICES:?select two allocated B300 devices}"
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^([^,[:space:]]+),([^,[:space:]]+)$ ]]
    [[ "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] ;;
esac
export PYTHONPATH="$WF_RUNTIME/src:$WF_RUNTIME"
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_EMULATE_SINGLE_PRECISION=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd -- "$WF_RUNTIME"
[[ "$(git rev-parse HEAD)" == 8626000a4b965763a70011a0291a891aa6535d7e ]]
[[ -z "$(git status --porcelain --untracked-files=all -- src scripts utils.py)" ]]
work="$WF_RUNS/stages"
mkdir -p -- "$work"
stage="$work/${cell:-panel}-$mode"
mkdir -p -- "$stage"
exec 9>"$stage/lock"
flock -n 9 || { echo "Stage already active: $stage" >&2; exit 1; }
[[ ! -f "$stage/completed" ]] || { echo "Stage already completed: $stage" >&2; exit 1; }
attempt=$(mktemp -d "$stage/attempt.XXXXXXXX")

printf 'owner=%s\nmode=%s\ncell=%s\ntimeout_seconds=%s\nstarted_utc=%s\n' \
  "$WF_OWNER" "$mode" "$cell" "$seconds" "$(date -u +%FT%TZ)" > "$attempt/status.txt"
# Keep the GNU timeout exit status and the complete journal, including failed attempts.
trap 'status=$?; printf "exit_code=%s\nfinished_utc=%s\n" "$status" "$(date -u +%FT%TZ)" >> "$attempt/status.txt"; if ((status == 0)); then printf "%s\n" "$attempt" > "$stage/completed"; fi' EXIT
timeout --kill-after=60s "${seconds}s" "$WF_PYTHON" "$packet/receiver.py" "$mode" "$cell" \
  2>&1 | tee "$attempt/console.log"
