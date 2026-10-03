#!/usr/bin/env bash
# Four shared CPU slots, each including its command's children and notebook kernel.
set -euo pipefail
cd "$(dirname "$0")/../../.."
state=research/campaign_2026-10-03/continuation
role=${1:?usage: run_cpu.sh ROLE COMMAND...}
shift
[[ $role =~ ^[a-zA-Z0-9_-]+$ ]] || exit 64
[[ $# -gt 0 ]] || exit 64
[[ -f $state/READY ]] || { echo 'resource admission is not ready' >&2; exit 75; }
while true; do
  [[ ! -f $state/RESOURCE_FAILURE ]] || { echo 'resource failure needs review' >&2; exit 75; }
  selected=
  for slot in 0 1 2 3; do
    exec {lease}>"$state/locks/cpu-$slot.lock"
    if flock -n "$lease"; then selected=$slot; break; fi
    exec {lease}>&-
  done
  [[ -n $selected ]] && break
  sleep 2
done
exec {admission}>"$state/locks/admission.lock"
flock "$admission"
[[ -f $state/READY && ! -f $state/RESOURCE_FAILURE ]] || exit 75
unit="wnn-cpu-${selected}-${role}-$(date +%s)-${BASHPID}"
# A killed wrapper may leave its scope alive. Never reuse that slot until it exits.
if ! scopes=$(systemctl --user list-units --type=scope --state=active --no-legend); then
  echo 'cannot inspect CPU scopes; admission blocked' >&2; exit 75
fi
if [[ $scopes == *"wnn-cpu-${selected}-"* ]]; then
  echo 'CPU slot has an orphan scope; retry after controller review' >&2; exit 75
fi
child=
cleanup() {
  trap - TERM INT HUP
  systemctl --user stop "$unit.scope" >/dev/null 2>&1 || true
  [[ -z $child ]] || wait "$child" || true
  exit 143
}
trap cleanup TERM INT HUP
set +e
env WNN_SLICE=wnn-campaign.slice WNN_SCOPE_UNIT="$unit" WNN_RUNTIME_MAX=900 \
  WNN_THREADS=1 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  POLARS_MAX_THREADS=1 ARROW_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  scripts/limited "$@" &
child=$!
# Commit admission while holding the same lock used to publish failure barriers.
# The slot lease remains held through command completion.
for attempt in {1..50}; do
  kill -0 "$child" 2>/dev/null || break
  systemctl --user is-active --quiet "$unit.scope" && break
  sleep 0.1
done
flock -u "$admission"
wait "$child"
result=$?
set -e
trap - TERM INT HUP
printf '%s role=%s slot=%s unit=%s exit=%s\n' "$(date -Is)" "$role" "$selected" "$unit" "$result" >> "$state/logs/cpu_commands.log"
journalctl --user -u "$unit.scope" --no-pager -o short-iso > "$state/logs/$unit.terminal.log" 2>&1 || true
# Started-unit descriptions include command arguments such as --timeout600.
# Diagnose scope failures from systemd's unit-specific messages, not that text.
failure_message="$unit\\.scope: (A process of this unit has been killed by the OOM killer|Failed with result '(timeout|oom-kill|signal|resources)')"
if [[ $result -ge 128 ]] || rg -qi "$failure_message" "$state/logs/$unit.terminal.log"; then
  flock "$admission"
  printf '%s killed or expired CPU scope %s exit=%s; diagnose before retry\n' "$(date -Is)" "$unit" "$result" >> "$state/RESOURCE_FAILURE"
  printf '%s CPU resource failure: %s\n' "$(date -Is)" "$unit" >> research/campaign_2026-10-01/QUEUE_PAUSE
  flock -u "$admission"
fi
exit "$result"
