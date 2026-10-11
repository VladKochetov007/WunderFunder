#!/usr/bin/env bash
set -uo pipefail
umask 077

FIT_TIMEOUT_SECONDS="${1:-}"
if [[ ! "$FIT_TIMEOUT_SECONDS" =~ ^[0-9]{2,4}$ ]] || (( 10#$FIT_TIMEOUT_SECONDS < 60 || 10#$FIT_TIMEOUT_SECONDS > 7200 )); then
    printf 'H391_FIT_R1_START_FAILED: provide an explicitly admitted timeout from 60 through 7200 seconds\n' >&2
    exit 64
fi

LAUNCH_SCRIPT=/root/run/H391_FIT_LAUNCH_R1.sh
LAUNCH_SHA256=2456c216b27b6a53114ced2b1a6e71648c00b33229a599f66ce7b8d66654c6a9
RUN_DIR=/root/run/h391_fit_r1
SUPERVISOR_LOG=/root/run/h391_fit_r1_supervisor.out
SUPERVISOR_PIDFILE=/root/run/h391_fit_r1_supervisor.pid
SUPERVISOR_STARTED=/root/run/h391_fit_r1_supervisor_started_utc.txt
FIT_TIMEOUT_FILE=/root/run/h391_fit_r1_admitted_timeout_seconds.txt

fail() {
    printf 'H391_FIT_R1_START_FAILED: %s\n' "$*" >&2
    exit 70
}

[[ -f "$LAUNCH_SCRIPT" ]] || fail "missing launch script: $LAUNCH_SCRIPT"
[[ ! -e "$RUN_DIR" ]] || fail "fit run directory already exists: $RUN_DIR"
[[ ! -e "$SUPERVISOR_LOG" ]] || fail "fit supervisor log already exists: $SUPERVISOR_LOG"
[[ ! -e "$SUPERVISOR_PIDFILE" ]] || fail "fit supervisor pidfile already exists: $SUPERVISOR_PIDFILE"
[[ ! -e "$SUPERVISOR_STARTED" ]] || fail "fit start receipt already exists: $SUPERVISOR_STARTED"
[[ ! -e "$FIT_TIMEOUT_FILE" ]] || fail "fit timeout receipt already exists: $FIT_TIMEOUT_FILE"

for tool in bash nohup sha256sum awk date mv; do
    command -v "$tool" >/dev/null || fail "required command is unavailable: $tool"
done
[[ "$(command -v nohup)" == /usr/bin/nohup ]] || fail "unexpected nohup path: $(command -v nohup)"
actual_sha="$(sha256sum "$LAUNCH_SCRIPT" | awk '{print $1}')" || fail 'cannot hash fit launch script'
[[ "$actual_sha" == "$LAUNCH_SHA256" ]] || fail "fit launch script hash mismatch: $actual_sha"
/usr/bin/bash -n "$LAUNCH_SCRIPT" || fail 'fit launch script failed bash -n'

printf '%s\n' "$FIT_TIMEOUT_SECONDS" > "$FIT_TIMEOUT_FILE"
nohup /usr/bin/bash "$LAUNCH_SCRIPT" "$FIT_TIMEOUT_SECONDS" > "$SUPERVISOR_LOG" 2>&1 < /dev/null &
supervisor_pid=$!
pid_tmp="${SUPERVISOR_PIDFILE}.tmp.$$"
printf '%s\n' "$supervisor_pid" > "$pid_tmp" || fail 'cannot write fit supervisor pidfile'
mv "$pid_tmp" "$SUPERVISOR_PIDFILE" || fail 'cannot publish fit supervisor pidfile'
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$SUPERVISOR_STARTED"
printf 'fit_timeout_seconds=%s\n' "$FIT_TIMEOUT_SECONDS"
printf 'supervisor_pid=%s\n' "$supervisor_pid"
printf 'launch_script_sha256=%s\n' "$actual_sha"
