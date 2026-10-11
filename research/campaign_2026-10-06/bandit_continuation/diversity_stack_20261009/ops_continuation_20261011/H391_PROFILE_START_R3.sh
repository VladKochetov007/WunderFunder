#!/usr/bin/env bash
set -uo pipefail
umask 077

LAUNCH_SCRIPT=/root/run/H391_PROFILE_LAUNCH_R3.sh
LAUNCH_SHA256=1081f02497f46df36b2ed028d8f04a22b0f22e2ee6c3b4cce7c601f8f5afb516
RUN_DIR=/root/run/h391_profile_r3
SUPERVISOR_LOG=/root/run/h391_profile_r3_supervisor.out
SUPERVISOR_PIDFILE=/root/run/h391_profile_r3_supervisor.pid
SUPERVISOR_STARTED=/root/run/h391_profile_r3_supervisor_started_utc.txt

fail() {
    printf 'H391_PROFILE_R3_START_FAILED: %s\n' "$*" >&2
    exit 70
}

[[ -f "$LAUNCH_SCRIPT" ]] || fail "missing launch script: $LAUNCH_SCRIPT"
[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
[[ ! -e "$SUPERVISOR_LOG" ]] || fail "supervisor log already exists: $SUPERVISOR_LOG"
[[ ! -e "$SUPERVISOR_PIDFILE" ]] || fail "supervisor pidfile already exists: $SUPERVISOR_PIDFILE"
[[ ! -e "$SUPERVISOR_STARTED" ]] || fail "supervisor start receipt already exists: $SUPERVISOR_STARTED"

for tool in bash nohup sha256sum awk date mv; do
    command -v "$tool" >/dev/null || fail "required command is unavailable: $tool"
done
[[ "$(command -v nohup)" == /usr/bin/nohup ]] || fail "unexpected nohup path: $(command -v nohup)"
actual_sha="$(sha256sum "$LAUNCH_SCRIPT" | awk '{print $1}')" || fail 'cannot hash launch script'
[[ "$actual_sha" == "$LAUNCH_SHA256" ]] || fail "launch script hash mismatch: $actual_sha"
/usr/bin/bash -n "$LAUNCH_SCRIPT" || fail 'launch script failed bash -n'

nohup /usr/bin/bash "$LAUNCH_SCRIPT" > "$SUPERVISOR_LOG" 2>&1 < /dev/null &
supervisor_pid=$!
pid_tmp="${SUPERVISOR_PIDFILE}.tmp.$$"
printf '%s\n' "$supervisor_pid" > "$pid_tmp" || fail 'cannot write supervisor pidfile'
mv "$pid_tmp" "$SUPERVISOR_PIDFILE" || fail 'cannot publish supervisor pidfile'
date -u +'%Y-%m-%dT%H:%M:%SZ' > /root/run/h391_profile_r3_supervisor_started_utc.txt
printf 'supervisor_pid=%s\n' "$supervisor_pid"
printf 'launch_script_sha256=%s\n' "$actual_sha"
