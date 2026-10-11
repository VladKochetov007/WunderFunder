#!/usr/bin/env bash
set -uo pipefail
umask 077

LAUNCH_SCRIPT=/root/run/H391_EXPORT_LAUNCH_R1.sh
LAUNCH_SHA256=bd489d9ca7ee8091b9a00cdc8f8256a7a7d82fa170b8202d8cde867ff60958ff
RUN_DIR=/root/run/h391_export_r1
SUPERVISOR_LOG="$RUN_DIR/supervisor.log"
SUPERVISOR_PIDFILE="$RUN_DIR/supervisor.pid"
SUPERVISOR_STARTED="$RUN_DIR/supervisor_started_utc.txt"

fail() {
    printf 'H391_EXPORT_R1_START_FAILED: %s\n' "$*" >&2
    exit 70
}

for tool in bash nohup sha256sum awk date mv mkdir; do
    command -v "$tool" >/dev/null || fail "required command is unavailable: $tool"
done
[[ "$(command -v nohup)" == /usr/bin/nohup ]] || fail 'unexpected nohup path'
[[ -f "$LAUNCH_SCRIPT" && ! -L "$LAUNCH_SCRIPT" ]] || fail "missing or symlinked launch script: $LAUNCH_SCRIPT"
actual_sha="$(sha256sum "$LAUNCH_SCRIPT" | awk '{print $1}')" || fail 'cannot hash launch script'
[[ "$actual_sha" == "$LAUNCH_SHA256" ]] || fail "launch script digest mismatch: $actual_sha"
/usr/bin/bash -n "$LAUNCH_SCRIPT" || fail 'launch script failed bash -n'
[[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || fail "unique export receipt directory already exists: $RUN_DIR"
mkdir -m 700 "$RUN_DIR" || fail "cannot create unique export receipt directory: $RUN_DIR"

{
    printf 'run=vt_f1_h391_dsq_h384_t3_a9_e6_s3_r2\nepoch=5\nphysical_gpu=0\nlaunch_script_sha256=%s\n' "$actual_sha"
    date -u +'%Y-%m-%dT%H:%M:%SZ'
} > "$RUN_DIR/start_intent.txt" || fail 'cannot write start intent receipt'

nohup /usr/bin/bash "$LAUNCH_SCRIPT" > "$SUPERVISOR_LOG" 2>&1 < /dev/null &
supervisor_pid=$!
pid_tmp="$SUPERVISOR_PIDFILE.tmp.$$"
printf '%s\n' "$supervisor_pid" > "$pid_tmp" || fail 'cannot write supervisor PID'
mv "$pid_tmp" "$SUPERVISOR_PIDFILE" || fail 'cannot publish supervisor PID'
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$SUPERVISOR_STARTED" || fail 'cannot record supervisor start time'
printf 'run_dir=%s\nsupervisor_pid=%s\nlaunch_script_sha256=%s\n' "$RUN_DIR" "$supervisor_pid" "$actual_sha"
