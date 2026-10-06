#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave18"
while systemctl --user is-active --quiet wunder-wave15-composite256-3 || systemctl --user is-active --quiet wunder-wave15-composite256-4; do sleep 10; done
WNN_CPU_PCT=6 WNN_MEM_MAX=24G scripts/limited .venv/bin/python "$D/prepare_composite_teachers.py" > "$D/prepare-composite-teachers.log" 2>&1
systemd-run --user --unit=wunder-wave18-composite-student-3 --property=RuntimeMaxSec=10800 /bin/bash "$D/composite_student_lane.sh" 3 2
systemd-run --user --unit=wunder-wave18-composite-student-4 --property=RuntimeMaxSec=10800 /bin/bash "$D/composite_student_lane.sh" 4 4
