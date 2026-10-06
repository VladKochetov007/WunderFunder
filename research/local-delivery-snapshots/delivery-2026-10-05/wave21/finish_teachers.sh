#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave21"
while systemctl --user is-active --quiet wunder-wave21-teacher-3 || systemctl --user is-active --quiet wunder-wave21-teacher-4; do sleep 10; done
WNN_CPU_PCT=6 WNN_MEM_MAX=24G scripts/limited .venv/bin/python "$D/prepare_teachers.py" > "$D/prepare.log" 2>&1
systemd-run --user --unit=wunder-wave21-final-student-5 --property=RuntimeMaxSec=10800 /bin/bash "$D/final_student_lane.sh" 5 0
systemd-run --user --unit=wunder-wave21-final-student-6 --property=RuntimeMaxSec=10800 /bin/bash "$D/final_student_lane.sh" 6 4
