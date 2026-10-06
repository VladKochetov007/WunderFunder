#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave22"
while systemctl --user is-active --quiet wunder-wave22-parent-3 || systemctl --user is-active --quiet wunder-wave22-parent-4; do sleep 10; done
WNN_CPU_PCT=6 WNN_MEM_MAX=24G scripts/limited .venv/bin/python "$D/prepare_teacher.py" > "$D/prepare-teacher.log" 2>&1
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$PWD/research/iterations/2026-10-05-wave6/scripts:$HOME/WunderFunder/src"
WNN_CPU_PCT=6 WNN_MEM_MAX=16G scripts/limited .venv/bin/python "$D/widen.py" > "$D/widen.log" 2>&1
systemd-run --user --unit=wunder-wave22-wide-3 --property=RuntimeMaxSec=14400 /bin/bash "$D/student_lane.sh" 3 5 wide
systemd-run --user --unit=wunder-wave22-wide-4 --property=RuntimeMaxSec=14400 /bin/bash "$D/student_lane.sh" 4 6 wide
systemd-run --user --unit=wunder-wave22-control --property=RuntimeMaxSec=18000 /bin/bash "$D/control_pair.sh"
