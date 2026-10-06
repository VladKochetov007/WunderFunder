#!/usr/bin/env bash
set -euo pipefail
D="$HOME/WunderFunder-creative/research/iterations/2026-10-05-wave22"
for seed in 3 4; do /bin/bash "$D/student_lane.sh" "$seed" 1 control; done
