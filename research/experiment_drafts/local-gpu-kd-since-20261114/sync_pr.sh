#!/usr/bin/env bash
# Copy this live folder into the research-team draft worktree, commit and push. Usage: sync_pr.sh "message"
set -euo pipefail
O=$(dirname "$(readlink -f "$0")"); W=/home/vlad/development/wunder/research_tmp/wt_local1114
D=$W/research/experiment_drafts/local-gpu-kd-since-20261114
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python "$O/make_results.py"
mkdir -p "$D"
cp "$O"/*.md "$O"/*.sh "$O"/*.py "$O"/queue.txt "$O"/queue.log "$O"/*.log "$D"/
[ -e "$O/thermal_guard.log" ] && cp "$O/thermal_guard.log" "$D"/
cd "$W" && git add -A research/experiment_drafts && git diff --cached --quiet && { echo "no changes"; exit 0; }
git commit -q -m "$1

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>" && git push -q origin HEAD && git log -1 --format='%h %s'
