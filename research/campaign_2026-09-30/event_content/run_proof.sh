#!/usr/bin/env bash
set -u
own=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-09-30/event_content
repo=/home/vlad/development/wunder/wnn_connectome_starterpack
cd "$repo" || exit 2
date --iso-8601=ns > "$own/start_v2.txt"
cp "$own/results_v1.json" "$own/results_v1_preserved.json"
"$repo/scripts/limited" "$repo/.venv/bin/python" "$own/proof.py"
code=$?
mv "$own/results_v1.json" "$own/results_v2.json"
cp "$own/results_v1_preserved.json" "$own/results_v1.json"
printf '%s\n' "$code" > "$own/exit_v2.txt"
date --iso-8601=ns > "$own/end_v2.txt"
exit "$code"
