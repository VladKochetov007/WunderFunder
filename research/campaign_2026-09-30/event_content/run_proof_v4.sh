#!/usr/bin/env bash
set -u
own=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-09-30/event_content
repo=/home/vlad/development/wunder/wnn_connectome_starterpack
cd "$repo" || exit 2
date --iso-8601=ns > "$own/start_v4.txt"
"$repo/scripts/limited" "$repo/.venv/bin/python" "$own/proof.py"
code=$?
printf '%s\n' "$code" > "$own/exit_v4.txt"
date --iso-8601=ns > "$own/end_v4.txt"
exit "$code"
