#!/usr/bin/env bash
set -u
own=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-09-30/event_content/exposure
repo=/home/vlad/development/wunder/wnn_connectome_starterpack
cd "$repo" || exit 2
date --iso-8601=ns > "$own/start.txt"
"$repo/scripts/limited" "$repo/.venv/bin/python" "$own/probe.py"
code=$?
printf '%s\n' "$code" > "$own/exit.txt"
date --iso-8601=ns > "$own/end.txt"
exit "$code"
