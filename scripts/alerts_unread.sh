#!/usr/bin/env bash
# Print watchdog ALERT/RECOVERED rows not yet seen by the session; advances a cursor. Run at the start of every loop pass.
set -u
F="$(dirname "$0")/../research/campaign_2026-10-06/bandit_continuation/watchdog/ALERTS.jsonl"
C="$(dirname "$0")/../.runtime/alerts_cursor"
mkdir -p "$(dirname "$C")"
n=$(wc -l < "$F"); seen=$(cat "$C" 2>/dev/null || echo 0)
[ "$seen" -gt "$n" ] && seen=0
tail -n +$((seen+1)) "$F" | python3 -c '
import json
import sys

for line in sys.stdin:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    timestamp = str(row.get("utc") or row.get("ts") or "")[:19]
    event = str(row.get("event") or row.get("kind") or "")
    severity = str(row.get("severity") or row.get("level") or "")
    label = " ".join(part for part in (event, severity) if part)
    key = str(row.get("key") or row.get("id") or "")
    message = str(row.get("message") or row.get("msg") or "")[:170]
    print(timestamp, label, key, "|", message)
'
echo "$n" > "$C"
