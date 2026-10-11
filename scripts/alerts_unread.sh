#!/usr/bin/env bash
# Print watchdog ALERT/RECOVERED rows not yet seen by the session; advances a cursor. Run at the start of every loop pass.
set -u
F="$(dirname "$0")/../research/campaign_2026-10-06/bandit_continuation/watchdog/ALERTS.jsonl"
C="$(dirname "$0")/../.runtime/alerts_cursor"
mkdir -p "$(dirname "$C")"
n=$(wc -l < "$F"); seen=$(cat "$C" 2>/dev/null || echo 0)
[ "$seen" -gt "$n" ] && seen=0
tail -n +$((seen+1)) "$F" | python3 -c "
import sys,json
for l in sys.stdin:
    try: d=json.loads(l)
    except Exception: continue
    print(d.get('ts','')[:19], d.get('level',d.get('kind','')), d.get('key',d.get('id','')), '|', str(d.get('message',d.get('msg','')))[:170])
"
echo "$n" > "$C"
