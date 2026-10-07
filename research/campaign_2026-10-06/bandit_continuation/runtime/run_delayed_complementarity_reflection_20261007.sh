#!/usr/bin/env bash
set -uo pipefail

repo=/home/vlad/development/wunder/wnn_connectome_starterpack
prompt_file="$repo/research/campaign_2026-10-06/bandit_continuation/reviews/GROK_BANDIT_REFLECTION_AFTER_NEW_STEPS_PROMPT_20261007.md"
output_file="$repo/research/campaign_2026-10-06/bandit_continuation/reviews/GROK_BANDIT_REFLECTION_AFTER_NEW_STEPS_20261007.md"
receipt_file="$repo/research/campaign_2026-10-06/bandit_continuation/runtime/REFLECTION_AFTER_NEW_STEPS_RECEIPT_20261007.json"
started_utc=$(date -u --iso-8601=seconds)
grok_status=0

{
    printf '# Delayed bandit-tree reflection after new steps\n\nStarted UTC: %s\n\n' "$started_utc"
    /usr/bin/timeout --signal=TERM --kill-after=10s 600s \
        /home/vlad/.grok/bin/grok \
        --cwd "$repo" \
        --model grok-4.7 \
        --reasoning-effort high \
        --no-subagents \
        --disable-web-search \
        --permission-mode plan \
        --max-turns 32 \
        --single "$(cat "$prompt_file")"
    grok_status=$?
    printf '\n\nGrok CLI exit status: %s\nCompleted UTC: %s\n' \
        "$grok_status" "$(date -u --iso-8601=seconds)"
} >> "$output_file" 2>&1

/usr/bin/python3 - "$receipt_file" "$started_utc" "$grok_status" "$output_file" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
receipt = {
    "scheduled_review": "two-hour read-only bandit allocation reflection after E-508-V4/E-463-V2 and complementarity scout steps",
    "provider": "native Grok CLI via grok.com",
    "model": "grok-4.7",
    "mode": "single-turn, read-only plan permission, no subagents, web disabled",
    "started_utc": sys.argv[2],
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "exit_code": int(sys.argv[3]),
    "output": sys.argv[4],
    "scientific_fit_or_submission": False,
    "numerical_prior_update": None,
    "public_gain_verified": False,
}
path.write_text(json.dumps(receipt, indent=2) + "\n")
PY

exit "$grok_status"
