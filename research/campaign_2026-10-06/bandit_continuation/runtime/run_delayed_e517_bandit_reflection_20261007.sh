#!/usr/bin/env bash
set -uo pipefail

repo=/home/vlad/development/wunder/wnn_connectome_starterpack
prompt_file="$repo/research/campaign_2026-10-06/bandit_continuation/reviews/GROK_BANDIT_REFLECTION_AFTER_E517_PROMPT_20261007.md"
output_file="$repo/research/campaign_2026-10-06/bandit_continuation/reviews/GROK_BANDIT_REFLECTION_AFTER_E517_20261007.md"
receipt_file="$repo/research/campaign_2026-10-06/bandit_continuation/runtime/REFLECTION_AFTER_E517_RECEIPT_20261007.json"
started_utc=$(date -u --iso-8601=seconds)
grok_status=0

{
    printf '# Delayed bandit-tree reflection after E-517\n\nStarted UTC: %s\n\n' "$started_utc"
    /usr/bin/timeout --signal=TERM --kill-after=10s 600s \
        /home/vlad/.grok/bin/grok \
        --cwd "$repo" \
        --model grok-4.7 \
        --reasoning-effort high \
        --no-subagents \
        --disable-web-search \
        --permission-mode plan \
        --max-turns 4 \
        --prompt-file "$prompt_file"
    grok_status=$?
    printf '\n\nGrok CLI exit status: %s\nCompleted UTC: %s\n' \
        "$grok_status" "$(date -u --iso-8601=seconds)"
} >> "$output_file" 2>&1

/usr/bin/python3 - "$receipt_file" "$started_utc" "$grok_status" "$output_file" "$prompt_file" <<'PY'
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1])
output_path = Path(sys.argv[4])
prompt_path = Path(sys.argv[5])
result_path = Path(
    "/home/vlad/development/wunder/wnn_connectome_starterpack/research/"
    "campaign_2026-10-06/bandit_continuation/disagreement_guided_H204/E517_RESULT.json"
)
e518_result_path = result_path.with_name("E518_RESULT.json")
repo_path = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
tree_config_path = repo_path / "research/bandit_tree/config.json"
tree_journal_path = repo_path / "research/bandit_tree/journal.jsonl"
experiment_ledger_path = repo_path / "research/experiments.jsonl"
hypothesis_ledger_path = repo_path / "research/hypotheses.jsonl"
current_state_path = repo_path / "specs/current-state.md"
receipt = {
    "scheduled_review": "read-only research-tree reallocation after the E-517 exposed disagreement-gate screen",
    "provider": "native Grok CLI via existing grok.com authentication",
    "model": "grok-4.7",
    "mode": "plan permission, one bounded prompt, no subagents, web disabled",
    "started_utc": sys.argv[2],
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "exit_code": int(sys.argv[3]),
    "output": str(output_path),
    "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
    "e517_result_sha256": (
        hashlib.sha256(result_path.read_bytes()).hexdigest() if result_path.exists() else None
    ),
    "e518_result_sha256": (
        hashlib.sha256(e518_result_path.read_bytes()).hexdigest() if e518_result_path.exists() else None
    ),
    "bandit_tree_config_sha256": hashlib.sha256(tree_config_path.read_bytes()).hexdigest(),
    "bandit_tree_journal_sha256": hashlib.sha256(tree_journal_path.read_bytes()).hexdigest(),
    "experiment_ledger_sha256": hashlib.sha256(experiment_ledger_path.read_bytes()).hexdigest(),
    "hypothesis_ledger_sha256": hashlib.sha256(hypothesis_ledger_path.read_bytes()).hexdigest(),
    "current_state_sha256": hashlib.sha256(current_state_path.read_bytes()).hexdigest(),
    "scientific_fit_or_submission": False,
    "allocation_changed": False,
    "numerical_prior_update": None,
    "public_gain_verified": False,
}
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
PY

exit "$grok_status"
