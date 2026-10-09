#!/usr/bin/env bash
# Re-grade a fold's prediction bank when the set of produced prediction files changes.
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
BANKDIR=$ROOT/research/campaign_2026-10-06/bandit_continuation/prediction_bank
for spec in "1 BANK_FOLD1.jsonl fold1" "0 BANK.jsonl fold0"; do
  set -- $spec; fold=$1; bank=$BANKDIR/$2; split=$ROOT/research/campaign_2026-09-29/artifacts/splits/$3.json
  key=$(python3 -c "import json,os,sys;r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()];print(sorted((x['name'],os.path.getmtime(os.path.join(sys.argv[2],x['prediction']))) for x in r if os.path.isfile(os.path.join(sys.argv[2],x['prediction']))))" "$bank" "$ROOT" | sha256sum | cut -c1-16)
  state=$BANKDIR/.graded_fold${fold}
  [ -f "$state" ] && [ "$(cat "$state")" = "$key" ] && continue
  out=$BANKDIR/GRADE_FOLD${fold}_$(date -u +%Y%m%dT%H%M%SZ)_${key}.json
  "$ROOT/.venv/bin/python" "$ROOT/scripts/grade_prediction_bank.py" --bank "$bank" --fold "$fold" --split "$split" --output "$out" --skip-missing >"$out.stdout" 2>&1
  cp "$out" "$BANKDIR/LATEST_GRADE_FOLD${fold}.json"
  echo "$key" >"$state"
done
