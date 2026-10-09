#!/usr/bin/env bash
# Re-grade a fold's prediction bank when the set of produced prediction files changes.
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
BANKDIR=$ROOT/research/campaign_2026-10-06/bandit_continuation/prediction_bank
for spec in "1 BANK_FOLD1.jsonl fold1 FOLD1" "0 BANK_FOLD0_FULL.jsonl fold0 FOLD0FULL" "0 BANK.jsonl fold0 FOLD0" "1 BANK.jsonl fold1 SMALLFOLD1" "3 BANK.jsonl fold3 SMALLFOLD3"; do
  set -- $spec; fold=$1; bank=$BANKDIR/$2; split=$ROOT/research/campaign_2026-09-29/artifacts/splits/$3.json; tag=$4
  python3 -c "import json,sys;sys.exit(0 if any(json.loads(l)['fold']==int(sys.argv[2]) for l in open(sys.argv[1]) if l.strip()) else 1)" "$bank" "$fold" || continue
  key=$(python3 -c "import json,os,sys;r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()];print(sorted((x['name'],os.path.getmtime(os.path.join(sys.argv[2],x['prediction']))) for x in r if os.path.isfile(os.path.join(sys.argv[2],x['prediction']))))" "$bank" "$ROOT" | sha256sum | cut -c1-16)
  state=$BANKDIR/.graded_${tag}
  [ -f "$state" ] && [ "$(cat "$state")" = "$key" ] && continue
  out=$BANKDIR/GRADE_${tag}_$(date -u +%Y%m%dT%H%M%SZ)_${key}.json
  "$ROOT/.venv/bin/python" "$ROOT/scripts/grade_prediction_bank.py" --bank "$bank" --fold "$fold" --split "$split" --output "$out" --skip-missing >"$out.stdout" 2>&1
  cp "$out" "$BANKDIR/LATEST_GRADE_${tag}.json"
  echo "$key" >"$state"
done
# Cross-fold blend transfer runs sequentially after grading (never concurrently with it).
cf_key=$(cat "$BANKDIR/.graded_FOLD0FULL" "$BANKDIR/.graded_FOLD1" 2>/dev/null | sha256sum | cut -c1-16)
if [ "$(cat "$BANKDIR/.crossfold_key" 2>/dev/null)" != "$cf_key" ]; then
  out=$BANKDIR/CROSSFOLD_F0_F1_$(date -u +%Y%m%dT%H%M%SZ)_${cf_key}.json
  "$ROOT/.venv/bin/python" "$ROOT/scripts/crossfold_blend.py" --bank-a "$BANKDIR/BANK_FOLD0_FULL.jsonl" --fold-a 0 \
    --bank-b "$BANKDIR/BANK_FOLD1.jsonl" --fold-b 1 --output "$out" >"$out.stdout" 2>&1 && cp "$out" "$BANKDIR/LATEST_CROSSFOLD_F0_F1.json"
  echo "$cf_key" >"$BANKDIR/.crossfold_key"
fi
