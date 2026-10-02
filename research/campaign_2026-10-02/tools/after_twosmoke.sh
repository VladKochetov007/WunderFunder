#!/usr/bin/env bash
# When the twoclock smoke job finishes with exit 0, append its two ep2 fold screens to the end of queue6.
cd "$(dirname "$0")/../../.."
rec=research/campaign_2026-09-29/jobs/c20261002_dtwosmoke_f0.json
until [[ -f $rec ]] && python3 -c "import json,sys;d=json.load(open('$rec'));sys.exit(d['status']=='running')"; do sleep 30; done
if python3 -c "import json,sys;d=json.load(open('$rec'));sys.exit(0 if d.get('exit_code')==0 else 1)"; then
  printf "c20261002_dtwo_f0 6600 8 research/campaign_2026-10-01/jobs/dtwo_fold.sh 0 0 screen\nc20261002_dtwo_f1 6600 8 research/campaign_2026-10-01/jobs/dtwo_fold.sh 1 0 screen\n" >> research/campaign_2026-10-01/jobs/queue6.txt
  echo "APPENDED twoclock screens"
else
  echo "SMOKE FAILED: $(python3 -c "import json;d=json.load(open('$rec'));print(d.get('exit_code'), d.get('failure'))")"
fi
