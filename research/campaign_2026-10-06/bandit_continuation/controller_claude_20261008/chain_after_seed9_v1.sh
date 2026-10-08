#!/usr/bin/env bash
# Durable chain: wait for the seed9 parent attempt to end, reclaim slice file cache, launch the guarded local warm queue worker.
set -uo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
ST=$ROOT/research/campaign_2026-10-06/bandit_continuation/runtime
C=$ROOT/research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008
G=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice
LOG=$C/chain_after_seed9_v1.log
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
att=$ST/attempts/h230-strong-parent-seed9-muon-local-v1-20261008.json
for i in $(seq 1 360); do
  st=$($ROOT/.venv/bin/python -c "import json;print(json.load(open('$att')).get('status'))")
  [[ $st != running && $st != reserved ]] && break
  sleep 60
done
log "seed9 status=$st"
for try in $(seq 1 15); do
  cur=$(cat $G/memory.current); file=$(awk '$1=="file"{print $2}' $G/memory.stat)
  want=$(( cur - 8000000000 )); (( want > file - 3000000000 )) && want=$(( file - 3000000000 ))
  if (( want > 0 )); then echo "$want swappiness=0" > $G/memory.reclaim 2>>"$LOG"; fi
  log "reclaim try=$try before=$cur requested=$want after=$(cat $G/memory.current)"
  if WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=300s \
     $ROOT/scripts/limited $ROOT/.venv/bin/python $ROOT/scripts/bandit_resources.py --root $ROOT --state $ST launch \
     --attempt warm-queue-local-v1-20261008 --kind gpu --cpus 2 --reserve-bytes 9000000000 \
     --profile H230-ALLDATA-STUDENT-GRU256-SEED3-FIT \
     --contract $C/WARM_QUEUE_LOCAL_V1_ROOT_AUTHORIZATION_20261008_R1.json --admission-timeout 1800 -- \
     /usr/bin/timeout --signal=TERM --kill-after=30s 21600s $ROOT/.venv/bin/python $ROOT/scripts/warm_queue_worker.py \
     --queue $ST/warm_queue_local_v1 --poll-seconds 20 --idle-exit-seconds 1800 >> "$LOG" 2>&1; then
    log "launched warm queue worker"; exit 0
  fi
  log "launch refused; retry in 120s"; sleep 120
done
log "giving up after retries"; exit 1
