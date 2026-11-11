#!/usr/bin/env bash
# Finish the teacher track (user: finish current ideas, then pause): wait for the rep-teacher
# chain, then the two declared seed-3 distillation pilots D1 (alpha 0.9) and D2 (alpha 0.6).
# Each fit's runtime args are checked from its log header; a mismatch is reported for manual stop.
set -uo pipefail
O=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-29/teacher_track_r1
C=/home/vlad/development/wunder/research_tmp/chrono_root/cache
while systemctl --user is-active -q wnn-tt29-rep; do sleep 60; done
[ -f $C/pred_tt29rep_fbc1k_s5_train.npy ] || { echo "FINISH_ABORT teacher train export missing"; exit 1; }
for spec in "d1 0.9" "d2 0.6"; do
  set -- $spec; arm=$1; a=$2
  $O/run_child.sh 3 $arm --distill tt29rep_fbc1k_s5 --distill-alpha $a &
  pid=$!; log=$O/tt29_child_${arm}_s3.fit.log
  for _ in $(seq 60); do [ "$(wc -l < $log 2>/dev/null || echo 0)" -ge 2 ] && break; sleep 15; done
  if rg -q "\"distill\": \"tt29rep_fbc1k_s5\"" $log && rg -q "\"distill_alpha\": $a" $log && rg -q '"init": "tt29_parent_s3@ep3"' $log; then
    echo "CONFIG_OK $arm alpha=$a"
  else
    echo "CONFIG_MISMATCH $arm (fit left running for manual stop)"
  fi
  wait $pid
done
echo FINISH_DONE
