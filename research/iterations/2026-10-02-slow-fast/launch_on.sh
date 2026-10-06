cd ~/WunderFunder-creative; D=research/iterations/2026-10-02-slow-fast
EXTRA_ARGS="--online-norm" setsid nohup $D/final_mu.sh fmuON_s3 3 0 h128 > /dev/null 2>&1 < /dev/null &
EXTRA_ARGS="--online-norm" setsid nohup $D/final_mu.sh fmuON_s4 4 0 h128 > /dev/null 2>&1 < /dev/null &
sed "s/\$2>7500/\$2>4500/" $D/gpu_queue.sh > $D/gpu_queue_small.sh; chmod +x $D/gpu_queue_small.sh
printf "muON 0 --optimizer muon --muon-lr-mult 6 --online-norm\nmuON 1 --optimizer muon --muon-lr-mult 6 --online-norm\nmuONs4 0 SEED=4 --optimizer muon --muon-lr-mult 6 --online-norm\nmuONs4 1 SEED=4 --optimizer muon --muon-lr-mult 6 --online-norm\n" > $D/queue.txt
setsid nohup $D/gpu_queue_small.sh > artifacts/logs/gpu_queue6.log 2>&1 < /dev/null &
