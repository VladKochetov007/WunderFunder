cd ~/WunderFunder-creative; S=research/iterations/2026-10-02-slow-fast/final_mu.sh; D=research/iterations/2026-10-02-slow-fast; chmod +x $S
pkill -u $USER -f "name dbgQ" ; rm -rf artifacts/runs/dbgQ artifacts/logs/dbgQ.log
for n in fmuQ_s3 fmuQ_s4 fmuK_s3 fmuK_s4; do mv artifacts/logs/$n.log artifacts/logs/$n.broken.log 2>/dev/null; done
EXTRA_ARGS="--seq-repeat $D/rep_quiet.npz" setsid nohup $S fmuQ_s3 3 2 h128 > /dev/null 2>&1 < /dev/null &
EXTRA_ARGS="--seq-repeat $D/rep_quiet.npz" setsid nohup $S fmuQ_s4 4 3 h128 > /dev/null 2>&1 < /dev/null &
EXTRA_ARGS="--seq-repeat $D/rep_classmix.npz" setsid nohup $S fmuK_s3 3 4 h128 > /dev/null 2>&1 < /dev/null &
EXTRA_ARGS="--seq-repeat $D/rep_classmix.npz" setsid nohup $S fmuK_s4 4 5 h128 > /dev/null 2>&1 < /dev/null &
