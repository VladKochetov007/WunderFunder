cd ~/WunderFunder-creative
for x in drop1_f0 drop1_f1 drop2_f1; do mv artifacts/logs/$x.log artifacts/logs/$x.oom.log; done
L(){ n=$1 f=$2 g=$3; shift 3; WNN_GPU_FRACTION=0.45 setsid nohup scripts/fold_job.sh $n $f $g "$@" > /dev/null 2>&1 < /dev/null & }
L drop1 0 0 --dropout 0.1; L drop1 1 1 --dropout 0.1; L drop2 1 2 --dropout 0.2
