cd ~/WunderFunder-creative
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo
L(){ n=$1 f=$2 g=$3; shift 3; WNN_GPU_FRACTION=0.3 setsid nohup scripts/fold_job.sh $n $f $g "$@" > /dev/null 2>&1 < /dev/null & }
for f in 0 1; do
 L drop1 $f 3 --dropout 0.1
 L drop2 $f 3 --dropout 0.2
 L dist2 $f 5 --distill h256_f$f,h256s4_f$f
 L dist4 $f 6 --distill h256_f$f,h256s4_f$f,d256_f$f,d256s4_f$f
done
