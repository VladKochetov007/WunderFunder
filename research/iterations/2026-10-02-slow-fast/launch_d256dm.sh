cd ~/WunderFunder-creative
for f in 0 1; do g=$([ $f = 0 ] && echo 2 || echo 4)
 WNN_GPU_FRACTION=0.45 setsid nohup scripts/fold_job.sh d256dm $f $g --hidden 256 --distill h256_f$f,h256s4_f$f --a-mode demean > /dev/null 2>&1 < /dev/null &
done
