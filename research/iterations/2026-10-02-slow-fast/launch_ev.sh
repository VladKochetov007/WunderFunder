cd ~/WunderFunder-creative
.venv/bin/python research/iterations/2026-10-02-slow-fast/patch_norow.py || exit 1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo
D=research/iterations/2026-10-02-slow-fast
for f in 0 1; do g=$((5+f));
 WNN_GPU_FRACTION=0.45 setsid nohup scripts/fold_job.sh evdual $f $g --seq-weight $D/seqw_b0.npz --prop-cache --a-mode demean --arch dual --event-hidden 128 > /dev/null 2>&1 < /dev/null &
 WNN_GPU_FRACTION=0.45 setsid nohup scripts/fold_job.sh evonly $f $g --seq-weight $D/seqw_b0.npz --prop-cache --a-mode demean --arch dual --event-hidden 128 --no-row-gru > /dev/null 2>&1 < /dev/null &
done
