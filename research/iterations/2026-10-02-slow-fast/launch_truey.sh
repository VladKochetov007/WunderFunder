cd ~/WunderFunder-creative
.venv/bin/python research/iterations/2026-10-02-slow-fast/patch_truey.py || exit 1
for f in 0 1; do
 WNN_GPU_FRACTION=0.3 setsid nohup scripts/fold_job.sh i0ty $f 3 --i1swap 4557 --i1swap-noswap --i1swap-truey > /dev/null 2>&1 < /dev/null &
done
