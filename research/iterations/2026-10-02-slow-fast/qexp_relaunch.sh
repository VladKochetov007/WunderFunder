cd ~/WunderFunder-creative
pkill -u $USER -f "fold_job.sh qexp"; pkill -u $USER -f "name qexp_f"; sleep 8
pgrep -u $USER -af "qexp" | grep -v qexp_relaunch | head -3
for f in 0 1; do [ -f artifacts/logs/qexp_f$f.log ] && mv artifacts/logs/qexp_f$f.log artifacts/logs/qexp_f$f.bad.log; done
cd research/iterations/2026-10-02-slow-fast
~/WunderFunder-creative/.venv/bin/python - <<'PY'
import numpy as np
R="/home/grinchenkoeg/WunderFunder-creative/"
cls=np.load(R+"research/artifacts/market-classes-v1.npz");Z=np.load("quiet_z.npz");W={}
for sp in ("train","valid"):
    w=np.clip(np.exp(-3*Z[sp]),0.05,20);w[cls[sp]!=3]=0.0;W[sp]=w
    c3=cls[sp]==3;print(sp,"class3 weight pct",np.percentile(w[c3],[10,50,90]).round(2),"nonzero non-c3",(w[~c3]>0).sum())
np.savez("seqw_q3.npz",**W)
PY
cd ~/WunderFunder-creative; D=research/iterations/2026-10-02-slow-fast
for f in 0 1; do g=$((5+f)); WNN_GPU_FRACTION=0.3 nohup scripts/fold_job.sh qexp $f $g --seq-weight $D/seqw_q3.npz --prop-cache --a-mode demean --train-classes 3 > /dev/null 2>&1 & done
sleep 60; tail -qn 1 artifacts/logs/qexp_f0.log; grep -c "seqw_q3" artifacts/logs/qexp_f0.log
