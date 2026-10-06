cd ~/WunderFunder-creative; D=research/iterations/2026-10-02-slow-fast
until [ -f $D/ds_cos_f0.npy ] && [ -f $D/ds_cos_f1.npy ]; do
  grep -q Traceback artifacts/logs/ds_score_f?.log && { echo FAILED; tail -5 artifacts/logs/ds_score_f?.log; exit 1; }; sleep 60; done
tail -n 1 artifacts/logs/ds_score_f0.log artifacts/logs/ds_score_f1.log
.venv/bin/python - <<'PY'
import numpy as np
D="research/iterations/2026-10-02-slow-fast/"
for f in (0,1):
    c=np.load(D+f"ds_cos_f{f}.npy");ok=~np.isnan(c);r=np.zeros(len(c));r[ok]=np.argsort(np.argsort(c[ok]))/(ok.sum()-1)
    drop=np.where(ok&(r>=0.2),1.0,0.0);soft=np.where(ok,0.25+1.5*r,0.0);uni=np.where(ok,1.0,0.0)
    for name,w in (("drop20",drop),("soft",soft),("uni",uni)):np.savez(D+f"seqw_ds{name}_f{f}.npz",train=w,valid=np.ones(1873))
    print(f,"cos pct",np.nanpercentile(c,[5,20,50,80,95]).round(3))
PY
L(){ n=$1 f=$2 g=$3; shift 3; WNN_GPU_FRACTION=0.45 setsid nohup scripts/fold_job.sh $n $f $g "$@" > /dev/null 2>&1 < /dev/null & }
for f in 0 1; do
 L dsDrop20 $f $f --optimizer muon --muon-lr-mult 6 --seq-weight $D/seqw_dsdrop20_f$f.npz
 L dsSoft $f $((2+2*f)) --optimizer muon --muon-lr-mult 6 --seq-weight $D/seqw_dssoft_f$f.npz
 L dsU $f $((5+f)) --optimizer muon --muon-lr-mult 6 --seq-weight $D/seqw_dsuni_f$f.npz
done
while [ $(grep -l DONE artifacts/logs/{dsDrop20,dsSoft,dsU}_f[01].log 2>/dev/null | wc -l) -lt 6 ]; do
  grep -q Traceback artifacts/logs/{dsDrop20,dsSoft,dsU}_f[01].log 2>/dev/null && echo FAILED; sleep 60; done
cd $D && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py muon6 muon6s4 dsU dsDrop20 dsSoft
