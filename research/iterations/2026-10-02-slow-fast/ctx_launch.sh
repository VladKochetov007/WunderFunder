cd ~/WunderFunder-creative; D=research/iterations/2026-10-02-slow-fast
until grep -q "^done fold" artifacts/logs/ctx_teacher_f0.log 2>/dev/null && grep -q "^done fold" artifacts/logs/ctx_teacher_f1.log 2>/dev/null; do
  grep -q Traceback artifacts/logs/ctx_teacher_f?.log && { echo FAILED; tail -5 artifacts/logs/ctx_teacher_f?.log; exit 1; }; sleep 60; done
tail -qn 1 artifacts/logs/ctx_teacher_f0.log artifacts/logs/ctx_teacher_f1.log
L(){ n=$1 f=$2 g=$3; shift 3; WNN_GPU_FRACTION=0.45 setsid nohup scripts/fold_job.sh $n $f $g "$@" > /dev/null 2>&1 < /dev/null & }
for f in 0 1; do
 L ctxW $f $([ $f = 0 ] && echo 5 || echo 6) --optimizer muon --muon-lr-mult 6 --distill muwarm_f$f --distill-alpha 0.5
 L ctxC $f 3 --optimizer muon --muon-lr-mult 6 --distill mucold_f$f --distill-alpha 0.5
done
while [ $(grep -l DONE artifacts/logs/{ctxW,ctxC}_f[01].log 2>/dev/null | wc -l) -lt 4 ]; do
  grep -q Traceback artifacts/logs/{ctxW,ctxC}_f[01].log 2>/dev/null && { echo FAILED; grep -A5 Traceback artifacts/logs/{ctxW,ctxC}_f[01].log | tail -10; exit 1; }; sleep 60; done
cd $D && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py muon6 muon6s4 ctxC ctxW
cd ~/WunderFunder-creative/$D && sed -e 's/exec(open("mix.py").read().split("def wp(")\[0\])/exec(open("mix.py").read().split("def wp(")[0])/' position.py > /dev/null
for r in ctxC ctxW; do sed "s/load(\"d256\")/load(\"$r\")/" position.py > position_$r.py; echo $r; ~/WunderFunder-creative/.venv/bin/python position_$r.py | head -3; done
