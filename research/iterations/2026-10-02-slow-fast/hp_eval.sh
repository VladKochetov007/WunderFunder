cd ~/WunderFunder-creative
while [ $(grep -l DONE artifacts/logs/hp_*_f[01].log 2>/dev/null | wc -l) -lt 12 ]; do
  if grep -q Traceback artifacts/logs/hp_*_f[01].log; then echo FAILED; grep -l Traceback artifacts/logs/hp_*_f[01].log; fi
  sleep 60; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py ref refs4 hp_lr2 hp_lr05 hp_bs64 hp_wu20 hp_b299 hp_clip03
