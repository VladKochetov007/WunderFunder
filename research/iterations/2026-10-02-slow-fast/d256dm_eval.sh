cd ~/WunderFunder-creative
while [ $(grep -l DONE artifacts/logs/d256dm_f[01].log 2>/dev/null | wc -l) -lt 2 ]; do
  if grep -q "Traceback" artifacts/logs/d256dm_f[01].log 2>/dev/null; then echo FAILED; grep -A5 Traceback artifacts/logs/d256dm_f[01].log | tail -20; exit 1; fi
  sleep 60; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py d256 d256s4 d256dm adem
