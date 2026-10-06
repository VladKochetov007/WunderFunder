cd ~/WunderFunder-creative
while [ $(grep -l DONE artifacts/logs/i1sF_f[01].log 2>/dev/null | wc -l) -lt 2 ]; do
  if grep -q "Traceback" artifacts/logs/i1sF_f[01].log 2>/dev/null; then echo FAILED; grep -A5 Traceback artifacts/logs/i1sF_f[01].log | tail -20; exit 1; fi
  sleep 60; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py ref refs4 i1sA i1sF
