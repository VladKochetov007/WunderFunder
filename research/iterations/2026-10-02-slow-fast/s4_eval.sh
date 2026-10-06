cd ~/WunderFunder-creative
while [ $(grep -l DONE artifacts/logs/qw0s4_f[01].log artifacts/logs/adems4_f[01].log 2>/dev/null | wc -l) -lt 4 ]; do
  if grep -q "Traceback" artifacts/logs/qw0s4_f[01].log artifacts/logs/adems4_f[01].log 2>/dev/null; then echo FAILED; grep -A5 Traceback artifacts/logs/qw0s4_f[01].log artifacts/logs/adems4_f[01].log | tail -20; exit 1; fi
  sleep 60; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py qw0 adem qw0s4 adems4 adrop
