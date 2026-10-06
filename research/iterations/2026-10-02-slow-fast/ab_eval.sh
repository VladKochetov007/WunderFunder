cd ~/WunderFunder-creative
while [ $(grep -l DONE artifacts/logs/drop1_f[01].log artifacts/logs/drop2_f[01].log artifacts/logs/dist2_f[01].log artifacts/logs/dist4_f[01].log 2>/dev/null | wc -l) -lt 8 ]; do
  if grep -q "Traceback" artifacts/logs/drop1_f[01].log artifacts/logs/drop2_f[01].log artifacts/logs/dist2_f[01].log artifacts/logs/dist4_f[01].log 2>/dev/null; then echo FAILED; grep -A5 Traceback artifacts/logs/drop1_f[01].log artifacts/logs/drop2_f[01].log artifacts/logs/dist2_f[01].log artifacts/logs/dist4_f[01].log | tail -20; exit 1; fi
  sleep 60; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py ref refs4 drop1 drop2 dist2 dist4
