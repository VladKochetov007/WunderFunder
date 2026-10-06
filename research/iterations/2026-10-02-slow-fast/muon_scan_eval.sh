cd ~/WunderFunder-creative
while [ $(grep -l DONE artifacts/logs/muon2s4_f[01].log artifacts/logs/muon3_f[01].log artifacts/logs/muon4_f[01].log 2>/dev/null | wc -l) -lt 6 ]; do
  if grep -q Traceback artifacts/logs/muon2s4_f[01].log artifacts/logs/muon3_f[01].log artifacts/logs/muon4_f[01].log; then echo FAILED; fi; sleep 60; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py ref refs4 muon muon2 muon3 muon4 muon2s4
