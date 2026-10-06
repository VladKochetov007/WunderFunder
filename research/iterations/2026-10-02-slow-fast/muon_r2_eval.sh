cd ~/WunderFunder-creative
while [ $(grep -l DONE artifacts/logs/muon4s4_f[01].log artifacts/logs/muon6_f[01].log artifacts/logs/d256mu_f[01].log 2>/dev/null | wc -l) -lt 6 ]; do
  if grep -q Traceback artifacts/logs/muon4s4_f[01].log artifacts/logs/muon6_f[01].log artifacts/logs/d256mu_f[01].log; then echo FAILED; fi; sleep 60; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py ref refs4 muon2 muon2s4 muon4 muon4s4 muon6 d256 d256s4 d256mu
