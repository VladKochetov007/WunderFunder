cd ~/WunderFunder-creative
while [ $(grep -l DONE artifacts/logs/{muon8,muon12,muon6s4,d256mu4,d256mu6}_f[01].log 2>/dev/null | wc -l) -lt 10 ]; do
  if grep -q Traceback artifacts/logs/{muon8,muon12,muon6s4,d256mu4,d256mu6}_f[01].log; then echo FAILED; grep -l Traceback artifacts/logs/{muon8,muon12,muon6s4,d256mu4,d256mu6}_f[01].log; fi; sleep 60; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py ref refs4 muon4 muon4s4 muon6 muon6s4 muon8 muon12 d256 d256s4 d256mu d256mu4 d256mu6
