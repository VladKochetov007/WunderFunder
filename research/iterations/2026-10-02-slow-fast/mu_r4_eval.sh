cd ~/WunderFunder-creative
while [ $(grep -l DONE artifacts/logs/{mu6inp,mu6wd05s4,d256mu8}_f[01].log 2>/dev/null | wc -l) -lt 6 ]; do
  if grep -q Traceback artifacts/logs/{mu6inp,mu6wd05s4,d256mu8}_f[01].log; then echo FAILED; fi; sleep 60; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py muon6 muon6s4 mu6wd05 mu6wd05s4 mu6inp d256mu6 d256mu8
