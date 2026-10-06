cd ~/WunderFunder-creative
while [ $(grep -l DONE artifacts/logs/{mu6ep6,mu6wd05,mu6m09}_f[01].log 2>/dev/null | wc -l) -lt 6 ]; do
  if grep -q Traceback artifacts/logs/{mu6ep6,mu6wd05,mu6m09}_f[01].log; then echo FAILED; fi; sleep 60; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py ref muon6 muon6s4 mu6ep6 mu6wd05 mu6m09
