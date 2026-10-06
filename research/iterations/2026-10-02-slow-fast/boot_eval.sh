cd ~/WunderFunder-creative
until grep -q "^DONE" artifacts/logs/d256boot_f0.log 2>/dev/null && grep -q "^DONE" artifacts/logs/d256boot_f1.log 2>/dev/null; do
  grep -q Traceback artifacts/logs/d256boot_f?.log && { echo FAILED; exit 1; }; sleep 60; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py d256mu6 d256mu6s4 d256boot d256boots4
