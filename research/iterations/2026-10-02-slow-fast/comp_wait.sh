cd ~/WunderFunder-creative
until [ $(grep -l "^DONE" artifacts/logs/cmp_*_s[34].log 2>/dev/null | wc -l) -ge 12 ]; do
  grep -l Traceback artifacts/logs/cmp_*_s[34].log 2>/dev/null | head -3 | while read f; do echo "TRACEBACK in $f"; done; sleep 120; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python comp_eval.py
