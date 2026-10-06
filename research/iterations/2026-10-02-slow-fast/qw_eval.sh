cd ~/WunderFunder-creative
while [ $(grep -l DONE artifacts/logs/qw[012]_f[01].log 2>/dev/null | wc -l) -lt 6 ]; do
  if grep -q -i "Traceback\|Error" artifacts/logs/qw*_f*.log; then echo FAILED; grep -i -A3 "Traceback\|Error" artifacts/logs/qw*_f*.log | tail -20; exit 1; fi
  sleep 60; done
cd research/iterations/2026-10-02-slow-fast
sed -e 's/^runs=sorted.*/import sys;runs=sys.argv[1:]/' -e 's/^runs=\[r for r in runs.*//' tailrank.py > tailrank_runs.py
~/WunderFunder-creative/.venv/bin/python tailrank_runs.py qw0 qw1 qw2 ref refs4
