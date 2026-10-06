cd ~/WunderFunder-creative
ok(){ for n in muKs4 muV0s4 muP1s4; do for f in 0 1; do grep -q "^DONE" artifacts/logs/${n}_f$f.log 2>/dev/null || return 1; done; done; }
until ok; do sleep 120; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py muon6 muon6s4 muK muKs4 muV0 muV0s4 muP1 muP1s4
