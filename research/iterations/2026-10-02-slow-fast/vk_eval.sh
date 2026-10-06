cd ~/WunderFunder-creative
ok(){ for n in d256mu6s4 d256boot d256boots4 d256vk d256vks4; do for f in 0 1; do grep -q "^DONE" artifacts/logs/${n}_f$f.log 2>/dev/null || return 1; done; done; }
until ok; do sleep 120; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py d256 d256s4 d256mu6 d256mu6s4 d256boot d256boots4 d256vk d256vks4
