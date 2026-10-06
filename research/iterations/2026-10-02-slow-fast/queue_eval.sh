cd ~/WunderFunder-creative
names="ctxWs4 ctxCs4 h256mu muema mu4k tfmmu tfmad"
done_all(){ for n in $names; do for f in 0 1; do grep -q "^DONE" artifacts/logs/${n}_f$f.log 2>/dev/null || return 1; done; done; return 0; }
until done_all; do sleep 120; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py muon6 muon6s4 ctxC ctxW ctxCs4 ctxWs4 muema mu4k h256mu d256mu6 tfmmu tfmad ref
