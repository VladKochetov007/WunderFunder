cd ~/WunderFunder-creative
for p in $(pgrep -u $USER -f "research/iterations/2026-10-02-slow-fast/bip.py"); do [ "$p" != "$$" ] && kill $p 2>/dev/null; done
sleep 5; pgrep -u $USER -fc "2026-10-02-slow-fast/bip.py [01]"
rm -f cache/propcache_bip_f*_*.npy cache/propcache_bip_f*_*.json
for f in 0 1; do CUDA_VISIBLE_DEVICES=3 setsid nohup .venv/bin/python research/iterations/2026-10-02-slow-fast/bip.py $f --write-cache > artifacts/logs/bip2_f$f.log 2>&1 < /dev/null & done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py ref refs4 muon4 muon4s4 muon6 muon6s4 muon8 d256 d256s4 d256mu d256mu4 d256mu6
