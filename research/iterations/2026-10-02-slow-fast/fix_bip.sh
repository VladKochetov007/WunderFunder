cd ~/WunderFunder-creative/research/iterations/2026-10-02-slow-fast
for p in $(pgrep -u $USER -f "2026-10-02-slow-fast/bip.py [01]"); do kill $p 2>/dev/null; done
sleep 5
python3 patch_bip.py && grep -n "class BiProp" -A2 bip.py | head -3 && grep -n "m=BiProp" bip.py
cd ~/WunderFunder-creative && rm -f cache/propcache_bip_f*_*
for f in 0 1; do CUDA_VISIBLE_DEVICES=3 setsid nohup .venv/bin/python research/iterations/2026-10-02-slow-fast/bip.py $f --write-cache > artifacts/logs/bip3_f$f.log 2>&1 < /dev/null & done
