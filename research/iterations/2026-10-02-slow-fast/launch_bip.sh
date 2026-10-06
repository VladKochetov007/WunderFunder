cd ~/WunderFunder-creative
for f in 0 1; do CUDA_VISIBLE_DEVICES=3 setsid nohup .venv/bin/python research/iterations/2026-10-02-slow-fast/bip.py $f --write-cache > artifacts/logs/bip_f$f.log 2>&1 < /dev/null & done
