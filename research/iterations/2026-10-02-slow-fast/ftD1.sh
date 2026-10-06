cd ~/WunderFunder-creative
mv artifacts/logs/ftD_f1.log artifacts/logs/ftD_f1.oom.log
bash research/iterations/2026-10-02-slow-fast/ft_sweep.sh ftD d256 1 5 "--epochs 2 --lr 1e-4" > /dev/null 2>&1 < /dev/null
tail -n 1 artifacts/logs/ftD_f1.log
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python tailrank_runs.py d256 ftA ftB ftC ftD
