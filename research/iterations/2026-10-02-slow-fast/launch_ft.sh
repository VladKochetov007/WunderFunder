cd ~/WunderFunder-creative; S=research/iterations/2026-10-02-slow-fast/ft_sweep.sh; chmod +x $S
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo
for f in 0 1; do
 setsid nohup $S ftA d256 $f 5 "--freeze inp,gru --epochs 2 --lr 5e-4" > /dev/null 2>&1 < /dev/null &
 setsid nohup $S ftB d256 $f 6 "--freeze inp,gru --epochs 4 --lr 5e-4" > /dev/null 2>&1 < /dev/null &
 setsid nohup $S ftC d256 $f 2 "--freeze inp,gru --epochs 2 --lr 5e-4" "--epochs 1 --lr 5e-5" > /dev/null 2>&1 < /dev/null &
 setsid nohup $S ftD d256 $f 4 "--epochs 2 --lr 1e-4" > /dev/null 2>&1 < /dev/null &
done
