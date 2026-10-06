cd ~/WunderFunder-creative; F=research/iterations/2026-10-02-slow-fast/final_ft.sh
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr "\n" " "; echo
g=0
for s in 3 4; do
 setsid nohup $F ffA_s$s fmu128_s$s $s $g "--epochs 2 --lr 2e-4" > /dev/null 2>&1 < /dev/null &
 setsid nohup $F ffB_s$s fmu128_s$s $s $g "--epochs 4 --lr 1e-4" > /dev/null 2>&1 < /dev/null &
 setsid nohup $F ffD_s$s fmu128_s$s $s $g "--epochs 2 --lr 1e-4 --optimizer muon --muon-lr-mult 6" > /dev/null 2>&1 < /dev/null &
done
echo launched
