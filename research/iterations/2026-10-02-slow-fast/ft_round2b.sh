cd ~/WunderFunder-creative; F=research/iterations/2026-10-02-slow-fast/final_ft.sh
for n in ffA ffB ffD; do for s in 3 4; do pkill -u $USER -f "name ${n}_s$s " ; pkill -u $USER -f "final_ft.sh ${n}_s$s " ; done; done
sleep 5; for n in ffA ffB ffD; do for s in 3 4; do rm -f artifacts/logs/${n}_s$s.log; rm -rf artifacts/runs/${n}_s$s; done; done
setsid nohup $F ffA_s3 fmu128_s3 3 1 "--epochs 2 --lr 2e-4" > /dev/null 2>&1 < /dev/null &
setsid nohup $F ffA_s4 fmu128_s4 4 2 "--epochs 2 --lr 2e-4" > /dev/null 2>&1 < /dev/null &
setsid nohup $F ffB_s3 fmu128_s3 3 3 "--epochs 4 --lr 1e-4" > /dev/null 2>&1 < /dev/null &
setsid nohup $F ffB_s4 fmu128_s4 4 4 "--epochs 4 --lr 1e-4" > /dev/null 2>&1 < /dev/null &
setsid nohup $F ffD_s3 fmu128_s3 3 5 "--epochs 2 --lr 1e-4 --optimizer muon --muon-lr-mult 6" > /dev/null 2>&1 < /dev/null &
setsid nohup $F ffD_s4 fmu128_s4 4 6 "--epochs 2 --lr 1e-4 --optimizer muon --muon-lr-mult 6" > /dev/null 2>&1 < /dev/null &
echo relaunched
