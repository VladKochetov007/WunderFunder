#!/usr/bin/env bash
set -euo pipefail
cd /home/grinchenkoeg/WunderFunder-creative
p=research/iterations/2026-10-01-overnight/runtime-optimization
q=research/iterations/2026-10-01-overnight/critic/runtime-probe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for trial in base3-r0 order3-r0 base4-r0 order4-r1 order4-r2 base4-r1 order3-r1 base3-r1; do
 if [[ $trial == order* ]]; then native=$p/native_order/build/v3; else native=$q/fixed_v1/v3; fi
 if [[ $trial == *3-* ]]; then members=1,2,3; else members=0,1,2,3; fi
 taskset -c 22 /home/grinchenkoeg/scorer_image/run.sh "$p/order_probe.py" --package "$q/fixed_v1" --native "$native" --raw "$q/raw.npy" --members "$members" --output "$p/$trial.json"
done
taskset -c 22 /home/grinchenkoeg/scorer_image/run.sh "$p/order_invariants.py" --package "$q/fixed_v1" --native "$p/native_order/build/v3" --raw "$q/raw.npy" --output "$p/order-invariants.json"
