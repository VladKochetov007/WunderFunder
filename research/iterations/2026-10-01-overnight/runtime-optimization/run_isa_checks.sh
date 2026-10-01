#!/usr/bin/env bash
set -euo pipefail
cd /home/grinchenkoeg/WunderFunder-creative
p=research/iterations/2026-10-01-overnight/runtime-optimization
q=research/iterations/2026-10-01-overnight/critic/runtime-probe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for level in v2 avx2; do
 for count in 3 4; do
  if [[ $count == 3 ]]; then members=1,2,3; else members=0,1,2,3; fi
  for variant in base order; do
   if [[ $variant == base ]]; then native=$q/fixed_v1/$level; else native=$p/native_order/build/$level; fi
   taskset -c 22 /home/grinchenkoeg/scorer_image/run.sh "$p/order_probe.py" --package "$q/fixed_v1" --native "$native" --raw "$q/raw.npy" --members "$members" --rows 2000 --output "$p/$variant$count-$level.json"
  done
 done
 taskset -c 22 /home/grinchenkoeg/scorer_image/run.sh "$p/order_invariants.py" --package "$q/fixed_v1" --native "$p/native_order/build/$level" --raw "$q/raw.npy" --output "$p/order-invariants-$level.json"
done
