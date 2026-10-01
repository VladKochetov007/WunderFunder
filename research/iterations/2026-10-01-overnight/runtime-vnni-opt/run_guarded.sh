#!/usr/bin/env bash
set -euo pipefail
cd /home/grinchenkoeg/WunderFunder-creative
p=research/iterations/2026-10-01-overnight/runtime-vnni-opt
v=research/iterations/2026-10-01-overnight/runtime-vnni
q=research/iterations/2026-10-01-overnight/critic/runtime-probe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for condition in initial:0 packed:0 register:0 block128:0 block128:1 register:1 packed:1 initial:1; do
 variant=${condition%:*}; repeat=${condition#*:}
 case $variant in
  initial) native=$v/native_vnni/build/v3 ;;
  *) native=$p/native_$variant/build/v3 ;;
 esac
 taskset -c 18 /home/grinchenkoeg/scorer_image/run.sh $p/raw_probe_monitored.py --native "$native" --package $q/fixed_v1 --raw $q/raw.npy --reference $v/dynamic-r0.npy --mode dynamic --output "$p/guarded-$variant-r$repeat.json" > "$p/guarded-$variant-r$repeat.log"
done
python3 - <<'PY'
import pathlib,json
for p in pathlib.Path('research/iterations/2026-10-01-overnight/runtime-vnni-opt').glob('guarded-*-r*.json'):
 d=json.loads(p.read_text()); print(p.name, d['wall_us'], d['host_busy_fraction']['cpu58'], d['prediction_sha256'])
PY
