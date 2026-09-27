#!/usr/bin/env bash
# Final candidate: EMA-4 final fit (train + dev valid, holdout never trained on) -> LP-FT on dev valid rows ->
# predictions -> holdout blend check vs the current LB best -> package 0.65 GRU + 0.35 baseline -> docker verify.
#   bash scripts/build_final.sh NAME SEED "extra train flags"      e.g. build_final.sh ab70_td5 4 "--td-alpha 0.5"
set -u
cd "$(dirname "$0")/.."
NAME=$1; SEED=$2; EXTRA=${3:-}
PROP=${PROP:-prop_v1}
export WNN_GPU_FRACTION=0.9 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=60
PY=".venv/bin/python"
FIN="--train-source train+valid --eval-source holdout --full-valid --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity $PROP --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 --hidden 128 --epochs 4 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --fast"
FT="--train-source valid --eval-source holdout --full-valid --mask-weight --loss wmse --aux-struct 0.3 --aux-next 0.3 --hidden 128 --seed 3 --epochs 1 --fast"
ENC=$(echo "$EXTRA" | rg -o -- '--enc \w+' || true)
WNN_MEM_MAX=20G scripts/limited $PY scripts/train_rnn.py --name $NAME --seed $SEED $FIN $EXTRA > artifacts/logs/$NAME.log 2>&1
echo "FIT $NAME $(rg '^BEST' artifacts/logs/$NAME.log)"
WNN_MEM_MAX=12G scripts/limited $PY scripts/train_rnn.py --name ${NAME}_head --init $NAME $FT $ENC --freeze inp,gru --lr 5e-4 > artifacts/logs/${NAME}_head.log 2>&1
WNN_MEM_MAX=12G scripts/limited $PY scripts/train_rnn.py --name ${NAME}_lpft --init ${NAME}_head $FT $ENC --lr 2e-4 > artifacts/logs/${NAME}_lpft.log 2>&1
echo "LPFT ${NAME}_lpft $(rg '^BEST' artifacts/logs/${NAME}_lpft.log)"
WNN_MEM_MAX=10G scripts/limited $PY scripts/predict_run.py ${NAME}_lpft valid >> artifacts/logs/${NAME}_lpft.log 2>&1
OMP_NUM_THREADS=2 WNN_CPU_PCT=15 WNN_MEM_MAX=8G scripts/limited $PY - <<PYEOF
import sys, numpy as np
sys.path.insert(0, "scripts")
import decide
from decide import wp, build_sums
decide.GROUPING = "dev"
hold = decide._splits_plain(np.load("cache/valid_targets.npz")["row_group"])["holdout"]
keep = decide._stress_keep()
ref = build_sums(None, [("ab60_final_ema4_p1_lpft", 0.65), ("baseline", 0.0875)])
new = build_sums(None, [("${NAME}_lpft", 0.65), ("baseline", 0.0875)])
rng = np.random.default_rng(0)
for lam in (0.0, 0.02):
    for tag, idx in (("holdout", hold), ("holdout-stress", hold[keep[hold]])):
        d = wp(new[idx], lam) - wp(ref[idx], lam)
        sd = np.std([wp(new[b], lam) - wp(ref[b], lam) for b in (rng.choice(idx, len(idx)) for _ in range(200))])
        print(f"HOLDOUT lam={lam} {tag}: new {wp(new[idx], lam):.4f} vs LB-0.6618 package {wp(ref[idx], lam):.4f} delta {d:+.4f}+-{sd:.4f}")
PYEOF
if [ -z "$ENC" ]; then
  $PY scripts/package_submission.py final_${NAME} ${NAME}_lpft:0.65 baseline:0.0875 2>&1 | tail -1
  mv submissions/final_${NAME}/submission.zip submissions/final_${NAME}/final_${NAME}.zip
  docker run --rm --cpus 1 --memory 4g -v "$PWD:/w" -w /w python:3.11-slim-bookworm bash -c "pip install -q numpy==2.2.6 onnxruntime==1.23.2 pyarrow==19.0.1 >/dev/null 2>&1; python scripts/verify_submission.py submissions/final_${NAME}/final_${NAME}.zip cache/valid_subset20.parquet cache/verify_final_${NAME}.npz" 2>&1 | tail -1
  OMP_NUM_THREADS=1 WNN_CPU_PCT=6 WNN_MEM_MAX=4G scripts/limited $PY scripts/check_package.py cache/verify_final_${NAME}.npz ${NAME}_lpft:0.65 baseline:0.0875
  sha256sum submissions/final_${NAME}/final_${NAME}.zip
else
  echo "encoder $ENC: package after the native port lands"
fi
echo FINALDONE
