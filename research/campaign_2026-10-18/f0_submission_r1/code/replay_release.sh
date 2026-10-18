#!/bin/sh
# Fixed, serial CPU-only release checks; invoked inside the bounded campaign unit.
set -eu
base=/home/grinchenkoeg/computer-vision-homework-20261018
work=$base/submission-f0-r1
py=$work/py311/bin/python
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=$base/research_tmp
unset PYTHONPATH LD_LIBRARY_PATH WNN_PRECISION WNN_NATIVE_LEVEL WNN_BUDGET_S
"$py" "$work/check_submission_abi.py" "$work/mix75_q8.zip" --output "$work/Q8_ABI.json"
for level in v2 avx2; do
  "$py" "$work/check_native_panel.py" \
    --zip "$work/mix75_q8.zip" --panel "$work/evidence/panel.parquet" \
    --reference "$work/q8_panel.npy" --max-groups 2 --level "$level" \
    --output "$work/Q8_${level}_PANEL.json" --predictions "$work/q8_${level}_panel.npy"
done
"$py" "$work/check_native_panel.py" \
  --zip "$work/mix75_q8.zip" --panel "$base/data/datasets/valid.parquet" \
  --fold "$work/evidence/fold0.json" --reference "$work/mix75_reference.npy" \
  --output "$work/Q8_FULL_F0.json" --predictions "$work/q8_full_f0.npy"
