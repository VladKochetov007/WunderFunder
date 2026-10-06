cd ~/WunderFunder-creative
while [ $(grep -l "^DONE" artifacts/logs/findm128_s3.log artifacts/logs/findm128_s4.log artifacts/logs/findmc500_s3.log 2>/dev/null | wc -l) -lt 3 ]; do
  if grep -q "Traceback" artifacts/logs/findm*.log; then echo FAILED; grep -A8 Traceback artifacts/logs/findm*.log | tail -30; exit 1; fi
  sleep 60; done
ls artifacts/runs/findm128_s3h2 artifacts/runs/findm128_s4h2 artifacts/runs/findmc500_s3h2
CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/predict_run.py findm128_s3h2 valid --epoch 1 2>&1 | tail -2
ls -la cache/pred_findm128_s3h2_valid.npy
