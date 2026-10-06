cd ~/WunderFunder-creative
while [ $(grep -l "^DONE" artifacts/logs/findm256_s3.log artifacts/logs/findm256_s4.log 2>/dev/null | wc -l) -lt 2 ]; do
  if grep -q "Traceback" artifacts/logs/findm256_s?.log; then echo FAILED; grep -A8 Traceback artifacts/logs/findm256_s?.log | tail -20; exit 1; fi
  sleep 60; done
ls artifacts/runs/findm256_s3h2 artifacts/runs/findm256_s4h2
