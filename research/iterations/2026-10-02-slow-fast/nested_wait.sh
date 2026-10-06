cd ~/WunderFunder-creative
while [ $(ls cache/pred_nft0_f3_valid.npy cache/pred_nft1_f3_valid.npy cache/pred_nft2_f3_valid.npy cache/pred_nft0_f2_valid.npy cache/pred_nft1_f2_valid.npy cache/pred_nft3_f2_valid.npy 2>/dev/null | wc -l) -lt 6 ]; do
  if grep -q Traceback artifacts/logs/nested_T*_K*.log artifacts/logs/nest?_f[23].log 2>/dev/null; then echo FAILED; grep -h -A6 Traceback artifacts/logs/nested_T*_K*.log artifacts/logs/nest?_f[23].log | tail -15; exit 1; fi; sleep 60; done
cd research/iterations/2026-10-02-slow-fast && ~/WunderFunder-creative/.venv/bin/python nested_eval.py
