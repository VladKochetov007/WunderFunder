# Delivered checkpoints and pending final packaging

Snapshot: 2026-11-01 00:38 UTC. This is a progress record, not a final score report.

## Actual delivery

[Download the 12-checkpoint archive](https://github.com/VladKochetov007/WunderFunder/releases/download/tabdata-pr54-full-corpus-checkpoints-20261031-r1/tabdata-pr54-completed-checkpoints-r2.zip).
The archive is18,501,727bytes. The actual GitHub download, CRC and all checkpoint/fit-receipt hashes were verified locally. It contains12 configured-final model artifacts from the18-role dependency graph, including intermediate teachers; it is not a submission ZIP.

All12,480 official train+valid sequences are fitting members. There are zero independent evaluation sequences after the submission-only refit. Historical train-only valid results are not scores of these new weights.

## Remaining execution

Three actual fits are running concurrently on separate B300 GPUs: fin256ft_s3, fin256ft_s4 and fbc1k_s5. The two tiedlead fits and W18 student remain dependent on this work. The frozen18-role graph, epochs, numerical training source and objectives are unchanged.

R1 controller bookkeeping and shared-device admission failures were retained. Qualified completed fits and feature exports were adopted without retraining. The separate R2 controller preserves fit-command hashes across exports and requires a B300 with at least32GiB free under the GPU lock. R1 files and terminal outcomes remain preserved.

The receiving R8 campaign is detached and bounded by the original deadline1793523959.0380824. No foreign processes are signaled.

## Serving evidence

The full four-member reference mixture projects above the60-minute CPU limit. The existing W18 student is therefore an explicitly separate compact fallback, with no new scientific fit or coefficient search.

Pinned historical W18, exposed full valid, identical official mask:

| Runtime | Weighted Pearson |
|---|---:|
| FP32 exact |0.6914165042737994|
| Q8 fast |0.6914021819338281|
| Q8 minus FP32 |−0.00001432233997134|

The prefrozen degradation budget is0.0002. Q8 is not FP32-equivalent: the original0.02 pointwise allclose diagnostic failed and remains recorded. Final-package RMS/p99/max ceilings were frozen at2× the old-reference calibration errors before any new-weight package check.

The old Q8 W18 reference measured12.88microseconds/row on one pinned Xeon6767P CPU under Python3.11, projecting about508seconds for1,970sequences. This is engineering evidence; actual new-weight ZIP checks and platform results remain pending.

## Detached artifact delivery

Worker source: receiving/operational-source/finalize_and_publish_r1.py.txt.
Actual deployed worker SHA256:9e13bfd2764d11480cccc641cf23aa18b7d365ed4c956d7e8f8690a69c3aaa8f.

The worker is running in phase waiting_for_terminal_fit_graph. It verifies frozen helper/library/contract hashes, waits for terminal fitting, publishes completed checkpoints, mints the18-fit model receipt, builds the full mixture and W18 ZIPs, checks the W18 ZIP under Python3.11, verifies uploaded downloads and posts results to PR54. The complete mixture remains runtime-unadmitted. A failed numerical/size/time/memory gate cannot be labeled eligible. Automatic Wundernn platform submission is not performed.

The worker uses a separate owned0600 publication credential file with inode-bound cleanup. No credential values are included in source or artifacts.

