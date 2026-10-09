# D02 R4 device-gate output recovery

R4 is a recovery-only successor to D02 R3. R3 reached the WMSE actual-T4
device-shape/update gate and recorded a passing 10/10 finite-update result.
The subsequent Huber gate failed before either full fit because both gates
opened `out/logs/device_shape_update_gate.log` with exclusive creation. The
collected failure is pinned at
`cloud_delivery/d02_r3/terminal_collect_r1/attempt_001/outputs/failure.json`
(SHA-256 `6764ad5d5f1aa2da49f0ecee22eaf9947749f23d551fe3a45822fe8e81cc2c09`).
No D02 R3 full fit started; its first-gate device evidence remains valid and
the R3 packet is preserved unchanged.

R4 assigns each loss-specific gate a distinct log path, command label, and
sampling label. Each receipt also hashes its own log. The gate directories,
reload receipts, update traces, prefit artifacts, and per-loss JSON receipts
were already separate; the combined `shared/device_profile_gate.json` is
written once after both gates pass. Resource samples in each per-loss receipt
now cover only that loss gate.

The optimizer, WMSE/Huber definitions, FP32 settings, Muon policy, sampler,
membership, exact 220-update fit argv, R5 AdamW control binding, and output
contract are unchanged. The R4 regression calls the production gate helper
twice consecutively. It uses the production `_wait_one` and `_launch` path so
exclusive log creation, command recording, receipt hashing, and per-loss
artifact locations are exercised; a short disposable child writes only the
profile artifacts required by the helper. It does not launch a trainer, use a
GPU, or claim profile evidence.

The sealed candidate remains pending root review and exact execution
admission. This package authorizes no dataset creation, kernel push, profile,
or fit.
