# Deep terminal verifier evidence

Exact owned support unit: `computer-vision-homework-r01-f3-deep-terminal-verifier-t20261019025238689375-8afa51a4.service`; invocation `t20261019025238689375-8afa51a4`. It launched at `2026-10-19T02:52:38.689758+00:00` and its durable remote launch record is included as `REMOTE_LAUNCH_V1.json`. The remote terminal capture was recorded at `2026-10-19T02:53:02.902652+00:00` with explicit SubState `exited`, Result `success`, ExecMainStatus `0`, and unit_success `True`. ExecMainCode is `1`; systemd reported `SubState=exited` and `ExecMainStatus=0`.

Support envelope: CPU quota `500ms` (expected `500ms`), MemoryMax `1500000000` bytes (expected `1500000000`), MemorySwapMax `0`, RuntimeMaxUSec `2min`. The systemd unit MemoryPeak was `1396736` bytes; this is the unit cgroup scope. Monotonic start-to-exit duration was 6.755915 seconds. CPUUsageNSec was `3330657000`. This MemoryPeak is distinct from process `peak_ram_bytes` recorded by the native training receipt.

Verification receipt SHA-256: 8bf4cb320a6c9787201dfe3655833bbe38f75910b5c83a647097229ccf080cba. Status: passed; next_epoch=6; fold3 manifest SHA-256=21c7fb52e8e69f38126362cbc5dc60dbc46539041d649b9206ec35e0f21ada1f.

The terminal capture response and the exact raw admission, launch, and helper receipt bytes are stored alongside this README. All encoded raw-file byte lengths and SHA-256 values were verified locally before the exact owned exited unit was stopped. Stop receipt: `2026-10-19T02:53:16.256309+00:00`, returncode `0`, capture SHA-256 `b366f035404c857ecc56f11993160e34bfcd21bb5089bc060c5825a18ea41e4a`. Parent memory events before and after capture were unchanged: `low 0; high 2179; max 561; oom 0; oom_kill 0; oom_group_kill 0`. Populated campaign descendants at capture: `., computer-vision-homework-r01-f3-sequence.service.

No model, checkpoint, array, or prediction payload was transferred to the local machine. For the collector, the checkpoint-support helper performed its bounded B-side work. This record is evidence preservation and helper validation, not a model-quality or grade conclusion.
