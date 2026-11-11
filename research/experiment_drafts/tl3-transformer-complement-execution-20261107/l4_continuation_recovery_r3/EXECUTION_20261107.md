# R3 active native execution, 7 November2026

**Status: ACTIVE at20:26:51UTC.** ROOT executed recovery on the existing L4 host.
Both canonical CPU and real CUDA restoration probes passed at the original
8E/51,680 frontier. Both external archivers became ready, and both production
continuations reached successful update51,750:70 new updates per original fit.
These are active trajectories, not completed12E endpoints or quality results.

The unchanged raw native capture is
[receipt113](evidence/113-recovery-r3-real-probe-status.json), SHA256
`078227bdd0995051b4feb52d37eda50e8c2f144d8e91ea1db845c2e25eebbf75`.
It was copied byte-for-byte from ROOT's local RPC result
`architecture-execution-20261107/l4-rpc/results/113-recovery-r3-real-probe-status.json`.
The successful RPC envelope records capture epoch1794083211.7452276; its raw
stdout includes all four native probe receipts, logs, transition/archive
readiness and successful training progress for both seeds.

| Seed | CPU probe | Real CUDA probe | Successful production update | Archive PID / trainer PID |
|---|---|---|---:|---|
|101001|passed2.30s|passed3.21s|51,750|2238610 /2238626|
|101002|passed2.18s|passed3.31s|51,750|2238608 /2238624|

Both probe modes reported zero forward calls and zero optimizer updates.
CPU receipts explicitly report the CUDA RNG setter stub and no actual CUDA
restoration. CUDA receipts explicitly report no stub and real CUDA restoration;
the code checks actual source/config/software/precision/physical GPU identity,
model/optimizer/scheduler tensors, RNG and safe journal offsets. Production
progress follows these distinct zero-update gates and matching live archivers.

| Seed | Migrated S4 checkpoint SHA256 | Transition proof SHA256 |
|---|---|---|
|101001|61aa2a6edabe8da024bc1e8aab9e0db2787e55bb12d212853ee75abf887b09b2|ddb5d02cde8d1a1d38c3243550fcafd64d6e0ddd642e0d75d510435655a1f9ff|
|101002|ddf864820729b7564b9c484af519007933c8923cba11ae678a9936882c84e9e5|e4e9ec474bdf37b633d6054f79ca412689d82f5b6a2b10af2747ad86adae513d|

ROOT's operator handoff additionally binds controller PID2238454,
birth2148028384 and plan SHA256
`58eb238d83144937555fa839993592892b40c5096940eef706474fedc480df82`.
The unchanged S4 trainer source identity is
`5bbbf68eccdc8c8773ff4b4eda2c4026d9c3f31b145dfc0d683b17d20a1987d0`;
its620-file inventory SHA is
`28ac81b46517f1923d4ffd9751bd0d615f2657c5721d2e3459726e8c05ef4182`.
These launch/source bindings come from ROOT's native operator report; receipt113
itself supplies the probe, checkpoint/proof, archive and progress measurements.

The new outputs are
`R/runs/gru_s101001_monitor_opt_r3/fit` and
`R/runs/gru_s101002_monitor_opt_r3/fit`, controlled by
`R/control/gru-monitor-recovery-r3`, where
`R=/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1`.
The original S1E8 checkpoint hashes, S4 source/config/profiles, official train-only
membership, full valid1873/3,528,273 and original deadline1794154066.717658 remain.
No extra fit, seed or deadline was added. Original S1, failed S3 and failed R2
artifacts remain historical and unchanged.

Still pending: every later epoch's full validation and immutable safe-state
archive, final native completed12E/77,520 plus archive_complete, and collection
and quality interpretation. The displayed window throughput is not a measured
whole-fit estimate. ROOT owns continued execution; the preparation agent only
read/copied this local evidence and updated documentation.
