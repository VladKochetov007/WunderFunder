Both original GRU cells resumed production training under S4 at approximately
2026-11-07T20:26:44UTC. The [actual native113 receipt](native_receipts/113-recovery-r3-real-probe-status.json)
shows successful updates beyond the original safeE8/51680 boundary: both cells
reached51750 by20:26:51UTC. Both canonical CPU probes and both real CUDA probes
passed first, with zero model forwards and zero optimizer updates in each probe.
Both external full-state archivers reported readiness before production resumed.
This is an active same-fit continuation, not a new scientific fit. Epoch12
completion and a quality improvement are not established by this receipt.

| Original seed | CPU canonical restore | Real CUDA canonical restore | Production step | Latest forwarded rows/s |
| --- | --- | --- | ---: | ---: |
| 101001 | Passed,2.30s | Passed,3.21s | 51750 | 405612.44366123027 |
| 101002 | Passed,2.18s | Passed,3.31s | 51750 | 398075.477724333 |

The CPU probes restored the real canonical model, optimizer, scheduler, RNG and
safe native log prefixes, stubbing only the CUDA RNG setter and checking its
serialized bytes. The CUDA probes used the actual GPU and unstubbed CUDA RNG
restoration. Each reported next_epoch8/global_step51680, zero forward calls and
zero optimizer updates. The later training-phase progress records distinguish
production updates from those zero-update engineering probes. The reported
rows/s values are individual progress windows, not whole-run timing evidence.

S4 canonical trainer source remains
`5bbbf68eccdc8c8773ff4b4eda2c4026d9c3f31b145dfc0d683b17d20a1987d0`.
The exact cumulative four-file amendment and source-qualified profiles remain
those approved in [104–107](NATIVE_QUALIFICATION_S4_R1.md); no scientific source
was changed to repair the later probe assertion. The normalized recipe differs
from the original only in the explicitly authorized retention amendment;
the numerical recipe, batch16,12E schedule, seeds, split/data/runtime and
optimizer/scheduler/RNG state remain unchanged.

The active recovery controller is PID2238454, birth2148028384. Its
[native launch112](native_receipts/112-launch-reviewed-recovery-r3.json)
pins plan
`58eb238d83144937555fa839993592892b40c5096940eef706474fedc480df82`.
The current output directories are:

- R/runs/gru_s101001_monitor_opt_r3/fit
- R/runs/gru_s101002_monitor_opt_r3/fit

Here `R=/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1`.
The respective transition proof SHA256 values are:

- 101001:ddb5d02cde8d1a1d38c3243550fcafd64d6e0ddd642e0d75d510435655a1f9ff
- 101002:e4e9ec474bdf37b633d6054f79ca412689d82f5b6a2b10af2747ad86adae513d

Receipt113 binds each CPU/CUDA probe, migrated checkpoint, transition readiness
and archive readiness to the same proof. The archived initial epoch is8; the
archiver PIDs are2238610 and2238608. Production trainer progress identifies
PIDs2238626 and2238624. These are receipt-time ownership/liveness observations;
future liveness or final completion requires later native evidence.

Every failed preparation remains preserved. The schema errors091/092/094
preceded continuation stops. S3's None-returning callback failure099 occurred
before new updates; both original E8 parents and failed S3 siblings remain
historical. Recovery R2's [failed CPU assertion110](native_receipts/110-recovery-r2-real-probe-status.json)
compared a plain model-state dict with a checkpoint OrderedDict after
load_resume returned the correct frontier. The corrected R3 probe preserves
strict per-state checks; neither the migration equality helper nor the
scientific source was weakened. [Audit R1](RECOVERY_PROBE_AUDIT_R1.json) retains
the failed R2 and corrected R3 launch snapshots through112.

Original deadline1794154066.717658 remains binding, with the configured epoch12
endpoint and supervised immutable full-state archival of subsequent safe epochs.
The total cohort stays eight original L4 fits. Official chronological primary
validation stays10325 fitting members, zero valid members in fitting ancestry,
and1873 full-valid sequences/3528273 scored rows at each complete evaluation.
Earlier validation remains exposed development evidence. There is no new seed,
automatic additional fit, checkpoint ranking, LR change or public-score claim.

[ACTIVE_CONTINUATION_S4_R3.json](ACTIVE_CONTINUATION_S4_R3.json) records the exact
native probe/transition/archive/progress objects and raw113 SHA
`078227bdd0995051b4feb52d37eda50e8c2f144d8e91ea1db845c2e25eebbf75`.
All native execution, approval and process actions were performed by ROOT;
the packet actor only preserved and documented the supplied evidence.
