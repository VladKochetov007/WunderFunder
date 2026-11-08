ROOT prepared S3 and qualified the two existing GRU cells for a monitored
same-fit continuation. Both native engineering profiles passed20 updates with
errors=[] and no scientific checkpoint. The supervisor launch receipt records
an armed process at2026-11-07T20:03:45.519122UTC. These receipts do not yet verify
a checkpoint migration, a resumed trainer update, future epoch archival or fit
completion. No additional architecture, seed, scientific fit or quality claim is
introduced.

The immutable local copies are exact bytes of ROOT's file-RPC result envelopes.
The corresponding full source manifests, native profile receipts, approved
profiles and supervisor plan remain on the receiver at the paths below. This
packet does not claim to contain those complete remote files. Their reported
identities and the copied response hashes are indexed in
[NATIVE_PREPARATION_R1.json](NATIVE_PREPARATION_R1.json).

S3 is the fresh620-file capsule
`R/source-gru-monitor-archive-r1`, where
`R=/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1`.
The [native builder result080](native_receipts/080-build-gru-archive-source.json)
binds parent inventory
`7d3324230effb050ff083b939e26ad638e5df5c5b72ea70b0f8aed169997db4a`
to S3 manifest
`a28f86a74c6d8617c6b4df37fbb9c972dd9148e2ea1e520a27888109a744e8f6`.
The actual profile identity reports canonical trainer source
`99ca81ddbe76a1a4004b8c0395edc67e7977428b7c41f384750619e9f5a4796e`.
Exactly three files differ from S1: bounded GRU gate replay, its diagnostic
population metrics, and admission of extended final_only monitoring without
early_stopping. S1 and S2 remain unchanged.

The [R5 native profile result089](native_receipts/089-gru-r5-profile-results.json)
records these engineering measurements. The R5 helper used the final raw r3
configs and r3 output namespace; helper revision and config revision are distinct.

| Existing cell | Physical GPU UUID | Native20-update seconds | Owned-command seconds | Peak reserved bytes |
| --- | --- | ---: | ---: | ---: |
| GRU seed101001 | 222adc44-ecd3-a645-cf7e-ddda9d495f7a | 3.37751130387187 | 21.812608238309622 | 1306525696 |
| GRU seed101002 | bce838c4-097e-0623-7b67-6a97f3778510 | 3.34066566452384 | 21.914484050124884 | 1306525696 |

Both receipts report world_size=1 and77520 total scheduled updates over the
unchanged12 epochs. Device memory is23670685696 bytes. The native elapsed time
and wrapper wall time measure different scopes; neither establishes a full-fit
throughput improvement. Both briefly paused originals have explicit resume
receipts in089. No continuation stop had been admitted by the failed approval
preflights below.

The normalized final_only config hashes are
`46364a0f68eefea9464918e848bf2af5166af4e2d65fbfb78d16d7c71d969325`
for101001 and
`fccc9a5bd2bd10eac158c6a647a5feea5368e18e88c4d0ed62bf3d71188d875b`
for101002. The sole normalized config amendment is retention
`all_epochs -> final_only`. Input data, masks, batch16, chunks, schedule,
optimizer, seed and FP32 policy are unchanged. All old inference snapshots and
indexes stay historical in the original run; future safe full resume states
must be archived externally under the approved continuation protocol.

ROOT preserved three failed approval preparations. They are audit/schema
mistakes before continuation stops, rather than failed training updates:

| Receipt | Preserved failure | Correct interpretation |
| --- | --- | --- |
| [091](native_receipts/091-approve-gru-profiles-build-plan.json) | TypeError multiplying device_total_memory_bytes by0.9 | The native field is a per-rank list; use the recorded device bound. |
| [092](native_receipts/092-approve-gru-profiles-r2.json) | KeyError reading identity.devices | Devices are nested under identity.runtime; the RPC wrapper's ok marker encloses inner returncode1. |
| [094](native_receipts/094-approve-gru-profiles-r3.json) | Assertion expecting20 rank0 log rows | Telemetry cadence10 emits exactly two rows for20 successful updates, at steps10 and20; inner returncode1 is preserved. |

[Schema probe093](native_receipts/093-profile-schema.json) retains the actual
memory and identity field shapes.
[Row probe095](native_receipts/095-profile-row-schema.json) retains the exact
source-paired logged loss and preclip gradient values:

| Seed | Step | S1 and S3 label loss | S1 and S3 preclip gradient norm |
| --- | ---: | ---: | ---: |
| 101001 | 10 | 2.0991626187196406 | 1.0469131469726562 |
| 101001 | 20 | 2.0732859440426683 | 1.3745020627975464 |
| 101002 | 10 | 1.915838230954347 | 1.0170482397079468 |
| 101002 | 20 | 1.883859800183235 | 0.8805330991744995 |

The corrected [approval096](native_receipts/096-approve-gru-profiles-r4.json)
returned0, checked these exact logged steps, and reported zero maximum absolute
loss and gradient differences for each seed. This sampled parity is additional
engineering evidence. It does not replace source qualification, preservation of
the complete optimizer/scheduler/RNG state, or later canonical resume admission.
Source, data and checkpoint-state checks were not weakened to fix the schema
errors.

Approval096 created the actual supervisor plan
`R/control/gru-monitor-transition-r1/supervisor-plan.json`, SHA
`dfc7a770c5b5ac14e0b09edc3395d50aae7fdcf10924130966dd961104f55a83`.
It retains the original L4 deadline/reserve boundary
1794154066.717658, or2026-11-08T16:07:46.717658UTC. No fresh24-hour clock is granted.

[Launch097](native_receipts/097-arm-gru-transition-supervisor.json) returned0
and records supervisor PID2230567, process birth2147894167 and PGID2230567 with
that exact plan SHA. The point-in-time owner status is running. Its two planned
cells are101001/GPU0 and101002/GPU1. The supported status is **launched/armed**;
there is no later supervisor waiting-state, transition proof, archiver readiness
or resumed-training receipt in this documentation cutoff. A subsequent status
must cite new native evidence before being called migrated or resumed.

The scientific cohort remains eight existing L4 fits with a fixed12E endpoint.
Official chronological primary validation stays10325 training members, zero
valid members in fitting ancestry,1873 full-valid sequences and3528273 scored
rows per complete evaluation. Repeatedly inspected valid remains exposed
development evidence. No local delta is a public-score claim.

The packet's focused CPU verification uses the unchanged canonical checkpoint,
metadata and monitor readers:17 tests cover config admission for both actual
recipes, tensor/optimizer/scheduler/RNG preservation, exact native byte prefixes,
source/config/offset/hash refusal, old-file immutability and safe-state archival.
Independent review also ran all17 tests. No SSH, GPU, authentication or process
action was performed by the documentation actor.
