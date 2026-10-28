# Receiving engineering evidence

These files were collected from the authorized eight-L4 host through the single
persistent SSH session. [source-r2.identity.json](source-r2.identity.json) binds
the extracted source archive to canonical commit b930df6d. The
[receiving check log](receiving-r2-tests-corrected.log) records38 CPU checks
passing on the actual Torch2.8.0+cu128 runtime. This is compatibility evidence,
not scientific training or exact original GALAM reproduction.

The24 `shape-r2-g*-b*.json` files use the canonical model/optimizer adapters on
their assigned GPUs, strict FP32 flags, two warm-up plus three measured updates,
primed causal attention state and discarded synthetic weights. They report
finite losses, parameters, gradients and FP32 optimizer states. They omit the
full20,000-row input/prefetch footprint and actual source decoding; actual-data
profiles remain required. R1 GALAM timings used default cuDNN precision flags
and are retained remotely as superseded engineering attempts; the Muon R1 probe
rejected those flags. R2 fixes the probe initialization using the same strict
precision function as real training.

Representative R2 results (host-observed rows/s; allocated peak):

| Configuration | Batch128 | Batch256 | Batch384 |
|---|---:|---:|---:|
| GALAM LSTM64 |12,119|23,377|34,156|
| GALAM LSTM128 |11,901|24,120|35,767|
| GALAM depth2 |11,661|23,559|34,800|
| GALAM GRU72 |12,390|25,055|36,594|
| GALAM history256 |12,007|23,877|31,290|
| GALAM TBPTT256 |11,281|see receipt|34,820|
| GALAM hybrid Muon |11,418|23,059|34,540|

TBPTT256 at384 allocates10,992,075,776B and reserves12,257,853,440B in the
synthetic probe. Two full raw112 input batches add approximately6.88GB before
targets/masks and allocator overhead, which the real-data profile must check.
Plain GRU32/64/128 measured approximately539k/659k/770k rows/s; retaining32
preserves its audited scratch batch and more optimizer updates per epoch.

The numbers above cannot establish a full training ETA, validation score or
quality-optimal batch. No scientific fit or checkpoint is represented here.

## Actual valid timing and receiving launcher

The16 `evaluation-r1-g*-b*.json` receipts run the canonical evaluator on384
whole official-valid sequences using fresh untrained weights and identity
normalization. They prove bounded evaluator behavior/finite outputs and measure
128 versus384 batches; their scores are not trained-model results.
[Timing summary](EVALUATION_THROUGHPUT_SUMMARY.json) records approximately69–78s
for GALAM384, versus190–200s at128. Linear full-valid projections are estimates;
actual epoch evaluation remains complete1873-group validation.

Source-r4/r5/r6 archive identity receipts bind the staged versions. The
[R4 evaluation tests](receiving-r4-probe-tests.log) pass4 checks; the
[R6 launcher checks](receiving-r6-launcher-tests.log) pass8 checks on Torch2.8.
[R6 host inventory](receiving-r6-host.json) and its [log](receiving-r6-host.log)
verify all eight L4 UUIDs, current CUDA-usable memory and driver570.172.08.
The [R5 inventory failure](receiving-r5-host.log) and
[UUID diagnostic](receiving-r5-uuid-diagnostic.json) are preserved: CUDA returns
raw UUIDs, whereas SMI prefixes them with `GPU-`. The operational join now
normalizes physical identity while preserving raw UUIDs in the exact runtime
identity. SMI physical23034MiB and CUDA usable23,670,685,696B are distinct facts.

[Independent prelaunch review](PRELAUNCH_REVIEW.json) identifies the remaining
PCI ordering issue in the waiting profile owner. The
[exact waiting-owner stop receipt](actual-profile-cohort-r1.stop.json) shows
zero profile/scientific starts; its replacement explicitly selects PCI_BUS_ID.
All actual-source profiles and scientific admission are still pending data
delivery and independent artifact review.

[Training source equivalence](training-source-equivalence.json) verifies identical
224-file training identities across R4/R5/R6: SHA256
`168561624b88828b343445b26e2a2409408c6a90ba7ca0a5ecc8a6f15fa916a9`.
[Official mask coverage](valid-scoring-mask-coverage.json) directly reads all1873
valid groups:3,528,273 official scored rows, zero in each sequence's first99.
The runtime's warm-up exclusion therefore removes no official scored-valid row.
[Publisher credential metadata](pr46-publisher-credential-receipt.json) records
only account/ownership/private-file mode and public transport-key identity; no
credential value is present. The publisher is prepared, not yet running.
