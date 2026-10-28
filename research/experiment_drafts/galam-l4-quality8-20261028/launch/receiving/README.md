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
