# E465 full-shape TPU worker draft

This is a source-only worker design. It has not been imported with JAX, run on
TPU, uploaded, or fitted. Root must first seal an input manifest from the
exact checkpoint/cache bundle and check it against the queued E463 one-device
TPU real-data compatibility record. E463 inventories eight TPU v5 lite
devices but runs its bounded update on one device; it checks TPU/data/objective
compatibility without demonstrating multi-device training. The archived
E463 CPUv3 result is CPU correctness evidence, not the production TPU gate.
Root must review E465's own all-eight-device full-shape profile before fit.
The JAX code in `rnn.py`,
`bridge.py`, `distributed_combo.py`, and `onecycle.py` remains unchanged.

`scripts/run_e465_tpu.py --phase profile` runs two actual global-32 batches
per precision on complete 20,000-row sequences. Both FP32 and BF16 compile and
execute all 20 detached 1,000-row optimizer chunks per batch from fresh state.
It reports first/second-batch timings, group and content hashes, peak process
RSS, and allocator HBM telemetry. The second precision must reproduce the
same two host-batch content hashes. This is a systems profile, not a fitted
model or predictive result. `--phase fit` requires the matching profile report
and independently checks all identities before beginning the registered
single epoch.

The fold-0 fit pool is 10,535 TRAIN groups plus 1,405 fit-VALID groups. Seed 3,
epoch 0 shuffles their ordered union into 373 full batches of 32 and drops the
last four groups. Each of the two precision arms starts from the same
`c20261003_combo_s3_f0` epoch-3 evaluation-EMA NPZ, with a fresh FP32 AdamW
state. Each arm makes 7,460 updates. Its input width is 211 (`raw112` plus the
registered engineered features), hidden size 256, two GRU layers, and the
original 23-output head. The NPZ's mean/std are frozen and reused unchanged.

Each mapped global batch has four sequences on each of eight TPU devices.
Recurrent carries reset to zero for each new sequence batch. Within the batch,
student and incoming-EMA states carry through all 20 chunks and detach between
chunks. Every chunk computes the original combo objective: fixed alpha-0.5,
stopped horizon-8 SBD target from the equal mean of the two fold-0 SBD
teachers; incoming-EMA consistency coefficient 1; and the original 0.3
structural plus 0.3 next-change losses. Main row weights preserve warmup 99,
propensity squared plus 0.03, the 0.03 scored-VALID correction, and the
`since_i0px` scale (32, floor 0.25). The helper pools each numerator and
denominator over all eight replicas before ratio formation. Gradients are
`pmean`-reduced, globally clipped at norm 1, then passed to AdamW; EMA updates
once per successful attempt at decay 0.999. No AMP skips are used. Thus
attempted and completed updates must both equal 7,460 per arm, and the AdamW
state count must agree.

The LR and beta1 at attempt indices 0–7,459 come from the exact OneCycle port
validated against the repository's PyTorch scheduler at all indices 0–7,460.
The last index is recorded as unused; no 7,461st optimizer update occurs.
AdamW is a deliberate optimizer change from the parent's Muon fit, not an
equivalent reproduction. The only planned contrast is recurrent compute
precision (FP32 versus BF16) within this fresh-AdamW JAX continuation.

After each full fit, the worker saves FP32 Torch-layout student and EMA NPZs,
the AdamW count and moments, per-update JSONL history, hashes for the parent,
all data/cache inputs and the staged source closure, and an epoch-0 EMA
prediction cache. Inference always uses FP32 EMA weights, begins each sequence
with zero state, and predicts all 1,873 VALID sequences in cache row-group
order. The official score mask is recorded as
`need_prediction AND is_scored AND row >= 99`; only the 468 fold evaluation
groups belong in paired scoring. No evaluation group is read by the fit
loader, and no checkpoint is selected using evaluation scores.

The worker keeps at most the active global host batch plus one prefetched
batch. It fails closed above 12 GiB process RSS, at or above 90% of any TPU
device's reported memory limit, or after eight hours wall time. It requests an
0.80 XLA allocator fraction and records telemetry; the allocator request is
not a substitute for the measured HBM gate. Eight host CPUs are used at most,
with BLAS/OpenMP thread pools set to one. The eight-hour wall cap starts before
the worker imports JAX and includes input hashing, compilation, both full
fits, all-VALID inference, and exports.

## Input manifest supplied by root

The root-owned manifest must conform to `input_manifest.schema.json`. It
contains relative paths and SHA-256 hashes for the fold split, both parquet
files, both target-index arrays, both teacher arrays/sidecars/checkpoints,
both propensity arrays/sidecars/checkpoints, and the parent NPZ, its sidecar,
and source checkpoint. The worker rehashes this closure before constructing
cache readers. Bridge validation additionally checks teacher row-group order,
checkpoint/cache metadata, absolute propensity row-group indexing, and
exclusion of corrupt TRAIN group 528. Root should compare the input manifest
to the queued TPU E463 preflight's source hashes before sealing either worker
phase.

## Measurements still required

There is no TPU throughput, compile-time, RSS, or per-device HBM measurement
for this full-shape worker yet. The two-batch profile's two-arm training
projection is a lower bound: it applies a 20% margin to measured profile
walltime, and excludes the later all-VALID inference and output I/O. It is not
an admission decision. Root must review the profile, code and cache lineage,
and maintain the eight-hour whole-process hard timeout. The one-device E463
TPU update does not establish multi-device training throughput or show that the full sequence's
backward pass fits within the 0.80 allocator target. E465's own all-eight-
device profile measures those conditions, while the full epoch remains
unmeasured until a separate fit is admitted.
