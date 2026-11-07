# L4 throughput pulse and bounded GRU gate telemetry

Current receipt-backed status: both original GRU cells are **actively training
under S4** at the native113 cutoff,20:26:51UTC on7November. Both real canonical
CPU and CUDA zero-update resume probes passed, both archivers became ready,
and production training advanced fromE8/51680 to51750 updates. See
[active continuation evidence](ACTIVE_CONTINUATION_S4_R3.md). The final12E
endpoint is unchanged and completion is not yet claimed.

S4 source preparation and both native20-update profiles had already passed
ROOT approval with zero logged S1 loss/gradient differences. That earlier
[qualification107](NATIVE_QUALIFICATION_S4_R1.md) and its pending-resume wording
remain a historical cutoff, superseded by the actual113 probe/update evidence.

The preceding S3 attempt: both copies migrated atE8/51680, but canonical
resume failed before new updates because the monitor restoration callback
returned None. ROOT preserved both original S1 safe checkpoints and cleaned up
the failed S3 processes. The [S4 resume fix](RESUME_FIX_S4_R1.md) records that
diagnosis and the exact four-file amendment.
The exact native output, including approval preflight failures091/092/094 and
resume failure099, is retained in [native_receipts](native_receipts/).

[Native preparation R1](NATIVE_PREPARATION_R1.md) retains the earlier successful
profile/approval/launch evidence through097. [Resume fix R1](RESUME_FIX_S4_R1.json)
records the later failure, safe parent pins and bounded recovery scope. Earlier
passed engineering profiles do not establish a working monitored resume.

The following S1/S2 proposal text is historical preparation context. Its
unimplemented/unrun statements describe that earlier stage. The qualified S3
route and its explicit retention amendment are documented historically in
[CONTINUATION_S3.md](CONTINUATION_S3.md). Its monitored-resume callback failure
requires the separate S4 source; no quality gain is established.

The new user request authorizes optimization now. ROOT alone owns SSH, GPU
execution, exact PID/birth checks, pause/resume and the independent SIGCONT
watchdog. This packet prepares engineering work; it changes no active fitting
source, configuration, weight or optimizer state. Existing twelve-epoch fits and
their old receipts remain unchanged.

The [12-config pulse plan](configs/PULSE_PLAN_R1.json) is ready for immediate
ROOT staging: four seed101001 families times batches16/32/64. Its SHA256 is
`3bb9f2898f1e5aeb7e24baf8951f0a1814c182961d9eae9ecc6108bd9f6dde65`.
Every config was statically matched against actual mirrored fitting metadata,
then changed only in batch_size. The plan carries byte and normalized-config
hashes, complete native CLI argv/env, GPU0/2/4/6, output directories and200-second
per-profile bounds. All receiver command paths are PurePosixPath strings.
No Torch import, model test, model evaluation or fitting was executed locally.

ROOT stages the12config files under
`R/configs/throughput-probe-r1`, retaining the plan alongside its new optimization
authority/ownership receipts. This is a separate engineering allocation with
twelve profiles maximum and zero scientific fits; do not silently rewrite the
original cohort's cap. A representative exact native command is:

```bash
R=/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1
S="$R/source-r1"
PY=/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python
env -u CUBLAS_WORKSPACE_CONFIG -u CUBLAS_EMULATE_SINGLE_PRECISION \
  -u RANK -u WORLD_SIZE -u LOCAL_RANK CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  PYTHONPATH="$S/src" PYTHONDONTWRITEBYTECODE=1 \
  "$PY" "$S/scripts/train_sequence.py" \
  --config "$R/configs/throughput-probe-r1/gru_s101001_b16.json" \
  --data-root "$R/project" --split-manifest "$R/project/chronological_primary.json" \
  --output "$R/runs/throughput-probe-r1/gru_b16" --profile-updates 20
```

Execute these argv through existing run_owned_command with timeout200, using the
plan's commands directly rather than a shell-built string. The same-thread
canonical prefetcher has no separate DataLoader processes, and world_size=1.
The receiving ResourceGuard has no stale-heartbeat kill condition: it checks
deadline/disk/RSS and treats stopped processes as alive. ROOT can therefore
pause only the exact owned trainer PID while leaving observer/relay/controller
running, verify queued GPU work settles, and use a birth-checked480-second
SIGCONT watchdog plus try/finally. SIGSTOP retains GPU allocations; it does not
create VRAM headroom. Record retained bytes and physical UUID, observe idle SM,
and account for foreign GPU4 services without signalling them.

Bound each family's entire pulse to420seconds, with each next profile timeout
`min(200,420-elapsed)` and cleanup before SIGCONT. Three200-second upper bounds
would otherwise exceed the480-second watchdog. Other paired-seed fits remain
running as references. Skip B128 until actual retained memory plus measured
profile reserve stays below90% of actual device capacity and pulse time remains.
The current observed train peaks are only1.2GB GRU,2.29GB local,2.26GB Mamba and
2.82GB EMA, but these are allocated peaks, not a proof of B64 reserved memory.

Canonical profile20 includes per-process cold shape/kernel startup. It is not
twenty post-warmup updates: no warmup-exclusion CLI exists. Keep both full owned
wall time and native elapsed time, ordinary step10 and sampled diagnostic step20,
peak allocated/reserved bytes, finite updates and actual rows. Batch changes also
change the scientific update schedule: B16 has77,520updates/12E, B32 has38,760,
B64 has19,440. Nominal rows/update are32k/64k/128k. Do not automatically scale LR,
continue an existing B16 optimizer at another batch, or claim equivalent quality.

Actual [mirrored timing evidence](MIRRORED_TIMING_R1.json) identifies a concrete
GRU cost: ordinary windows are about.080s, while every twentieth diagnostic
window takes about1.83–1.85s. Correcting the native log's10-step sampling, the
extra diagnostic replay represents about52% of estimated GRU window time. The
source replays all2000rows/two GRU layers with Python-dispatched gate operations.
Its latest epoch cycle is about1200s, including61–65s full validation. EMA cycles
are about745s with77–78s evaluation; Mamba about785s with80–82s evaluation.
Optimizer time is below about1% of ordinary windows. The data path already has
one prefetched pinned full batch and asynchronous H2D; window timers exclude
batch waiting. Residual cycle time also includes checkpoint and post-window
monitoring, so these receipts do not isolate loader time. Increasing CPU threads
past8/worker has no demonstrated benefit;64requested compute threads fit the
192CPU host allocation.

The separate [source builder](prepare_monitor_source.py) prepares the next
monitor revision without touching source-r1:

```bash
"$PY" "$Q/prepare_monitor_source.py" --source "$S" \
  --inventory "$R/manifests/l4-source-r1-manifest.json" \
  --output "$R/source-monitor-prefix-r1"
```

It verifies all620parent files and permits exactly two changed diagnostic files.
GRU gate replay uses at most the first2sequences and contiguous first64rows of
the current chunk, initialized from those streams' actual provided chunk-start
hidden state. It does not skip intervening rows or reset carried state. LSTM
diagnostics, objective, backward, optimizer and scheduler code remain unchanged.
New replay_sequence_count/replay_prefix_rows metrics make the reduced scope
explicit. Reuse B16 with only this new source/PYTHONPATH and a fresh output for
one separately bounded diagnostic probe; it is not part of the S1 twelve-profile
plan and needs its own recorded ROOT engineering scope.

The patch uses no RNG calls and operates on detached read-only diagnostic
tensors, but actual timing/gradient/trajectory qualification remains a receiver
test. Reduced allocation/kernel interleaving can affect nondeterministic CUDA
execution; no bitwise continuation claim is made. Static source/AST/prefix/state
checks and Ruff passed without importing Torch or executing a model.

Do not migrate active checkpoints by relabelling source SHA or bypassing strict
resume identity. A qualified migration needs the exact safe checkpoint parent
hash, a two-file source/config whitelist, unchanged tensors/optimizer/scheduler/
RNG/data/runtime, safe-prefix log copies, new independently approved native
profile and explicit lineage in new metadata/artifact closure. That receiving
migration has not been implemented or qualified here. Keep the healthy eight
fits intact while measuring the new source for subsequent admitted work; an
instant speedup of the already running L4 fits is not promised.
