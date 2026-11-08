# Architecture execution,7 November2026

The user explicitly authorized setup and launch on both existing hosts. The
initial [contract](CONTRACT.md) remains frozen; this is the dated receiving and
execution record. No previous stopped job was resumed.

| Device allocation | Fresh fits | Fixed endpoint | Actual phase |
|---|---:|---|---|
|8L4, one fit/GPU |4 families × seeds101001/101002 =8 |epoch12 |All eight started; native epoch/full-valid events and actual PNG/SVG plots exist |
|8B300, four two-GPU pairs |labels/complement × same two seeds =4 |epoch30 /774600 updates |All four started18:36UTC after twelve native profiles; actual updates observed |

L4 families: GRU256x2; GRU256x2 plus causal local attention; genuine pinned
Mamba-2 d128×6/state16/head32; causal EMA plus a custom TabM-style head. The last
head is not a reproduction of official TabM. This is not an exact GALAM replica.
All use label-only AdamW and fresh initialization. Actual node configs are in
[configs-runtime-r3](l4/configs-runtime-r3); preparation configs are historical.
Batch16, chunk2000, accumulation1, raw112, AdamW3e-4, decay.01, clip1, five-percent
warmup/cosine, fixed12E. L4 FP32 compute excludes AMP/TF32. Mamba also binds
TRITON_F32_DEFAULT=ieee; the actual production cache contained240PTX, zeroTF32
instructions, retained with its [qualification](receiving-evidence/mamba-production-qualification-r1.json).

B300 source is6ab2f679e70928f913b97d5a29f0d1e09ceacaa3, capsule manifest
9a3b147c20e902786f026a29741cd9f5135de91183e305105b2039f735905d7a.
The separate receiving packet is
research/experiment_drafts/tl3-transformer-complement-execution-20261107.
Each model has170198786 trainable parameters: T768/L24/H12, context1024,
chunk256, B4/A4/world2. BF16 forward/backward plus BF16x9, FP32 master/gradient/
state/loss tensors and Muon3e-4; BF16x9 is not IEEE-FP32 arithmetic parity.
The comparison is label fitting versus curvature-scaled fitting of the fixed
0.9cleanTL3+0.1Transformer output; no blend weights are fitted on valid.

All fitting uses10325 qualified train sequences, zero valid fitting; full
official valid1873/3528273 scored rows. Split SHA256:
0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c.
Every row advances causal state; official scored mask applies to valid. The
32-sequence diagnostic train-eval uses all rows after99; absolute train/valid
score gaps therefore mix distributions and masks and are not an overfit test.
Valid remains development-exposed. Primary metric is pooled clipped weighted
Pearson, global_wp_clip2_v1; fixed final is the comparison endpoint.

The exact legacy Git blobs had to be restaged without Windows EOL conversion.
Earlier CRLF exports and socket-path failures are preserved on the node, not
relabelled successful. The correct LF predictor SHA is
981d7557082f8e33e025741dad2b2ab2e86fb20eb4347ab58eca1d53f00c9520.
The clean TL3 full-valid replay is WP .6923774248169172, t0 .6887196914805331,
t1 .6960351581533012; score receipt SHA
3c309abba8e86f9361dcb60953437f4c7f5d4ada6ee687d29be84437f88f9cff.
This is engineering reproduction, not a new improvement.

Actual B300 warm windows in production were~.71–.73s/update. Complete schedules
are25820updates/epoch ×30 =774600 per fit, in parallel. Arithmetic implies~6.4
days core fitting before full-valid/data/checkpoint overhead. Planning estimate
is6–8days, not a guarantee. The initial24h envelope was insufficient. Before any
fit, ROOT published a finite9-day total bound with the same four-fit/twelve-profile
caps and final1h collection reserve. Amendment SHA:
097a50483f1842d1c79e0e60410bb9c6b8d77e3a74e560657b613a83051bb9fb.
Absolute deadline2026-11-16T18:03:59.189580Z; fitting closes one hour earlier.
The frozen native observer has a7-day per-worker cap; its expiry stops only the
owned fit family. Safe epoch checkpoints are written independently of final-only
inference retention. Any needed same-fit continuation must verify actual expiry,
ownership, exact source/config/runtime/checkpoint identity and remaining budget;
it is not permission to launch new seeds or historical fits. Current source is
unchanged. L4 remains under its original24h receiving bound.

Controllers are detached with stdin detached, source and owner PID/birth receipts,
resource guards and cleanup limited to owned descendant process families. One
persistent nonterminal SSH file-RPC stream per host replaces the old interactive
ConPTY transport; old session IDs in the frozen contract are historical.
The current local RPC helper has a24h transport lifetime; disconnect does not
stop native fitting. Continuous9-day external reporting cannot be claimed from
that transport alone; renewal/collection remains ROOT's operational responsibility.

Online dashboard: http://127.0.0.1:8769/ (local loopback). It shows selectable
run/config curves, objective terms, train-eval/full-valid metrics, LR/preclip
gradient/clip incidence, sampled update/weight diagnostics and real per-GPU/host
telemetry. Native metrics remain on both nodes. Epoch watcher saves immutable
native metric-prefix snapshots, hashes and real PNG/SVG plots locally under
C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/plots.
No synthetic graph is scientific evidence. SSM hidden-state RMS is not instrumented.

GitHub progress is a separate owner: verify SemyonEpanov before every write,
deduplicate terminal/30-minute updates, distinguish fit complete, fixed endpoint,
byte verification and artifact upload. See [the reporter](GITHUB_PROGRESS_REPORTER.md).
No complete final result, new public score, significance, student benefit or
winner is asserted at this launch snapshot. Two seeds do not establish robust
training variance; eventual comparisons must retain both paired outcomes.
