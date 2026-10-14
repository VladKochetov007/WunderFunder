# Live CPU/student frontier

Entry: 2026-10-13T23:43:28Z (October14 Europe/Kyiv). Root owns completion under [contract](CONTRACT.json).

## Current position — 2026-10-14 00:42 UTC

- The complete plan is saved. [A–B–C–D–A delivery](../../../submissions/10-14-cpu-policy-r1/README.md) is checked and independently reviewed; user uploads are asynchronous. All original weights and coefficients stay fixed. No target-host result has arrived.
- [All eight shapes](architecture_profile_r1/REPORT.md) have FP32/Q8 full-callback cost proxies with exact reset replay. One local Intel host, synthetic weights, no quality or competition capacity claim. At similar parameter counts, depth/width changes measured cost materially; retain both alternatives for fitting.
- [Packed W4](packed_w4_screen_r1/README.md) is exact on the tested matrices but vectorized decode remains1.3–2.3× slower than expanded VNNI here. The scalar lookup is not an optimized T-MAC reproduction. No W4 serving advantage is established.
- The first H192×2 B1 finite-update smoke passed, with the source-order claim corrected below. The proposed all-shape B16 launch was withheld before execution because larger activation bounds could exceed5GB observed VRAM. Source-qualified H640×1 and H448×3 B1 profiles are being sealed to choose a common safe fitting batch. No new eight-shape fit or quality winner exists yet.
- The causal initializer wrapper and ten implementation checks pass. A separate fitting-sequence diagnostic finds perturbation effects through several thousand rows, so warmup does not universally erase initial-state information. Native initialization support is now an independent implementation lane; trained quality remains unknown.
- At00:38UTC, repaired private Kaggle R3 profile remainsRUNNING and the TPU bridge remainsQUEUED. Terminal collection is prepared. Provider status alone is not evidence of an optimizer update.

## Entry and historical updates

- Complete plan recorded; current tree clean at entry revision7800d304. No new model/runtime results yet.
- Parent5CPU/18GB high/20GB maximum/no swap verified. Entry charged memory5.12GB; local GPU75MiB, no research worker observed; physical Btrfs has56GB available. These are snapshots, not future admission receipts.
- Inference actor: implement separately asserted A/B/C/D policies on original xteam_h4; exact complete-sequence checks, actual assembly, callback timings and packages.
- Training actor: bind existing clean F0 teacher cohort and architecture recipes; profile all eight shapes before matched fits. Local GPU admission is root-owned.
- Independent reviewer: review first-wave recipe, causality, native equivalence and delivered artifact evidence independently.
- Root: refresh Kaggle quota and TPU outcome, repair the cache containment successor, integrate/code-review/grade/record results. No paid capacity. User uploads CPU probes later; this does not gate local training or theory.
- Next nonblocking lines: state initialization and prefix-state perturbation theory, depth/width and cache capacity analysis, then precision and cheaper recurrence at matched measured cost.

Update 2026-10-14T00:11:07Z: the first real-data H192×2/B1/C2000 FP32 smoke completed one finite optimizer update under the shared controller (exit0,5.08s,2.092GB host peak; observed harness VRAM437MB, not the controller's sparse75MiB sample). This establishes only this small shape's update path. Actual fitting-batch/all-shape profiles and the strict new cache consumer are next. The [initializer wrapper](INITIALIZER_IMPLEMENTATION_R1.md) passes ten synthetic implementation checks; fitting and native export remain pending. Inference actor reports B/C/D bitwise equality to original A on two complete sequences and resets; independent package review is active. Local Intel short timings favor A but do not establish an AMD winner. Kaggle R3 remains RUNNING at the latest poll; TPU collection stays independent.

Post-smoke correction: the B1 loader confused overlapping train/valid row-group IDs, loading train37 instead of sampler-selected valid37. Both are fitting data, so the measured one-row resource result stands; the exact sampler/source-order claim is withdrawn. The immutable attempt is retained. B16 must use source-qualified positions with an overlapping-ID regression. The [separate state-decay diagnostic](STATE_DECAY_DIAGNOSTIC_R1.md) explicitly declares train37 and remains a one-fitting-sequence mechanism observation, not quality evidence.

Update 2026-10-14T00:00:24.698137+00:00: complete plan committed7e4e58da; independent specification reviewb36f86a8 identified shape-coupled cache validation and normalization/sampler confounds. Actor is building a distinct consumer recipe and an immediate small actual-data GPU profile. Native A/B/C/D synthetic matrix/recurrent tests pass in actor worktree; original full-sequence checks underway. R3 private Kaggle profile version1 isRUNNING after independentreviewa75fa132; TPU remainsQUEUED. [Theory](THEORY_AND_DISCRIMINATING_TESTS_R1.md) and exact derived shape counts are indexed. No fittedstudent or newwinner yet.
