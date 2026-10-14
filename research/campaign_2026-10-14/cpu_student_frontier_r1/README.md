# CPU student frontier campaign

The [complete user plan](PLAN.md) defines matched fixed-weight Q8 CPU policies, an eight-shape distilled-student ladder, precision and recurrence alternatives, causal state initialization, clean ancestry and staged cross-fold/seed validation. The follow-up explicitly asks for theory and independent work while CPU submissions await user upload. This is active implementation; no new winning architecture or target-host capacity is claimed.

- [Contract](CONTRACT.json): root-owned compute and comparison authority, exact first-wave scope, exposure and deliverables. Local CPU/RAM/VRAM protections and no paid capacity remain binding.
- [Frontier](FRONTIER.md): completed stages, active owners, actual dependencies and the next runnable actions. Historical QAT packages and receipts remain immutable.

The canonical narrative entry is in [unified inference](../../unified/inference/README.md); sources, independent reviews and operations are indexed in their corresponding unified domains as they land.

- [Entry resource/quota and recovery receipt](OPERATIONS_ENTRY_R1.md): live GPU/TPU quota and statuses, effective limits, and the [contained-cache R3 successor](kaggle_profile_r3/README.md). Execution evidence only.

- [Theory and discriminating tests](THEORY_AND_DISCRIMINATING_TESTS_R1.md), with [shape accounting](SHAPE_ACCOUNTING_R1.csv), grounds the nonblocking architecture/initializer/precision work in primary sources and explicit formulas. Derived costs are not serving measurements.

- [Initializer implementation](INITIALIZER_IMPLEMENTATION_R1.md): separate zero/constant/first-row/prefix Python wrapper and ten causal, reset, gradient and long-sequence checks. Native export/callback support is in the linked native R1 entry below; fits and quality comparison remain pending.

- [Native causal state initialization R1](../cpu_state_initializer_native_r1/README.md): native callback now accepts exported per-member FP32 zero, bounded constant, first-row MLP and prefix residual states while retaining the original GRU. Twenty-five focused checks pass; Python wrapper/native exact activation error is at most 2.98e-8, zero equals omitted configuration bitwise, and the abi3 callback/reset path runs under Python 3.11. Synthetic implementation and transport evidence only; initializer fitting and quality comparison remain with root.

- [Actual H256×2 full-sequence callback check](../cpu_state_initializer_native_r1/README.md): on the pinned clean-F0 train row-group37 sequence, 20k-row native exact-vs-Python prediction error is at most1.91e-6 and fast-vs-exact is at most4.18e-7 for the deterministic initializer fixture. Prefix row99 and reset are checked; these are one fitting sequence and not a grade or trained-initializer result.

- [Actual-data state decay diagnostic](STATE_DECAY_DIAGNOSTIC_R1.md): one clean-F0 control and one fitting sequence show a random initial perturbation surviving warmup, with second-layer influence through several thousand rows. This is mechanism evidence only, not initializer quality or a population-level decay claim.

- [Student resource profiles](profiles/README.md): source/hash-bound actual-data smoke, controller terminal receipt and measured host/device consumption. B1 success does not establish the intended fitting-batch capacity.

- [Genuine packed-W4 matrix screen](packed_w4_screen_r1/README.md): identical represented weights across expanded VNNI, nibble decoding and bit-plane lookup. All tested dots are exact; the first vectorized packed path remains slower on local Intel. Matrix-only synthetic evidence, with no packed serving-capacity claim or optimized T-MAC result.

- [Inference policy R1](inference_policy_r1/README.md) contains the A/B/C/D native-policy implementation, build provenance, bounded matrix/cell/callback evidence and ready-package status. Its [measured report](inference_policy_r1/REPORT.md) and [results receipt](inference_policy_r1/RESULTS_R1.json) show exact full-sequence parity and fresh-reset parity on two complete xteam_h4 validation sequences, including a rebuilt-A versus original-A package control. On one pinned Intel i9-11900H core, B and C were slower than shipped A and D was about 5% slower; no local speed winner or AMD/public result is established. Ready B/C/D ZIP hashes and the A/B/C/D/A submission order are in the [package manifest](inference_policy_r1/PACKAGE_MANIFEST_R1.json); rebuilt A is identified separately and excluded from that order. Related hypothesis: H-CPU-POLICY-20261014. The parity claim is limited to tested callbacks and is not an all-input proof.

- [Architecture cost screen R1](architecture_profile_r1/README.md) measures FP32 and the sealed native Q8 VNNI implementation over all eight planned student shapes using deterministic synthetic weights, original normalization, and a validated complete raw-feature callback fixture. Its [report](architecture_profile_r1/REPORT.md), [16-worker summary](architecture_profile_r1/runs_r1/LADDER_RESULTS_R1.json) and [verification receipt](architecture_profile_r1/runs_r1/RESULTS_VERIFICATION_R1.json) record reset bit parity, Q8 constructor packing, callback time and model storage. On one i9-11900H under a one-core quota, Q8 was 4.7–7.7× faster than FP32 for this random-weight proxy. This is runtime-only evidence; no fitting, quality ranking, W4 or trained-package size is claimed.

- [Kaggle R3 terminal diagnosis and R4 repair](kaggle_profile_r4/README.md): the cache copy succeeded, then a stale symlink requirement blocked every candidate before updates. The sealed R4 specs route exactly three regular files through full hash validation; actual-producer runner checks pass without worker execution.

- [Matched B8 fitting recipe](recipes/README.md): frozen source-qualified F0 seed3 consumer contract for all eight architectures, common fitting-derived normalization and teacher targets, 14,920 FP32 updates and fixed checkpoint0. The existing cache producer validator is retained. Actual trainer CLI profile and fitted predictions are the next owned delivery stages; no quality rank yet.

- [Distillation and recurrence theory supplement](DISTILLATION_AND_RECURRENCE_THEORY_R1.md) separates capacity from finite-budget optimization, derives clipped weighted-correlation mixture sensitivity, and examines minGRU memory and zero-state scan semantics. The linked synthetic algebra receipt and reproducible script verify the derivative and one-layer perturbation formula; no new fit or quality result.

- [Clean F3 seed3 checkpoint tranche R5](f3_teacher_fit_r5/README.md), with [offline verification](f3_teacher_fit_r5/DRAFT_STATUS_R5.md), prepares the missing source-qualified teacher independently of CPU uploads. It preserves the earlier three-epoch fit and actual-T4 profile but defers all exports to stay within remaining quota. Draft source/recipe/bootstrap checks pass; quota is intentionally stale and no provider action is admitted yet.

- [Actual weight geometry](WEIGHT_GEOMETRY_R1.md): pinned historical H256 control/H384 teacher spectra and exact row-quantizer errors. Their stacked recurrent95%-energy ranks exceed the arithmetic-saving threshold; this cautions against aggressive post-hoc SVD, without rejecting trained low-rank or W4 models. Static weight evidence only, with no score or state-distribution claim.
