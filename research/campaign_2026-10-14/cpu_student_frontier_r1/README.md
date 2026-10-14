# CPU student frontier campaign

The [complete user plan](PLAN.md) defines matched fixed-weight Q8 CPU policies, an eight-shape distilled-student ladder, precision and recurrence alternatives, causal state initialization, clean ancestry and staged cross-fold/seed validation. The follow-up explicitly asks for theory and independent work while CPU submissions await user upload. This is active implementation; no new winning architecture or target-host capacity is claimed.

- [Contract](CONTRACT.json): root-owned compute and comparison authority, exact first-wave scope, exposure and deliverables. Local CPU/RAM/VRAM protections and no paid capacity remain binding.
- [Frontier](FRONTIER.md): completed stages, active owners, actual dependencies and the next runnable actions. Historical QAT packages and receipts remain immutable.

The canonical narrative entry is in [unified inference](../../unified/inference/README.md); sources, independent reviews and operations are indexed in their corresponding unified domains as they land.

- [Entry resource/quota and recovery receipt](OPERATIONS_ENTRY_R1.md): live GPU/TPU quota and statuses, effective limits, and the [contained-cache R3 successor](kaggle_profile_r3/README.md). Execution evidence only.

- [Theory and discriminating tests](THEORY_AND_DISCRIMINATING_TESTS_R1.md), with [shape accounting](SHAPE_ACCOUNTING_R1.csv), grounds the nonblocking architecture/initializer/precision work in primary sources and explicit formulas. Derived costs are not serving measurements.

- [Initializer implementation](INITIALIZER_IMPLEMENTATION_R1.md): separate zero/constant/first-row/prefix wrapper and ten passing causal, reset, gradient and long-sequence checks. Code evidence only; fits, native export and quality comparison remain pending.

- [Actual-data state decay diagnostic](STATE_DECAY_DIAGNOSTIC_R1.md): one clean-F0 control and one fitting sequence show a random initial perturbation surviving warmup, with second-layer influence through several thousand rows. This is mechanism evidence only, not initializer quality or a population-level decay claim.

- [Student resource profiles](profiles/README.md): source/hash-bound actual-data smoke, controller terminal receipt and measured host/device consumption. B1 success does not establish the intended fitting-batch capacity.

- [Inference policy R1](inference_policy_r1/README.md) contains the A/B/C/D native-policy implementation, build provenance, bounded matrix/cell/callback evidence and ready-package status. Its [measured report](inference_policy_r1/REPORT.md) and [results receipt](inference_policy_r1/RESULTS_R1.json) show exact full-sequence parity and fresh-reset parity on two complete xteam_h4 validation sequences, including a rebuilt-A versus original-A package control. On one pinned Intel i9-11900H core, B and C were slower than shipped A and D was about 5% slower; no local speed winner or AMD/public result is established. Ready B/C/D ZIP hashes and the A/B/C/D/A submission order are in the [package manifest](inference_policy_r1/PACKAGE_MANIFEST_R1.json); rebuilt A is identified separately and excluded from that order. Related hypothesis: H-CPU-POLICY-20261014. The parity claim is limited to tested callbacks and is not an all-input proof.

- [Architecture cost screen R1](architecture_profile_r1/README.md) measures FP32 and the sealed native Q8 VNNI implementation over all eight planned student shapes using deterministic synthetic weights, original normalization, and a validated complete raw-feature callback fixture. Its [report](architecture_profile_r1/REPORT.md), [16-worker summary](architecture_profile_r1/runs_r1/LADDER_RESULTS_R1.json) and [verification receipt](architecture_profile_r1/runs_r1/RESULTS_VERIFICATION_R1.json) record reset bit parity, Q8 constructor packing, callback time and model storage. On one i9-11900H under a one-core quota, Q8 was 4.7–7.7× faster than FP32 for this random-weight proxy. This is runtime-only evidence; no fitting, quality ranking, W4 or trained-package size is claimed.
