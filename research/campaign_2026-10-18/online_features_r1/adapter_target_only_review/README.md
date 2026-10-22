# Target-only feature-adapter implementation review

This packet records an independent, read-only implementation critique of the target-only loss mode for the frozen-backbone geometry adapter. It is an ABI/training-code review, not an empirical-quality verdict. V1 missed cache-backed and frozen-initializer teachers in `main_loss`; V2 adds those guards and passed mocked objective checks pinned to its integration receipt. V2 still has a main-loss-normalization scope caveat and has not been fit-admitted by this review.

- [Review](REVIEW.md) — V1 counterexamples, V2 conditional pass, remaining normalization and fail-fast caveats, and serving-consumer inspection.
- [Receipt](receipt.json) — V1/V2 source identities, checker outputs, scope, resource bounds, and non-admission status.
- [V1 counterexample checker](check_target_only_guard.py) and [V2 guard checker](check_target_only_v2.py) — standard-library-only AST/mock checks; neither imports Torch or reads model/data payloads.

The root data audit is indexed separately at [online_features_r1](../README.md); it defines callback observability and exposure limits, not whether this training-objective patch is correct.
