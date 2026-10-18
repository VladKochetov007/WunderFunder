# Clean propensity capability for the modified TL3 counterpart

Scope: opt-in engineering implementation; current review status is recorded below; **no fit, cache generation, GPU profile or scientific admission**. Root owns integration. Branch: `experiment/tl3-clean-propensity-20261018`, based on research-team `5665d7b7c76adf7f80b7599307a636d3ac80dd19`. H/E IDs: `H-TL3-CLEAN-PROPENSITY-20261018` / `E-TL3-CLEAN-PROPENSITY-20261018`.

The finite [TL3 ancestry search](../../campaign_2026-10-18/quality_gap_r1/tl3_closure_decision_r1/README.md) did not recover the actual creative propensity producer. The [readiness map](../../campaign_2026-10-18/quality_gap_r1/tl3_modified_counterpart_preparation_r1/README.md) therefore calls for rebuilding learned preprocessing in a separately labeled modified clean counterpart. This patch supplies an opt-in `strict_fp32_v1` propensity path; it does not reconstruct the historical producer or change any frozen Beleriand source.

The strict path requires a canonical split and fixed endpoint without outer-evaluation selection, fits deterministic normalization over all ordered fit-valid rows, disables autocast/TF32, and binds checkpoint/cache source, membership, normalization and output identities. Cache inference reads feature columns only. Fit membership and cache output membership remain distinct: a fold-clean model may infer features of evaluation groups without fitting their targets. Legacy defaults are retained.

Strict run directories and cache outputs are create-only. Nonfinite training/inference values and invalid probabilities are rejected. Producer and consumer must agree on the source/data/split and exact ordered output rows. Tiny synthetic checks test these contracts; they are not a real-data replay, GPU numerical parity or a quality comparison.

## Evidence and remaining work

- [PR.md](PR.md) gives the eleven required workflow answers and bounded local verification commands.
- [recipe.json](recipe.json) records the engineering scope, branch ancestry and explicit missing campaign work. It is not a launch plan.
- [VALIDATION.json](VALIDATION.json) records the original fifteen passing synthetic/static checks. [VALIDATION_V2.json](VALIDATION_V2.json) records successor checks and source hashes after review fixes, when present.
- [INDEPENDENT_REVIEW_V1.json](INDEPENDENT_REVIEW_V1.json) rejected the initial implementation because the cache name could escape its data root and the portable-data verifier was absent from source pins. [INDEPENDENT_REVIEW_V2.json](INDEPENDENT_REVIEW_V2.json), when present, defines the successor decision; no PASS is inferred from the first test suite.
- [PR access observation](PR_ACCESS_OBSERVATION_V1.json) records missing CLI authentication. A published Git branch is distinct from an actual PR.
- `revisions/v1/` retains the originally reviewed code/test and packet bytes so the first review remains reproducible.

Before a campaign fit, add reviewed optimizer attempt/success/nonfinite accounting and resume/failure evidence, verify an actual bounded runtime/data profile, and complete the modified scientific recipe and tied-lead adapter. The current code is **not campaign-ready**. A 30-cell quality study is separate from this capability patch; existing F0/F1 scores and the running F3 pair do not become results of it. The initial proposed comparison with rank1 had unequal member/training budgets and was rejected as a matched scientific control. A successor must define matching member budgets and an explicit estimand; this capability patch does not settle that design.

No training commands are authorized by this document. No model score, public gain, draft PR URL or publication is claimed until supported by its corresponding receipt.
