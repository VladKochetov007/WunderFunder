# Fixed causal feature screening

`feature_plan.py` deterministically divides the canonical outer-held-out
components into two head partitions using SHA-256 of the component identifier.
The same frozen, fixed-epoch GRU supplies both partitions. It emits reciprocal
fit/evaluation manifests: fit A and evaluate B, then fit B and evaluate A.
Different backbone states must never be pooled; repeat on another outer fold
with independent heads and pool only final prediction metrics. These folds are
historically exposed, not newly untouched data. Extract the two partitions once
and reuse them in reversed input positions for the prespecified reciprocal
contrast; cache origin-role labels are descriptive, while disjoint components
and the sealed command establish fit/evaluation separation for each head.

`feature_cache.py` verifies split source identities and all supplied learned
ancestor checkpoint hashes, recovers fitting membership from embedded
provenance, and rejects any ancestor trained on the held-out components. All
checkpoint ancestors (parents, propensity, and distillation teachers) must be
supplied explicitly. Missing provenance is an error, never an inferred absence.
It supports the actual linear-encoder GRU with causal default inputs and optional
volume28/current; other architectures/input transformations are refused.
Privileged future-input teachers are refused as screening backbones.

Features and recurrent state are computed over complete sequences before row
selection, with recurrent state carried through chunks. The frozen row sample
uses one uniformly jittered row per 128-row stratum, starting at row227, with
a SHA-256 seed derived from the sequence ID and271828. The same sequence always
gets the same rows in both directions. Selection then intersects the explicit
scoring mask; the first99 rows are never evaluated. Jitter prevents periodic
sampling from observing only one phase of an eight-row cycle. The mask is used
for fitting/evaluation selection, never as a
predictor. Every placebo is a causal 128-row lag computed on the dense sequence,
not on the selected-row cache and never rolled across a sequence boundary.

The nine fixed families cover auxiliary regime-offset proxies, multiplicative
auxiliary/state proxies, observable-age clock proxies, activity interactions,
EMA-centered i1-delta proxies, event age, observable transitions, ordinary
EMA8/32 smoothing proxies, and volume28. Reports name each proxy explicitly.
Exact four-warmup-regime intercept/amplitude routing, target-specific row-versus-
event EMA, phase-aware cycle8 smoothing, and learned conditional i1 innovations
remain pending. These proxy tests do not fulfill or close those exact proposals.
Names do not establish physical clock/market semantics. Prefix mutation tests
check causality for all families. The lagged placebo may itself be informative;
it tests recent alignment, not an assumption of pure noise.

`feature_screen.py` fits every prespecified arm with one fixed ridge coefficient
(0.01): cue, lagged placebo, equal-width extra state capacity, and unchanged
backbone. Fitted normalization/projection matrices are exported with coefficients.
The fitted arms each have 99 design columns: intercept, two frozen
predictions, 64 fixed nonlinear state projections and 32 extra projections.
Regime-head extra projections multiply 32 state projections in all three arms.
Normalization uses only fitting rows, with fixed seeds and no outer-set tuning.
All arms and contrasts are saved. The head optimizes clipped-target weighted
residual MSE; it does not claim to optimize Pearson directly.

Primary scoring uses `reference_global_wp` and stable paired component moments,
2000 bootstrap draws, seed271828. Scores are exact for the selected rows only;
they do not establish full-validation or end-to-end model gains. Positive screens
need the reciprocal direction, an independent outer fold/seed, full-row scoring,
class guards and matched trainable-GRU adapters. A nominal +0.0005 screen is a
lead, not an automatic promotion or a multiple-testing-adjusted claim. Historical
negative results remain controls rather than eligibility exclusions.

`feature_recipes.py` generates content-sealed executable CPU/GPU drafts once the
two caches exist. It refuses missing cache bytes instead of producing a claimed
ready experiment. GPU uses the same float64 normal equations as CPU. The JAX TPU
solver is explicit float32 and cannot be scientifically launched before hardware
inventory and a matched numerical pilot; current TPU capacity is still unknown.
The tiny linear solve is not evidence of efficient TPU use. Full cache extraction
currently supports CPU/CUDA only.

Example preparation after controller review (paths are explicit dependencies):

```bash
PYTHONPATH=src python scripts/feature_plan.py --split research/campaign_2026-09-29/artifacts/splits/fold0.json --checkpoint artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt --ancestor artifacts/runs/c20260929_since_s3_f0/model_ep3.pt --ancestor artifacts/runs/c20260929_since_s4_f0/model_ep3.pt --ancestor artifacts/runs/c20260929_prop_f0/model.pt --output research/campaign_2026-10-04/feature_plan_f0
PYTHONPATH=src python scripts/feature_cache.py --manifest research/campaign_2026-10-04/feature_plan_f0/direction0_fit.json --output cache/feature_f0_a.npz --device cuda
PYTHONPATH=src python scripts/feature_cache.py --manifest research/campaign_2026-10-04/feature_plan_f0/direction0_evaluation.json --output cache/feature_f0_b.npz --device cuda
PYTHONPATH=src python scripts/feature_recipes.py --fit cache/feature_f0_a.npz --evaluation cache/feature_f0_b.npz --cue i1_innovations --backend gpu --name feature_i1_f0_ab --output research/campaign_2026-10-04/specs/feature_i1_f0_ab.json
```

Use the campaign controller for resource admission and launch. No scientific
cache extraction or fit was run while implementing this harness.
