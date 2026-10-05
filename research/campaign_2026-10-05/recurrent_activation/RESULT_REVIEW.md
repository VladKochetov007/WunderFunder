# Independent result review: E471 recurrent candidate pilot

## Provenance and completion

The sealed E471 six-fit spec has 14 code entries and 11 inputs; every current
entry matches its bound SHA-256 and size. The runner attempt embeds the same
spec, finished with exit 0 and verified teardown, and its sole fit-manifest
artifact hash/size matches `runs/six_v1/result.json`. It charged 191.7076
seconds, peaked at 371,195,904 observed VRAM bytes and 1,792,749,568 worker
RAM bytes, and recorded zero worker OOM/kill events. All six checkpoints load
with 8,235 finite parameter values; each result row reports 474 updates.

The fit receipt records the unchanged candidate trainer, tests and probe
hashes; the frozen E470 recipe hash; and the content-bound precision wrapper
hash. The fit source, tests, probe, wrapper and campaign runtime hashes match
the sealed spec. The common policy fields are `cuda.matmul.allow_tf32=False`,
`cudnn.allow_tf32=False`, cuDNN enabled, and FP32 matmul precision `highest`.
The v2 production-shape probe passed alpha-zero output/state/input-gradient/
parameter-gradient parity under the original tolerances before fitting. This
is a changed backend policy from historical E470's cuDNN default, so the two
historical tanh controls below are descriptive backend diagnostics, not exact
same-policy fit controls.

All six selected-prediction archives match the grader's stored hashes, contain
finite arrays, share the same 26,990 selected targets/row segmentation, and
match selected source labels and official `need_prediction & is_scored`
parquet rows. I independently recomputed each pooled and per-target
`global_wp_clip2_v1` result from the exports; all values match the fit and
grading receipts. I then independently formed per-component centered weighted
moments and recomputed all nine 2,000-draw paired-component contrasts using
the preregistered seed and common component resamples. Candidate/reference
points, intervals, bootstrap standard deviations, and sign frequencies all
match `RESULT.json` exactly. No grader or model source was changed.

## Recomputed scores and contrasts

The six new arms score:

| Candidate | Optimizer | GlobalWP | t0 | t1 |
|---|---|---:|---:|---:|
| α = −0.1 | AdamW | 0.4887901 | 0.5010069 | 0.4765732 |
| α = −0.1 | MuonAdamW | 0.4971020 | 0.4991056 | 0.4950985 |
| α = 0 | AdamW | 0.4822645 | 0.4987989 | 0.4657301 |
| α = 0 | MuonAdamW | 0.5054955 | 0.5055565 | 0.5054344 |
| α = +0.1 | AdamW | 0.4793591 | 0.4945869 | 0.4641314 |
| α = +0.1 | MuonAdamW | 0.5116375 | 0.5086565 | 0.5146186 |

The four α-versus-zero differences cross by optimizer: α=−0.1 is +0.006526
with AdamW (conditional component interval `[+0.000389,+0.010991]`) and
−0.008393 with MuonAdamW (`[−0.012829,−0.004752]`). α=+0.1 is −0.002905
with AdamW (`[−0.006452,+0.001970]`) and +0.006142 with MuonAdamW
(`[+0.002880,+0.010190]`). The directions therefore do not support a
optimizer-independent sign for this small candidate perturbation.

MuonAdamW minus AdamW is positive at all three α values: +0.008312 for
−0.1 (interval crosses zero), +0.023231 for zero, and +0.032278 for +0.1.
These are fixed nominal-LR recipe comparisons, not an optimizer-family
ranking. α changes gradient distributions and Adam/Muon effective updates;
the results do not equalize optimizer- or activation-specific update scales.

The two zero-alpha scores differ from the prior tanh controls by only
`+0.00000505` for AdamW (interval `[+0.00000124,+0.00000859]`) and
`−0.00000294` for MuonAdamW (interval crosses zero). These tiny differences
are numerically descriptive. They compare the E471 scripted cell under its
common FP32 policy with E470's fused cuDNN tanh control under its earlier
backend policy and should not be interpreted as performance evidence.

## Scientific limits and disposition

The custom scan changes only the post-tanh GRU candidate to
`u + α sin(πu)` while retaining the standard sigmoid reset/update gates. It
uses a one-layer width-32 tanh-embedding/readout model and 8,235 parameters,
not the two-layer width-256 GELU incumbent. This one seed, one epoch, fixed
24-fit/16-exposed-component screen has seven primary conditional contrasts
plus two historical backend diagnostics. The component intervals do not
measure training-seed uncertainty; no multiplicity correction or untouched
component evaluation is present. Do not select an α/optimizer winner, claim
optimizer superiority, or extrapolate to the incumbent or public score.

The result is a valid bounded pilot under the amended common FP32 policy. Its
strongest finding is optimizer-dependent direction: both nonzero α values
help in one optimizer and hurt or remain uncertain in the other. That calls
for a fresh, predeclared replication of all six arms at new training seeds,
with no outcome-based pruning, while preserving the same topology, fit recipe,
precision policy, and all arms. Even repeated seed directions on these same
exposed components would remain exploratory and would not create independent
component evidence.

Evidence: `PREREG.md`, `FP32_POLICY_PREREG.md`,
`e471_six_fit_v1.sealed.json`, `runs/six_v1/result.json`,
`runs/six_v1/*.pt`, the six selected-prediction exports, `RESULT.json`,
`score_recurrent.py`, and `runner/attempts/e471-recurrent-six-fit-v1.json`.
