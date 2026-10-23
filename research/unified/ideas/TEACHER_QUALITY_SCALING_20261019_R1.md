---
kind: idea
status: proposed
created_utc: 2026-10-19
updated_utc: 2026-10-19
hypothesis_ids: [H-TEACHER-QUALITY-SCALING-20261019-R1]
experiment_ids: []
evidence_paths: []
---

# Teacher quality first: an empirical scaling programme

## Question and latest instruction

The user now prioritizes progressively stronger teachers. Distillation is optional
future engineering, not a criterion for selecting, training or promoting a teacher.
This amendment supersedes that strategic part of the earlier large-teacher design;
its architecture descriptions and frozen proposal history remain useful.
No experiment is admitted here. The B300 trainer is still unimplemented. Existing
local start/budget, strict FP32 and stopped-controller restrictions remain binding.

Goal: find the best held-out official prediction quality obtainable from permitted
data and allocated compute. Estimate the marginal value of additional parameters,
training, history and dynamics supervision before committing most server time.

## Sources and what transfers

Primary abstracts inspected on October 19 using web retrieval; no full-paper
replication or arXiv helper invocation is claimed in this amendment.

- [Chinchilla, 2203.15556](https://arxiv.org/abs/2203.15556): balance model size and
  training data under a compute budget. Its language-token ratios and exponents
  are not constants for our rows, objectives or correlated sequences.
- [Data-constrained scaling, 2305.16264](https://arxiv.org/abs/2305.16264): repeated
  data can help, with diminishing returns. Its observed repetition behaviour is
  external LLM evidence, not our epoch recommendation.
- [Time-series scaling, 2405.13867](https://arxiv.org/abs/2405.13867): decoder-only
  forecasting models show scaling across a heterogeneous corpus. This motivates
  measurement, but does not establish scaling on this small supplied corpus.
- [Forecasting scaling and look-back, 2405.15124](https://arxiv.org/abs/2405.15124):
  context length and granularity affect the scaling picture; longer history can
  hurt. History is a separate axis from parameter count.
- [Tensor Programs V, 2203.03466](https://arxiv.org/abs/2203.03466): parameterization
  can stabilize hyperparameter transfer across width. This is not proof that
  standard muP rules transfer unchanged to Muon or simultaneous depth changes.

See the [prior source register](../sources/TRANSFORMER_B300_SCALE_SOURCES_20261019_R2.md)
for TradeFM, Muon internals, hardware, attention and the asynchronous pipeline paper.

## What we actually measure

Keep separate coordinates:

- N: actual trainable parameters, plus trunk/head counts and architecture.
- D: distinct fitting rows, sequences and source-qualified dependency components.
- T: cumulative supervised core-row presentations; T/D records reuse, not new data.
- I: processed input rows including repeated left context and auxiliary tails.
- K: causally visible history; distinguish dense context, slow-memory summaries,
  forward state and gradient horizon. They are not interchangeable.
- C: measured GPU-seconds on the same device allocation; also record wall time,
  per-update phases, peak memory, communication and a documented FLOP estimate.
- U: successful updates, actual global core-row batch and independent component
  count per update. Eight devices do not require an eightfold larger global batch.

One time step with 112 columns is not 112 independent language tokens. Overlapping
windows, repeated prefixes and epochs do not multiply independent information.
A learned tokenizer changes attention cost and must retain a declared raw-row
accounting. Unique component counts are essential even when raw row counts are huge.

The often-used C approximately 6*N*T is only a dense matrix-work proxy here. It
omits repeated context, attention score work, Muon orthogonalization, auxiliary
heads, recomputation and communication. Use I and measured phases for estimates;
never infer B300 wall time from an advertised low-precision FLOP rate.

## Scaling design: learn a frontier before a law

Freeze one causal backbone, objective, encoding, data membership, context and
checkpoint policy. Initial ladder uses the existing trunk specifications:

| Approximate trunk parameters | Layers | Width | FF width | Role |
|---|---:|---:|---:|---|
| 0.206B | 16 | 1024 | 2816 | optimization and data-reuse anchor |
| 0.679B | 24 | 1536 | 4096 | medium teacher |
| 1.644B | 32 | 2048 | 5632 | first large teacher |
| 3.172B | 40 | 2560 | 6912 | subsequent extension if informative |

These change width and depth together: conclusions concern scale RECIPES, not a
pure parameter-count causal effect. Input/readout parameters are additional.
Keep head dimension 64 and SwiGLU initially. Do not fill 288 GB just because it is
available. Allow 3B+ if its validation trajectory and optimization support it.

**Proposed replacement first capacity panel: six total new fits**, three sizes
0.206/0.679/1.644B times two matched fold/seed cells. Reuse the previously proposed
F0/1001 and F2/1002 identities only after verifying ownership and clean ancestry.
With two cells neither seed variance nor a scaling exponent is well identified.
This proposal extends the old four-fit capacity panel; it does not authorize both
panels. Before launch amend the machine recipe explicitly, bind sources and budget,
and label the old recipe superseded. The event-objective panel remains separate.

Save resumable stable-prefix checkpoints and predictions at predeclared budgets,
for example 1/2/4/8/12 fitting passes, subject to measured cost. Evaluate larger
models after sufficient optimization; a weak first epoch is not a scale verdict.
There are two legitimate comparisons:

1. Equal T/D: same fitting-row presentations and membership, revealing scale at
   a fixed data-reuse budget while compute differs.
2. Equal measured C: compare learning curves at overlapping GPU-hour budgets,
   revealing the best quality for the available compute. Larger models see fewer
   updates/rows at that budget; report that fact rather than hiding it.

For WSD, a stable-prefix snapshot and a cooled terminal model are different
endpoints. Use stable-prefix curves consistently; a cooled fork at an early budget
requires its own recorded extra updates/GPU-hours. Such forks share ancestry, are
not independent fits/replications, and must be listed as extra training work.
Never compare a cooled small model to an uncooled large model without labeling it.

After the initial panel, choose ONE informative 4-fit follow-up: size contrast,
training length, context, or objective. It is separately allocated, not hidden in
the six-fit screen. Keep exposed folds labeled exposed; no newly pristine holdout
is created by larger models, another seed or server execution.

## How to estimate scaling without fooling ourselves

Start with learning curves and a Pareto table of official WP against GPU-hours,
model size and reuse. Primary utility is the unchanged global_wp_clip2_v1; report
per-target, component and sequence-position behaviour and matched differences.
Track a fixed supervised validation loss separately from auxiliary NLL. Improving
next-event likelihood alone is not evidence of better competition predictions.

A possible later descriptive loss model is

    L(N, D, R) = L_floor + A*N^(-alpha) + B*D_eff(D,R)^(-beta)
    D_eff(D,R) = D * [1 + rho*(1 - exp(-(R-1)/rho))], R >= 1.

This is a proposed saturation parameterization, not a fitted or validated law.
Repeated rows still supply optimization exposure; D_eff is a fitted proxy, never
an asserted count of independent samples. At fixed D the repetition scale rho
is not identifiable from a few endpoints. Context and objectives require separate
strata or much richer designs. Do not jointly fit floor/alpha/beta/rho from six
fits and report a confident optimum. Checkpoint points share a trajectory and do
not increase the number of independent fits.

Use later evidence across several sizes, fitting-component fractions and budgets
to compare a power law with a simple monotone/saturating alternative. Weight and
resample at dependency-component level, report fitting-seed variation separately,
and assess forecasts on a subsequently allocated size/budget. Initially extrapolate
at most one size/budget rung. Do not translate a fitted loss delta into a public
score, or assume official correlation utility obeys a power law.

Data-scaling follow-up can compare 25/50/100 percent of fitting COMPONENTS with
entire sequences retained. Use nested membership, matched clocks and controls;
fractions change distribution as well as size. Adding valid.parquet to final
all-data fitting removes its held-out interpretation. No extra external market
data or synthetic-data benefit is assumed.

## Parameters to tune, in priority order

| Priority | Axis | Initial candidates | Interpretation |
|---|---|---|---|
| 1 | LR and parameter update scale | Muon base 0.5/1/2 times 7.5e-5; head AdamW LR independently recorded | rule out an optimization-limited large model |
| 2 | Capacity versus reuse | ladder above; predeclared learning-curve budgets | locate compute frontier |
| 3 | Context | 1024 vs 4096 initially; 8192/full-sequence slow memory later | test useful long-history signal |
| 4 | Dynamics objective | coefficient 0 vs 0.05; event horizon 64 fixed | teach predictive dynamics without confounding size |
| 5 | Numeric encoding | raw/group encoder vs raw plus numerical embeddings | help continuous value and cross-group reasoning |
| 6 | Generalization | AdamW decay 0.01 vs 0.1; dropout 0 vs 0.05 when indicated | address observed train/validation divergence |
| 7 | Shape/attention | matched-parameter depth/width, FF ratio, QK norm or gated attention | tune after signal and stable scale exist |

Candidates are a menu for successive bounded panels, NOT a Cartesian sweep or
an assertion all will run. Muon uses match_rms_adamw convention; original Muon
LR 0.02 is not interchangeable with these bases. Norm/bias/embedding/readout groups
remain explicit. Strict FP32 requires auditing the Newton-Schulz internals.

Recommended optimization starting point: Muon momentum 0.95, five orthogonalization
iterations, global clipping 1; AdamW betas (0.9,0.95), eps 1e-8 on remaining groups.
Retain already frozen optimizer settings when reusing an existing control. Changing
them creates a new recipe. Log groupwise update/weight RMS, preclip norm, clipping
frequency, nonfinite events, activation RMS and attention entropy at bounded
frequency. Diagnose large norms and frequent clipping before changing capacity.

Use WSD as the extensible default: warmup approximately 5 percent of the declared
initial update budget, stable prefix, final 20 percent cooldown. Freeze integer
boundaries and ending LR before fitting; the old 1500-update warmup is not suitable
for every short diagnostic. Compare cosine only in a separate schedule panel.
Checkpoint-averaged endpoints are cheap to evaluate but require predeclared weight
averaging windows and cannot be counted as independent models.

Start from the prior nominal 65,536 new core rows/update only if a real-shape
profile supports it; it is not an optimum. Screen half that batch if optimization
or component diversity suggests it. Change accumulation to preserve global batch
when comparing 4-way with 8-way DDP. Optimize completed comparable fits/hour, and
choose 8-way for an individual large fit only when measured scaling justifies it.
muP is a possible later width-transfer method, not a guarantee for this Muon recipe;
do not introduce muP and a new optimizer simultaneously into the scale control.

## What the observations would mean

- Larger train AND validation improve, with usable slope: extend capacity/compute.
- Larger train improves but validation worsens: investigate reuse, component
  diversity, regularization and objectives; do not simply add more epochs.
- Larger train is worse: investigate optimization/update scale and sufficient
  updates before interpreting it as a statistical capacity limit.
- Long context hurts: check masks, effective row budget, prefix cost and optimization;
  compare useful memory at matched compute before rejecting long-range dynamics.
- Auxiliary loss improves but official utility is flat: representation changed,
  yet target relevance is unproven; measure target effects and reduce its weight.
- Predictive signal improves at higher cost: retain the quality endpoint and Pareto
  tradeoff. No GRU-transfer gate is required by the latest teacher-first objective.

## Next concrete handoff

Amend PR31 with this teacher-first priority. Implement/profile the causal trainer
before fitting; freeze the six-fit ladder OR a separately chosen four-fit LR/event
screen according to the measured blocker. Receiver must declare a server GPU-hour
allocation and preserve actual run/evidence/weights manifests. No launch time,
new benchmark, scaling exponent or better submission is claimed by this document.
