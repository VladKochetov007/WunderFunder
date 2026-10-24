# B300 campaign with abundant compute

**Precision correction, 24 October:** [The explicit BF16/BF16x9 amendment](PRECISION_AMENDMENT_20261024.md)
supersedes the strict-FP32 wording in this historical proposal for every new
R4 cell, including G reference controls and H students. FP32 master/update state
remains. The team’s active W2 teacher fits are a separate source-bound execution.

**Implementation update, 24 October:** [R4 executable packet](high_compute_r4/README.md)
implements the conditional B–H stages below, including inner manifests, complete
configs, receiver commands and three-teacher cache composition. The text below
preserves the original proposal. R4 freezes H students at 30 epochs and adds 180
student fit-epochs; receiving-host acceptance and every scientific result remain
pending. The four historical R3 configs are unchanged.

User steering, 24 October 2026: compute is plentiful. Optimize for the strongest
transferable teacher and final CPU student, with more informative comparisons
and replication. The four R3 fits are the first capacity panel, not a campaign
ceiling. This is a **proposed continuation**, not extra runnable R3 cells or a
record of launches. R3 source/configs and its four-fit accounting stay frozen.

## Concrete continuation

R3 and G use full outer fitting membership. B–F use all inner-training membership
after reserving inner component validation from the outer fitting membership;
this requires new manifests and normalization caches. Every teacher fit uses
30 epochs over its own declared membership, strict FP32, Muon and fixed-final
scoring. Equal epoch counts do not mean equal rows or compute. No 10%/30% data proxy. Existing
outer F0/F2 remain exposed; inner search is development, not pristine evidence.
Freeze two source-qualified inner cells before search and pair every arm's seed,
data, masks and endpoint. Count failed attempts as attempted configurations.

| Stage | Comparison | New fits | Exact reuse |
|---|---|---:|---|
| A: existing R3 | 630M vs 1.54B; F0/F2 | 4 teachers already planned | None |
| B: LR | 1.54B; `1e-4`, `3e-4`, `1e-3`; 5% warmup; two inner cells | 6 teachers | None: R3 outer fits cannot substitute for inner-fit controls |
| C: warmup | Selected LR; 1%, 5%, 10% (about 0.3, 1.5, 3 epochs) | 4 teachers | Two exact 5% endpoints from B |
| D: forward context | 1024, 2048, 4096 rows; chunk512 / accumulation8 unchanged | 4 teachers | Two selected context1024 endpoints from B/C |
| E: chunk/horizon recipe | Selected context; chunk512/accumulation8 vs chunk2048/accumulation2 | 2 teachers | Two exact chunk512 endpoints from D or its selected ancestor |
| F: targeted capacity follow-up | 1.54B vs approximately 3.15B: width2560, depth40, heads40, FF multiplier4 | 2 teachers | Two selected 1.54B endpoints from E or its selected ancestor |
| G: robustness | Frozen selected recipe vs original 1.54B R3 recipe; 3 paired new seeds on each of F0/F2 | 12 teachers | None; train both arms on complete outer fitting membership |
| H: student transfer | Fixed raw112 GRU512x2: labels, single-teacher KD, three-teacher mean KD; two outer cells | 6 students | Candidate teacher checkpoints from G; no new teachers |

If every stage is retained and every stated control is reusable, the route is
**40 fits: 34 teachers and 6 students**, including the original four R3 fits.
The extension adds 30 teachers and 6 students, not 40 extra fits. Teacher cost is
1,020 teacher epochs across the whole route (900 additional), not comparable
wall time across model/context sizes. Student duration and optimizer must be
frozen in its own packet; student cost is not hidden inside teacher accounting.
Engineering checks/profiles, exports, replay and failed attempts are additional
costs, logged separately. A broken exact-reuse contract requires an explicit
replacement-control allocation, not pretending the old control is matched.

Stages are conditional, not an automatic queue or an instruction to complete
all 40 regardless of evidence. Initial LR discovery uses six new fits; later
stages answer separate questions and retain every result. The robustness stage
is a separately frozen confirmation attempt, not a 12-fit initial screen.
If the selected configuration is exactly the R3 reference, omit duplicate arm
training and amend the inventory. Extend a promising family when uncertainty
and likely endpoint benefit justify it; a zero-crossing interval is not a stop
rule. Freeze the selected recipe before reading stage G outcomes. Because F0/F2
are already exposed, G measures robustness, not untouched generalization.
The sequential LR then warmup search can miss interactions. It is a broad
coordinate search, not a joint optimum; a later LR-by-warmup factorial would be
a separately declared follow-up with its own cost, not hidden inside these 40.
The same limitation applies to selected context/scale ancestors. F is a targeted
extension of the existing capacity family, not a new 4–6-fit initial screen.

## What extra compute buys

- Search LR and warmup directly, rather than label the current `3e-4 / 5%`
  optimum. The `1e-3` arm is intentionally aggressive and needs its own peak-LR
  stress profile; unstable attempts remain in the search history.
- Separate forward memory from gradient horizon. A 4096-row KV context still has
  a 512-row backward graph in D. E keeps 4096 new rows per sequence per full
  update while extending backpropagation; attention context stays fixed.
  Padding, target denominators and final short windows retain actual weights.
  E also changes query chunk shape, activation memory and accumulation. Attribute
  its outcome to that complete recipe, not an isolated gradient-horizon effect.
- Inspect a larger teacher after a tuned 1.54B reference exists. The same LR at
  3.15B is a matched transfer of settings, not proof it is optimal at that scale.
  Add a separately declared larger-model LR follow-up if its curves justify it.
- Cross three seeds with two folds to distinguish their variation better than
  the original two confounded cells. Report paired deltas by fold and seed;
  neither component bootstrap nor six correlated cells guarantees significance.
- Use a fixed equal mean of all three frozen selected-recipe teacher seeds from G
  per outer fold for H. The single-teacher arm uses the first seed in G's frozen
  manifest, selected before seeing any G outer score. Both KD arms use G's same
  selected-recipe branch. Measure the
  student's gain against the same student's label-only control. A better teacher
  or a larger ensemble alone is not a submission improvement.
  H's F0/F2 student scores are also exposed, directional evidence, not untouched
  student generalization.

## Compute placement and implementation boundary

The confirmed recipe targets one eight-B300 group. Additional groups and their
available duration are not yet inventoried. Parallelize independent complete
fits across actually allocated disjoint groups; do not silently change an
eight-rank recipe to one/two/four ranks or increase batch to fill memory.
Different rank counts/global batches require separately profiled configurations.
Reuse immutable data, normalization and teacher caches only when their complete
source/split/ancestry identities match. No GPU job is launched by this document.

R3 already implements fit/profile/recovery, the stated optimizer and model
configuration fields. B–G still need frozen cell manifests/configs and launch
adapters: R3 `run.sh` deliberately accepts only its original four cell IDs.
Every changed architecture/context/chunk/LR needs real-shape acceptance and
resource profiling; do not bypass R3 guards or call these stages launch-ready.
H additionally needs a provenance-bound three-teacher cache merge, fixed student
recipe and its own CPU export/serving acceptance. The raw112 student is not the
native211/TL3 adapter; no compatibility or submission-size claim is made.

Use actual GPU-hours, optimizer/window time, total completed data passes, memory
and fixed endpoint quality to allocate the next stage. A fit count is not a GPU
budget or a duration estimate. Preserve some available capacity for exports,
student training and CPU verification instead of spending everything on teachers.
Data and causal representation can limit benefit even with abundant compute.
Keep 30 epochs as requested; any longer-duration comparison is a new arm, never
an extension selected using outer-fold checkpoints.

## Research basis and evidence

[Muon scaling research](https://arxiv.org/abs/2502.16982) supports examining
matrix update scaling and larger models, but its language-model results do not
establish a speed or quality advantage for this strict-FP32 time-series trainer.
[Data-constrained scaling research](https://arxiv.org/abs/2305.16264) motivates
measuring the value of repeated data and excess capacity; its numerical epoch
conclusions are not imported into this task. These are design motivations.

No new Python/runtime code, tests, training, inference or GPU profiles in this
amendment. Prior R3 static evidence and GitHub lint belong to their exact source;
actual B300 acceptance and all teacher/student outcomes remain pending.
