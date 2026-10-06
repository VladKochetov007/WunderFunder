# E-496 supervision frontier

Status: fixed design and unsealed job specifications only. No fitting, prediction,
calibration, resource probe, scoring, or ledger write has been launched here.
The controller owns E-496 and the shared campaign ledger. H-180 is registered
for the alpha-by-bootstrap-horizon mechanism; this directory also covers the
registered H-126 denominator and H-134 direction-auxiliary hypotheses.
H-124 remains the separate E-494 surviving-span reanchor experiment.

The question is whether the strong two-teacher GRU recipe is sensitive to the
fixed distillation blend/horizon, benefits from a small current-target direction
auxiliary, or benefits from a fitting-pool-frozen update denominator. The causal
mechanism is a change to training targets, a discarded direction head, or the
main loss divisor respectively; none changes inference inputs. Verification is
Tier B for component-excluded comparisons, with public transfer still Tier C.

## Frozen parent and readout

Each block is `(seed=3, fold=0)` screen, `(seed=4, fold=0)` confirmation, and
`(seed=3, fold=1)` confirmation. Every parent uses the current strong recipe
from `scripts/continuation_recipes.py`: train+valid fit source with the fold
manifest, H256 two-layer GRU, B32/T1000, four epochs, fixed epoch 3, Muon on RNN
matrices, EMA 1 / .999 with eval-EMA, `wmse`, structural and next-event weights
0.3 each, propensity power 2 / epsilon .03 / true-mask epsilon .03, since
weight 32 / floor .25, fold-specific seed-3/seed-4 since teachers, and seeded
batch ordering. All runs preserve component exclusions, no evaluation during
fit, the same fold-specific teacher and propensity caches, and identical
normalization membership. Readout is historical GELU; no reanchor, candidate
activation, schedule, or other auxiliary is added.

Each parent is paired with a frozen-input head fit from its own parent epoch 3:
valid fit source under the same fold, batch 16/chunk 2000, one AdamW epoch at
.0005, input and GRU frozen, fixed epoch 0 and seed 3. Primary complete endpoint
is the fixed 0.5 parent / 0.5 head prediction mixture; parent-only predictions
are a separate endpoint. No head or arm is chosen from evaluation performance.

## Fixed arm blocks and contrasts

The H-180 block is a complete six-cell grid, repeated at all three seed/fold
blocks: alpha `{0.50, 0.70, 0.85}` crossed with bootstrap horizon `{0, 8}`.
All cells use chunk denominator, direction auxiliary 0 and reanchor 0. The
predeclared contrasts are boot-8 minus boot-0 at each alpha; alpha .70 and .85
minus .50 at each horizon; and the two corresponding difference-in-differences
against alpha .50. All six cells and contrasts are reported regardless of
screen outcome; no hierarchical gate or post-hoc arm selection is allowed.

The H-134 direction block holds alpha=.70, horizon=8, denominator=chunk and all
other settings fixed. Compare `--aux-sign 0` with `--aux-sign 0.1`. The .1
coefficient is a fixed proposal for this preregistration, not a recovered
historical setting and not a sweep. The target is the sign of the current
clipped contrast `clip(y0)-clip(y1)>0`. Its BCE uses ordinary main row weight
times `min(abs(clip(y0)-clip(y1)), 2)`, divided by that weight sum. The added
output head is discarded at inference.

The H-126 denominator block holds alpha=.70, horizon=8, direction auxiliary 0
and all other settings fixed. Compare `chunk`, `fixed`, and `shuffled`, with
predeclared fixed-minus-chunk, shuffled-minus-fixed, and shuffled-minus-chunk
contrasts. The fixed reference is the mean main-loss update mass measured by
`--calibrate-distill-reference` on a complete fitting-only loader pass for that
same seed/fold and fit membership. The calibration uses the actual recipe,
propensity and teacher input closure, CPU device, chunk denominator, and
calibration epoch 1; it exits before optimizer construction. The shuffled sham
uses existing `--denominator-shuffle-seed 314159`. Do not enter a scalar until
the calibration report is present and its format, fold, seed, fitting
membership, source and consumed-input hashes are checked. The supplied resolver
performs those structural checks and writes a downstream draft bound to the
calibration report hash; it does not estimate or invent mass. Calibration
reports carry every measured update mass as a canonical float64 little-endian
vector and its digest. The resolver verifies that digest and recomputes count,
mean, minimum, maximum, standard deviation and zero-mass count before resolving
either denominator template.

The common alpha=.70 / horizon=8 / aux=0 / chunk arm is shared across blocks.
Together these are nine unique parents per seed/fold: six grid cells, one
direction-on arm, and fixed and shuffled denominator arms. Each has its own
fixed epoch-0 head and parent/head prediction jobs. The same exact arm identity
may serve only the contrasts stated above.

## Evaluation and stop rules

Use frozen `global_wp_clip2_v1`, common fold-excluded components, both targets,
class-by-event-age diagnostics and 2,000 paired component bootstrap draws.
Report each seed/fold and the parent and complete endpoints separately. Keep
per-class harm at -0.002 as the existing guard. A failed lineage, missing or
mismatched calibration artifact, non-finite objective, invalid fit membership,
or resource-limit breach stops the affected arm and is recorded as such.
Calibration and resource profiles do not score hypotheses. A null result limits
only the registered fixed arms. No public-score forecast follows from these
proxy panels.

Before any run: independent source/spec review, one-worker resource admission,
fresh quota/budget/deadline check, and controller sealing. The current contract's
local start cutoff applies. Parent/head fit drafts use the continuation recipe's
measured full-shape resource envelope. Each full calibration pass is also its
own resource and timing observation: CPU only, one worker, 18,000,000,000-byte
host cap, zero device reservation, bounded timeout, and the runner must hide
accelerators and enforce the host-memory guard. Calibration is not a sampled
subset; if it cannot fit this reviewed envelope, fixed/shuffled arms stay
blocked. The authoring tests themselves are restricted to one CPU and 2 GiB.
All drafts remain unsealed and are not launch authority.

## Dependency graph and unresolved inputs

`dependency_graph.json` is emitted by the materializer and records every
parent, head, prediction, comparison endpoint, calibration and probe node. Some
fold-specific caches or raw data may be absent locally; each such path stays in
the `missing_inputs` array and remains an explicit provider dependency. The
fixed/shuffled parent jobs are `.template.json` files until the calibration
report resolves their reference-mass token. Calibration and parent runs depend
on fitting-only fold assets and fold-specific teacher/propensity artifacts;
heads depend on their corresponding parent checkpoint; prediction endpoints
depend on both fixed checkpoints. Missing inputs are never replaced by
historical scores or made-up constants.

The added probe plans are exact one-worker, production-shape resource profiles
(B32/T1000, one warm-up plus two measured steps) for the alpha/horizon envelope,
direction head, fixed denominator and shuffled denominator. They are plans only;
no local or provider process is launched by this materializer.
