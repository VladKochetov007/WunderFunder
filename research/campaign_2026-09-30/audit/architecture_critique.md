# Independent specification critique, 2026-09-30

Verdict: retain A/B/C as distinct proposed families; approve CPU implementation
and falsification, not GPU fitting or promotion. B has the cleanest first
comparison once its scan, precision and training chronology pass independent
checks. A needs an effective-capacity control specification. C needs a router
gradient specification and stronger parity preparation. This critic authored no
architecture and consumed no GPU time.

Evidence inspected: `proposals/architecture_cards.{md,json}`, the resume
contract, the frozen evaluator, and the new data reconstruction contracts,
source and raw JSON. Arithmetic/graph identity checks are saved in
`data_spec_audit.json`; this is a partial independent audit, not a rerun of the
entire decoder on raw parquet. Public transfer remains verification tier C.

## Shared conditions before any fit

Freeze implementation, configuration, seed, component manifest, normalization,
propensity, features, auxiliary targets, update order and fixed checkpoint epoch
before viewing evaluation outputs. Every arm processes the same dense timeline,
including warmup and unscored rows. Scoring masks may select loss/output rows but
must never select memory writes or define the event clock. Scored-only hidden
caches are usable as independent readout examples, never as adjacent time steps.

The clean pair is a two-model anchor. A single new backbone has a different
training budget, so the causal mechanism claim comes from the matched A/B/C
control, not from an allegedly equal-cost comparison with the pair. Report both
arm-alone and fixed 80/20 blend scores; compare candidate and matched-control
blends with the exact same pair predictions. Do not compare selected/best model
checkpoints with fixed checkpoints, or unequal successful update counts when one
arm hits its runtime cap. A timeout makes that comparison inconclusive.

An identical tensor count does not ensure identical effective capacity. Record
trainable parameters, parameters receiving nonzero gradients, state bytes,
per-row operations including all state updates, optimizer steps, examples, and
observed resource/time consumption. A dummy projection executed on zeros can
inflate the nominal control budget without matching statistical capacity.

The apparent 0.6778 fold-0 score is a clean recipe analog, not a public forecast.
The new mechanisms have no established public transport coefficient. Keep the
0.6660 expectation gate separate from screen and confirmation effects. Fold-2/3
are less-exposed confirmation, not untouched data. The existing fixed metric
and centered component-bootstrap source hashes remain the general score grader.

## A: normalized decayed content memory

The position control needs an exact, persisted input basis and dimensions before
implementation. Learned constants plus a few clocks can leave most q/k weights
unidentifiable while content uses the entire hidden vector. Disclose that
difference or supply a clock/event basis that exercises comparable projections;
neither equal tensor shapes nor equal multiply counts alone resolves it. Keep
values, gate, readout, insertion timing, half-lives and event flags identical.
Current-row insertion must be included in both candidate and control. A gain
then supports content selection conditional on this specified control.

The event clock definition must match the causal feature implementation; raw
representation-code changes are not automatically quote-price events. Before
any quote event, event-head state/output is zero. On quiet rows event heads have
`a=1,w=0`; decaying them by row count changes the proposed mechanism.

Required CPU falsifiers: independently step the matrix/normalizer recurrence and
compare carried-state microblocks at 1/17/64/2000, nonzero incoming states, and
events exactly on boundaries. Use zero events, a first-row event, 20,000 quiet
rows, dense events, repeated contexts, and extreme finite q/k inputs. For
positive q/k maps and nonnegative normalizer, each retrieved value coordinate
must lie within the coordinate-wise range of written values plus zero when
the denominator is clamped. Require finite nonnegative Z and finite outputs.
Under FP16 autocast, S/Z updates and normalizer arithmetic must really execute
in FP32, not merely store the result afterward. Compare gradients through
microblocks with a sequential reference, including incoming S/Z gradients.

A CPU null result on frozen random/fixed representations does not reject the
learned memory family. A positive readout result does not validate dense causal
memory or the proposed GPU training kernel.

## B: structured state model

Both multi and single arms may learn any half-life in [1,8192], so their
comparison tests initialization/optimization. It cannot isolate whether the
multiscale hypothesis class itself beats a single-scale hypothesis class. Keep
that narrow claim; a future fixed/tied-decay control would answer a different
question and requires a separate preregistration. Freeze the identical phase
grid, injection normalization, residual gates and initialization for B/C/output
maps. Changing only rho can still change gradient scales and initial output
variance; report those measurements instead of silently retuning one arm.

Required CPU falsifiers: independently verify scalar and real 2x2 recurrence,
impulse decay/rotation, analytical incoming-state propagation, and constant
inputs. Include rho near the upper half-life bound, positive/negative signals,
zero and nonzero initial states, all sequence partitions, and gradients with
respect to rho, B, C, gates and incoming states. Scan composition must respect
time order. Each block is linear in its own recurrence; the two nonlinear
blocks cannot generally be collapsed into one global linear convolution.

Compute constrained rho and slow-state transitions in FP32. Casting a rho near
one to FP16 can erase the intended decay. Use a numerically stable formula for
small `1-rho²`; do not rely on subtraction in reduced precision. Variance-scaled
injection does not bound the mean response to constant input: its steady-state
gain grows as rho approaches one. Compare observed magnitudes with the explicit
constant-input bound, not an invented requirement that every state remain
below one. A learned positive gap effect and long-mode occupancy are diagnostics
under frozen slice definitions, not validation-driven parameters or new loss
terms. Freeze the long-gap event/threshold definition using fitting data only.

The optional ridge probe needs fixed causal projection/state matrices,
normalization from fitting rows, a declared fit-group selection and regularizer,
and dense warmup/history. Do not choose fixed modes, projection seeds or ridge
strength on the evaluation fold. Its null result is evidence only for the
fixed-state feature construction.

## C: sparse mixture of operators

Specify the router gradient estimator before implementation. Argmax itself has
no useful derivative; selected softmax weight, a straight-through estimator,
and a soft/dense training path produce different algorithms. Persist the exact
selection, probability weighting, balancing loss coefficient, gradient rule
and inference behavior. Test that router parameters receive task gradients and
that expert states/readouts receive the expected gradients. A train/inference
top1 declaration is incomplete without these details.

The fixed hash router cannot optimize its assignment or occupancy. A learned
balancing loss versus an unchangeable hash balance has different attainable
constraints. Measure initial/evolved occupancy and task exposure by expert;
freeze the hash salt and quantization before labels/evaluation. Explain whether
hash features depend on the trainable shared backbone: quantized learned hidden
vectors can change route discontinuously as the backbone trains. A feature-only
hash and a learned-hidden hash are different controls. Universal-only comparison
also differs in stored parameter count; keep that disclosed.

Required CPU falsifiers: all four expert states update on every row, even if
their readout is never chosen. Change the selected route on the next row and
verify the previously unselected states match full-update references. Compare
training/evaluation/streaming routes and probability weights. Include exact ties,
near ties, quantization boundaries, signed zero, quiet prefixes and reset after
each expert has accumulated state. Lower-index argmax tie-breaking cannot prevent
an arithmetic perturbation from reversing a near tie. Persist route margins;
later native AVX2/v2/v3 checks must agree on route identity, not merely average
prediction closeness. Dense evaluation of all expert readouts changes the claimed
active work unless every comparator uses it and the budget claim is revised.

## Data reconstruction: supported scope and falsifiers

An independent undirected adjacency traversal of `cache/chains.npz` verified that
the 64 fit groups occupy 64 graph components and the 32 test groups occupy 32
disjoint graph components. Current metadata, source and contract hashes match
the three result artifacts. Selection precedes target/mask access in the script.

Mapped interior-code spread values transfer exactly on this test panel. However,
all-row map coverage is 80.54%, 95.85%, 99.90%, 99.99% in classes 0..3. Outside
offset-two exactness is 57.60%, 71.57%, 98.78%, 99.75%; the simple law is not
universal. Classes 0/1 have only 4/3 test components. Three class-1 target changes
occur without a representation-price change. Exact repeated rows and supported
votes within one plateau are not independent observations. Report unique code,
event and component support alongside row-weighted quantities.

The target-derived class and target-selected interior branch are diagnostic
oracles. A deployed feature pipeline cannot choose them using targets. Require
a causal feature-only class decision or a class-free decoder, fitted outside
evaluation components, with unknown confidence/coverage explicit. The transition
map uses target differences and assumes a common future reference. Its exactness
does not independently identify physical bid/ask movement. Nonzero supervised
cycles reject a potential for that supervised table; contamination of movement
by changing future references remains an alternative explanation.

The adaptive barrier overshoot model loses constant2 on known origins for
classes 0/1/2: 72.68% vs 81.61%, 72.90% vs 85.56%, 96.25% vs 98.10%. On the
known offset>2 subset it matches zero of 556/497/125 origins. Class 3 ties at
100% on the eligible subset, with no eligible offset>2 examples. Known-origin
coverage is only 25.27%/55.27%/82.55%/88.76%. Treat this as counterevidence to
this stated overshoot rule, not proof against every stopping/horizon mechanism.

The barrier contract promises unknown and right truncation separately, but the
implementation combines them with fully observed no-hit cases in
`no_known_first_barrier_within64_origins`. Split these outcomes before a further
physical claim. Derive quote/target signs and the threshold on independent toy
bid/ask paths: the contract uses entry/current spread+2 while implementation
uses proxy displacement>=2, so the algebra and proxy coordinate must be shown.
Also count target/branch changes inside a code plateau before assigning all its
rows the target at the event origin. Report first/last-row horizon availability.

Next CPU falsifiers should use a newly frozen, component-disjoint TRAIN panel
with accepted mappings and exact formulas unchanged; show per-component coverage
and failures. The adaptive offset/barrier followups reused the original panel,
so they are exploratory. Persist fit code/movement dictionaries and hash them,
not only the scripts; save imported decoder-source hashes in followup sidecars.
External quote paths may falsify an algebraic mechanism, but their numeric
transforms must not become competition inference mappings without authorization.

## Independent grading preparation

`architecture_cpu_grading_contract.json` freezes generic causality, reset,
partition, forward/gradient and numerical-stability criteria before actor fits.
It is a protocol commitment, not a claim that a grader implementation or any
actor has passed. The existing metric evaluator bytes remain frozen. The grader
will consume immutable actor modules/checkpoints and cannot be edited by actors.
Publish actual probe inputs, outputs, discrepancies and source identities when
runnable actors arrive; do not accept a self-reported actor pass as independent
evidence. A controller may amend the protocol before fitting with a new dated
artifact; post-fit amendments need an explicit reason and retained old protocol.

## Pooled-map reproduction and bounded label screen

The later pooled lookup resolves the oracle-class concern for this sampled
coordinate. `replay_pooled_code.py` independently rebuilt all 1,026 accepted
code-to-(class,spread) entries and every accepted training vote/total from raw
TRAIN parquet, without importing actor code. All 32 group/class evaluation
counts match exactly in `pooled_independent_replay.json`. The durable CPU job
finished exit0, 9.19s compute wall time, about 1.04GB peak RSS, one thread/2GiB.
Its inference lookup accepts only current rounded code plus the frozen fitting
table; labels determine offline supervision and retrospective grading only.
Unknown-code coverage and small class0/1 component counts remain limitations.

After the four-fold clean controls finish, a two-arm fold0 label-only screen is
justified as a bounded falsifier. It must start both arms from exactly the same
clean seed3 parent fixed-epoch3 checkpoint bytes and run exactly one fixed
continuation epoch, with identical optimizer reset/resume policy, learning rate,
data order, random seed, batches/accumulation, EMA initialization/update and
number of steps. Keep the backbone, 12-bin spread head, branch/class labels and
main/other losses identical. Original target-proxy spread labels are the control;
pooled current-code interior-derived spread labels are the candidate.

Build the static lookup only on the fold's permitted TRAIN fitting components,
with no validation labels or masks. Freeze its table bytes before continuation.
Use the exact same code-known mask, same loss normalization/weight and same
12-bin clipping in both arms. An unknown code disables that spread loss in
both arms; it must never revert to an oracle class or silently use the target
proxy only in the candidate. This mask-matched original-label arm differs from
an unchanged full-mask continuation. The shared unchanged parent/pair remains
an anchor, and a full-mask continuation would be a separately budgeted control
for the masking change. Preserve every original default/control artifact.

The candidate map changes training supervision only in this screen. No new
class/branch inference feature, main target transformation, post-processing or
validation weight choice is authorized by the label comparison. Freeze the
single-member and fixed-pair blend formula before evaluation. A renamed or more
accurate physical coordinate does not imply better prediction: the original
`-loser-2` proxy contains extra future-tail information on outside rows. Require
the matched control win and existing >=+0.0005 screen gate before confirmation.

Defer a separate future-excess-offset head until it has a matched active head
with the same parameters, loss weight, training rows and update count. Adding
an unused head to the control does not match effective extra capacity. Any
third-arm target/parameter change needs its own prior contract and cost; do not
interpret a three-arm win as solely the effect of current-spread correctness.
