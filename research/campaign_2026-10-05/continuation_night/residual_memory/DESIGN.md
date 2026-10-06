# E-497: input-only residual matrix memory

Status: isolated prototype and resource plan only. No trainer/native integration,
resource probe, fit, export, or score has run. This block continues H-148 and
H-153; it does not claim that either mechanism improves the predictor.

## Mechanism

The module is a causal adapter around the existing predictor's row outputs. The
base GRU still computes its ordinary prediction at each row. A compact
external associative matrix stores only key/value features computed from the
current causal input. The current base prediction supplies the query. Each row
first reads the matrix left by earlier rows, adds a zero-initialized residual
projection to the base output, and only then writes the current input. Thus the
first residual is exactly zero for the tested FP32 and FP16 paths, writes from
row `t` cannot change the output at `t`, and no label, future row, target
transition, or dense time-by-time attention matrix enters this mechanism.

The state is one float32 matrix per sequence with shape `[batch, key_dim,
value_dim]`; key and value dimensions default to 16. The delta arm predicts the
value already associated with the normalized input key and writes the
input-conditioned error back along that key:

`M <- M + beta_t * key_t outer (value_t - key_t @ M)`.

The matched additive control uses the same query/key/value projections, event
and quiet gates, state shape, and residual output, but updates
`M <- decay * M + beta_t * key_t outer value_t` with fixed per-row decay
`0.999`, a frozen nominal `e^-1` retention horizon of about 1,000 rows. This is
an a-priori control constant, not a tuned result. A learned decay can be
enabled in the module for a later registered contrast; it is not part of this
two-arm resource plan. The module derives an
event flag from a configured column of the current-row input to select separate
event and quiet write-rate projections. Both paths can write on every row; an
event is not assumed to mean the target changed and quiet rows are not frozen.

The runner-facing adapter accepts base predictions and input features; it has
no label, future-row, or external event-mask argument. A future integration
must bind its event index to the current-row observable `ev_i0px` feature
(column 112 in standard `build_inputs`) and must call the explicit
`detach_state` at the registered recurrent-credit boundary. Passing
`state=None` resets memory between independent sequences. Carry the returned
matrix only across chunks of the same sequence.

## Falsifiers and checks

The source-level correctness gate is: exact base-output equality at zero
residual initialization for FP32 and FP16; no prefix change after perturbing
future inputs; same output and final state whether a sequence is passed whole
or as carried chunks; nonzero residual-head gradients on the first update and
nonzero query/key/value/write-gate gradients after that head moves; and state
storage bounded by `batch * key_dim * value_dim`, independent of sequence
length.
CPU tests exercise both delta and additive modes. These checks do not measure
predictive utility, full-shape throughput, or integration correctness.

The primary scientific contrast after integration is delta write versus
additive fixed decay, each paired with the same untouched strong GRU anchor.
Both compare parent predictions and the campaign's fixed complete head
endpoint on clean component-excluded folds. Preserve the registered
seed-3/fold-0 screen and seed-4/fold-0 plus seed-3/fold-1 confirmations, common
data/order/teacher lineage, target and market-class breakdowns, and 2,000
paired component-bootstrap draws. Event age is the number of rows since the
most recent top-of-book bid/ask change, with fixed bins `[0,4)`, `[4,16)`,
`[16,64)`, and `[64,infinity)` matching the existing event-age audit. The
24-fit/16-test cohort is not a substitute for these full-recipe lineages. No
in-sample fit score can promote the branch.

## Probe and integration boundary

`resource_probe_plan.json` records the future production-shape profile:
batch 32, 20,000 rows, chunk 1,000, base GRU H256/two layers, and matrix
dimensions 16 by 16, using only causal feature inputs and event column 112.
Measure both update modes' forward and backward wall time, peak
allocated/reserved device bytes, process-tree RAM, state bytes, and the
host/device OOM counters under the campaign limits. This is deliberately not
an executable training argv:
`train_rnn.py` exposes neither a memory-adapter hook nor a state-reset/detach
contract for this external state. Inventing a trainer flag would make a false
readiness claim. Integrate and review that interface before any resource
profile; only a passing probe and a new admission can precede science fits.

Native export, production model wrapping, checkpoint provenance, trainer
resume, and fit integration are unsupported and pending. The prototype is not
a submission model and supplies no score or deployment evidence.
