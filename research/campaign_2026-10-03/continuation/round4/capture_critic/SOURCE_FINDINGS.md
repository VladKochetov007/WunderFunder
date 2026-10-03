Independent source findings and acceptance protocol
=================================================

Blindness and tier
TESTS.json and initial PREMORTEM.md were written before actual trainer inspection
or execution. No round4 builder file/result has been read. Permitted prior reads
were only objective_state_gate/export_schema.json and REPORT.md, plus resource
contracts. No checkpoint values, datasets, supplied-data scores or GPU launches.
This is tierA source-bound fixture evidence. It does NOT admit tierB actual
CUDA/fp16 online replay, tierC training improvement or any GPU candidate.

Actual source ordering (scripts/train_rnn.py)
-927..928: --fast sets sync_free/fused_adam/pin_memory/prop_cache. It does not
 substitute a separate GRUfast recurrent implementation; make_model returns
 RNNModel with nn.GRU (src/wnn/nn.py:14..80). Under optim=muon the MuonAdamW
 branch at1237..1240 wins, so fused_adam does not select fused AdamW there.
-1254..1259: fast/sync_free chooses FastEMA; otherwise AveragedModel. Both teacher
 models are eval; raw is train at1303. Modes must be captured, not guessed from
 equality of weights. Normalization mean/std are model buffers.
-1393..1404: raw forward then detached outgoing raw hidden; lagged EMA forward
 under no_grad then detached outgoing EMA hidden. These are separate histories.
 h=h_ema=None only at the start of each batch; later chunks carry historical
 states. Incoming hidden cannot be recomputed by applying current weights to
 invented history or zeroed because the weights happen to match.
-1511..1516: consistency uses the PRE-update teacher prediction; target remains
 detached, unclipped, and is not recomputed after the weight update. Weight
 broadcasting/denominator and optional metric-weight scaling must match source.
-1518..1538: zero_grad; scale/backward; unscale; record AMP found_inf; optional
 grad stats; clip; scaler.step; scaler.update; scheduler.step; attempted
 step_count++; optional SWA; EMA.update_parameters; EMA counter++. This exact AST
 block is executed in fixtures.py, with only profiling timer a no-op.
-364..387: FastEMA copies PARAMETERS on its first update (n==0), then foreach
 lerp with 1-decay; n increments on every call. It does NOT refresh buffers.
 AveragedModel is a different branch with different state/buffer semantics.
-src/wnn/optim.py:59..104: Muon has momentum_buffer, orthogonalization and group
 blocks/ns_steps/rms; Adam groups also have exp_avg/exp_avg_sq/step. Muon has no
 per-parameter step field. Group betas/LR can change with the scheduler. CPU Muon
 orthogonalization uses FP32; CUDA uses BF16 internally, even in fp16 training.
-src/wnn/run_identity.py:49..80: completed optimizer count is attempts minus
 observed found_inf skips; this does not force scheduler/EMA clocks to agree.

Required captured state and boundary
Capture before BOTH forwards, at a precisely named post-input/post-label/pre-LR
or post-LR phase; source-selected current LR must be present. Bind exact chunk
position, batch sample order, reset boundary, input processing/normalization,
final labels/main weights, clipped cached teacher values and bootstrap survival,
structural labels/weights and next-change labels/masks. Targets/aux labels are
training-only captured objective inputs, never inference features.
Capture raw+EMA weights/buffers/modes and detached incoming h/h_ema; CPU and all
used device RNG states; optimizer implementation and name-bound group/moment
mapping; scaler state/growth tracker; scheduler state/current group settings;
EMA decay/n and branch; attempted/completed/EMA/scheduler clocks. Record precision,
autocast device/dtype, cuDNN/determinism/TF32 settings, versions and source hashes.
Capture immutable owned tensor copies, not aliases. If capture is before the
source zero_grad and there is no gradient accumulation, stale grads are cleared;
do not invent an accumulation state. Any alternate boundary must specify this.

Acceptance before comparing aligned/permuted arms
Restore pristine state independently for each arm; first compare DEFAULT replay
with the actual online step. A replay-to-itself comparison cannot certify capture.
Capture-disabled and capture-enabled trainer semantics must also match. Neither
is tested here because builder remains blind/unread and no online GPU step exists.
Require raw/teacher predictions, outgoing h/h_ema, frozen teacher target, every
loss component and total, unscaled gradients, clip norm and post-clip gradients,
parameter DELTAS, optimizer post moments/groups, scaler, scheduler and EMA state.
Every name/order/discrete clock/source/input identity is exact. Nonfinite skip
must leave raw parameters/moments unchanged while preserving SOURCE advances.
Inspect found_inf after unscale before update; step() return None is not a skip
detector. Comparing only final EMA or aggregate counts is an automatic rejection.

Tolerance commitment
For a deterministic fixed backend and operation order, require tensor BYTE
equality, including shape/dtype, plus exact discrete state. fixtures.py compares
byte views, and a signed-zero corruption confirms sensitivity. Same snapshots
on our deterministic CPU backend passed five repeats and serialization exactly.
Independent algebraic comparisons are separate and may use documented dtype
summation bounds; they do not relax default replay parity.
For a genuinely nondeterministic actual backend, first run >=5 independently
restored DEFAULT repeats before inspecting arms. Freeze each output tensor's
maximum pairwise absolute repeat error floor f; use <=10*f maximum absolute
error as the predeclared empirical envelope (f==0 means byte equality).
Also preserve elementwise/relative errors and gradient/delta norms, and freeze
the manifest before arms. Any exceeding tensor fails. A noisy envelope unable
to distinguish the intended intervention is inconclusive, not a pass. This
empirical protocol means bounded reproducibility, not exact same-step identity.
CPU FP32 floors cannot authorize CUDA/BF16-Muon/fp16/cuDNN tolerances.

Measured fixtures and limitations
Actual RNNModel hidden4/layers2, active dropout.35, two outputs, 2x5 synthetic
chunk, four warm steps; actual MuonAdamW, source-extracted FastEMA, actual enabled
CPU GradScaler and StepLR. Frozen main weighted_mse_loss plus source-equivalent
consistency stress loss; no full23-head combo objective, distillation/bootstrap,
structural or next-label scoring. CPU FP32 is deliberately NOT fp16 combo replay.
The scheduler fixture is StepLR, not the actual selected OneCycle/WSD schedule;
it demonstrates unconditional source ordering, not schedule-specific parity.

13 controls pass; 11 adversaries rejected:
1 wrong raw hidden; 2 wrong EMA hidden; 3 swapped hidden trajectories;
4 altered Muon momentum; 5 altered Adam momentum; 6 post-forward RNG;
7 optimizer ID/name order reversed (same-shaped GRU matrices);
8 recomputed teacher from updated EMA; 9 skip count falsely completed;
10 EMA/scheduler held on skip with aggregate clocks unchanged;
11 attached teacher prediction target (actual EMA forward with autograd enabled).
Moment corruptions retain identical gradients AND clocks, but change parameters.
Six state corruptions retain aggregate clocks; updated-EMA teacher error
is .02670019865 despite identical counters and immutable original teacher value.
For the forced nonfinite loss, optimizer attempts1/completed0/skips1/EMA1;
attempted step_count goes4->5, scheduler advances, scaler scale halves, raw weights and
optimizer moments stay byte-identical; lagged EMA weights still change.
Equal weights/hidden AND modes give exact teacher identity; equal weights alone
with different hidden states do not. Mode equality matters under dropout.

All mismatches, source hashes, prereg hash and resource cost are in results.json.
run_attempt1/results_attempt1/resource_attempt1 retain the first passing run.
Second run tightened numerical equality to actual byte equality and added the
precommitted attached-teacher corruption. Final source/comment provenance rerun
is retained separately; no outcome or tolerance was changed after scoring.

Next independent review
When controller supplies frozen builder, first map its capture boundary/state
against TESTS.json, then adapt these adversaries to its validator/replayer without
training/scoring. Demand actual default-online comparison before using an arm's
gradient difference. Check buffer handling, fast initial-copy n, Muon name map,
scaler found_inf timing, scheduler beta/LR and skip clocks first. Unsupported
actual combination must reject explicitly rather than fall back to this CPU toy.
No candidate approval or training recommendation follows from these fixtures.
