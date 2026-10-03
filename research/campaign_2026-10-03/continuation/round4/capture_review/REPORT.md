Independent frozen implementation review
========================================

Verdict: FAIL AS FROZEN for a scientifically admissible bounded GPU measurement
under the blind same-online-step acceptance specification. The normal source
arithmetic and pristine CPU control pass, but malformed optimizer identity and
non-exact discrete comparison can be admitted. Fix the gates below, refreeze and
retest only the decisive controls before controller authorization. This is not a
claim that the builder's 25 CPU tests failed, that current captures are corrupt,
or that CUDA parity is impossible. No receipt was written and no GPU was launched.

Measured decisive findings
-------------------------
1. Optimizer parameter-ID corruption is accepted (blocking specification gap).
   Swapping first-layer GRU ih/hh optimizer IDs [10,11] to[11,10], retaining the
   unchanged optimizer_names and same-shaped warm moments, passes validate AND
   restore. The resulting student/EMA weights, two momentum buffers and parameter
   deltas differ in eight saved fields. Preclip gradients and all clocks remain
   byte-identical. live_capture.py:306–335 derives ID->name by zipping the corrupted
   IDs with asserted names; restore:645–650 checks names, then load_state_dict maps
   states positionally. The manifest can therefore describe the wrong moments.
   Fix: bind independent canonical ID/name correspondence at capture, and verify
   each group's serialized IDs against the reconstructed optimizer's canonical
   state_dict IDs/names BEFORE load. Alternatively serialize moments directly by
   trusted parameter name and rebuild the ID state. Keep same-shape swap rejection.
   Actual normal writer produces canonical IDs; this corruption is an integrity
   adversary, not evidence of a spontaneously corrupted normal capture.

2. Comparison policy is weaker than the blind gate (blocking specification gap).
   compare:709 applies atol/rtol to every tensor, including integer clocks. With
   the prospective CUDA envelope abs1e-5/rel1e-4, int64 counter100000 versus100001
   is ACCEPTED; counter4 versus5 is rejected. A parameter-delta corruption of
   4.999999873689376e-6 is also accepted. This is expected float-envelope behavior,
   not a CUDA observation. Python integer clocks compare exactly, but tensor skips
   and other discrete state must not inherit floating tolerances.
   Fix: exact equality for integer/bool/RNG/name/order state; byte equality for
   deterministic same-backend floating replay. Before relaxing CUDA comparisons,
   measure >=5 independently restored default repeats and freeze per-tensor
   floors before interpreting online/replay or intervention differences. The
   present single repeat and prospective fixed envelope are not an empirical
   backend floor. If noise masks a relevant delta, verdict is inconclusive.
   Maintain explicit nonfinite mask checks; do not summarize only finite deltas.

Controls and source semantics that PASS
--------------------------------------
Our pristine CPU post-state replay matches BYTE-FOR-BYTE (independent recursive
comparator, not builder assert_close). Independent AST removal of guarded hooks
and capture conditional RHS expressions reconstructs the original chunk AST
exactly. This confirms addition order and disabled-hook math, not CUDA kernels.
Pending author review_request.json is correctly rejected by verify_review.
CPU execution used builder's actual trainer import/RNN/Muon/OneCycle/FastEMA and
four warmed historical chunks, full23-head four-loss recipe. It did not execute
runner.main (which rewrites frozen generated artifacts) or duplicate 25 tests.
PeakRSS789104KiB inside Python, wall1.98s; resource.log records external process
cost. No datasets, full prediction caches or mature model states were read.

Pre-forward capture owns detached raw and lagged EMA state_dict copies including
normalization buffers; distinct incoming h/h_ema; full processed inputs, raw y,
averaged cached teach, next-change bootstrap indices and final main/struct/next
weights/labels. It records CPU/Python/NumPy/device RNG after deterministic copying
and before forward, modes enforced as raw train/EMA eval, warm optimizer moments,
current OneCycle state/groups, GradScaler READY boundary and source/backend flags.
Restore constructs before resetting RNG, preserving the captured pre-forward
position. Stale gradients need not be captured: source clears them before backward.
The AST executes actual zero_grad/backward/unscale/found_inf observation/clip/
scaler.step/update/scheduler/attempted-clock/EMA order. CPU skip behavior was
independently established in the protected blind report; builder source preserves
it and has explicit same-prestate injected-inf replay. Nonfinite clipping can
yield NaNs while the skipped optimizer leaves raw weights/moments unchanged and
scheduler/EMA advance. Those source semantics must remain unnormalized on CUDA.

Additional necessary capture/output tightening
---------------------------------------------
- Capture.post omits raw out, lagged out_ema/frozen prediction target and returned
  preclip norm (only losses, gradients and hidden states are saved). Persist and
  compare these detached values before/after the update. Recompute neither the
  teacher target from updated EMA nor the clipping norm from already clipped grads.
- ema_decay is captured but restore hardcodes .999 and ignores that field;
  validate does not bind it to args. Require exact agreement with the fixed recipe
  or restore and validate the captured value. Warm n is correctly restored.
- Bind pre.pt/post.pt/components.pt/results content hashes and their source/input
  lineage after measured execution. Exact tensor captures permit reproduction,
  but the runner currently emits no prospective GPU-output hash manifest.
- Source drift binding covers nine actual training modules and passes here.
  Runner.main:243 calls generate_copy AFTER receipt validation and overwrites
  protected trainer.patch/instrumented_train_rnn.py/chunk_source.py.txt. Static
  identical regeneration may preserve bytes, but should be verification-only for
  frozen measurement; do not rewrite reviewed source or rely on that side effect.

Real-mode interpretation and input lineage
-----------------------------------------
The frozen panel's exact28TRAIN+4fitVALID identities, provenance and sidecars are
bound. Real entry hashes mature teacher arrays/target maps and checks dataset/
propensity stats; prop_cache_path independently binds propensity sidecar checkpoint
hash, but does not validate propensity array row order. Dataset and propensity
contents are not fully hashed; selected tensors
and metadata disclose this limit. No scientific claim upgrades those stat identities
to full-file content identity. Our protected lineage audit found no reported fit
component intersections, but older teachers' executed source/membership remain
unknown. This does not automatically invalidate a shared paired contrast.
Fresh seed3, full fixed-panel normalization, four warm chunks then chunk4000 are
valid for a CONTROLLED SMALL TRAJECTORY. They are not incoming states of today's
full fit and cannot recover mature raw/EMA/history from evaluation EMA exports.
Full20000-row labels preserve bootstrap/next/end boundaries; normalization uses
fit-panel data by deliberate setup, not causal prediction-time normalization.
Cached distillation bypasses running norm_ema under this recipe, as source does.

Concrete controller runtime integration
---------------------------------------
The actor gpu_smoke_review_only.sh directly calls scripts/limited in the parent
slice: FAIL as the launch path; it lacks exclusive campaign admission/monitoring.
The supplied controller_gpu_payload.sh + RUNTIME_INTEGRATION.md correct that
architecture WITHOUT changing frozen Python. Static review of campaign_job.py
shows nonblocking exclusive gpu.lock admission, rejection of still-active prior
GPU scopes, family slice validation, persistent records, source fingerprints,
periodic total-device memory measurement and stop/incident handling. run_gpu.sh
sets8cores/24GiB, family inheritance, no swap and appropriate thread pools. The
payload requires campaign marker+GPU slice. PASS for this runtime chain design,
provided the controller actually uses campaign_job launch --kind gpu --timeout600
--threads8 -> run_gpu.sh -> controller_gpu_payload.sh RECEIPT FRESH_OUTPUT.
Markers alone are not authorization; direct execution of the payload is excluded.

The payload sends TERM at590s; the outer job owns600s AFTER admission, with separate
startup and teardown grace. These are compatible NOMINAL child/job budgets, not
a proved strict590s hard kill: timeout has no --kill-after and campaign monitor can
stop after600 plus cleanup (3s process-group grace; scope/service guards also have
grace). To guarantee GPU child lifetime below10minutes, use a controller-only
--kill-after=5s with TERM590 (<=595 nominal child seconds) and retain outer600s.
Do not relabel cleanup-inclusive wall time as590s. Timeouts/nonzero exits are
failures with retained pre-captures, not parity results.

Receipt currently binds seven candidate files and nine training sources, not the
controller wrapper/run_gpu.sh/scripts/limited/campaign_job command. Campaign
source_fingerprints records common runtime scripts but does not include nested
round4/controller_gpu_payload.sh. Bind its current hash and exact timeout/resource
CLI in a controller-owned runtime manifest/job attachment (or reviewed receipt
extension); retain Python candidate hashes. Approval identity/queue_finished are
trusted controller assertions, not independent live queue evidence in the runner.
Before any receipt the controller MUST inspect terminal actual units, job records
and queue state after fold3 drain, then let exclusive admission recheck at launch.
This review neither inspected nor attested that live drain; no receipt exists here.

GPU measurement gates after remediation
---------------------------------------
Synthetic actual CUDA architecture FIRST; only a passing source-bound DEFAULT
online/replay plus uninstrumented-original comparison permits real-panel assembly.
Require the same default/skip gates on the controlled real panel. Failed state,
source, memory, timeout or backend parity must block interpretation and preserve
failure records. No aligned/permuted training arm is approved. Four-term AMP
decomposition may have different summation rounding; do not conflate its separate
oracle tolerance with exact default step reproduction. CPU25/zero deltas establish
neither CUDA parity nor mature/full-fit state nor score gains.

Protected artifacts and deliverables
------------------------------------
verify_protected.py verified17 author-protected entries,54 author artifact hashes,
17 blind artifacts,12 lineage artifacts and9 source bindings: ALL MATCH. Full
expected/actual hashes are in protected_verification.json and its log; reverify
at review freeze. Author files, blind/lineage artifacts and controller integration
files were not edited. Review-owned runnable adversaries.py, results.json,
cpu_capture/pre.pt/post.pt, source/resource logs, TEST_PLAN.json, STATE.json,
REPORT.md, runtime hashes and commands preserve these narrow findings. No own
GPU candidate approval, receipt write, queue mutation or live source edit.
