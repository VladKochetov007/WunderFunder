# Partial common-macrostep credit plan — NOT EXECUTED

User STOP/WRAP takes precedence. This is a partial design saved for later controller
review, not an implemented instrument, preregistered scientific experiment or launch.
No code, schema, tests, forwards, training, actual capture read or GPU validation ran.

Inspected before stop:
- round6/state_drift_next_options.md, prospective option3.
- round5/credit_control/check.py, its source AST extraction, common loss/forward/step
  helpers and the existing58-check foundation. Its controls were NOT rerun.
- round4/live_capture_v4/live_capture.py, warm capture/restore and source chunk helpers.
- scripts/train_rnn.py, original four-loss/update branches and one-attempt clock order.
No live source or previously frozen artifact was edited.

Intended contrast, still unimplemented:
Start from exactly the same genuine warm student parameters, incoming student h,
laggedEMA parameters AND incoming h_ema, optimizer named moments, scheduler/scaler,
RNG, batch, student normalization and source-bound teacher/auxiliary targets.
Both arms perform two unchanged-weight1000-row student calls and two EMA calls.
Only student midpoint detach differs. Pool one original four-loss functional over
4000:6000, then one clip/optimizer/scheduler/EMA attempt. No fresh/stale-h factor,
scoring/full fit or per-component optimizer attribution.

Required identities before any scientific gradient interpretation:
Forward outputs, EMA outputs, stopped blended targets, clipped-label weights,
loss coefficients, branch masses and total loss must match exactly across arms.
Common denominators are over the complete macrostep, not a mean of independently
normalized half-losses. Retain actual main clipped-|y| weight and strict bootstrap8,
source structural/next heads and coefficients, and singleton-row-mass default EMA
broadcast semantics. Preserve all attempt/skip/optimizer/scheduler/EMA clocks.
Incoming h must be the shared warm state, never silently recomputed/reset.

Unfinished items for a later separately authorized continuation:
1. Freeze executable prereg/schema with exact supported source/recipe/state bindings.
2. Implement the minimal source-faithful pooled loss/macrostep adapter and CPU-only
   boundary guards before restore/privateCUDA validation or any capture access.
3. Implement genuine warm tiny source controls, scalar/fixed-weight null and planted
   cross-boundary credit, with positive second-half and first-half-only null losses.
4. Verify source coefficients/target/normalizer arithmetic independently; reject
   half-loss averaging and ensure one common update attempt per arm.
5. Execute tiny CPU controls only after authorization resumes; preserve failures.
6. Independently review the frozen prototype. Actual B32,2x1000 connected graph VRAM
   is UNKNOWN; a single-query2GB cost is insufficient. No silent smaller-batch or
   shorter-graph substitute. CUDA/real payload access requires controller admission,
   independent PASS and a sufficient absolute completion/deadline guard.

No source AST rewrite, trainer integration, launch command or memory estimate was
implemented. The22:45 prototype target was superseded by the user's stop. Existing
v4, landed index, month options and completed ARTBP oracle remain frozen separately.
Controller owns staging/commits; actor will not stage or commit.
