# From activation pilots to a useful predictor

The public anchor is the complete seed4 combo at0.6660. These activation
studies do not establish a0.7000 public expectation. Their models use24 fitting
components, one epoch and much smaller capacity than the submitted model.
Their16 evaluation components are already exposed. A large pilot delta can
reflect faster early optimization rather than a higher final forecasting limit.

Keep negative and seed-dependent cells. The candidate-shape, embedding,
mixed-readout, optimizer and schedule mechanisms have separate evidence;
none is a reason to close an entire architecture family.

Before a full-data activation submission:

1. Finish the complete registered matrices and independent export checks.
   Preserve fixed/learned initial-shape controls, the matched linear addition
   and the matched-center cubic. Candidate bounds only describe a scalar map.
2. Freeze one transfer recipe and test it against an equally trained complete
   strong recipe, including teacher targets, EMA and the same final head
   procedure. Compare both target correlations and class slices diagnostically;
   global WP is not a weighted average of class correlations.
3. Confirm on multiple matched seeds/folds without selecting epochs on the
   evaluation set. Report training-seed variation separately from conditional
   component-bootstrap intervals. Use genuine component separation for any
   newly constructed cohort; label previously explored folds honestly.
4. Measure throughput and resource use before committing full-data fits. A
   scripted candidate correction replaces the optimized cuDNN recurrent path;
   the pilot's small VRAM peak is not a full-data memory or runtime guarantee.
   A JAX/compiled scan is a separate implementation requiring forward,
   gradient and optimizer-step parity before scientific comparisons.
5. Benchmark the complete deterministic native predictor on one CPU. Sin/cos
   calls have a different deployment cost from a polynomial or lookup map.
   Any approximation is a distinct predictor until score and parity are tested.
   Check the actual archive and ELF dependencies in the offline scorer image
   before calling a package ready. Do not repeat the libpython3.13 failure.

Optimizer parameters are another controlled axis. In the current MuonAdamW
implementation, changing the shared first beta also changes AdamW's first
moment for non-recurrent weights. A recurrent-momentum experiment must isolate
that parameter in the Muon groups or explicitly admit both changes. Matching
LR sums alone does not match effective update area; fixed telemetry describes
this discrepancy without tuning on exposed scores.

Learned per-layer scalars, causal context-dependent shape routing, fixed
parallel activation channels and activation-aware compression are different
methods. Dynamic routing must use only information available at the current
row. Keep batch-statistic/future-row effects out of a streaming activation.

Current admission remains within the existing device-hour budget, one local
GPU worker, actual aggregate12CPU/30decimalGB RAM/no swap, and observed VRAM
headroom. New admissions close07:00UTC/10:00Kyiv; admitted jobs then drain.
Neither a tiny probe nor a failed grader is counted as a scientific full fit.
