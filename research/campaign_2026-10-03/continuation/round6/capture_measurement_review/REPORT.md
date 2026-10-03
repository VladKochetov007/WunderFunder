Independent capture measurement review — narrow measured-parity PASS
====================================================================

The recorded v4 synthetic actual-architecture and fixed real-panel warm
unchanged/default trajectories support the claimed measured parity. No blocking
artifact or metadata issue was found. This review authenticated the completed
measurement; it did not independently replay GRU/CUDA, validate private CUDA
engines, authorize scientist arms or infer scientific component bounds.

Bytes, lineage and actual job
----------------------------
Exactly one full output-byte pass streamed40 indexed files,2,380,024,182bytes,
in1MiB chunks. Every SHA256 matches; no missing/unindexed capture appeared.
output_byte_verification.json retains each hash/size/mtime. Receipt candidate9,
source9, current runtime4, generated artifacts, panel identity and prior critic
bindings match. Relevant source hashes also match the actual monitor job snapshot.
The job command names the exact controller receipt and fresh output directory;
receipt predates admission. Current wrapper hash is4d54fc09... selecting v4.
The output index's inherited textual 'v3' lineage is historical; schema4,
candidate and runtime hashes establish the executed version.

c20261003_capture_v4_default_2120r retained status finished, exit0/no failure,
31.285s and peak observed VRAM3,709,861,888bytes, below5.5decimalGB. Job metadata
records the8core/24GiB/zero-swap GPU slice and600s outer budget. Completion precedes
the23:00Kyiv deadline. The controller receipt and runtime bind TERM585/KILL5.
Current watched guards are healthy, barriers absent and active GPU list empty at
21:31:39Kyiv. Kernel GPU-family oom/oom_kill counters are0; parent counters remain
the previously reported11. Historical RAM peaks are not attributed to this job.
Transient units now show not-found/inactive: their default Result field alone is
not completion proof; retained job metadata/log and receipt supply that evidence.

Numeric policies and order
--------------------------
Both result JSONs have finite numeric summaries and zero normal/default/repeat/
skip maxima, including every recorded per-field delta. Four frozen policies each
bind SIX original/default sample hashes to verified artifacts:24samples total.
All empirical floors and limits are0. Fixed ceilings1e-5 absolute/1e-4 relative,
multiplier2 and no automatic relaxation match the frozen procedure.

Bound source passes PRESTATE ONLY to original calibration, writes its floor, then
compares online/default results. Both normal and skip source order is checked;
sample mtimes precede floor files and floor mtimes precede final results. This
is source-plus-artifact evidence, not an independent comparison-event timestamp
trace. The measured zero floor is specific to this warmed backend/trajectory;
it is not a universal cold-cache or arbitrary-input determinism guarantee.

CPU mmap-only state inspection
------------------------------
Four authenticated pre/post pairs were loaded with mmap=True/map_location=CPU,
sequentially, with no actor imports, model construction, private engine validation
or replay. Peak RSS545232KiB; metadata/small-target checks took0.176s inside Python.
GRU256x2/23heads, Muon rnn-scope, fp16/fast, OneCycle, enabled scaler, fixed EMA,
distillation/boot horizon and both auxiliary weights match the supported recipe.
Incoming raw and EMA hidden states are detached CPU-stored FP16 shape2x32x256;
weights/moments are FP32. Outputs are FP16 shape32x1000x23; stopped targets are
FP32 shape32x1000x2 and exactly match the clipped-label/teacher blend with the
source's8-row alive-path teacher bootstrap. No full input arrays were copied.

Every incoming state is warm after FOUR actual attempts/updates: step_count,
attempts, EMA clock/n and native Adam steps are4. Normal post advances to5.
Forced skip increments skips0->1 and backoffs the scaler; Adam steps remain4,
while the source-defined attempt/EMA/scheduler clocks advance. Inf clipping norm
on deliberately nonfinite skip gradients is expected, not a JSON integrity fault.
Recorded backend is Torch2.14.0+cu130/CUDA13/cuDNN92400, cudnn_deterministic=true,
benchmark=false, deterministic_algorithms=false; the zero floors are measured,
not assumed from a blanket backend label.

Panel and limits
----------------
Captured real rows/teacher bindings match the frozen28TRAIN+4fitVALID panel and
provenance/split identities. Permitted membership and VALID evaluation exclusion
are checked; no new graph-component or historical teacher-lineage guarantee is
inferred. Small sidecars/stat identities were rechecked. Full external target/
prediction cache hashes are enforced by the hash-bound successful real entry;
this reviewer did not reread those large caches or the raw datasets.

Fresh seed3/panel normalization and four warm chunks do not reconstruct today's
mature full-fit state. Parity does not establish estimator benefit, causal quote
teachers, public score, or any scientific gradient/component bound.

The initial metadata audit used a nonexistent evaluation.train key. That local
inspection error is retained in byte_audit.log; the byte pass had already finished
and was saved. Corrected inspection resumed from authenticated hash/stat records
without a second full byte pass. Captures, receipts and earlier reviews remain
unchanged. Final STATE.json/HASHES.json bind this terminal bounded review.
