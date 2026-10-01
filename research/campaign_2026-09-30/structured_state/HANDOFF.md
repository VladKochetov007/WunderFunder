# Frozen Card B implementation handoff

Final source SHA256:
`227a5b8fe8ada7b34d61463ad2f022a0d802011aa985b215ef880d5a469c1756`.
Integration is unapplied, GPU runner unlaunched, and scorer/launcher/native/live
trainer/model files unchanged. This actor read no competition data or scores.
The shared dirty worktree is preserved: base Git identity plus content hashes
and source snapshots identify the implementation, rather than a new Git commit.

Independent primary evidence: `../audit/structured_state_grade_v2.json` passes
all8cases with unchanged numeric tolerances. Its independently written recurrence
does not call actor recurrence helpers. It covers matched controls, FP64 forward
and all gradients, FP32 partitions/stream, prefix causality, reset/batch isolation,
autocast FP32 state and long20k half-life8192 impulse. Long stream/FFT error is
4.26e-6, maximum tolerance ratio0.213, analytical serialized-A error<1.2e-7.

Original failure is preserved at `../audit/structured_state_grade_v1.json` and
old code in `revisions/`. Arithmetic amendment v4 fixes unrounded FFT powers
versus rounded real recurrence coefficients. The revised analytical oracle uses
the serialized FP32A, with unchanged tolerances; ideal continuous-radius deviation
1.282e-4 at20k remains diagnostic. Native must serialize those same real A values.
Original CPU smoke v1 also initialized CUDA via CPU optimizer health-check;
subsequent enforced CPU smokes hide GPUs and verify CUDA stays uninitialized.
Both original and corrected synthetic checkpoints are preserved with exact SHA.

## Actor checks and bounded CPU profile

`smoke_results_v5.json` passes all actor checks. FP64 parameter-gradient error
8.33e-17, FP32 parameter-gradient error6.71e-8. Chunk1/17/64/2000 outputs within
3.28e-7; carried state within1.44e-5. Future perturbation max2.98e-7. Reset and
outer-autocast comparison exact. Long quiet state finite, constant-input bounds
not violated. Actor's stricter long impulse stream/FFT error6.79e-6, ratio0.160.

Actual recipe dimension input211/output23:127511parameters,126912denseMAC/row
(FFT/activations excluded),1024FP32 recurrent state bytes/sequence. Arms have
identical counts/operations and phase buffers; only raw decay initialization
differs. The scientific comparison remains initialization/optimization.

CPU batch2/chunk2000 measured forward0.061–0.089s, backward+AdamW0.096–0.114s,
full20k forward0.831s, on the loaded shared host. Entire actor smoke16.76s,
peak RSS908.2MiB, one thread, campaign slice, MemoryMax2GiB, no GPU initialization.
These are CPU engineering measurements, not GPU-fit or native deployment claims.
Batch32 two-block state timelines alone are62.5MiB; scalar+rotation FFT spectra
are64.016MiB/block/tensor before graph copies. Total GPU memory remains unmeasured.

`integration_cpu_check_v2.json` passes proposed factory, strict checkpoint loading,
actual trainer FastEMA/forward_chunk/flat detach, inherited single-init metadata,
and exactly four trainable head tensors under freeze `inp,blocks`. Legacy `inp,gru`
is rejected. Integration patch dry-run and ruff pass. Read README for metadata,
prediction state/reset, FT freeze, reproduction commands and native limitations.

## Root-owned next steps

Freeze remains closed until a graded mechanism or controller instruction requires
a follow-up. Root finishes controls, then owns synthetic GPU runs and paired fits.
GPU smoke IDs are `gpu_smoke_multi` and `gpu_smoke_single`, as preregistered;
runner writes corresponding JSON/PT and root durable jobs capture logs/exits/VRAM.
Batch32/full20k/chunk2000/output23 includes actual student/EMA shapes, outer
FP16+GradScaler and inner FP32, all10 updates. Synthetic core profiling excludes
loader/feature-construction intermediates, so root's observed total-device monitor
and real-batch checks remain mandatory. No special batch/accumulation workaround.

Paired fit currently retains1080GPU-second cap/arm, fixed4epochs/epoch3, no eval
during fit, same clean fold0 recipe and fixed80/20 blends against the completed
FT pair. If either arm cannot finish all updates, no mechanism comparison is valid.
Root may preregister an equal1440 cap based on cost measurement **before** fits;
the existing cards/cap record are retained. Future FT of both arms must be separately
predeclared before evaluation and use the explicit `inp,blocks` head freeze.

Numerical implementation validity is independently supported. Competitive quality,
long-gap gain, retained slow modes, GPU fit cost/memory, native parity/runtime,
confirmation and expected-public >=0.6660 are still unknown. No promotion claimed.
