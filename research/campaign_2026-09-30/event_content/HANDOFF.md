# D1 numerical v4 implementation — CPU grades pass

Model SHA256: `75beff387c4f9312dc3bc7e065d76e7df2ceeb666fe1bb3e93cc245dfca445cd`.
Preregistration v4 SHA256: `9f3325206d41a9fe4118aada9b967b6fc574e24a97af81a86d48cd77e5ce5a61`.

All 32 actor cases pass: durable exit 0, 139.744 seconds, peak RSS 792.13 MiB,
CUDA uninitialized, one CPU, hard 2 GiB, shared campaign slice, 300-second cap.
The model has 121,687 trainable parameters at input 211/output 23 and 2,664 state
bytes per sequence. CPU batch 2/chunk 2000 took 0.165 seconds forward and 0.383
seconds backward. These measurements do not establish GPU cost.

## Preserved failure and numerical revision

Independent v1 grading failed long 20,000-row state parity: content/raw-code
0.0061264 and position 0.0129395 exceeded the unchanged 0.003 tolerance. The original
actor short checks were insufficient. Sources, proofs, patch, manifest and critic
failure are preserved under revisions/. No period, gate, dimension, parameter map
or tolerance was retuned.

V4 uses the same stored FP32 base decay coefficient in streaming and nonnegative
integer scan powers. S/Z, write injections and denominator floors are scaled by
stored `c = 1 - base_alpha`. Exact real normalized attention, including floor
behavior, remains unchanged. New base_decay/norm_scale buffers and scaled S/Z
contents require strict checkpoint and native representation handling. Read
numerical_amendment_v4.md. The six-field state layout and byte size stay the same.

Stronger actor checks compare all arms against literal sequential recurrence over
20,000 rows, both quiet after one actual event and eventful on every row. Seed 534
was fixed before checks. Maximum long output difference is 6.892e-8 and state
difference 7.20024e-5. Independent critic v2 grading passes 25/25 cases at unchanged tolerances, including
long quiet/eventful paths and exact real floor/ratio equivalence near 1e-16.
See ../audit/event_content_grade_v2.json. This is independent implementation
evidence, not an actor performance promotion.

## Integration and exposure

The regenerated integration patch is unapplied; syntax and git apply --check pass.
No live trainer, model, SSM, ledger or scheduler file changed. The builder repeated integration CPU proof with the amended source and passed
all proposed families/default loss/gradient/inherited-CLI checks. See
../integration/cpu_proof_v2.json.

exposure/ freezes 16 unique TRAIN components selected from fold 0 allowed fitting
groups using metadata alone, disjoint from components containing any of the prior
64 fitting/32 test groups. After both CPU proofs passed, feature-only exposure executed with recorded
authorization. All selected components pass queried/written rank 96 and at least
128 writes under each event clock. Counts range 178–6579 endpoint and 337–10127
raw-code. Unknown-class rows (8454/320000) remain in the gate. Read exposure/REPORT.md
for initial empty histories, origins, rare quiet endpoint and modulo alias coverage.
The existing TRAIN class table is diagnostic only and never reaches D1. Fresh
independent raw replay is requested; no labels/masks/VALID/GPU or scores were read.

## Unlaunched GPU profiler

The runnable gpu_smoke.py was prepared, linted and parsed without executing GPU
APIs. Three exclusive arm CLI values share student/FastEMA/fused AdamW/AMP and ten
chunks at batch 32, sequence 20,000, chunk 2000, input 211/output 23. Raw features,
causal event inputs and synthetic loss arrays remain resident. It records allocator
peaks, sampled total device memory with a 5e9-byte cap, attempted/successful updates,
AMP skips and EMA steps. Final weights, exact synthetic inputs and reset FP32/AMP
readouts/states are saved for later independent parity. Precision switches remain
at inherited process defaults; earlier unlaunched TF32-high assumption is corrected
and archived. Primary CPU parity tolerances stay 3e-5 output/.003 floating states,
with counters/raw prices exact. Root owns launch after CPU grades, all four controls and
an equal cost/budget amendment. No GPU fitting is authorized by this handoff.

Passing CPU/exposure gates establishes implementation and support evidence only.
Competitive quality, physical quote/time/source identity, native runtime and public
score remain unknown. Endpoint/raw-code changes both event writes and decay;
content/position compares content with a periodic 128-position control.
