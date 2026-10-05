# Blind specification critique: E487 Muon precision diagnostic

Reviewed `E487_DESIGN.md` at SHA-256 `c937a9d6772f70cd56d83e7c9d6bfdde47a52c0e4bb9c66b0ca05a4e29fe8bd6` before inspecting any new
implementation. The question is properly narrow: hold the captured native and
custom gradients, model bytes, optimizer buffers, clipping, and groups fixed;
vary only the Newton--Schulz intermediate dtype; test whether the observed
one-step discrepancy changes. This can explain one numerical mismatch. It
cannot establish that FP32 orthogonalization improves training or prediction,
and it must not reopen or reinterpret the failed E485 admission.

The central control is the 2×2 cross-replay. Capture one finite gradient
snapshot from the native recurrence and one from the custom alpha-zero scan,
then replay each snapshot through BF16 and FP32 orthogonalization from the
same parameter bytes and fresh zero optimizer state. For every fixed gradient
source, the clipped gradient tensor bytes must be identical across the two
precision variants. Save and compare hashes of the pre-step state, all raw
gradients, clipped gradients, and optimizer buffers so a changed gradient or
state cannot masquerade as an orthogonalizer effect. No forward or backward
pass may occur between capture and replay.

Fresh zero momentum is the right control for this first-update attribution,
but it removes dependence on any trained Muon history. Results must therefore
be described as a zero-buffer one-step diagnostic; they do not establish how
BF16 and FP32 differ after momentum has accumulated during a trajectory.

For each variant, verify the four Muon tensors independently:
`gru.weight_ih_l0`, `gru.weight_hh_l0`, `gru.weight_ih_l1`, and
`gru.weight_hh_l1`, each split into three gate blocks. Define the AdamW groups
explicitly and report them separately for input projection, recurrent biases,
and head parameters. Do not aggregate a failing Muon matrix into an apparently
passing model-wide RMS. For each tensor/family report reference delta RMS,
error delta RMS, their ratio, and max absolute delta error; apply the already
registered strict gates independently to each matrix and AdamW family. Record
the unclipped norm and post-clip norm for both gradient sources and require
clipping to activate, otherwise the registered clipping path has not been
tested.

The shared-gradient control should use deterministic, non-collinear gradients
on every parameter and exceed global norm 1.0. Run identical gradient bytes
through cloned native and custom parameter sets under each precision path.
Require byte-identical parameters and optimizer states between the duplicate
models. Compare BF16 directly with the frozen E474 helper. For FP32, compare
the new helper with the existing CPU FP32 orthogonalizer on identical synthetic
matrices. Because CPU and CUDA matrix kernels can differ in reduction order,
the design should state a fixed CPU/GPU parity tolerance before any GPU result
is observed; exact equality is appropriate only for same-device duplicate
replays.

Interpret the crossed replay as two separate comparisons: BF16-versus-FP32
within a fixed captured gradient source estimates the effect of the
orthogonalizer arithmetic for that source; native-versus-custom gradient
sources within one precision setting measures the update consequence of their
gradient difference. Do not attribute a difference across both changing axes
to precision alone. The strict native/custom update gate is evaluated inside
each precision condition, and no bound may be widened after the replay.

The FP32 diagnostic must preserve the same Newton--Schulz polynomial, five
iterations, normalization, epsilon, momentum update, final FP32 cast, and
group/block layout as the frozen BF16 implementation. Change only the
intermediate orthogonalization dtype. Check the BF16 branch still delegates to
the frozen implementation without monkeypatching `src/wnn/optim.py`. Capture
all backend precision flags while the full-IEEE context is active; keep the
intentional BF16 CUDA orthogonalizer distinction explicit in the receipt.

Use only seed 271836's first canonical four fit roles and first 256 rows, with
fit-only normalization and target/mask loading. Verify the cohort order, raw
finite features and labels before clipping, exact state keys and H128
parameter bytes, train-mode native/custom forwards and output/state/input and
parameter gradients under the same IEEE context. The input state must be zero
for both recurrent implementations. A source/data/cohort identity mismatch,
nonfinite tensor, clipping mismatch, incomplete parameter coverage, changed
initial bytes, resource overage, or any strict update gate failure must stop
the diagnostic.

Before CUDA admission, require CPU synthetic tests for exact update-helper
controls, both gradient-source replays, the FP32-vs-CPU calculation and every
reported delta metric. The CUDA probe must use B4×256 at H128, configure the
allocator before model allocation, enforce 2-GiB allocator and
3,000,000,000-byte physical-device ceilings before and after each replay,
record teardown, and finish within the 600-second maximum. No full fit,
score, provider action, or test-role read is authorized by this design.

**Admission verdict:** suitable as a source-and-one-probe numerical diagnostic
if the CPU/GPU FP32 parity tolerance and exact optimizer-replay invariants are
registered before implementation freeze. An FP32 pass only supports the
limited attribution stated in the design; it does not admit E485 or authorize
FP32 Muon training.
