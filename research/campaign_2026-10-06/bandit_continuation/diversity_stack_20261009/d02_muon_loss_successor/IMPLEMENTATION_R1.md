# D02 isolated source successor

This implements the equal-LR, recurrent-only, gatewise Muon arm frozen in
`RECIPE_R1.md`; it is not a best-tuned Muon comparison. Two new WMSE/Huber cells
reuse the completed matched AdamW controls, once their corrected producer exists.
No provider, local GPU, device profile or production fit has run from this source.

The initial source copy was checked against all 61 entries of GPU R2's frozen
source manifest at commit `8924deec1eaf1a50cb31d4c42e042d24093b4be6`. Changes are
isolated to this directory. Top-level trainers, the TPU closure and current jobs
are unchanged. AdamW construction and math remain unchanged; focused tests compare
its actual updates against native Torch AdamW exactly.

Muon now explicitly orthogonalizes in FP32 on all devices and records matrix
precision separately from model/autocast precision. It retains the existing
five-step polynomial, per-gate geometry, Nesterov update, RMS scaling and equal
group peak LR0.001. The source exposes `--muon-ortho-dtype-policy fp32`; other
policies fail closed. The model remains FP32/TF32 off. Both Muon/AdamW groups must
match the original control's LR/beta1 at all 220 attempted updates.

Before updates, the trainer checks exact pre-fit tensor keys/dtypes/shapes/values,
normalization, all epoch/source IDs, the raw first-batch inputs/labels/mask and
pre-update schedule against the completed, root-pinned AdamW contract. NPZ
container timestamps are not mistaken for tensor identity. Control files are
read-only producer artifacts; no AdamW-trained checkpoint initializes D02.

The existing GPU worker supplies its resource guards, finite deadline, cache and
auxiliary-asset validation, checkpoint/prediction collection and per-update
unscaled gradients/LR/beta/clip/skip diagnostics. The D02 discarded T4 gate performs
one complete B32/C2000/GRU256×2 batch (ten updates) for each loss, requires finite
loss/gradients and records explicit precision, exact control bindings and actual
model/optimizer/scheduler reload plus state-reset, prefix and carried-state checks.
Diagnostics and gate checkpoints are retained. Production fit remains conditional
on embedded root admission; profile-only admission returns a separate receipt.

The operator diagnosed GPU v2's actual terminal failure as the trainer's argparse
error `profiling/parity early exits cannot save epoch continuation state`.
The common command had `--save-training-state` even for `--profile-steps`.
D02 carries the concrete repair: omit that flag only from disposable profile
commands and retain it for both full fits. No GPU v2 profiles or fit updates
completed. The failed packet is execution evidence, not a negative loss result.
Its immutable scientific source remains the starting point; the final control
kernel/source identity must bind the corrected R3 producer before launch.

Current bounded CPU evidence (1 CPU, 1.5GB, 120 seconds per check): five tiny
real-optimizer/reload/schedule/precision tests, two actual pre-fit/profile exporter
tests and two control-binding/mutation tests passed. The exporter fixture uses
synthetic binaries, not actual data or accelerator parity. The unchanged large
auxiliary-asset validation is covered by root's prior actual mount checks and is
not rerun as part of these tiny support tests. Actual T4 claims remain pending.

`seal_d02_candidate.py` reuses the existing GPU packet/asset/bootstrap approach,
with a unique title-derived ID and hash-bound source archive. An embedded root
record must bind the exact code manifest and completed control contract before
runtime admission. Its pending form permits no execution. Root also owns source
asset creation, provider quota, dispatch and independent result review.
