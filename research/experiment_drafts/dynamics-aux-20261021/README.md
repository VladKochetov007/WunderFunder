# Future-input dynamics auxiliary screen

**Status:** code implemented on `experiment/dynamics-aux-20261021`; four focused
CPU tests pass; static Ruff and training-structure checks pass. No sustained
auxiliary profile or scientific fit has run. This packet is cold-screen only.
A strong-parent continuation would need a separate conditional extension; the
current CLI rejects `--init` for this head.

Implementation source: `b83147cf3264df68d6fe77b827fcedbca0be6a50`, based on
`3eeb2a9c2a0f456112f76175122deab376c764b3`.

The accepted [October 21 local/Kaggle plan](../../../specs/plans/2026-10-21-next-local-kaggle-cycle.md)
allocates four new cold fits: auxiliary-loss weight 0 and 0.05 on matched
F0/seed1001 and F2/seed1002 cells. Both arms allocate the same 52-output head
from the same initialization seed. The shared backbone is the small i1 screen's
H512×1 GRU with all 367 `i1_history` slots allocated and zeroed by
`--research-feature-control all`; `--drop-i1` stays false. Training uses three
epochs and fixed zero-based CP2, AdamW/OneCycle, batch 8, TBPTT chunk 2,000,
and unchanged WMSE.

The mechanism teaches each recurrent state to summarize future changes in the
raw input stream. `wnn.kalman_group_features.FEATURE_GROUPS` supplies the 13
source-qualified raw groups spanning channels 0:112. For each group and
horizon `h ∈ {16, 256}`, compute `z=(raw-model.mean[:112])/model.std[:112]`, then
`mean(d)` and `log(mean(d²)+1e-8)` for `d[k]=z[t+k]-z[t+k-1]`, `k=1..h`.
The registry-group order, followed by those two summaries, yields 26 outputs
per horizon and 52 total. Target origin `t` is valid only when `t >= warmup`
(the existing row-valid mask, normally 99) and `t+h < sequence_length`; the
future raw rows may cross a TBPTT chunk boundary. Targets come from complete,
unthinned feature-only sequence reads. They never enter model inputs. Any
retained-mass thinning mask applies to target origins; targets still use raw
future observations.

Each output is standardized by its mean and population standard deviation over
the exact unique selected fitting sequences; the standard deviation floor is
`1e-3`. Statistics are computed with `SequenceFile.read(targets=False,
mask=False)`, so training labels and scoring masks are never loaded. In each
training window, the auxiliary objective is mean Huber loss (`delta=1`) across
all valid horizon/group/summary cells. The total loss is unchanged WMSE plus
`weight × auxiliary_loss`. Weight zero retains the same initialized head and
target-statistics pass but contributes no auxiliary gradient. Evaluation emits
only the unchanged two WMSE outputs.

Membership is frozen, not regenerated here. F0 reuses
`research/campaign_2026-10-20/plateau_review_r1/fit_f0_10pct_s1001.json`
(SHA-256 `706181dbb9a096653b9c63af58f8f66b70ffea9b3f522a3467ff3fd6f720d313`,
subset ID `E-PLATEAU-I1-20261020-R1`). F2 uses
`research/campaign_2026-10-21/next_cycle_r1/i1/fit_f2_10pct_s1002.json`
(SHA-256 `841576ddd469befa98b37d1a567e47462979f298314b2a4cf4a399beb5ddc908`,
subset ID `E-PLATEAU-I1-20261021-R1`). The F2 manifest binds split SHA-256
`a9973e5798d3109362571c299b2d08eca71d01e92008da39b0d0ddde54c93aa9` and
315/3,145 components (1,148 train and 141 valid fitting sequences); its
evaluation has 443 groups. F0 binds split SHA-256
`09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d` and has
1,118 train plus 118 valid fitting sequences; its evaluation has 468 groups.
The F2 membership file is frozen in sibling branch
`experiment/i1-screen-20261021`, commit
`b1b680d4a700d353f3e6db4f761f6670bd02fc7f`; root must integrate that exact
artifact before running F2. Do not regenerate it.
No model fit is reused. There are two new fits per arm, four total, and zero
teacher or student fits.

[recipe.json](recipe.json) carries the four exact fit commands and full-fold
prediction/official-score commands. [PR.md](PR.md) is the review handoff.
Prediction comparison must retain each paired F0/F2 cell and the fixed 90/10
control/candidate mixture; two fold/seed cells cannot separate fold
from seed variation or establish significance. All folds are exposed, and no
public transfer claim follows from this screen. The practical `+0.0003` mean
official improvement is a follow-up target, not an automatic stop rule.

The auxiliary head and its target-normalization buffers remain in the training
checkpoint for restoration and provenance. `scripts/package_submission.py` and
`scripts/export_native_weights.py` remove the auxiliary head and buffers before
serving export. Improved auxiliary prediction alone does not earn follow-up;
target quality and fixed complete-endpoint contribution must support it. Any
strong-parent continuation requires a new recipe and explicit compatibility
work before it can be run.

The focused CPU command `uv run pytest tests/test_dynamics_auxiliary.py -q`
passed all four tests in2.35s under root's serialized 1-core/1.8-GB CPU slot,
with `PYTHONPATH` pointed at this worktree's `src`. Ruff check and
`scripts/check_training_structure.py` also pass. No model-quality, runtime,
B300 profile, parity, or export-grade result is claimed here.
